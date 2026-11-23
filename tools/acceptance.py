#!/usr/bin/env python3
"""验收测量：`docs/route-D-implementation.md` §8 里能离线量的两项。

为什么要有这个脚本
------------------
§8 的 6 条验收标准一直没有实测过，全靠"看起来还行"。这里把能离线量的两条
做成脚本，以后每次改渲染/追踪都能重跑一遍，而不是靠肉眼看。

    python3 tools/acceptance.py parallax [--asset portrait3] [--gain 40]
    python3 tools/acceptance.py latency  [--asset portrait3] [--seconds 20]

parallax
    用 `memorial_viewer --snapshot` 渲染 eye=±e 三帧，光流测出人脸（近景）
    与背景（远景）各自的位移，换算成"观众头动 ±15cm 时，人物相对背景移动
    多少屏幕像素"，与 §8 要求的 ≥8px 比对。
    背景层增益恒为 0（drawPass(0)），所以"相对位移"就是人脸自己的位移。

latency
    起 tracker(--stamp) + viewer，读 viewer 打印的 UDP 接收延迟，再叠加
    追踪器自报的"相机帧龄"与推理耗时，给出端到端分解。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "ui", "build", "memorial_viewer")
TRACKER = os.path.join(ROOT, "track", "eye_tracker.py")

# 相机内参（config/depth_calib.json）：fx 决定"横向 1 m 在画面里有多少像素"
FX_PX = 1198.7
CAM_W = 1280            # 追踪用左目图像宽度
SENS = 0.3              # eye_tracker 的 --sens 默认值（必须与那边一致）
DEFAULT_GAIN = 62.0     # dibrwidget.h 里 m_gain 的默认值（必须与那边一致）

# §8 第 1 条：头动 ±15 cm，人物相对背景反向位移 ≥ 8 px
CRITERIA_PARALLAX_PX = 8.0
# §8 第 2 条：端到端延迟 < 100 ms
CRITERIA_LATENCY_MS = 100.0


def eye_per_cm(dist_m: float, sens: float = SENS) -> float:
    """观众横向移动 1 cm 时，追踪器输出的 eye 增量。

    追踪器：eye_x = (nose_x - cx) / (sens * w/2)；相机：Δpx = fx * ΔX / d。
    """
    return (FX_PX * 0.01 / dist_m) / (sens * CAM_W / 2.0)


def _clean_env() -> dict:
    """给 Qt 子进程用的干净环境。

    pip 版 OpenCV 在被 import 时会往 os.environ 里塞 QT_QPA_PLATFORM_PLUGIN_PATH
    和 QT_QPA_FONTDIR，指向它自带的 Qt 插件目录。本进程一旦 import cv2，再起
    memorial_viewer 就会去加载 OpenCV 那套插件，直接 SIGABRT：
    "Could not load the Qt platform plugin xcb ... even though it was found"。
    """
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    env.pop("QT_QPA_FONTDIR", None)
    env.setdefault("DISPLAY", ":0")
    env.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    return env


def _render(asset: str, eye: float, out: str, look: int = 0,
            gain: float | None = None) -> bool:
    args = [VIEWER, "--asset", asset, "--snapshot", out,
            "--look", str(look), "--eye", f"{eye},0", "--port", "9973"]
    if gain is not None:
        args += ["--gain", str(gain)]
    # stdin 必须断开：Qt 程序从管道里继承了 SSH/heredoc 的 stdin 时会挂住
    r = subprocess.run(args, env=_clean_env(), timeout=90, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.isfile(out):
        print(f"  viewer 退出码 {r.returncode}，输出文件 "
              f"{'存在' if os.path.isfile(out) else '不存在'}")
        return False
    return True


def _segments(asset_dir: str, shape):
    """用资产自己的深度图分出"近景人像"和"远景背景"。"""
    meta = json.load(open(os.path.join(asset_dir, "meta.json"), encoding="utf-8"))
    w, h = meta["width"], meta["height"]
    depth = np.fromfile(os.path.join(asset_dir, "depth.f32"),
                        dtype=np.float32).reshape(h, w)
    near_z = meta["z_subject"] + 0.25
    far_z = meta["z_subject"] + 1.2
    return (depth < near_z), (depth > far_z)


def _flow_median(a, b, mask_half):
    """a→b 的位移，返回掩码区域内的中位水平位移（半分辨率像素）。

    用 goodFeaturesToTrack + LK 而不是稠密光流：视差量只有 1~3 px，稠密光流
    在低纹理的大片暗色衣服上会被量化到 0，中位数直接归零（实测踩过）。
    稀疏角点 + LK 亚像素精度才量得准这种小位移。
    """
    m = (mask_half.astype(np.uint8) * 255)
    pts = cv2.goodFeaturesToTrack(a, maxCorners=600, qualityLevel=0.01,
                                  minDistance=6, mask=m)
    if pts is None or len(pts) < 5:
        return 0.0
    p1, st, _err = cv2.calcOpticalFlowPyrLK(a, b, pts, None,
                                            winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (p1 is not None)
    if ok.sum() < 3:
        return 0.0
    d = (p1[ok, 0, 0] - pts[ok, 0, 0])
    return float(np.median(d))


def cmd_parallax(a) -> int:
    asset_dir = a.asset if os.path.isdir(a.asset) else os.path.join(ROOT, "assets", a.asset)
    if not os.path.isfile(os.path.join(asset_dir, "meta.json")):
        print(f"✗ 资产不存在: {asset_dir}")
        return 1
    if not os.path.isfile(VIEWER):
        print(f"✗ 未构建: {VIEWER}")
        return 1

    tmp = "/tmp/accept_par"
    os.makedirs(tmp, exist_ok=True)
    frames = {}
    for tag, e in (("neg", -a.eye), ("zero", 0.0), ("pos", a.eye)):
        p = os.path.join(tmp, f"{tag}.png")
        if not _render(asset_dir, e, p, gain=a.gain):
            print(f"✗ 渲染失败: eye={e}")
            return 1
        frames[tag] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)

    hh, hw = frames["zero"].shape[:2]
    near, far = _segments(asset_dir, (0, 0))
    near = cv2.resize(near.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST) > 0
    far = cv2.resize(far.astype(np.uint8), (hw, hh), interpolation=cv2.INTER_NEAREST) > 0

    # 负位移用负号折算成"eye 增大时往哪边走"，两次取平均压噪声
    d_sub = (_flow_median(frames["zero"], frames["pos"], near)
             - _flow_median(frames["zero"], frames["neg"], near)) / 2.0
    d_bg = (_flow_median(frames["zero"], frames["pos"], far)
            - _flow_median(frames["zero"], frames["neg"], far)) / 2.0

    # 快照是资产尺寸的一半，换算回屏幕像素（板端全屏时约 1:1 显示资产）
    scale = 2.0
    k_sub = d_sub * scale / a.eye          # px per unit eye
    k_bg = d_bg * scale / a.eye
    rel = abs(k_sub - k_bg)                # px per unit eye（相对位移）

    gain = a.gain if a.gain is not None else DEFAULT_GAIN
    print(f"资产 {os.path.basename(asset_dir)}   gain={gain:.0f}   "
          f"渲染 {hw*2}x{hh*2}（快照 1/2，已换算回屏幕像素）")
    print(f"  人脸位移 {k_sub:+.2f} px / 单位 eye")
    print(f"  背景位移 {k_bg:+.2f} px / 单位 eye（背景层增益 0，应≈0）")
    print()
    print("  头动 ±15 cm 时，人物相对背景位移：")
    ok_any = False
    for d in a.dist:
        e = eye_per_cm(d) * 15.0
        px = rel * e
        verdict = "达标" if px >= CRITERIA_PARALLAX_PX else "不达标"
        print(f"    观众距相机 {d:3.1f} m → eye={e:+.3f}  位移 {px:5.2f} px   {verdict}")
        ok_any = ok_any or px >= CRITERIA_PARALLAX_PX

    need = CRITERIA_PARALLAX_PX / (rel * eye_per_cm(a.dist[0]) * 15.0) * gain
    print()
    print(f"  §8 要求 ≥{CRITERIA_PARALLAX_PX:.0f} px（{a.dist[0]} m 处）: "
          f"{'✅ 达标' if ok_any else '❌ 不达标'}"
          f"   若要达标，gain 需 ≥ {need:.0f}（当前 {gain:.0f}）")
    return 0 if ok_any else 2


def cmd_latency(a) -> int:
    asset_dir = a.asset if os.path.isdir(a.asset) else os.path.join(ROOT, "assets", a.asset)
    port = 9971
    env = _clean_env()

    vlog = open("/tmp/accept_viewer.log", "w")
    tlog = open("/tmp/accept_tracker.log", "w")
    viewer = subprocess.Popen([VIEWER, "--asset", asset_dir, "--port", str(port)],
                              env=env, stdout=vlog, stderr=subprocess.STDOUT)
    time.sleep(2.0)
    tracker = subprocess.Popen([sys.executable, "-u", TRACKER,
                                "--port", str(port), "--stamp", "--print"],
                               env=env, stdout=tlog, stderr=subprocess.DEVNULL)
    time.sleep(a.seconds)
    for p in (tracker, viewer):
        p.terminate()
    for p in (tracker, viewer):
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    vlog.close()
    tlog.close()

    lat = [float(m) for m in re.findall(r"latency avg=([\d.]+) ms",
                                        open("/tmp/accept_viewer.log").read())]
    ages = [float(m) for m in re.findall(r"帧龄 (\d+)ms",
                                         open("/tmp/accept_tracker.log").read())]
    infers = [float(m) for m in re.findall(r"infer ([\d.]+)ms",
                                           open("/tmp/accept_tracker.log").read())]
    fps = [float(m) for m in re.findall(r"([\d.]+) fps",
                                        open("/tmp/accept_tracker.log").read())]
    if not lat:
        print("✗ 没收到带时间戳的数据包（追踪器没检测到人？相机被占用？）")
        print("  看 /tmp/accept_tracker.log 与 /tmp/accept_viewer.log")
        return 1

    cam = statistics.median(ages) if ages else float("nan")
    inf = statistics.median(infers) if infers else float("nan")
    net = statistics.median(lat)
    present = 8.0        # 60fps 下一次绘制平均 8ms（最坏 16ms）
    total = cam + net + present

    print(f"资产 {os.path.basename(asset_dir)}   追踪 {statistics.median(fps):.1f} fps"
          if fps else "")
    print("  端到端分解：")
    print(f"    相机帧龄（读到时这帧已经过了）   {cam:7.1f} ms   ← 12fps 消费 31fps 流的代价")
    print(f"    推理 + UDP + Qt 接收            {net:7.1f} ms   （其中推理 ~{inf:.0f} ms）")
    print(f"    渲染呈现（60fps，均值）          {present:7.1f} ms")
    print(f"    ────────────────────────────────────────")
    print(f"    合计                            {total:7.1f} ms   "
          f"§8 要求 <{CRITERIA_LATENCY_MS:.0f} ms → "
          f"{'✅ 达标' if total < CRITERIA_LATENCY_MS else '❌ 不达标'}")
    return 0 if total < CRITERIA_LATENCY_MS else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="§8 验收测量")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parallax", help="视差：头动 ±15cm 的位移 ≥8px")
    p.add_argument("--asset", default="portrait3")
    p.add_argument("--eye", type=float, default=0.4, help="渲染用的视点幅度")
    p.add_argument("--gain", type=float, default=None, help="覆盖视差增益（默认用程序内 40）")
    p.add_argument("--dist", type=float, nargs="+", default=[1.0, 1.2, 1.5, 2.0],
                   help="观众到相机的距离（米）")
    p.set_defaults(func=cmd_parallax)

    q = sub.add_parser("latency", help="延迟：端到端 <100ms")
    q.add_argument("--asset", default="portrait3")
    q.add_argument("--seconds", type=float, default=15.0)
    q.set_defaults(func=cmd_latency)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
