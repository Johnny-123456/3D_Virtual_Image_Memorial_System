#!/usr/bin/env python3
"""生成渲染资产：一帧 RGB + 对应米制深度图。

产物（assets/<name>/）：
    rgb.png     校正并裁切后的左目图
    depth.f32   float32 小端，W*H，单位米（0 = 无效）
    meta.json   宽高、焦点深度、深度范围

深度来源优先用单目 DA（稠密、边缘好），再用 config/mono_depth_calib.json
标定成米制；若标定缺失则退化为相对深度映射到 0.5~8 m。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera           # noqa: E402
from capture.mono_depth import MonocularDepth        # noqa: E402
from capture.rectify import CONFIG_PATH, Rectifier, load_depth_calib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "depth_anything_v2_small.rknn")
MONO_CFG = os.path.join(ROOT, "config", "mono_depth_calib.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成渲染资产")
    ap.add_argument("--name", default="demo")
    ap.add_argument("--frames", type=int, default=3, help="深度取多帧中位")
    ap.add_argument("--allow-no-subject", action="store_true",
                    help="跳过主体覆盖守卫（拍空景、或拿 MODNet 硬抠动物做实验时用）")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "assets", a.name)

    rect = Rectifier(CONFIG_PATH)
    x, y, w, h = rect.valid_roi
    mono = MonocularDepth(MODEL)

    rgbs, rels = [], []
    with StereoCamera() as cam:
        for _ in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            eL, _ = rect.rectify_pair(f.left, f.right)
            crop = eL[y:y + h, x:x + w].copy()
            rgbs.append(crop)
            rels.append(mono.infer(crop))
    mono.close()

    rgb = rgbs[len(rgbs) // 2]
    rel = np.median(np.stack(rels, 0), axis=0).astype(np.float32)

    # 相对深度 -> 米制
    if os.path.isfile(MONO_CFG):
        c = json.load(open(MONO_CFG, encoding="utf-8"))
        A, B = float(c["a"]), float(c["b"])
        depth = 1.0 / np.clip(A * rel + B, 1e-4, None)
        src = "mono_calibrated"
    else:
        lo, hi = np.percentile(rel, [2, 98])
        n = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, 1)
        depth = 0.5 + (1.0 - n) * 7.5          # rel 大=近
        src = "mono_relative"

    depth = np.clip(depth, 0.3, 20.0).astype(np.float32)

    # 焦点：中央 50%x65% 的中位深度
    ch, cw = depth.shape
    fx0, fx1 = int(cw * 0.25), int(cw * 0.75)
    fy0, fy1 = int(ch * 0.10), int(ch * 0.75)
    z_focus = float(np.median(depth[fy0:fy1, fx0:fx1]))

    # 采集真的拿到画面之后才建目录：读帧失败时不留下空资产目录
    existed = os.path.isdir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb)

    # 生成人像 matte（MODNet），供渲染器做精确前景合成
    matte = None
    try:
        from capture.matte import PortraitMatte
        mp = os.path.join(ROOT, "models", "modnet.rknn")
        if os.path.isfile(mp):
            print("  生成 matte ...")
            with PortraitMatte(mp) as pm:
                matte = pm.infer(rgb)
            cv2.imwrite(os.path.join(out_dir, "matte.png"),
                        (matte * 255).round().astype(np.uint8))
            print("  matte 覆盖 %.1f%%" % (float((matte > 0.5).mean()) * 100))
    except Exception as e:
        print("  matte 生成失败（不影响主流程）:", e)

    # 受采者检测：只认画面中央的那个人（直接 pb.detect() 在多人画面里会选中
    # 背景路人——实测 portrait3 检出 3 人，眨眼就贴到别人脸上了）。
    # 只推一次，守卫和眨眼贴图共用。
    det = pair = None
    try:
        from capture.landmarks import detect_main_person, eye_pair
        from capture.pose import load_pose
        pb = load_pose(imgsz=640, conf_threshold=0.4)
        try:
            det = detect_main_person(pb, rgb)
        finally:
            pb.release()
        pair = eye_pair(det)
    except Exception as e:
        print("  受采者检测不可用（姿态后端缺失？）:", e)

    # 守卫：画面里必须真的有一个够近的受采者。
    # 没有它的时候镜头前空无一人也会照常写出资产（blink.png 照样生成），
    # 渲染端拿到的是一张废图，首页还多一个点进去全黑的素材。
    #
    # 一开始想用 matte 覆盖率当门限，**不够**：实测空场景 MODNet 也会给到
    # 10.6%~13.4%（桌椅杂物），真人 25%+，区间重叠。真正能分开的是瞳距——
    # 空场景里姿态模型编出来的"双眼"瞳距只有 0.9px，而 1.2m 处的真人 76px。
    # 所以以瞳距 >=45px 为准（阈值与 capture_wizard.py / mono_calib.py 一致），
    # matte 覆盖率只作为辅助信息打出来。
    cov = 0.0 if matte is None else float((matte > 0.5).mean()) * 100.0
    iod = 0.0 if pair is None else float(pair[2])
    if iod < 45.0 and not a.allow_no_subject:
        print(f"\n✗ 画面中央没有够近的受采者（瞳距 {iod:.1f}px < 45px，"
              f"matte 覆盖 {cov:.1f}%），未生成资产")
        print("  请受采者站到镜头前 1.2~1.5m 处、位于画面中央后重试；"
              "只想拍空景/动物实验时加 --allow-no-subject")
        if not existed:
            shutil.rmtree(out_dir, ignore_errors=True)
        return 3

    # 眨眼贴图（复用上面的检测结果）
    if pair:
        try:
            from capture.blink import make_blink
            le, re_, inter, _mid = pair
            cv2.imwrite(os.path.join(out_dir, "blink.png"),
                        make_blink(rgb, [(le["x"], le["y"]),
                                         (re_["x"], re_["y"])], inter))
            print("  blink.png 已生成")
        except Exception as e:
            print("  眨眼贴图生成失败（不影响主流程）:", e)
    else:
        print("  没检测到成对的眼睛：跳过 blink.png（视差不受影响）")

    # 顺手生成背景板（去掉主体），供 DIBR 背景层使用，避免鬼影
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from make_bg import make_background
        bg = make_background(rgb, depth,
                             max(float(np.percentile(depth, 15)) * 1.8, 1.0),
                             matte=matte)
        cv2.imwrite(os.path.join(out_dir, "bg.png"), bg)
    except Exception as e:
        print("  背景板生成失败（不影响主流程）:", e)
    depth.tofile(os.path.join(out_dir, "depth.f32"))

    meta = {
        "width": int(cw),
        "height": int(ch),
        "z_focus": round(z_focus, 3),
        "z_min": round(float(np.percentile(depth, 2)), 3),
        "z_max": round(float(np.percentile(depth, 98)), 3),
        "depth_source": src,
        "created": cv2.getTickCount() and __import__("time").strftime("%Y-%m-%d %H:%M"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("资产已生成:", out_dir)
    print("  rgb.png   %dx%d" % (cw, ch))
    print("  depth     焦点 %.2f m  范围 %.2f ~ %.2f m  来源 %s"
          % (z_focus, meta["z_min"], meta["z_max"], src))
    print("  文件大小: rgb %.0f KB, depth %.0f KB"
          % (os.path.getsize(os.path.join(out_dir, "rgb.png")) / 1024,
             os.path.getsize(os.path.join(out_dir, "depth.f32")) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
