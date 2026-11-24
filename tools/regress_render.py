#!/usr/bin/env python3
"""渲染回归：拿改动前的二进制当基线，逐资产 × 逐参数渲染快照并逐像素比较。

为什么要有这个脚本
------------------
改渲染是"牵一发动全身"：给狗做的照片导入、视差归一、补边，都可能顺手把
人像改坏。肉眼看不出来，但用户能一眼看出来。所以每次动 dibrwidget/studio，
先跑一遍这个，用像素数据说话。

用法
----
    # 1) 取一个基线版本编出来（改动前的提交）
    git worktree add /tmp/base <commit>
    cd /tmp/base/ui && mkdir -p build && cd build && cmake .. && make -j4 memorial_viewer

    # 2) 跑回归
    python3 tools/regress_render.py --baseline /tmp/base/ui/build/memorial_viewer

    # 只跑当前版本做冒烟（不比较）
    python3 tools/regress_render.py

判据
----
同一窗口尺寸下，若只是"外围补边/着色器精度"级别的改动，中部差异应该肉眼为 0：
平均 < 1.0 灰阶、p99 差值 ≤ 24。超过就打印 <-- 中部差异偏大。

注意：动画（呼吸/点头）是逐帧的，同一版本连跑两次也有 p99=1 的抖动，判据留了余量。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "ui", "build", "memorial_viewer")

CASES = [
    ("eye0",  ["--eye", "0,0"]),
    ("eyeL",  ["--eye", "-0.6,0"]),
    ("eyeR",  ["--eye", "0.6,0"]),
    ("eyeUp", ["--eye", "0,0.5"]),
    ("gaze",  ["--eye", "0.4,0", "--gaze", "1"]),
    ("blink", ["--eye", "0,0", "--blink", "1"]),
    ("look1", ["--eye", "0,0", "--look", "1"]),
    ("look2", ["--eye", "0,0", "--look", "2"]),
]

MEAN_MAX = 1.0
P99_MAX = 24


def clean_env() -> dict:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    env.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    # pip 版 cv2 import 时会塞这两个变量，Qt 子进程加载 OpenCV 的插件目录会 abort
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    env.pop("QT_QPA_FONTDIR", None)
    return env


def render(binary: str, asset: str, tag: str, args, out: str, port: int) -> bool:
    cmd = [binary, "--asset", asset, "--snapshot", out, "--port", str(port)] + list(args)
    try:
        r = subprocess.run(cmd, env=clean_env(), capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return os.path.isfile(out) and "snapshot OK" in (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染回归对比")
    ap.add_argument("--baseline", default=None, help="基线版本的 memorial_viewer 路径")
    ap.add_argument("--out", default="/tmp/verify/reg", help="快照输出目录")
    ap.add_argument("--assets", default=None, help="只测这些资产（逗号分隔）")
    a = ap.parse_args()

    if not os.path.isfile(VIEWER):
        print("✗ 当前版本没编译：", VIEWER)
        return 1
    os.makedirs(a.out, exist_ok=True)

    base = os.path.join(ROOT, "assets")
    names = sorted(d for d in os.listdir(base)
                   if os.path.isfile(os.path.join(base, d, "meta.json")))
    if a.assets:
        names = [n for n in a.assets.split(",") if n in names]
    if not names:
        print("✗ assets/ 下没有可用资产")
        return 1

    print("资产:", ", ".join(names))
    print("基线:", a.baseline or "（无，只做冒烟）")
    port = 9100
    bad = []
    for name in names:
        asset = os.path.join(base, name)
        for tag, args in CASES:
            port += 1
            new_png = os.path.join(a.out, "%s_%s_new.png" % (name, tag))
            if not render(VIEWER, asset, tag, args, new_png, port):
                print("  %-10s %-6s 当前版本渲染失败" % (name, tag))
                bad.append((name, tag, "render"))
                continue
            if not a.baseline:
                img = cv2.imread(new_png)
                print("  %-10s %-6s %dx%d  均值 %.1f"
                      % (name, tag, img.shape[1], img.shape[0], img.mean()))
                continue

            port += 1
            old_png = os.path.join(a.out, "%s_%s_old.png" % (name, tag))
            if not render(a.baseline, asset, tag, args, old_png, port):
                print("  %-10s %-6s 基线渲染失败" % (name, tag))
                bad.append((name, tag, "baseline"))
                continue
            o = cv2.imread(old_png).astype(np.int16)
            n = cv2.imread(new_png).astype(np.int16)
            if o.shape != n.shape:
                print("  %-10s %-6s 尺寸不同 %s vs %s" % (name, tag, o.shape, n.shape))
                bad.append((name, tag, "size"))
                continue
            d = np.abs(o - n).max(axis=2)
            h, w = d.shape
            core = d[int(h * 0.12):int(h * 0.88), int(w * 0.10):int(w * 0.90)]
            cm, cp = float(core.mean()), float(np.percentile(core, 99))
            flag = ""
            if cm > MEAN_MAX or cp > P99_MAX:
                flag = "   <-- 中部差异偏大"
                bad.append((name, tag, "diff"))
            print("  %-10s %-6s %4dx%-4d 全图 mean %5.2f p99 %3d | 中部 mean %6.3f p99 %3d%s"
                  % (name, tag, w, h, d.mean(), np.percentile(d, 99), cm, cp, flag))

    print()
    if a.baseline:
        print("判定:", "全部通过" if not bad else "有问题 → %s" % (bad,))
    else:
        print("冒烟:", "全部出图" if not bad else "失败 → %s" % (bad,))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
