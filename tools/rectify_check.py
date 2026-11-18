#!/usr/bin/env python3
"""Step 1 验收：确认 config/rectify.json 里的校正变换真的把对应点拉到了同一行。

不依赖棋盘格——用 SIFT 匹配 + RANSAC，对任意有纹理的场景都能给结论。

判定标准（文档 §5 Step 1）：
    校正后中位 |Δy| < 1 px，且 |Δy| < 1 px 的匹配占比 > 80%

用法：
    python3 tools/rectify_check.py
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera                    # noqa: E402
from capture.rectify import CONFIG_PATH, Rectifier, epipolar_error  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="极线校正校验")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()

    rect = Rectifier(a.config)
    s = rect.summary()
    print("=== 极线校正校验 ===")
    print(f"  配置     : {a.config}")
    print(f"  方法     : {s['method']}   生成于 {s['created']}")
    print(f"  图像     : {s['image_size'][0]}x{s['image_size'][1]}")
    x, y, w, h = s["valid_roi"]
    print(f"  公共有效区: x={x} y={y} w={w} h={h}  "
          f"({w * h / (1280 * 720) * 100:.1f}% 画面)")
    print(f"  记录指标 : 中位|Δy| {s['epipolar_dy_median']} px, "
          f"{s['epipolar_within_1px']}% <1px")

    raw, cor = [], []
    with StereoCamera() as cam:
        for _ in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            gL = cv2.cvtColor(f.left, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(f.right, cv2.COLOR_BGR2GRAY)
            raw.append(epipolar_error(gL, gR))

            rL, rR = rect.rectify_gray(f.left, f.right)
            cor.append(epipolar_error(rect.crop(rL), rect.crop(rR)))

    def med(lst, k):
        vals = [r[k] for r in lst if r["n"] > 0]
        return float(np.median(vals)) if vals else float("nan")

    print()
    print(f"  [校正前] 匹配 {med(raw, 'n'):5.0f}   中位|Δy|={med(raw, 'dy_median'):7.3f} px   "
          f"<1px {med(raw, 'within_1px') * 100:5.1f}%")
    print(f"  [校正后] 匹配 {med(cor, 'n'):5.0f}   中位|Δy|={med(cor, 'dy_median'):7.3f} px   "
          f"<1px {med(cor, 'within_1px') * 100:5.1f}%")

    dy = med(cor, "dy_median")
    within = med(cor, "within_1px")
    before = med(raw, "dy_median")
    print()
    print(f"判定: 中位|Δy|={dy:.3f} px (<1.0),  <1px 占比 {within * 100:.1f}% (>80%)")
    if before == before:      # 非 NaN
        print(f"      校正前 {before:.2f} px → 校正后 {dy:.2f} px "
              f"(改善 {before / max(dy, 1e-3):.1f} 倍)")
    ok = dy < 1.0 and within > 0.80
    print("✓ Step 1 极线校正通过" if ok else "✗ 未达标")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
