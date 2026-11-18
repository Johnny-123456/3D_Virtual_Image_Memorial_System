#!/usr/bin/env python3
"""Step 1：从图像自身估计极线校正变换，写入 config/rectify.json。

不依赖标定内参——本相机的标定文件实测不可用（详见 capture/rectify.py 模块注释）。

用法：
    python3 tools/rectify_selfcal.py            # 估计并保存
    python3 tools/rectify_selfcal.py --frames 8 # 多帧择优
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera                    # noqa: E402
from capture.rectify import (CONFIG_PATH, estimate_rectification,  # noqa: E402
                             save_rectification)


def main() -> int:
    ap = argparse.ArgumentParser(description="自校正估计")
    ap.add_argument("--frames", type=int, default=6, help="采样帧数，取质量最好的一帧")
    ap.add_argument("--out", default=CONFIG_PATH)
    a = ap.parse_args()

    best = None
    with StereoCamera() as cam:
        for _ in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            gL = cv2.cvtColor(f.left, cv2.COLOR_BGR2GRAY)
            gR = cv2.cvtColor(f.right, cv2.COLOR_BGR2GRAY)
            try:
                r = estimate_rectification(gL, gR, method="uncalibrated")
            except RuntimeError as e:
                print(f"  跳过一帧: {e}")
                continue
            e = r["epipolar"]
            print(f"  {r['method']:<13} 匹配 {e['n']:4d}  中位|Δy|={e['dy_median']:6.3f} px  "
                  f"<1px {e['within_1px'] * 100:5.1f}%")
            if best is None or e["dy_median"] < best["epipolar"]["dy_median"]:
                best = r

    if best is None:
        print("✗ 所有帧都估计失败")
        return 1

    e = best["epipolar"]
    ok = e["dy_median"] < 1.0 and e["within_1px"] > 0.80

    print()
    print("=== 选定 ===")
    print(f"  方法      : {best['method']}")
    print(f"  中位 |Δy| : {e['dy_median']:.3f} px   (阈值 1.0)")
    print(f"  <1px 占比 : {e['within_1px'] * 100:.1f}%  (阈值 80%)")
    x, y, w, h = best["valid_roi"]
    print(f"  公共有效区: x={x} y={y} w={w} h={h}  "
          f"({w * h / (1280 * 720) * 100:.1f}% 画面)")

    h1 = np.asarray(best["H_left"])
    h2 = np.asarray(best["H_right"])
    print(f"  H_left  与单位阵最大差: {np.abs(h1 - np.eye(3)).max():.4f}")
    print(f"  H_right 含平移分量     : ({h2[0, 2]:+.1f}, {h2[1, 2]:+.1f}) px")

    if not ok:
        print("\n✗ 未达标，不写入配置")
        return 2

    save_rectification(best, a.out)
    print(f"\n✓ 已写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
