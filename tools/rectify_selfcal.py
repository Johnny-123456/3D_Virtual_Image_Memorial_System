#!/usr/bin/env python3
"""Step 1：从图像自身估计极线校正变换，写入 config/rectify.json。

不依赖标定内参——本相机的标定文件实测不可用（详见 capture/rectify.py 注释）。
默认 method=auto：先试相似变换，也试 stereoRectifyUncalibrated，
用「视差尺度变化」闸门剔除退化单应后取极线误差最小者。

用法：
    python3 tools/rectify_selfcal.py             # 估计并保存
    python3 tools/rectify_selfcal.py --frames 8  # 多帧择优
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera                    # noqa: E402
from capture.rectify import (CONFIG_PATH, estimate_rectification,  # noqa: E402
                             save_rectification)

import cv2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="自校正估计")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default=CONFIG_PATH)
    ap.add_argument("--method", default="auto", choices=["auto", "similarity", "uncalibrated"])
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
                r = estimate_rectification(gL, gR, method=a.method)
            except RuntimeError as e:
                print(f"  跳过一帧: {e}")
                continue
            e = r["epipolar"]
            rej = r.get("rejected", [])
            print(f"  {r['method']:<13} 匹配 {e['n']:4d}  中位|Δy|={e['dy_median']:6.3f} px  "
                  f"<1px {e['within_1px'] * 100:5.1f}%  "
                  f"尺度变化 {r['scale_variation'] * 100:5.2f}%")
            for j in rej:
                print(f"      ✗ 否决 {j['method']}: {j['reason']}")
            if best is None or e["dy_median"] < best["epipolar"]["dy_median"]:
                best = r

    if best is None:
        print("✗ 所有帧都估计失败")
        return 1

    e = best["epipolar"]
    ok = e["dy_median"] < 1.0 and e["within_1px"] > 0.80 and \
        best["scale_variation"] <= 0.05

    print()
    print("=== 选定 ===")
    print(f"  方法      : {best['method']}")
    print(f"  中位 |Δy| : {e['dy_median']:.3f} px      (阈值 1.0)")
    print(f"  <1px 占比 : {e['within_1px'] * 100:.1f}%     (阈值 80%)")
    print(f"  尺度变化  : {best['scale_variation'] * 100:.2f}%    (阈值 5%，越小几何越健康)")
    x, y, w, h = best["valid_roi"]
    print(f"  公共有效区: x={x} y={y} w={w} h={h}  "
          f"({w * h / (1280 * 720) * 100:.1f}% 画面)")

    if not ok:
        print("\n✗ 未达标，不写入配置")
        return 2

    save_rectification(best, a.out)
    print(f"\n✓ 已写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
