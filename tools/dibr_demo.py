#!/usr/bin/env python3
"""Step 4 验证：用真实深度渲染多个视点，输出对比图。

流程：采集 → 纯旋转校正 → SGBM 米制深度 → DIBR 重投影 → 5 视点对比

用法：
    python3 tools/dibr_demo.py --out /tmp/dibr
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera            # noqa: E402
from capture.rectify import Rectifier                  # noqa: E402
from capture.sgbm_truth import SGBMTruth               # noqa: E402
from render.dibr import DibrParams, DibrRenderer       # noqa: E402

MIN_DISPARITY = -192
NUM_DISPARITIES = 160


def main() -> int:
    ap = argparse.ArgumentParser(description="DIBR 多视点演示")
    ap.add_argument("--out", default="/tmp/dibr")
    ap.add_argument("--gain", type=float, default=62.0)
    ap.add_argument("--eyes", type=float, nargs="+",
                    default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--focus", type=float, default=None,
                    help="焦点深度(米)，默认取主体区域中位")
    a = ap.parse_args()

    rect = Rectifier()
    x, y, w, h = rect.valid_roi
    sgbm = SGBMTruth(num_disparities=NUM_DISPARITIES, min_disparity=MIN_DISPARITY)

    # --- 采集 + 深度 ---
    with StereoCamera() as cam:
        f = cam.grab()
    if f is None:
        print("读帧失败")
        return 1
    eL, eR = rect.rectify_pair(f.left, f.right)
    rgb = eL[y:y + h, x:x + w].copy()
    gL = cv2.cvtColor(eL, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
    gR = cv2.cvtColor(eR, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]

    t0 = time.perf_counter()
    disp = sgbm.compute(gL, gR)
    depth = rect.disparity_to_depth(disp)
    t_depth = (time.perf_counter() - t0) * 1000
    print("深度: 有效 %.1f%%, 耗时 %.0f ms" %
          ((disp != 0).mean() * 100, t_depth))

    # --- DIBR 预处理 ---
    r = DibrRenderer(DibrParams(parallax_gain=a.gain, z_focus=a.focus))
    t0 = time.perf_counter()
    r.prepare(depth)
    print("DIBR 预处理 %.0f ms, 焦点深度 %.2f m" %
          ((time.perf_counter() - t0) * 1000, r.z_focus))

    # --- 多视点渲染 ---
    views = []
    for ex in a.eyes:
        t0 = time.perf_counter()
        v = r.render(rgb, (ex, 0.0), backfill_blur=6)
        ms = (time.perf_counter() - t0) * 1000
        views.append(v)
        dx, _ = r.parallax_field((ex, 0.0))
        print("  eye=%+.2f  位移 中位 %+.1f px  范围 %+.1f..%+.1f  渲染 %.0f ms"
              % (ex, np.median(dx), dx.min(), dx.max(), ms))

    # --- 视差强度自检：焦点前后位移是否反向 ---
    z = r.depth
    near = z < r.z_focus * 0.8
    far = z > r.z_focus * 1.2
    dxn, _ = r.parallax_field((1.0, 0.0))
    print("\n  近处(比焦点近)平均位移 %+.1f px" % dxn[near].mean() if near.any() else "")
    print("  远处(比焦点远)平均位移 %+.1f px" % dxn[far].mean() if far.any() else "")
    # 观察者右移时，近物的像相对远景向左移 —— 近处应为负位移、远处为正
    sign_ok = dxn[near].mean() < 0 < dxn[far].mean()
    print("  方向检查: %s（近处左移、远处右移 = 正确）" %
          ("✓ 通过" if sign_ok else "✗ 异常"))

    # --- 输出对比图 ---
    os.makedirs(a.out, exist_ok=True)
    strip = np.hstack([cv2.resize(v, (w // 2, h // 2)) for v in views])
    cv2.imwrite(os.path.join(a.out, "views.jpg"), strip)

    # 视差差异图：两端视点之差，越亮位移越大
    d = cv2.absdiff(views[0], views[-1])
    d = np.clip(d.astype(np.float32) * 3, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, "diff.jpg"),
                cv2.resize(d, (w // 2, h // 2)))

    # 红青立体
    ana = r.anaglyph(rgb, eye_sep=0.6)
    cv2.imwrite(os.path.join(a.out, "anaglyph.jpg"),
                cv2.resize(ana, (w * 3 // 4, h * 3 // 4)))

    # 深度伪彩
    zz = np.clip(z / max(np.percentile(z, 95), 1e-6) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, "depth.jpg"),
                cv2.resize(cv2.applyColorMap(zz, cv2.COLORMAP_TURBO),
                           (w // 2, h // 2)))

    print(f"\n输出目录: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
