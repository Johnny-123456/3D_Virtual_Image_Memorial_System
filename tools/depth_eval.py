#!/usr/bin/env python3
"""Step 1 验收 + Step 3 深度标定验证：SGBM 深度质量评测。

深度模型（不是教科书里的 Z = fB/d）：

    Z = K / (d - d_inf),   K = fx * B

`d_inf` 是本机固有的无穷远视差偏移（约 -156 px，来自两目主点不一致），
数值上比物体视差还大——漏掉它深度会差数倍。

用法：
    python3 tools/depth_eval.py                    # 文字报告
    python3 tools/depth_eval.py --visual /tmp/x    # 另存可视化三联图
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
from capture.rectify import Rectifier, load_depth_calib  # noqa: E402
from capture.sgbm_truth import SGBMTruth              # noqa: E402

# 视差搜索范围：必须覆盖相机固有偏移 d_inf 附近的整段
MIN_DISPARITY = -192
NUM_DISPARITIES = 160


def colorize_disp(disp: np.ndarray, lo: float, hi: float) -> np.ndarray:
    d = np.clip((disp - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


def main() -> int:
    ap = argparse.ArgumentParser(description="SGBM 深度质量评测")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--visual", default=None)
    a = ap.parse_args()

    dc = load_depth_calib()
    if dc is None:
        print("✗ 缺少 config/depth_calib.json，先做两点标定")
        return 1

    rect = Rectifier()
    s = rect.summary()
    print("=== 校正与标定 ===")
    print(f"  校正方法 {s['method']}   有效区 {s['valid_roi']}   "
          f"极线中位|Δy| {s['epipolar_dy_median']} px")
    print(f"  尺度变化 {s['scale_variation'] * 100:.2f}%   "
          f"有效区占比 {(s['valid_roi'][2] * s['valid_roi'][3]) / (1280 * 720) * 100:.1f}%")
    print(f"  深度标定 fx={dc['fx_px']}  B={dc['baseline_mm']} mm  "
          f"d_inf={dc['disparity_inf_px']} px  K={dc['k_px_m']} px·m")

    x, y, w, h = rect.valid_roi
    sgbm = SGBMTruth(num_disparities=NUM_DISPARITIES, min_disparity=MIN_DISPARITY)
    print(f"  SGBM minDisparity={MIN_DISPARITY} numDisparities={NUM_DISPARITIES} MODE_HH")

    portrait = (int(w * 0.25), int(h * 0.10), int(w * 0.50), int(h * 0.65))
    boxes = {"中央肖像区": portrait, "左侧": (0, 0, w // 4, h),
             "右侧": (w * 3 // 4, 0, w // 4, h),
             "上部(远)": (0, 0, w, h // 3), "下部(近)": (0, h * 2 // 3, w, h // 3)}

    valids, times, reg = [], [], {k: [] for k in boxes}
    dvals, zvals = [], []
    first = None

    with StereoCamera() as cam:
        for _ in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            eL, eR = rect.rectify_pair(f.left, f.right)
            gL = cv2.cvtColor(eL, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
            gR = cv2.cvtColor(eR, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]

            t0 = time.perf_counter()
            disp = sgbm.compute(gL, gR)
            times.append((time.perf_counter() - t0) * 1000)

            depth = rect.disparity_to_depth(disp)
            v = disp[disp != 0]
            valids.append(v.size / disp.size)
            dvals.append(v)
            zvals.append(depth[depth > 0])
            for k, (bx, by, bw, bh) in boxes.items():
                reg[k].append(float((disp[by:by + bh, bx:bx + bw] != 0).mean()))
            if first is None:
                first = (eL[y:y + h, x:x + w].copy(), disp, depth)

    print()
    print(f"=== 结果（{a.frames} 帧中位）===")
    vr = float(np.median(valids))
    print(f"  有效视差率   : {vr * 100:5.1f}%")
    print(f"  SGBM 耗时    : {np.median(times):5.1f} ms")

    d = np.concatenate(dvals)
    z = np.concatenate(zvals)
    print(f"  视差范围     : p5={np.percentile(d, 5):7.1f}  "
          f"p50={np.percentile(d, 50):7.1f}  p95={np.percentile(d, 95):7.1f} px")
    print(f"  深度范围     : p5={np.percentile(z, 5):6.2f}  "
          f"p50={np.percentile(z, 50):6.2f}  p95={np.percentile(z, 95):6.2f} m")

    print("\n  分区域有效率：")
    for k in boxes:
        r = float(np.median(reg[k]))
        flag = "  ← 弱纹理空洞" if r < 0.4 else ""
        print(f"    {k:<10} {r * 100:5.1f}%{flag}")

    print("\n  深度分布：")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)):
        m = (z >= lo) & (z < hi)
        if m.sum():
            print(f"    {lo:4.1f}-{min(hi, 99):4.1f} m : {m.sum() / z.size * 100:5.1f}%")

    print("\n  1-2 m 肖像距离的深度分辨率（视差量化 1/16 px）：")
    k = dc["k_px_m"]
    for zz in (1.0, 1.2, 1.5, 2.0):
        print(f"    Z={zz:.1f} m -> {zz ** 2 * (1 / 16) / k * 1000:.2f} mm")

    ok = vr >= 0.25
    print(f"\n判定：有效视差率 {vr * 100:.1f}% (阈值 25%) "
          f"—— {'✓ 双目可用' if ok else '✗ 弱纹理严重，转单目深度为主'}")

    if a.visual:
        os.makedirs(a.visual, exist_ok=True)
        img, disp, depth = first
        vis = colorize_disp(disp, MIN_DISPARITY, MIN_DISPARITY + NUM_DISPARITIES)
        cv2.imwrite(os.path.join(a.visual, "depth_eval.jpg"), np.hstack([img, vis]))
        print(f"\n  可视化: {a.visual}/depth_eval.jpg")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
