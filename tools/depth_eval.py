#!/usr/bin/env python3
"""Step 1 验收：SGBM 深度真值质量评测。

回答两个问题（文档 §9 D2 决策点的输入）：
  1. 这套双目在肖像距离上，稠密视差的有效率够不够？
  2. 纹理弱的区域（脸、白墙）空洞有多严重？

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
from capture.rectify import FX_HINT, Rectifier        # noqa: E402
from capture.sgbm_truth import SGBMTruth              # noqa: E402

# 两只候选基线（mm）：标定文件的 nominal 字段 vs 由 |T| 实测
BASELINES = {"nominal 52.0": 52.0, "measured 61.64": 61.6356}


def colorize_disp(disp: np.ndarray, vmax: float) -> np.ndarray:
    d = np.clip(disp / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


def main() -> int:
    ap = argparse.ArgumentParser(description="SGBM 深度质量评测")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--visual", default=None, help="可视化输出目录（默认不落盘）")
    ap.add_argument("--disparities", type=int, default=128)
    a = ap.parse_args()

    rect = Rectifier()
    s = rect.summary()
    print("=== 校正参数 ===")
    print(f"  方法 {s['method']}   有效区 {s['valid_roi']}   "
          f"极线中位|Δy| {s['epipolar_dy_median']} px ({s['epipolar_within_1px']}% <1px)")

    x, y, w, h = rect.valid_roi
    print(f"  评测区域 {w}x{h}   fx_hint={FX_HINT}")

    sgbm = SGBMTruth(num_disparities=a.disparities)
    print(f"  SGBM numDisparities={a.disparities} blockSize=7 MODE_HH")

    # 肖像区域：画面中央偏上，人物头部/胸部的典型位置
    portrait = (int(w * 0.25), int(h * 0.10), int(w * 0.50), int(h * 0.65))
    boxes = {
        "中央肖像区": portrait,
        "左侧": (0, 0, w // 4, h),
        "右侧": (w * 3 // 4, 0, w // 4, h),
        "上部（远）": (0, 0, w, h // 3),
        "下部（近）": (0, h * 2 // 3, w, h // 3),
    }

    valids, meds, times, reg = [], [], [], {k: [] for k in boxes}
    first_disp = first_img = None

    with StereoCamera() as cam:
        for i in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            gL, gR = rect.rectify_gray(f.left, f.right)
            cL, cR = rect.crop(gL), rect.crop(gR)

            t0 = time.perf_counter()
            disp = sgbm.compute(cL, cR)
            times.append((time.perf_counter() - t0) * 1000)

            v = disp[disp > 0]
            valids.append(v.size / disp.size)
            if v.size:
                meds.append(float(np.median(v)))
            for k, (bx, by, bw, bh) in boxes.items():
                reg[k].append(float((disp[by:by + bh, bx:bx + bw] > 0).mean()))

            if first_disp is None:
                first_disp = disp
                first_img = (f.left.copy(), cv2.cvtColor(cL, cv2.COLOR_GRAY2BGR))

    print()
    print("=== 结果（%d 帧中位）===" % a.frames)
    vr = float(np.median(valids))
    md = float(np.median(meds)) if meds else float("nan")
    print(f"  有效视差率   : {vr * 100:5.1f}%")
    print(f"  中位视差     : {md:6.1f} px")
    print(f"  SGBM 耗时    : {np.median(times):5.1f} ms  "
          f"(p95 {np.percentile(times, 95):.1f} ms)")

    v = first_disp[first_disp > 0]
    if v.size:
        print(f"  视差分布     : p5={np.percentile(v, 5):.1f}  "
              f"p50={np.percentile(v, 50):.1f}  p95={np.percentile(v, 95):.1f} px")

    print("\n  分区域有效率：")
    for k in boxes:
        r = float(np.median(reg[k]))
        flag = "  ← 弱纹理空洞" if r < 0.4 else ""
        print(f"    {k:<12} {r * 100:5.1f}%{flag}")

    print("\n  深度换算（尺度未标定，仅看量级）：")
    for name, b in BASELINES.items():
        near = FX_HINT * b * 1e-3 / max(np.percentile(v, 95), 1e-6)
        far = FX_HINT * b * 1e-3 / max(np.percentile(v, 5), 1e-6)
        print(f"    B={name:<18} 视差 p95→{near:5.2f} m   "
              f"p5→{far:6.2f} m")

    print()
    ok = vr >= 0.25
    print(f"判定：有效视差率 {vr * 100:.1f}% (阈值 25%)"
          f" —— {'✓ 双目可用' if ok else '✗ 双目弱纹理严重，转单目深度为主'}")

    if a.visual:
        os.makedirs(a.visual, exist_ok=True)
        vmax = float(np.percentile(v, 99)) if v.size else 128.0
        vis = colorize_disp(first_disp, vmax)
        tiles = [first_img[0], first_img[1], vis]
        tiles = [cv2.resize(t, (w, h)) for t in tiles]
        cv2.imwrite(os.path.join(a.visual, "depth_eval.jpg"),
                    np.hstack(tiles))
        print(f"\n  可视化已写入 {a.visual}/depth_eval.jpg")
        print("  （左：原始左目；中：校正后；右：视差伪彩）")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
