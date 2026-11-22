#!/usr/bin/env python3
"""Step 3：用双目 SGBM 给单目 DepthAnything 标定米制尺度（一次采集同时出资产）。

拟合  1/Z = a · rel + b      （rel 为 DA 原始输出，越大越近）

留出集用棋盘式采样（每 5 像素取 1），避免自评过拟合。

用法
----
    python3 tools/mono_calib.py --frames 4 --write --save-asset portrait
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera          # noqa: E402
from capture.mono_depth import MonocularDepth        # noqa: E402
from capture.rectify import CONFIG_PATH, Rectifier   # noqa: E402
from capture.sgbm_truth import SGBMTruth             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "depth_anything_v2_small.rknn")
OUT_CFG = os.path.join(ROOT, "config", "mono_depth_calib.json")

MIN_DISPARITY = -192
NUM_DISPARITIES = 160
Z_MIN, Z_MAX = 0.35, 12.0


def robust_fit(rel, inv_z, iters=4):
    keep = np.ones(rel.shape, bool)
    a = b = 0.0
    for _ in range(iters):
        A = np.stack([rel[keep], np.ones(int(keep.sum()))], 1)
        a, b = np.linalg.lstsq(A, inv_z[keep], rcond=None)[0]
        resid = inv_z - (a * rel + b)
        mad = np.median(np.abs(resid[keep])) or 1e-6
        keep = np.abs(resid) < 3.0 * mad
        if keep.sum() < 0.3 * rel.size:
            break
    return float(a), float(b), keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--save-asset", default=None, help="同时把最后一帧存成渲染资产")
    args = ap.parse_args()

    if not os.path.isfile(MODEL):
        print(f"✗ 缺少模型: {MODEL}")
        return 1

    rect = Rectifier(CONFIG_PATH)
    x, y, w, h = rect.valid_roi
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2

    print("[1/3] 加载模型与相机（这段时间可以先站好）...", flush=True)
    sgbm = SGBMTruth(num_disparities=NUM_DISPARITIES, min_disparity=MIN_DISPARITY)
    mono = MonocularDepth(MODEL)

    rels, invs, zs = [], [], []
    last_rgb = last_rel = None

    with StereoCamera() as cam:
        print(f"[2/3] 开始采集 {args.frames} 帧，请站住别动 ...", flush=True)
        t_start = time.perf_counter()
        for i in range(args.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            eL, eR = rect.rectify_pair(f.left, f.right)
            cropL = eL[y:y + h, x:x + w]
            cropR = eR[y:y + h, x:x + w]
            sqL = cropL[y0:y0 + s, x0:x0 + s]
            sqR = cropR[y0:y0 + s, x0:x0 + s]

            disp = sgbm.compute(cv2.cvtColor(sqL, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(sqR, cv2.COLOR_BGR2GRAY))
            z = rect.disparity_to_depth(disp)
            rel_full = mono.infer(cropL)
            rel_sq = rel_full[y0:y0 + s, x0:x0 + s]

            ok = (z > Z_MIN) & (z < Z_MAX)
            rels.append(rel_sq[ok])
            invs.append(1.0 / z[ok])
            zs.append(z[ok])
            last_rgb, last_rel = cropL, rel_full
            print("    帧%d: 双目有效 %5.1f%%" % (i, ok.mean() * 100), flush=True)
        print("[2/3] 采集完成，用时 %.1f s —— 可以坐下了" %
              (time.perf_counter() - t_start), flush=True)

    mono.close()

    rel = np.concatenate(rels)
    inv = np.concatenate(invs)
    z_st = np.concatenate(zs)
    if rel.size < 5000:
        print("✗ 共同有效像素不足")
        return 1

    print(f"\n[3/3] 标定（共同有效像素 {rel.size}）")
    a, b, inl = robust_fit(rel, inv)
    print(f"  1/Z = {a:.6f}·rel + {b:.6f}   (a>0? {'✓' if a > 0 else '✗'})  "
          f"内点 {inl.mean() * 100:.1f}%")

    idx = np.arange(rel.size)
    test = (idx % 5) == 0
    z_mono = 1.0 / np.clip(a * rel[test] + b, 1e-4, None)
    z_true = z_st[test]
    aerr = np.abs(z_mono - z_true)
    rerr = aerr / np.clip(z_true, 1e-6, None)
    r2 = 1 - np.var(inv[test] - (a * rel[test] + b)) / np.var(inv[test])

    print(f"  留出集: 中位误差 {np.median(aerr) * 1000:.0f} mm  "
          f"中位相对 {np.median(rerr) * 100:.1f}%  p95 相对 "
          f"{np.percentile(rerr, 95) * 100:.1f}%   R²={r2:.3f}")

    print("\n  分深度段（留出集）:")
    print("    区间          样本      中位误差     相对误差")
    for lo, hi in ((0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12)):
        m = (z_true >= lo) & (z_true < hi)
        if m.sum() < 200:
            print(f"    {lo:4.1f}-{hi:4.1f} m   {m.sum():6d}   样本不足")
            continue
        e = np.abs(z_mono[m] - z_true[m])
        print(f"    {lo:4.1f}-{hi:4.1f} m   {m.sum():6d}   "
              f"{np.median(e) * 1000:7.0f} mm   {np.median(e / z_true[m]) * 100:5.1f}%")

    if args.write:
        cfg = {
            "method": "stereo_to_mono_affine",
            "calibrated_on": time.strftime("%Y-%m-%d"),
            "model": os.path.basename(MODEL),
            "input_size": 518,
            "a": round(a, 8),
            "b": round(b, 8),
            "formula": "Z = 1 / (a * rel + b)",
            "n_pixels": int(rel.size),
            "inlier_ratio": round(float(inl.mean()), 4),
            "holdout": {
                "median_abs_err_mm": round(float(np.median(aerr) * 1000), 1),
                "p95_rel_err": round(float(np.percentile(rerr, 95)), 4),
                "median_rel_err": round(float(np.median(rerr)), 4),
                "r2": round(float(r2), 4),
            },
        }
        with open(OUT_CFG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 标定已写入 {OUT_CFG}")

    if args.save_asset and last_rel is not None:
        name = args.save_asset
        d = os.path.join(ROOT, "assets", name)
        os.makedirs(d, exist_ok=True)
        depth = 1.0 / np.clip(a * last_rel + b, 1e-4, None)
        depth = np.clip(depth, 0.3, 20.0).astype(np.float32)
        ch, cw = depth.shape
        fx0, fx1 = int(cw * 0.25), int(cw * 0.75)
        fy0, fy1 = int(ch * 0.10), int(ch * 0.75)
        zf = float(np.median(depth[fy0:fy1, fx0:fx1]))
        cv2.imwrite(os.path.join(d, "rgb.png"), last_rgb)
        depth.tofile(os.path.join(d, "depth.f32"))
        meta = {"width": cw, "height": ch, "z_focus": round(zf, 3),
                "z_min": round(float(np.percentile(depth, 2)), 3),
                "z_max": round(float(np.percentile(depth, 98)), 3),
                "depth_source": "mono_calibrated",
                "created": time.strftime("%Y-%m-%d %H:%M")}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"✓ 资产已存 assets/{name}  (焦点 {zf:.2f} m, "
              f"{meta['z_min']:.2f}~{meta['z_max']:.2f} m)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
