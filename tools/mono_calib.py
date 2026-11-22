#!/usr/bin/env python3
"""Step 3：用双目 SGBM 给单目 DepthAnything 标定米制尺度（命令行版）。

与 `tools/capture_wizard.py` 共用 `capture/pipeline.py` 的同一套逻辑，
区别只在于：向导带实时质检与引导，这个版本直接采固定帧数。

拟合  1/Z = a · rel + b      （rel 为 DA 原始输出，越大越近）

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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera                       # noqa: E402
from capture.matte import PortraitMatte                          # noqa: E402
from capture.mono_depth import MonocularDepth                    # noqa: E402
from capture.pipeline import (MIN_DISPARITY, NUM_DISPARITIES,     # noqa: E402
                              evaluate, fit_affine, process_frame,
                              write_asset, write_background)
from capture.rectify import CONFIG_PATH, Rectifier               # noqa: E402
from capture.sgbm_truth import SGBMTruth                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DA_MODEL = os.path.join(ROOT, "models", "depth_anything_v2_small.rknn")
MATTE_MODEL = os.path.join(ROOT, "models", "modnet.rknn")
OUT_CFG = os.path.join(ROOT, "config", "mono_depth_calib.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--write", action="store_true", help="写入标定配置")
    ap.add_argument("--save-asset", default=None, help="同时存成渲染资产")
    a = ap.parse_args()

    if not os.path.isfile(DA_MODEL):
        print(f"✗ 缺少模型: {DA_MODEL}")
        return 1

    rect = Rectifier(CONFIG_PATH)
    sgbm = SGBMTruth(num_disparities=NUM_DISPARITIES, min_disparity=MIN_DISPARITY)
    print("[1/3] 加载模型（可先站好）...", flush=True)
    mono = MonocularDepth(DA_MODEL)

    samples = []
    print(f"[2/3] 采集 {a.frames} 帧 ...", flush=True)
    t0 = time.time()
    with StereoCamera() as cam:
        for i in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            s = process_frame(rect, sgbm, mono, f.left, f.right)
            samples.append(s)
            print(f"    帧{i}: 双目有效 {s['valid_ratio'] * 100:5.1f}%", flush=True)
    print(f"[2/3] 完成，用时 {time.time() - t0:.1f} s", flush=True)

    rel = np.concatenate([s["rel_sq"][s["ok"]] for s in samples])
    inv = np.concatenate([1.0 / s["z_sq"][s["ok"]] for s in samples])
    z_st = np.concatenate([s["z_sq"][s["ok"]] for s in samples])
    if rel.size < 5000:
        print("✗ 共同有效像素不足")
        return 1

    print(f"\n[3/3] 标定（共同有效像素 {rel.size}）")
    A, B, inl = fit_affine(rel, inv)
    m = evaluate(rel, inv, z_st, A, B)
    print(f"  1/Z = {A:.6f}·rel + {B:.6f}   内点 {inl.mean() * 100:.1f}%  "
          f"(a>0? {'✓' if A > 0 else '✗'})")
    print(f"  留出集: 中位 {m['median_abs_err_mm']:.0f} mm  "
          f"相对 {m['median_rel_err'] * 100:.1f}%  "
          f"p95 {m['p95_rel_err'] * 100:.1f}%   R²={m['r2']:.3f}")
    print("\n  分深度段（留出集）:")
    for b in m["bands"]:
        if "note" in b:
            print(f"    {b['range']:>9} m  {b['n']:6d}  样本不足")
        else:
            print(f"    {b['range']:>9} m  {b['n']:6d}  "
                  f"{b['median_abs_mm']:7.0f} mm  {b['median_rel'] * 100:5.1f}%")

    if a.write:
        cfg = {
            "method": "stereo_to_mono_affine",
            "calibrated_on": time.strftime("%Y-%m-%d"),
            "model": os.path.basename(DA_MODEL),
            "input_size": 518,
            "a": round(A, 8), "b": round(B, 8),
            "formula": "Z = 1 / (a * rel + b)",
            "n_pixels": int(rel.size),
            "inlier_ratio": round(float(inl.mean()), 4),
            "holdout": m,
        }
        with open(OUT_CFG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 标定已写入 {OUT_CFG}")

    if a.save_asset:
        best = max(samples, key=lambda s: s["valid_ratio"])
        matte_net = PortraitMatte(MATTE_MODEL)
        matte = matte_net.infer(best["rgb"])
        matte_net.close()
        out = os.path.join(ROOT, "assets", a.save_asset)
        meta = write_asset(out, best["rgb"], best["rel_full"], A, B, matte=matte)
        depth = 1.0 / np.clip(A * best["rel_full"] + B, 1e-4, None)
        write_background(out, best["rgb"], depth.astype(np.float32),
                         meta["z_split"], matte)
        print(f"✓ 资产已存 {out}  (主体 {meta['z_subject']:.2f} m, "
              f"零视差面 {meta['z_focus']:.2f} m)")

    mono.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
