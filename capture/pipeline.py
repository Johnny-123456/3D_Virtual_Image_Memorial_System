"""采集流水线：单帧处理 / 尺度拟合 / 资产生成。

被 `tools/mono_calib.py` 和 `tools/capture_wizard.py` 共用，
避免两处各写一份标定与出资产的逻辑。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Z_MIN, Z_MAX = 0.35, 12.0
MIN_DISPARITY, NUM_DISPARITIES = -192, 160


def square_region(w: int, h: int) -> Tuple[int, int, int]:
    """取中央正方形区域（DA 与姿态模型都是方形输入，避免纵横比失真）。"""
    s = min(h, w)
    return (w - s) // 2, (h - s) // 2, s


def process_frame(rect, sgbm, mono, left: np.ndarray, right: np.ndarray) -> Optional[dict]:
    """校正 + 裁切 + 双目深度 + 单目相对深度。返回一帧的分析结果。"""
    x, y, w, h = rect.valid_roi
    eL, eR = rect.rectify_pair(left, right)
    cropL = eL[y:y + h, x:x + w]
    cropR = eR[y:y + h, x:x + w]
    x0, y0, s = square_region(w, h)

    disp = sgbm.compute(cv2.cvtColor(cropL[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(cropR[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2GRAY))
    z = rect.disparity_to_depth(disp)
    rel_full = mono.infer(cropL)
    rel_sq = rel_full[y0:y0 + s, x0:x0 + s]

    ok = (z > Z_MIN) & (z < Z_MAX)
    return {"rgb": cropL, "rel_full": rel_full, "z_sq": z, "rel_sq": rel_sq, "ok": ok,
            "square": (x0, y0, s), "valid_ratio": float(ok.mean())}


def fit_affine(rel: np.ndarray, inv_z: np.ndarray, iters: int = 4):
    """鲁棒拟合 1/Z = a·rel + b，迭代剔除离群点。"""
    keep = np.ones(rel.shape, bool)
    a = b = 0.0
    for _ in range(iters):
        if keep.sum() < 100:
            break
        A = np.stack([rel[keep], np.ones(int(keep.sum()))], 1)
        a, b = np.linalg.lstsq(A, inv_z[keep], rcond=None)[0]
        resid = inv_z - (a * rel + b)
        mad = np.median(np.abs(resid[keep])) or 1e-6
        keep = np.abs(resid) < 3.0 * mad
    return float(a), float(b), keep


def evaluate(rel: np.ndarray, inv: np.ndarray, z_st: np.ndarray,
             a: float, b: float) -> dict:
    """棋盘式留出集评估 + 分深度段误差。"""
    idx = np.arange(rel.size)
    test = (idx % 5) == 0
    z_mono = 1.0 / np.clip(a * rel[test] + b, 1e-4, None)
    z_true = z_st[test]
    aerr = np.abs(z_mono - z_true)
    rerr = aerr / np.clip(z_true, 1e-6, None)
    r2 = 1 - np.var(inv[test] - (a * rel[test] + b)) / max(np.var(inv[test]), 1e-9)

    bands = []
    for lo, hi in ((0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12)):
        m = (z_true >= lo) & (z_true < hi)
        if m.sum() < 200:
            bands.append({"range": f"{lo}-{hi}", "n": int(m.sum()), "note": "样本不足"})
            continue
        bands.append({
            "range": f"{lo}-{hi}", "n": int(m.sum()),
            "median_abs_mm": round(float(np.median(np.abs(z_mono[m] - z_true[m])) * 1000), 1),
            "median_rel": round(float(np.median(np.abs(z_mono[m] - z_true[m]) / z_true[m])), 4),
        })

    return {
        "median_abs_err_mm": round(float(np.median(aerr) * 1000), 1),
        "p95_abs_err_mm": round(float(np.percentile(aerr, 95) * 1000), 1),
        "median_rel_err": round(float(np.median(rerr)), 4),
        "p95_rel_err": round(float(np.percentile(rerr, 95)), 4),
        "r2": round(float(r2), 4),
        "bands": bands,
    }


def write_asset(out_dir: str, rgb: np.ndarray, rel_full: np.ndarray,
                a: float, b: float, matte: Optional[np.ndarray] = None,
                depth_source: str = "mono_calibrated") -> dict:
    """把一帧写成一个可渲染的资产目录。"""
    os.makedirs(out_dir, exist_ok=True)
    depth = 1.0 / np.clip(a * rel_full + b, 1e-4, None)
    depth = np.clip(depth, 0.3, 20.0).astype(np.float32)

    ch, cw = depth.shape
    x0, x1 = int(cw * 0.25), int(cw * 0.75)
    y0, y1 = int(ch * 0.10), int(ch * 0.75)
    z_subject = float(np.percentile(depth[y0:y1, x0:x1], 15))
    z_focus = float(np.percentile(depth, 60))

    cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb)
    depth.tofile(os.path.join(out_dir, "depth.f32"))
    if matte is not None:
        cv2.imwrite(os.path.join(out_dir, "matte.png"),
                    (np.clip(matte, 0, 1) * 255).round().astype(np.uint8))

    meta = {
        "width": int(cw), "height": int(ch),
        "z_focus": round(z_focus, 3),          # 零视差面 = 屏幕平面（放背景）
        "z_split": round(max(z_subject * 1.8, 1.0), 3),
        "z_subject": round(z_subject, 3),
        "z_min": round(float(np.percentile(depth, 2)), 3),
        "z_max": round(float(np.percentile(depth, 98)), 3),
        "depth_source": depth_source,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def write_background(out_dir: str, rgb: np.ndarray, depth: np.ndarray,
                     z_split: float, matte: Optional[np.ndarray]) -> None:
    """生成背景板 bg.png（挖掉主体后 inpaint）。"""
    if matte is not None:
        mask = (matte > 0.15).astype(np.uint8) * 255
    else:
        mask = ((depth > 0) & (depth < z_split)).astype(np.uint8) * 255
    if mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, k, iterations=2)
        bg = cv2.inpaint(rgb, mask, 7, cv2.INPAINT_TELEA)
    else:
        bg = rgb.copy()
    cv2.imwrite(os.path.join(out_dir, "bg.png"), bg)
