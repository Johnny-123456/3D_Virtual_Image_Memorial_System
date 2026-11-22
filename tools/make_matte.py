#!/usr/bin/env python3
"""为已有资产生成 matte.png（人像 alpha）。

用法：
    python3 tools/make_matte.py assets/portrait
    python3 tools/make_matte.py assets/portrait --preview /tmp/m.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capture.matte import PortraitMatte  # noqa: E402

MODEL = os.path.join(ROOT, "models", "modnet.rknn")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成人像 matte")
    ap.add_argument("asset_dir")
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()

    d = a.asset_dir
    rgb_path = os.path.join(d, "rgb.png")
    img = cv2.imread(rgb_path)
    if img is None:
        print("读不到", rgb_path)
        return 1
    if not os.path.isfile(MODEL):
        print("缺少模型:", MODEL)
        return 1

    with PortraitMatte(MODEL) as pm:
        alpha = pm.infer(img)

    out = os.path.join(d, "matte.png")
    cv2.imwrite(out, (alpha * 255).round().astype(np.uint8))

    soft = float(((alpha > 0.05) & (alpha < 0.95)).mean()) * 100
    cover = float((alpha > 0.5).mean()) * 100
    print(f"matte 已写 {out}")
    print(f"  主体覆盖 {cover:.1f}%   软边（羽化）像素 {soft:.2f}%")

    if a.preview:
        bg = np.zeros_like(img)
        for y in range(img.shape[0]):
            bg[y, :] = (70 + int(50 * y / img.shape[0]), 45, 35)
        comp = (img.astype(np.float32) * alpha[:, :, None] +
                bg.astype(np.float32) * (1 - alpha[:, :, None])).astype(np.uint8)
        cv2.imwrite(a.preview, np.hstack([
            img,
            cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            comp]))
        print("  预览:", a.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
