#!/usr/bin/env python3
"""为已有资产生成背景板 bg.png：把主体抠掉并修补。

DIBR 的鬼影来自"背景层里还留着主体"。用深度图把比 z_split 更近的区域
挖掉再填补，得到一张干净背景——渲染时背景层用它，主体移动就不再重影。

填洞算法与采集链路共用 `capture.pipeline.make_background`。

用法：
    python3 tools/make_bg.py assets/portrait
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.pipeline import make_background  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: make_bg.py <asset_dir>")
        return 1
    d = sys.argv[1]
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    w, h = meta["width"], meta["height"]
    z_split = float(meta.get("z_split", meta.get("z_subject", 2.0) * 1.8))

    rgb = cv2.imread(os.path.join(d, "rgb.png"))
    depth = np.fromfile(os.path.join(d, "depth.f32"), dtype=np.float32).reshape(h, w)
    if rgb is None:
        print("读不到 rgb.png")
        return 1

    matte = None
    mp = os.path.join(d, "matte.png")
    if os.path.isfile(mp):
        matte = cv2.imread(mp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        print("  使用 matte.png 作为掩码（比深度阈值精确）")
    bg = make_background(rgb, depth, z_split, matte=matte)
    out = os.path.join(d, "bg.png")
    cv2.imwrite(out, bg)

    covered = float(((depth > 0) & (depth < z_split)).mean()) * 100
    print(f"z_split={z_split:.2f} m  挖除区域 {covered:.1f}%  已写 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
