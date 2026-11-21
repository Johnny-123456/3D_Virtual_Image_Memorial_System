#!/usr/bin/env python3
"""生成渲染资产：一帧 RGB + 对应米制深度图。

产物（assets/<name>/）：
    rgb.png     校正并裁切后的左目图
    depth.f32   float32 小端，W*H，单位米（0 = 无效）
    meta.json   宽高、焦点深度、深度范围

深度来源优先用单目 DA（稠密、边缘好），再用 config/mono_depth_calib.json
标定成米制；若标定缺失则退化为相对深度映射到 0.5~8 m。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture.camera_io import StereoCamera           # noqa: E402
from capture.mono_depth import MonocularDepth        # noqa: E402
from capture.rectify import CONFIG_PATH, Rectifier, load_depth_calib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "depth_anything_v2_small.rknn")
MONO_CFG = os.path.join(ROOT, "config", "mono_depth_calib.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成渲染资产")
    ap.add_argument("--name", default="demo")
    ap.add_argument("--frames", type=int, default=3, help="深度取多帧中位")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "assets", a.name)
    os.makedirs(out_dir, exist_ok=True)

    rect = Rectifier(CONFIG_PATH)
    x, y, w, h = rect.valid_roi
    mono = MonocularDepth(MODEL)

    rgbs, rels = [], []
    with StereoCamera() as cam:
        for _ in range(a.frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            eL, _ = rect.rectify_pair(f.left, f.right)
            crop = eL[y:y + h, x:x + w].copy()
            rgbs.append(crop)
            rels.append(mono.infer(crop))
    mono.close()

    rgb = rgbs[len(rgbs) // 2]
    rel = np.median(np.stack(rels, 0), axis=0).astype(np.float32)

    # 相对深度 -> 米制
    if os.path.isfile(MONO_CFG):
        c = json.load(open(MONO_CFG, encoding="utf-8"))
        A, B = float(c["a"]), float(c["b"])
        depth = 1.0 / np.clip(A * rel + B, 1e-4, None)
        src = "mono_calibrated"
    else:
        lo, hi = np.percentile(rel, [2, 98])
        n = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, 1)
        depth = 0.5 + (1.0 - n) * 7.5          # rel 大=近
        src = "mono_relative"

    depth = np.clip(depth, 0.3, 20.0).astype(np.float32)

    # 焦点：中央 50%x65% 的中位深度
    ch, cw = depth.shape
    fx0, fx1 = int(cw * 0.25), int(cw * 0.75)
    fy0, fy1 = int(ch * 0.10), int(ch * 0.75)
    z_focus = float(np.median(depth[fy0:fy1, fx0:fx1]))

    cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb)
    depth.tofile(os.path.join(out_dir, "depth.f32"))

    meta = {
        "width": int(cw),
        "height": int(ch),
        "z_focus": round(z_focus, 3),
        "z_min": round(float(np.percentile(depth, 2)), 3),
        "z_max": round(float(np.percentile(depth, 98)), 3),
        "depth_source": src,
        "created": cv2.getTickCount() and __import__("time").strftime("%Y-%m-%d %H:%M"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("资产已生成:", out_dir)
    print("  rgb.png   %dx%d" % (cw, ch))
    print("  depth     焦点 %.2f m  范围 %.2f ~ %.2f m  来源 %s"
          % (z_focus, meta["z_min"], meta["z_max"], src))
    print("  文件大小: rgb %.0f KB, depth %.0f KB"
          % (os.path.getsize(os.path.join(out_dir, "rgb.png")) / 1024,
             os.path.getsize(os.path.join(out_dir, "depth.f32")) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
