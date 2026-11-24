#!/usr/bin/env python3
"""为已有资产生成 blink.png（闭眼帧）。

用法：
    python3 tools/make_blink.py assets/portrait
    python3 tools/make_blink.py assets/portrait --preview /tmp/blink.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capture.blink import make_blink  # noqa: E402
from capture.landmarks import detect_main_person, eye_pair  # noqa: E402

POSE_MODEL = None   # 由 capture.pose 统一解析（models/ 优先，~/rehab-coach 兜底）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()

    img = cv2.imread(os.path.join(a.asset_dir, "rgb.png"))
    if img is None:
        print("读不到 rgb.png")
        return 1

    from capture.pose import load_pose
    pose = load_pose(POSE_MODEL, imgsz=640, conf_threshold=0.4)
    det = detect_main_person(pose, img)
    pose.release()

    pair = eye_pair(det)
    if not pair:
        print("✗ 未检测到双眼关键点")
        return 2
    le, re_, iod, _mid = pair
    le = {"x": le["x"], "y": le["y"], "conf": le["conf"]}
    re_ = {"x": re_["x"], "y": re_["y"], "conf": re_["conf"]}
    eyes = [(le["x"], le["y"]), (re_["x"], re_["y"])]
    inter = iod
    print(f"左眼 ({le['x']:.0f},{le['y']:.0f})  右眼 ({re_['x']:.0f},{re_['y']:.0f})  "
          f"瞳距 {inter:.1f} px  置信度 {le['conf']:.2f}/{re_['conf']:.2f}")

    blink = make_blink(img, eyes, inter)
    if np.array_equal(blink, img):
        print("✗ 未产生任何变化（眼睛区域计算异常）")
        return 3

    out = os.path.join(a.asset_dir, "blink.png")
    cv2.imwrite(out, blink)
    changed = float((cv2.absdiff(blink, img).max(axis=2) > 8).mean()) * 100
    print(f"blink.png 已写 {out}   改动像素 {changed:.2f}%")

    if a.preview:
        # 眼部区域放大对比
        cx = int((le["x"] + re_["x"]) / 2)
        cy = int((le["y"] + re_["y"]) / 2)
        r = max(40, int(inter * 1.4))
        x0, x1 = max(0, cx - r), min(img.shape[1], cx + r)
        y0, y1 = max(0, cy - r // 2), min(img.shape[0], cy + r // 2)
        crop_a = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3,
                            interpolation=cv2.INTER_NEAREST)
        crop_b = cv2.resize(blink[y0:y1, x0:x1], None, fx=3, fy=3,
                            interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(a.preview, np.vstack([crop_a, crop_b]))
        print("  预览（上=睁眼 下=闭眼，3x 放大）:", a.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
