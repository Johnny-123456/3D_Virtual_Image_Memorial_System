#!/usr/bin/env python3
"""资产后处理：为资产补齐 blink.png 与 head.png。

`mono_calib.py --save-asset` 只生成 rgb/depth/matte/bg/meta，
眨眼与头部遮罩需要姿态模型，单独放在这里，由界面或脚本串起来调用。

用法：
    python3 tools/finalize_asset.py assets/portrait2
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_REHAB = os.path.expanduser("~/rehab-coach/src/backend")
sys.path.insert(0, os.path.join(_REHAB, "vision"))
sys.path.insert(0, _REHAB)

from capture.blink import make_blink                       # noqa: E402
from capture.landmarks import detect_main_person, eye_pair  # noqa: E402

POSE_MODEL = os.path.expanduser("~/rehab-coach/models/yolov8n-pose.rknn")


def make_head_mask(shape, eye_mid, iod, matte=None):
    """头部椭圆遮罩（与 tools/make_headmask.py 同一套公式）。"""
    h, w = shape
    cx, cy = eye_mid
    cy -= 0.55 * iod
    rx, ry = 1.40 * iod, 1.80 * iod
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2
    m = np.clip(1.0 - d, 0.0, 1.0) ** 0.45
    if matte is not None:
        m *= np.clip(matte, 0, 1)
    return m.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    a = ap.parse_args()

    img = cv2.imread(os.path.join(a.asset_dir, "rgb.png"))
    if img is None:
        print("读不到 rgb.png")
        return 1
    matte = None
    mp = os.path.join(a.asset_dir, "matte.png")
    if os.path.isfile(mp):
        matte = cv2.imread(mp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    from pose_rknn import RknnPoseBackend
    pose = RknnPoseBackend(POSE_MODEL, imgsz=640, conf_threshold=0.4)
    det = detect_main_person(pose, img)
    pose.release()

    pair = eye_pair(det)
    if not pair:
        print("✗ 未检测到双眼，跳过 blink/head")
        return 2

    le, re_, iod, mid = pair
    print(f"瞳距 {iod:.1f}px  中点 ({mid[0]:.0f},{mid[1]:.0f})")

    # 守卫：瞳距过小 = 检测到的是远处的人（或背景里的同事），不是受采者。
    # 实测踩过：镜头前没人时会选中画面深处 2.8px 瞳距的目标，
    # 流程照样产出资产，但 head.png 覆盖 0%，渲染完全无效。
    if iod < 45.0:
        print(f"✗ 受采者离得太远或不在画面中央（瞳距仅 {iod:.1f}px，至少 45px）")
        print("  请站到镜头前 1.2~1.5 m 处、位于画面中央后重试")
        return 3

    cv2.imwrite(os.path.join(a.asset_dir, "blink.png"),
                make_blink(img, [(le["x"], le["y"]), (re_["x"], re_["y"])], iod))
    print("  blink.png")

    m = make_head_mask(img.shape[:2], mid, iod, matte)
    cv2.imwrite(os.path.join(a.asset_dir, "head.png"),
                (m * 255).round().astype(np.uint8))
    print(f"  head.png（覆盖 {float((m > 0.5).mean()) * 100:.1f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
