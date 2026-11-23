#!/usr/bin/env python3
"""生成头部区域遮罩 head.png，供"眼神回望"做局部偏移。

为什么需要单独一张遮罩
----------------------
DIBR 的视差是全局的：观众右移，头会像"窗外的物体"一样向左滑。
但人转头看你是反过来的——你往右移，他的头朝右转。

两者要做相反方向的叠加，就必须知道"哪些像素属于头"。

遮罩生成
--------
用姿态关键点的双眼位置推算头部椭圆：
    中心  = 双眼中点，稍向上（眼睛大约在头顶往下 40% 处）
    半宽  = 1.15 × 瞳距
    半高  = 1.50 × 瞳距
边缘羽化，并与人像 matte 相交（避免把背景也算进头部）。

用法：
    python3 tools/make_headmask.py assets/portrait [--preview /tmp/h.jpg]
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
sys.path.insert(0, _REHAB)   # pose_rknn 多目标时会 import core.*

from capture.landmarks import detect_main_person, eye_pair  # noqa: E402

POSE_MODEL = os.path.expanduser("~/rehab-coach/models/yolov8n-pose.rknn")


def make_head_mask(shape, eye_mid, iod, matte=None) -> np.ndarray:
    """返回 float32 遮罩（0..1）。"""
    h, w = shape
    cx, cy = eye_mid
    cy -= 0.55 * iod            # 眼睛在头顶往下约 40%，中心要上移
    rx = 1.40 * iod             # 第一版 1.15 偏小，只盖住脸没包住头发
    ry = 1.80 * iod

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2
    m = np.clip(1.0 - d, 0.0, 1.0)
    m = m ** 0.45                     # 让中心区域更实，边缘渐隐

    if matte is not None:
        m *= np.clip(matte, 0, 1)     # 只在人像内部生效
    return m.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()

    img = cv2.imread(os.path.join(a.asset_dir, "rgb.png"))
    if img is None:
        print("读不到 rgb.png")
        return 1
    h, w = img.shape[:2]

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
        print("✗ 未检测到双眼")
        return 2
    _le, _re, iod, mid = pair
    print(f"瞳距 {iod:.1f}px  双眼中点 ({mid[0]:.0f},{mid[1]:.0f})")

    m = make_head_mask((h, w), mid, iod, matte)
    cv2.imwrite(os.path.join(a.asset_dir, "head.png"),
                (m * 255).round().astype(np.uint8))
    cov = float((m > 0.5).mean()) * 100
    print(f"head.png 已写   覆盖 {cov:.1f}% 画面")

    if a.preview:
        vis = (img.astype(np.float32) * (0.35 + 0.65 * m[:, :, None])).astype(np.uint8)
        cv2.imwrite(a.preview, np.hstack([
            img, cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), vis]))
        print("  预览（原图 | 遮罩 | 叠加）:", a.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
