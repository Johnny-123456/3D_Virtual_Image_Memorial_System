"""人体关键点：在主画面中心区域内检测，避开背景里的其他人。

背景
----
rehab-coach 的 `RknnPoseBackend` 在检测到多人时会用 `pick_person_index`
选一个目标（优先跟踪框，其次离画面中心最近）。我们的采集场景里
背景常有同事走动，实测出现过"选中了最右边远景人物"（瞳距仅 14.7px）
导致眨眼/头部遮罩算错的情况。

这里的做法更直接：只把画面中央的一条区域送去检测。我们的采集向导
本来就要求受采者站在中间，所以这样做既简单又确定。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def detect_main_person(pose, img: np.ndarray, center_frac: float = 0.62):
    """在画面中央区域检测，返回映射回原图坐标的结果。"""
    h, w = img.shape[:2]
    cw = max(64, int(w * center_frac))
    x0 = (w - cw) // 2
    sub = img[:, x0:x0 + cw]

    det = pose.detect(sub)
    if not det:
        return None

    det["bbox"][0] += x0
    det["bbox"][2] += x0
    for v in det.get("keypoints_2d", {}).values():
        v["x"] += x0
    return det


def eye_pair(det) -> Optional[tuple]:
    """从检测结果里取双眼坐标与瞳距。"""
    if not det:
        return None
    kp = det.get("keypoints_2d", {})
    if "left_eye" not in kp or "right_eye" not in kp:
        return None
    le, re_ = kp["left_eye"], kp["right_eye"]
    iod = float(np.hypot(le["x"] - re_["x"], le["y"] - re_["y"]))
    mid = ((le["x"] + re_["x"]) / 2.0, (le["y"] + re_["y"]) / 2.0)
    return le, re_, iod, mid
