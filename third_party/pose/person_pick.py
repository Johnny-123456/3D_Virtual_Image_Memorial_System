"""
视觉质量分析：遮挡 / 多人 / 逆光检测与会话级 quality_score。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 评估 / 上肢动作关键关节
_ARM_KEYS = (
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
)

# 下肢训练关键关节
_LOWER_KEYS = (
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
)

_OCCLUSION_MSG = {
    'upper': '请露出肩、肘和手腕',
    'lower': '请露出髋、膝和脚踝',
    'integration': '请确保肩、髋和四肢都在画面内且未被遮挡',
}


def bbox_iou(a: List[float], b: List[float]) -> float:
    """计算两个 xyxy bbox 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


TRACK_IOU_MATCH = 0.30      # 认定同一人的 IoU 下限
TRACK_IOU_REFRESH = 0.35    # 更新锁定框所需 IoU
TRACK_IOU_LOOSE = 0.10      # 丢失重关联时的最低重叠


def _bbox_center(b: List[float]) -> tuple:
    return (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5


def score_bbox_for_supine_patient(
    box: list,
    frame_size: Optional[tuple],
) -> float:
    """
    卧床场景：优先选「躺卧患者」而非站立护理者。
    躺卧者通常 bbox 更宽、中心更靠画面下方；站立者更高窄。
    """
    if not box or len(box) < 4 or not frame_size or len(frame_size) < 2:
        return 0.0
    fw, fh = float(frame_size[0]), float(frame_size[1])
    if fw <= 0 or fh <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    bcx, bcy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    aspect = w / h
    # 画面 Y 向下：bcy 越大越靠下（床面区域）
    lower_score = bcy / fh
    flat_score = min(aspect, 2.5) if aspect >= 1.05 else aspect * 0.35
    area_ratio = (w * h) / (fw * fh)
    # 站立者惩罚：明显竖长且中心偏高
    standing_penalty = 0.0
    if h > w * 1.25 and bcy < fh * 0.55:
        standing_penalty = 0.45 + min(0.35, (h / w - 1.25) * 0.2)
    return (
        lower_score * 2.2
        + flat_score * 0.55
        + area_ratio * 1.5
        - standing_penalty
    )


def pick_person_index(
    boxes_xyxy: np.ndarray,
    track_bbox: Optional[list] = None,
    frame_size: Optional[tuple] = None,
    pick_mode: str = 'default',
) -> int:
    """
    多人时：优先 IoU 跟踪锁定 bbox；
    无锁定时：选画面中心最近的人；
    有锁定但 IoU 不足时：按与锁定框中心距离选，避免跳到更大目标。
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return 0
    if len(boxes_xyxy) == 1:
        return 0

    if track_bbox:
        ious = [
            bbox_iou(box.tolist(), track_bbox)
            for box in boxes_xyxy
        ]
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= TRACK_IOU_MATCH:
            return best_idx

        tcx, tcy = _bbox_center(track_bbox)
        candidates = [
            i for i, iou in enumerate(ious) if iou >= TRACK_IOU_LOOSE
        ]
        if not candidates:
            candidates = list(range(len(boxes_xyxy)))

        def _center_dist(i: int) -> float:
            bcx, bcy = _bbox_center(boxes_xyxy[i].tolist())
            return (bcx - tcx) ** 2 + (bcy - tcy) ** 2

        return min(candidates, key=_center_dist)

    if pick_mode == 'supine_bed' and frame_size and len(frame_size) >= 2:
        scores = [
            score_bbox_for_supine_patient(box.tolist(), frame_size)
            for box in boxes_xyxy
        ]
        return int(np.argmax(scores))

    if frame_size and len(frame_size) >= 2:
        fw, fh = float(frame_size[0]), float(frame_size[1])
        cx, cy = fw * 0.5, fh * 0.5
        dists = []
        for box in boxes_xyxy:
            bcx, bcy = _bbox_center(box.tolist())
            dists.append((bcx - cx) ** 2 + (bcy - cy) ** 2)
        return int(np.argmin(dists))
    areas = [
        max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        for b in boxes_xyxy
    ]
    return int(np.argmax(areas))


