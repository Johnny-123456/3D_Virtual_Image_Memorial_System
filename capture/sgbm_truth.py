"""Step 1 交付物：SGBM 双目稀疏深度真值。

用途定位（重要）
----------------
SGBM **不是**每帧都要跑的实时模块。它只在 Step 3 做"单目相对深度 → 米制尺度"
的标定时用几次（拿稀疏但几何正确的真值当标尺）。因此优先精度而非速度。

实测性能（964x712 有效区，8 核，2026-11-18）
-------------------------------------------
====================================  =========  =============
参数                                  耗时        有效视差率
====================================  =========  =============
HH  nd=128 bs=7  全分辨率             510 ms     54.5%
SGBM nd=128 bs=7 全分辨率             512 ms     36.4%
SGBM nd=64  bs=7 全分辨率             276 ms     39.7%
SGBM nd=64  bs=5 半分辨率              60 ms     30.9%
SGBM nd=48  bs=5 半分辨率              49 ms     32.2%
====================================  =========  =============

默认选 HH + nd=128 + 全分辨率：有效视差率最高。若将来需要稠密视频深度，
再降到半分辨率换 8 倍速度。

参数说明
--------
- ``num_disparities`` 必须能被 16 整除。128 → 最小可测深度 ≈ fx*B/128 ≈ 0.44 m。
- ``P1/P2`` 平滑惩罚，经验值 8*ch*bs² 与 32*ch*bs²。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DisparityStats:
    valid_ratio: float          # 有效像素占比
    median_px: float
    p10_px: float
    p90_px: float
    min_depth_m: float
    max_depth_m: float


class SGBMTruth:
    """半全局块匹配立体。输入必须是**已校正**的灰度像对。"""

    def __init__(
        self,
        num_disparities: int = 128,
        min_disparity: int = 0,
        block_size: int = 7,
        mode: int = cv2.STEREO_SGBM_MODE_HH,
        uniqueness_ratio: int = 10,
        speckle_window: int = 100,
        speckle_range: int = 2,
        disp12_max_diff: int = 1,
    ) -> None:
        if num_disparities % 16 != 0:
            raise ValueError("num_disparities 必须能被 16 整除")
        p1 = 8 * block_size ** 2
        p2 = 32 * block_size ** 2

        self.num_disparities = num_disparities
        self.min_disparity = min_disparity
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window,
            speckleRange=speckle_range,
            disp12MaxDiff=disp12_max_diff,
            mode=mode,
        )

    def compute(self, gray_left: np.ndarray, gray_right: np.ndarray) -> np.ndarray:
        """返回 float32 视差图，单位像素。无效点为 0。"""
        raw = self._sgbm.compute(gray_left, gray_right)
        disp = raw.astype(np.float32) / 16.0
        # OpenCV SGBM: 有效值落在 [minDisparity, minDisparity+numDisparities)，
        # 无效值置为 minDisparity-1。这里只清无效值，**保留绝对视差坐标**。
        disp[disp <= (self.min_disparity - 1) + 1e-3] = 0.0
        return disp

    @staticmethod
    def stats(disp: np.ndarray) -> DisparityStats:
        v = disp[disp > 0]
        if v.size == 0:
            return DisparityStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return DisparityStats(
            valid_ratio=float(v.size) / disp.size,
            median_px=float(np.median(v)),
            p10_px=float(np.percentile(v, 10)),
            p90_px=float(np.percentile(v, 90)),
            min_depth_m=0.0,
            max_depth_m=0.0,
        )

    @staticmethod
    def depth_stats(disp: np.ndarray, fx_rect: float, baseline_mm: float) -> DisparityStats:
        s = SGBMTruth.stats(disp)
        v = disp[disp > 0]
        if v.size:
            z = fx_rect * baseline_mm * 1e-3 / v
            s.min_depth_m = float(np.percentile(z, 5))
            s.max_depth_m = float(np.percentile(z, 95))
        return s


def region_validity(disp: np.ndarray, boxes: dict) -> dict:
    """按矩形区域统计有效视差率。boxes: {name: (x, y, w, h)}"""
    return {name: float((disp[y:y + h, x:x + w] > 0).mean()) if dp.size else 0.0
            for name, (x, y, w, h) in boxes.items()
            for dp in [disp[y:y + h, x:x + w]]}
