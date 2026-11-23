"""头部椭圆遮罩：给渲染端的"眼神回望"做加权区域。

`tools/finalize_asset.py`（界面采集链路）和 `tools/capture_wizard.py`
（命令行向导）共用这一份公式——以前向导压根不生成 head.png，导致向导出的
资产没有眼神回望，而界面出的有，同一个系统两种资产形态。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def make_head_mask(shape: Tuple[int, int], eye_mid: Sequence[float], iod: float,
                   matte: Optional[np.ndarray] = None) -> np.ndarray:
    """以双眼中点为锚生成头部椭圆遮罩，返回 0..1 的 float32。

    shape    图像 (h, w)
    eye_mid  双眼中点 (x, y)
    iod      瞳距（像素）——遮罩尺寸按它等比缩放
    matte    人像 alpha，传入后会相乘，避免遮罩溢出到背景
    """
    h, w = shape
    cx, cy = float(eye_mid[0]), float(eye_mid[1])
    cy -= 0.55 * iod
    rx, ry = 1.40 * iod, 1.80 * iod
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2
    m = np.clip(1.0 - d, 0.0, 1.0) ** 0.45
    if matte is not None:
        m *= np.clip(matte, 0, 1)
    return m.astype(np.float32)
