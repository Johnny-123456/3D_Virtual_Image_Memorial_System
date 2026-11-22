"""程序化眨眼贴图生成。

没有真实眨眼素材时，用图像变形合成一张"闭眼"图，渲染时与睁眼帧交叉淡化。

做法（对每只眼睛）：
    1. 取眼睛上方一条**皮肤带**（眉骨/额头）
    2. 把它纵向拉伸覆盖整个眼部矩形
    3. 边缘羽化后贴回

这模拟的是"上眼睑下垂盖住眼球"的过程，在 140 ms 的眨眼时长下足够可信。

局限：眼睛特别大或眼周纹理复杂时会露馅；有真实眨眼素材时应改用素材帧。
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _feather(shape: Tuple[int, int], pad: int = 6) -> np.ndarray:
    """生成一个中间 1、边缘渐变的羽化掩码。"""
    h, w = shape
    m = np.ones((h, w), np.float32)
    pad = max(1, min(pad, h // 3, w // 3))
    ramp_v = np.minimum(np.arange(h), np.arange(h)[::-1]) / pad
    ramp_h = np.minimum(np.arange(w), np.arange(w)[::-1]) / pad
    m *= np.clip(ramp_v, 0, 1)[:, None]
    m *= np.clip(ramp_h, 0, 1)[None, :]
    return m


def make_blink(img: np.ndarray, eyes: List[Tuple[float, float]],
               inter_eye: float) -> np.ndarray:
    """合成闭眼图。

    img       原图 BGR
    eyes      双眼中心 [(x,y), ...]
    inter_eye 瞳距（像素），用于自适应眼睛大小
    """
    out = img.copy()
    h_img, w_img = img.shape[:2]

    # 眼睛矩形尺寸按瞳距缩放。人体测量：眼宽 ≈ 0.48×瞳距 → 半宽 ≈ 0.24×瞳距
    # （第一版用了 0.78，矩形大到把整张脸盖住，实测教训）
    ew = max(8, int(inter_eye * 0.26))       # 半宽
    eh = max(5, int(inter_eye * 0.17))       # 半高
    band_h = max(3, int(eh * 1.1))           # 上眼睑皮肤带高度

    for (ex, ey) in eyes:
        x0, x1 = int(ex - ew), int(ex + ew)
        y0, y1 = int(ey - eh), int(ey + eh)
        # 越界裁剪
        x0c, x1c = max(0, x0), min(w_img, x1)
        y0c, y1c = max(0, y0), min(h_img, y1)
        if x1c - x0c < 6 or y1c - y0c < 4:
            continue

        # 取眼睛上方的皮肤带（避开眉毛本身，从眉骨下方开始）
        sy0 = max(0, y0c - band_h)
        sy1 = max(sy0 + 2, y0c - 1)
        band = img[sy0:sy1, x0c:x1c]
        if band.size == 0:
            continue

        # 纵向拉伸覆盖眼部
        patch = cv2.resize(band, (x1c - x0c, y1c - y0c), interpolation=cv2.INTER_LINEAR)
        patch = cv2.GaussianBlur(patch, (0, 0), 1.2)

        # 羽化贴回
        m = _feather((y1c - y0c, x1c - x0c), pad=max(3, eh // 3))[:, :, None]
        roi = out[y0c:y1c, x0c:x1c].astype(np.float32)
        out[y0c:y1c, x0c:x1c] = (roi * (1 - m) + patch.astype(np.float32) * m
                                 ).astype(np.uint8)

    return out
