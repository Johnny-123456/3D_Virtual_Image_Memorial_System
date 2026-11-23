"""程序化眨眼贴图生成。

没有真实眨眼素材时，合成一张"闭眼"图，渲染时与睁眼帧交叉淡化。

做法（对每只眼睛）：
    1. 用周围皮肤把眼球区域填掉（金字塔外推，与背景板同一个算子）
    2. 整体压暗一点点——眼睑本来就比眼周皮肤暗
    3. 画一条两端收敛的睫毛线

第一版是把眼睛上方的皮肤带纵向拉伸盖住眼睛。实测不行：那条皮肤带正好在
刘海的阴影里，拉下来是一块**发黑的矩形**，四边硬、颜色比周围暗一档，
看上去像脸上贴了块黑胶布，不像闭眼。现在改成"填皮肤 + 画睫毛线"。

局限：眼睛特别大或眼周纹理复杂时会露馅；有真实眨眼素材时应改用素材帧。
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from capture.pipeline import fill_holes


def _feather(shape: Tuple[int, int], pad: int = 4) -> np.ndarray:
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
    ew = max(8, int(inter_eye * 0.26))       # 半宽
    eh = max(5, int(inter_eye * 0.17))       # 半高

    for (ex, ey) in eyes:
        x0, x1 = int(ex - ew), int(ex + ew)
        y0, y1 = int(ey - eh), int(ey + eh)
        x0c, x1c = max(0, x0), min(w_img, x1)
        y0c, y1c = max(0, y0), min(h_img, y1)
        if x1c - x0c < 6 or y1c - y0c < 4:
            continue

        # 局部裁剪一块比眼睛大的区域，用四周皮肤把眼球填掉。
        # 只在外扩的窗口里做，避免整幅金字塔（1272×712 全图约 0.2s，×2 眼）。
        px, py = int(ew * 1.8) + 4, int(eh * 2.4) + 4
        cx0, cx1 = max(0, x0c - px), min(w_img, x1c + px)
        cy0, cy1 = max(0, y0c - py), min(h_img, y1c + py)
        sub = out[cy0:cy1, cx0:cx1]
        hole = np.zeros(sub.shape[:2], bool)
        hole[y0c - cy0:y1c - cy0, x0c - cx0:x1c - cx0] = True
        fill = fill_holes(sub, hole)
        patch = fill[y0c - cy0:y1c - cy0,
                     x0c - cx0:x1c - cx0].astype(np.float32)

        # 眼睑比周围皮肤略暗
        patch *= 0.94

        # 睫毛线：横向深色带，两端收敛成弧（竖直方向高斯、水平方向四次幂）
        hh, ww = patch.shape[:2]
        yy = np.arange(hh)[:, None] - hh / 2.0
        xx = np.arange(ww)[None, :] - ww / 2.0
        la, lb = max(hh * 0.16, 1.2), max(ww * 0.62, 2.0)
        line = np.exp(-(yy / la) ** 2 - (xx / lb) ** 4)
        patch = patch * (1.0 - 0.62 * line[..., None])

        # 羽化贴回
        m = _feather((hh, ww), pad=max(2, min(hh, ww) // 5))[..., None]
        roi = out[y0c:y1c, x0c:x1c].astype(np.float32)
        out[y0c:y1c, x0c:x1c] = np.clip(roi * (1 - m) + patch * m,
                                        0, 255).astype(np.uint8)

    return out
