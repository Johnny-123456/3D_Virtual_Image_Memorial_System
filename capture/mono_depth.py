"""Step 2/3 交付物：单目相对深度推理（DepthAnything-V2-Small on RKNN）。

职责
----
只负责"图片 -> 相对深度图"。**不负责尺度**——DA 输出是无量纲的相对量
（越大越近，等价于逆深度），米制尺度由 `tools/mono_calib.py` 用双目标定。

关键约定
--------
- 归一化（ImageNet）已烘焙进 `.rknn`，这里直接喂 **uint8 RGB**，不要自己再减均值。
- 模型是静态 518×518 输入，要求输入宽高都能被 14 整除。
  输入是正方形，所以**必须先把图像裁成正方形**再喂，否则纵横比失真会显著掉精度。
- 输出语义：数值越大 = 越近（逆深度），与视差同向。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

MODEL_SIZE = 518        # 37 * 14
PATCH = 14              # 输入必须是 14 的倍数


class MonocularDepth:
    """DepthAnything-V2-Small 推理封装（RK3588 NPU）。"""

    def __init__(self, model_path: str, size: int = MODEL_SIZE) -> None:
        from rknnlite.api import RKNNLite

        self.size = size
        self.model_path = model_path
        self._rk = RKNNLite()
        if self._rk.load_rknn(model_path) != 0:
            raise RuntimeError(f"加载失败: {model_path}")
        if self._rk.init_runtime() != 0:
            raise RuntimeError("init_runtime 失败")

    def close(self) -> None:
        try:
            self._rk.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """BGR 图 -> 相对深度图，尺寸与输入图相同（越大越近）。"""
        h, w = bgr.shape[:2]
        # 裁成正方形再缩放，避免纵横比失真
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        sq = bgr[y0:y0 + s, x0:x0 + s]

        img = cv2.resize(sq, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)

        outs = self._rk.inference(inputs=[rgb], data_format="nhwc")
        if not outs:
            raise RuntimeError("推理返回空")
        d = np.asarray(outs[0]).squeeze().astype(np.float32)

        # 放大回正方形区域，再放回原图对应位置（边缘用复制填充）
        d_sq = cv2.resize(d, (s, s), interpolation=cv2.INTER_LINEAR)
        out = np.empty((h, w), dtype=np.float32)
        out[:, :] = float(np.median(d_sq))
        out[y0:y0 + s, x0:x0 + s] = d_sq
        return out


def load(path: str, size: int = MODEL_SIZE) -> MonocularDepth:
    return MonocularDepth(path, size)
