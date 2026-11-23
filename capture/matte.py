"""人像抠图（MODNet on RKNN）。

定位
----
替代 V1 里"用深度阈值当遮罩"的粗做法。深度阈值没有语义——
头发和背景深度接近时必然糊，而且没有 alpha 羽化。

MODNet 输出的是**连续 alpha**，专门为人像/视频抠图设计，发丝级边缘表现好。

性能
----
实测 RK3588：**≈287 ms / 帧**（512×512）。
比深度模型（763 ms）便宜，且**每个资产只需算一次**（离线），
所以这个耗时完全不影响产品。

归一化：MODNet 期望 (x/255 - 0.5) / 0.5，已烘焙进 .rknn（mean=std=127.5），
因此这里直接喂 uint8 RGB。
"""
from __future__ import annotations

import cv2
import numpy as np

MODEL_SIZE = 512


class PortraitMatte:
    """MODNet 人像抠图（RK3588 NPU）。"""

    def __init__(self, model_path: str, size: int = MODEL_SIZE) -> None:
        from .quiet import silence_stdout

        self.size = size
        # 见 capture/mono_depth.py：librknnrt 在加载时缓存 fd 1，必须连 import 一起静音
        with silence_stdout():
            from rknnlite.api import RKNNLite
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
        """BGR 图 -> float32 alpha（0..1），尺寸与输入一致。"""
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)

        outs = self._rk.inference(inputs=[rgb], data_format="nhwc")
        if not outs:
            raise RuntimeError("推理返回空")
        a = np.asarray(outs[0]).squeeze().astype(np.float32)
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(a, 0.0, 1.0)


def composite(bgr: np.ndarray, alpha: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """把前景按 alpha 叠到背景上。"""
    a = alpha[:, :, None]
    return (bgr.astype(np.float32) * a + bg_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)
