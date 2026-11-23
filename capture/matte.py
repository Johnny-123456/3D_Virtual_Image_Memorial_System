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


class SalientMatte:
    """通用显著目标抠图（U²-Netp，320×320）。

    为什么需要它：MODNet 是人像专用，换成猫狗会把耳朵、毛吃掉。U²-Netp 是
    通用显著目标分割，人和动物都能出合理的 alpha，代价是边缘比 MODNet 硬一些
    （发丝级别不如人像模型），实测板端 ~190–210 ms/帧。

    预处理是 U²-Net 官方的：RGB → /max → 逐通道减均值除标准差 → NCHW。
    （ONNX 里没有烘焙归一化，转换时 mean=0/std=1 是对的。）
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: str, size: int = 320) -> None:
        from .quiet import silence_stdout

        self.size = size
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
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb / max(float(rgb.max()), 1e-6)
        rgb = (rgb - self.MEAN) / self.STD
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)
        x = small.transpose(2, 0, 1)[None].astype(np.float32)

        outs = self._rk.inference(inputs=[x], data_format="nchw")
        if not outs:
            raise RuntimeError("推理返回空")
        m = np.asarray(outs[0]).squeeze().astype(np.float32)   # d0 = 融合输出
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / max(hi - lo, 1e-6)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(m, 0.0, 1.0)


def load_matte(model_path: str):
    """按文件名挑抠图后端：u2net* → 通用显著目标，其余 → MODNet 人像。"""
    import os
    if "u2net" in os.path.basename(model_path).lower():
        return SalientMatte(model_path)
    return PortraitMatte(model_path)
