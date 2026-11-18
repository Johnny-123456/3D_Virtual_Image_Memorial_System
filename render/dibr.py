"""Step 4 交付物：DIBR（Depth-Image-Based Rendering）参考实现。

设计原则
--------
1. **不做硬分层**。把深度切成 N 层会出现层间割裂。这里用连续的视差场，
   位移量平滑地随深度变化。
2. **先做可离屏验证的 CPU 参考实现**。GPU 版（Qt/OpenGL 顶点着色器）另写，
   但算法与参数先在这里调对——在板子上调 GLSL 代价太高。
3. **视差不与深度成正比，而与 1/z 之差成正比**：

       parallax_px = eye * gain * (1/z_focus - 1/z)

   `z_focus` 处位移为 0（"贴在屏幕上"），比它近的前移、比它远的后移，
   形成反向相对位移——这才是纵深感的来源。

参数标定
--------
`gain` 是唯一的艺术参数，单位 px·m：它把"景物比焦点近多少"换成位移像素数。
物理上应取 `fx·B·视点位移/屏距` 的组合，但直接算出来通常偏小或偏大，
实操上按观感调。建议初值 55~80，超过 100 会在物体边缘出现明显拉伸。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DibrParams:
    parallax_gain: float = 62.0      # px·m，视差强度
    z_focus: Optional[float] = None  # None = 自动取主体深度
    z_min: float = 0.3               # 深度夹紧
    z_max: float = 15.0
    max_shift_px: float = 60.0       # 位移硬上限，防炸
    solve_iters: int = 3             # 逆向映射定点迭代次数
    hole_inpaint_radius: int = 3
    depth_median_ksize: int = 5      # 深度中值滤波核（必须为奇数）
    depth_bilateral: float = 0.35    # 双边滤波 sigmaColor（米），0=关闭


class DibrRenderer:
    """把 (RGB, 深度) 渲染成任意视点的 2.5D 视图。"""

    def __init__(self, params: Optional[DibrParams] = None) -> None:
        self.p = params or DibrParams()
        self._z: Optional[np.ndarray] = None
        self._z_focus: float = 0.0
        self._grid: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ---------------- 预处理（每张图一次）----------------

    def prepare(self, depth_m: np.ndarray,
                focus_box: Optional[Tuple[int, int, int, int]] = None) -> None:
        """填洞 + 定焦点。之后每帧只调 render()。"""
        z = np.asarray(depth_m, dtype=np.float32).copy()
        z = np.clip(z, 0.0, self.p.z_max)

        valid = z > 0
        if valid.sum() < z.size * 0.05:
            raise ValueError("有效深度不足 5%，无法渲染")

        # 在 1/z 域填洞：视差与 1/z 线性相关，比在 z 域插值更符合几何
        inv = np.zeros_like(z)
        inv[valid] = 1.0 / z[valid]
        lo, hi = np.percentile(inv[valid], [2, 98])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)
        n8 = (norm * 255).astype(np.uint8)
        if (~valid).any():
            n8 = cv2.inpaint(n8, (~valid).astype(np.uint8) * 255,
                             self.p.hole_inpaint_radius, cv2.INPAINT_TELEA)
        norm = n8.astype(np.float32) / 255.0
        inv = lo + norm * (hi - lo)
        z = 1.0 / np.maximum(inv, 1e-4)

        # 平滑：SGBM 在弱纹理区（白墙、天花板）会给出抖动很深的伪深度，
        # 直接拿去做 DIBR 会把这些噪声拉伸成肉眼可见的花斑。
        # 中值先压离群点，双边再在保边的前提下抹平。
        k = self.p.depth_median_ksize
        if k and k >= 3:
            z = cv2.medianBlur(z.astype(np.float32), k if k % 2 else k + 1)
        if self.p.depth_bilateral > 0:
            z = cv2.bilateralFilter(z.astype(np.float32), 9,
                                    self.p.depth_bilateral, 9)

        self._z = np.clip(z, self.p.z_min, self.p.z_max)

        # 焦点深度：主体区域中位（默认画面中央 50%×65%）
        if self.p.z_focus:
            self._z_focus = float(self.p.z_focus)
        else:
            h, w = z.shape
            if focus_box is None:
                focus_box = (int(w * 0.25), int(h * 0.10),
                             int(w * 0.50), int(h * 0.65))
            bx, by, bw, bh = focus_box
            self._z_focus = float(np.median(self._z[by:by + bh, bx:bx + bw]))

        h, w = self._z.shape
        self._grid = (np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32)))

    @property
    def z_focus(self) -> float:
        return self._z_focus

    @property
    def depth(self) -> Optional[np.ndarray]:
        return self._z

    # ---------------- 渲染 ----------------

    def parallax_field(self, eye: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """给定归一化视点（-1..1），返回每像素的位移量(px)。"""
        if self._z is None:
            raise RuntimeError("先调用 prepare()")
        ex, ey = eye
        p = (1.0 / self._z_focus) - (1.0 / self._z)   # 焦点处为 0
        dx = np.clip(ex * self.p.parallax_gain * p,
                     -self.p.max_shift_px, self.p.max_shift_px)
        dy = np.clip(ey * self.p.parallax_gain * p,
                     -self.p.max_shift_px, self.p.max_shift_px)
        return dx.astype(np.float32), dy.astype(np.float32)

    def render(self, rgb: np.ndarray, eye: Tuple[float, float],
               backfill_blur: int = 0) -> np.ndarray:
        """把 rgb 按深度场重投影到视点 eye（归一化 -1..1）。

        backfill_blur > 0 时，未被覆盖的区域用模糊版本回填，
        模拟"背景虚化"，同时掩掉前景拉伸。
        """
        if self._z is None or self._grid is None:
            raise RuntimeError("先调用 prepare()")

        h, w = self._z.shape
        u, v = self._grid
        dx, dy = self.parallax_field(eye)

        # 逆向映射：正向是 src -> dst(src + d(src))，
        # 要得到 dst -> src 需解不动点；本量级位移 3 次迭代足够收敛
        sx, sy = u.copy(), v.copy()
        for _ in range(self.p.solve_iters):
            dxs = cv2.remap(dx, sx, sy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            dys = cv2.remap(dy, sx, sy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            sx = u - dxs
            sy = v - dys

        out = cv2.remap(rgb, sx, sy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

        holes = ((sx < 0) | (sx > w - 1) | (sy < 0) | (sy > h - 1)).astype(np.uint8)
        if holes.any():
            if backfill_blur > 0:
                bg = cv2.GaussianBlur(out, (0, 0), backfill_blur)
                m = cv2.GaussianBlur((holes * 255).astype(np.uint8), (0, 0), 2)
                a = (m.astype(np.float32) / 255.0)[:, :, None]
                out = np.clip(out.astype(np.float32) * (1 - a) +
                              bg.astype(np.float32) * a, 0, 255).astype(np.uint8)
            else:
                out = cv2.inpaint(out, holes * 255, self.p.hole_inpaint_radius,
                                  cv2.INPAINT_TELEA)
        return out

    def anaglyph(self, rgb: np.ndarray, eye_sep: float = 0.5) -> np.ndarray:
        """红青立体图，用来快速判断视差方向与强度是否正常。"""
        left = self.render(rgb, (-eye_sep, 0.0))
        right = self.render(rgb, (eye_sep, 0.0))
        out = np.zeros_like(rgb)
        out[:, :, 2] = left[:, :, 2]      # 红 = 左眼
        out[:, :, 1] = right[:, :, 1]
        out[:, :, 0] = right[:, :, 0]
        return out
