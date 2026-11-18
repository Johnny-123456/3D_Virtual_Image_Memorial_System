"""Step 1 交付物：双目极线校正（数据驱动）。

为什么不用标定文件的 stereoRectify
----------------------------------
`rehab-coach` 里的 `stereo_calib.json` 在本相机上**不可用**，实测证据：

- 外参 T = (-47.23, 2.34, -39.54) mm，基线与 x 轴夹角 **40°**；
- 据此 stereoRectify 得到 R1 含 40° 旋转、P1 的 cx = **-885.5**（负数）；
- alpha=0 时 fx 从 1198 被逼到 **5579（4.66 倍变焦）**，公共有效区只剩 56.9%；
- 校正后特征匹配直接失效。

而实测原始拆半图**本身已接近行对齐**（分块匹配 dy 中位 -1 px），
说明这是真立体且出厂已基本对准，只是存在约 143 px 水平偏移 + 0.36% 尺度差。

因此改为**从图像自身估计校正变换**（不信任内参）：

- 主路径：`cv2.stereoRectifyUncalibrated`（SIFT 匹配 → 本质矩阵 → H1/H2）
- 备选：`estimateAffinePartial2D`（相似变换，4 自由度）

标定文件里的 fx≈1198 仍可作为估计值用于量纲换算，见 `depth_note`。
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "rectify.json")

# 标定文件里唯一还可信的量：焦距量级（决定 FOV，1/4" 传感器 1280 宽 ≈ 1198 px）
FX_HINT = 1198.7


# ---------------------------------------------------------------- 估计

def _sift_match(gL: np.ndarray, gR: np.ndarray, nfeatures: int = 6000,
                ratio: float = 0.7):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(gL, None)
    kp2, des2 = sift.detectAndCompute(gR, None)
    if des1 is None or des2 is None:
        raise RuntimeError("特征不足：场景纹理太弱")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(des1, des2, k=2) if m.distance < ratio * n.distance]
    if len(good) < 30:
        raise RuntimeError(f"匹配对太少: {len(good)}")
    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return p1, p2


def epipolar_error(gL: np.ndarray, gR: np.ndarray) -> dict:
    """用匹配点的 |Δy| 衡量极线对齐程度。Δy 越小说明越接近"行对齐"。"""
    p1, p2 = _sift_match(gL, gR)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
    inl = (mask.ravel() == 1) if mask is not None else np.ones(len(p1), bool)
    dy = np.abs(p1[inl, 1] - p2[inl, 1])
    if dy.size == 0:
        return {"n": 0, "dy_median": float("nan"), "dy_p95": float("nan"),
                "within_1px": 0.0}
    return {
        "n": int(dy.size),
        "dy_median": float(np.median(dy)),
        "dy_p95": float(np.percentile(dy, 95)),
        "within_1px": float((dy < 1.0).mean()),
    }


def estimate_rectification(gL: np.ndarray, gR: np.ndarray,
                           method: str = "uncalibrated",
                           size: Tuple[int, int] = (1280, 720)) -> dict:
    """从一对灰度图估计校正变换，返回含 H1/H2 与质量指标的 dict。"""
    p1, p2 = _sift_match(gL, gR)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None or mask is None:
        raise RuntimeError("本质矩阵估计失败")
    inl = mask.ravel() == 1
    p1i = np.ascontiguousarray(p1[inl])
    p2i = np.ascontiguousarray(p2[inl])
    if len(p1i) < 20:
        raise RuntimeError(f"内点太少: {len(p1i)}")

    H1 = H2 = None
    used = method
    if method == "uncalibrated":
        ret, h1, h2 = cv2.stereoRectifyUncalibrated(p1i, p2i, F, size, threshold=1.0)
        if ret and h1 is not None and h2 is not None:
            H1, H2 = h1.astype(np.float64), h2.astype(np.float64)

    if H1 is None:                       # 备选：相似变换（只变换右目）
        used = "similarity"
        M, _ = cv2.estimateAffinePartial2D(p2i, p1i, method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
        if M is None:
            raise RuntimeError("相似变换估计失败")
        H1 = np.eye(3, dtype=np.float64)
        H2 = np.vstack([M, [0, 0, 1]]).astype(np.float64)

    wL = cv2.warpPerspective(gL, H1, size)
    wR = cv2.warpPerspective(gR, H2, size)
    err = epipolar_error(wL, wR)

    h, w = size[1], size[0]
    valid = _valid_roi(H1, H2, w, h)
    return {
        "method": used,
        "image_size": [w, h],
        "H_left": np.asarray(H1).tolist(),
        "H_right": np.asarray(H2).tolist(),
        "valid_roi": list(valid),
        "epipolar": err,
    }


def _valid_roi(H1, H2, w, h, margin: int = 4):
    """求两幅变换图都落在原画面内的公共区域（去掉黑边）。"""
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def bbox(H):
        pts = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        x0 = max(0, int(np.ceil(pts[:, 0].min()))) + margin
        y0 = max(0, int(np.ceil(pts[:, 1].min()))) + margin
        x1 = min(w, int(np.floor(pts[:, 0].max()))) - margin
        y1 = min(h, int(np.floor(pts[:, 1].max()))) - margin
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    a = bbox(H1)
    b = bbox(H2)
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2])
    y1 = min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


# ---------------------------------------------------------------- 校正器

class Rectifier:
    """加载（或现估）校正变换，把原始左右目映射到行对齐的像对。"""

    def __init__(self, config_path: str = CONFIG_PATH) -> None:
        self.config_path = config_path
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"缺少校正配置: {config_path}\n"
                f"  先运行: python3 tools/rectify_selfcal.py")
        with open(config_path, "r", encoding="utf-8") as f:
            c = json.load(f)
        self.H1 = np.asarray(c["H_left"], dtype=np.float64)
        self.H2 = np.asarray(c["H_right"], dtype=np.float64)
        self.image_size = tuple(c["image_size"])
        self.valid_roi = tuple(c["valid_roi"])
        self.meta = c

    # --- 校正 ---

    def rectify_pair(self, left: np.ndarray, right: np.ndarray):
        w, h = self.image_size
        return (cv2.warpPerspective(left, self.H1, (w, h), flags=cv2.INTER_LINEAR),
                cv2.warpPerspective(right, self.H2, (w, h), flags=cv2.INTER_LINEAR))

    def rectify_gray(self, left: np.ndarray, right: np.ndarray):
        gL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        gR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
        return self.rectify_pair(gL, gR)

    def crop(self, img: np.ndarray) -> np.ndarray:
        x, y, w, h = self.valid_roi
        return img[y:y + h, x:x + w]

    # --- 视差 → 深度（尺度待 Step 3 标定）---

    def disparity_to_depth(self, disp: np.ndarray,
                           baseline_mm: Optional[float] = None) -> np.ndarray:
        """Z[m] = fx * B / d。baseline 未标定前只作相对量使用。"""
        b = baseline_mm if baseline_mm is not None else self.meta.get("baseline_mm")
        out = np.zeros_like(disp, dtype=np.float32)
        if not b:
            return out
        valid = disp > 0
        out[valid] = (FX_HINT * b * 1e-3) / disp[valid]
        return out

    def summary(self) -> dict:
        e = self.meta.get("epipolar", {})
        return {
            "method": self.meta.get("method"),
            "image_size": self.image_size,
            "valid_roi": self.valid_roi,
            "epipolar_dy_median": round(e.get("dy_median", float("nan")), 3),
            "epipolar_within_1px": round(e.get("within_1px", 0) * 100, 1),
            "created": self.meta.get("created"),
        }


def save_rectification(result: dict, path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import datetime
    result = dict(result)
    result["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
