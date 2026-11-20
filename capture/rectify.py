"""Step 1 交付物：双目极线校正（数据驱动，恒定缩放）。

为什么不用标定文件的 stereoRectify
----------------------------------
`rehab-coach` 的 `stereo_calib.json` 在本相机上**不可用**（实测证据）：

- 外参 T = (-47.23, 2.34, -39.54) mm，基线与 x 轴夹角 **40°**；
- 据此 stereoRectify 得到 R1 含 40° 旋转、P1 的 cx = **-885.5**（负数）；
- alpha=0 时 fx 从 1198 被逼到 **5579（4.66 倍变焦）**，公共有效区只剩 56.9%；校正后特征匹配直接失效。

为什么也不用 stereoRectifyUncalibrated
--------------------------------------
它在 dy 指标上很好看（0.196 px），但实测其单应**局部退化**：

====================  ==========  ==========
位置                  H_left 缩放  H_right 缩放
====================  ==========  ==========
画面中心 (640,360)     1.279       1.000
有效区中心 (794,360)   1.755       1.300
左上 (312,4)           0.761       0.636
右下 (1276,716)        **13.65**   5.336
====================  ==========  ==========

左右缩放比在有效区内从 1.35 变到 2.56——**同一个视差值在不同位置对应不同深度**，
稠密深度不可用。它只保证了"行对齐"，却把几何搞坏了。

采用方案：单应性相似变换
------------------------
只变换**右目**（左目保持原始视角，作为渲染参考图），模型为
`p_left = s·R·p_right + t`（4 自由度：缩放、旋转、平移）。

- 恒定缩放 → 视差在所有位置等比，几何一致
- 左目不变 → 保留真实焦距 fx≈1198.7 与原始图像质量
- 把两目间约 143 px 的固定水平偏移归一化掉，SGBM 才能用 minDisparity=0

实测：中位 |Δy| **0.239 px**，96.4% < 1 px（优于 1 px 阈值）。
相似变换的缩放约 0.9964（0.36% 误差），旋转约 0.03°。

模块内的 `_disparity_scale_variation` 是防退化闸门：若某个候选单应让视差尺度
在画面内变化超过 5%，直接判为不可用并回退到相似变换。
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "rectify.json")
DEPTH_CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "config", "depth_calib.json")

# 标定文件里唯一还可信的量：焦距量级（决定 FOV，1/4" 传感器 1280 宽 ≈ 1198 px）
FX_HINT = 1198.7

# 视差尺度在画面内的允许变化比例，超过即判定单应退化
MAX_SCALE_VARIATION = 0.05

# 是否在相似变换中保留平移分量。必须为 False：平移会吸收场景中位视差，
# 破坏绝对尺度，使视差-深度关系随场景变化。
ZERO_TRANSLATION = True


# ---------------------------------------------------------------- 特征

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
    """用匹配点的 |Δy| 衡量极线对齐度（越小越接近行对齐）。"""
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


# ---------------------------------------------------------------- 几何健康度

def local_scale(H: np.ndarray, px: float, py: float) -> float:
    """单应在 (px,py) 处的局部缩放（雅可比行列式的平方根）。"""
    eps = 1.0

    def f(xx, yy):
        d = H[2, 0] * xx + H[2, 1] * yy + H[2, 2]
        if abs(d) < 1e-12:
            return np.array([np.inf, np.inf])
        return np.array([(H[0, 0] * xx + H[0, 1] * yy + H[0, 2]) / d,
                         (H[1, 0] * xx + H[1, 1] * yy + H[1, 2]) / d])

    dpx = (f(px + eps, py) - f(px - eps, py)) / (2 * eps)
    dpy = (f(px, py + eps) - f(px, py - eps)) / (2 * eps)
    J = np.array([dpx, dpy]).T
    det = abs(np.linalg.det(J))
    return float("inf") if det <= 0 else float(np.sqrt(det))


def disparity_scale_variation(H1: np.ndarray, H2: np.ndarray,
                              roi: Tuple[int, int, int, int], grid: int = 5) -> float:
    """左右目缩放比在有效区内的相对变化幅度。0 = 完美恒定缩放。

    这是判断单应是否退化的关键指标：只保证 dy 小是不够的，
    若缩放比随位置变化，视差与深度就不再是线性关系。
    """
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return float("inf")
    ratios = []
    for gx in range(grid):
        for gy in range(grid):
            px = x + (gx + 0.5) * w / grid
            py = y + (gy + 0.5) * h / grid
            s1 = local_scale(H1, px, py)
            s2 = local_scale(H2, px, py)
            if not np.isfinite(s1) or not np.isfinite(s2) or s2 <= 1e-9:
                return float("inf")
            ratios.append(s1 / s2)
    r = np.asarray(ratios)
    return float((r.max() - r.min()) / max(abs(r.mean()), 1e-9))


# ---------------------------------------------------------------- 估计

def _valid_roi(H1, H2, w, h, margin: int = 4):
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def bbox(H):
        pts = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        x0 = max(0, int(np.ceil(pts[:, 0].min()))) + margin
        y0 = max(0, int(np.ceil(pts[:, 1].min()))) + margin
        x1 = min(w, int(np.floor(pts[:, 0].max()))) - margin
        y1 = min(h, int(np.floor(pts[:, 1].max()))) - margin
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    a, b = bbox(H1), bbox(H2)
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2])
    y1 = min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _fit_rotation_only(p1: np.ndarray, p2: np.ndarray, w: int, h: int):
    """估计右目 → 左目的纯旋转，旋转中心固定在画面中心。

    为什么不拟合平移：平移量会收录场景的中位视差，使"视差→深度"的绝对关系
    随场景而变，无法用于标定。这里只求旋转（以及绕中心推出的定心平移），
    相机固有的视差偏移被完整保留。

    用 1-D 粗到细搜索最小化匹配点对的垂直残差中位数。
    """
    cx, cy = w / 2.0, h / 2.0
    ctr = np.array([cx, cy], dtype=np.float64)

    def score_deg(deg):
        th = np.deg2rad(deg)
        c, sn = np.cos(th), np.sin(th)
        R = np.array([[c, -sn], [sn, c]])
        q = (p2 - ctr) @ R.T + ctr
        return float(np.median(np.abs(p1[:, 1] - q[:, 1])))

    coarse = np.arange(-4.0, 4.001, 0.05)
    best_deg = min(coarse, key=score_deg)
    fine = np.arange(best_deg - 0.1, best_deg + 0.101, 0.005)
    best_deg = min(fine, key=score_deg)

    th = np.deg2rad(best_deg)
    c, sn = np.cos(th), np.sin(th)
    R = np.array([[c, -sn], [sn, c]])

    H1 = np.eye(3, dtype=np.float64)
    H2 = np.eye(3, dtype=np.float64)
    H2[:2, :2] = R
    H2[0, 2] = cx - (R[0, 0] * cx + R[0, 1] * cy)   # 绕画面中心旋转
    H2[1, 2] = cy - (R[1, 0] * cx + R[1, 1] * cy)
    return H1, H2, float(best_deg)


def estimate_rectification(gL: np.ndarray, gR: np.ndarray,
                           method: str = "auto",
                           size: Tuple[int, int] = (1280, 720)) -> dict:
    """估计校正变换。method: auto | similarity | uncalibrated。"""
    p1, p2 = _sift_match(gL, gR)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999)
    if F is None or mask is None:
        raise RuntimeError("本质矩阵估计失败")
    inl = mask.ravel() == 1
    p1i = np.ascontiguousarray(p1[inl])
    p2i = np.ascontiguousarray(p2[inl])
    if len(p1i) < 20:
        raise RuntimeError(f"内点太少: {len(p1i)}")

    w, h = size
    candidates = []      # (name, H1, H2)
    extra = {}

    if method in ("auto", "rotation"):
        _h1, _h2, _deg = _fit_rotation_only(p1i, p2i, w, h)
        candidates.append(("rotation", _h1, _h2))
        extra["rotation"] = round(_deg, 4)

    if method in ("auto", "similarity"):
        M, _ = cv2.estimateAffinePartial2D(p2i, p1i, method=cv2.RANSAC,
                                           ransacReprojThreshold=3.0)
        if M is not None:
            M = M.astype(np.float64).copy()
            if ZERO_TRANSLATION:
                # 关键：平移量是对**整个场景**拟合出来的，会连同场景中位视差一起抹掉，
                # 使"视差→深度"的绝对关系随场景而变，无法用于标定。
                # 只保留 2x2 部分（旋转+缩放），把相机固有的视差偏移保留下来，
                # 由一个显式常量描述（config 里的 disparity_offset）。
                M[0, 2] = 0.0
                M[1, 2] = 0.0
            candidates.append(("similarity", np.eye(3, dtype=np.float64),
                               np.vstack([M, [0, 0, 1]]).astype(np.float64)))
    if method in ("auto", "uncalibrated"):
        ret, h1, h2 = cv2.stereoRectifyUncalibrated(p1i, p2i, F, (w, h), threshold=1.0)
        if ret and h1 is not None and h2 is not None:
            candidates.append(("uncalibrated",
                               h1.astype(np.float64), h2.astype(np.float64)))

    if not candidates:
        raise RuntimeError("所有校正方法都失败")

    best = None
    best_key = None
    rejected = []
    for name, H1, H2 in candidates:
        roi = _valid_roi(H1, H2, w, h)
        if roi[2] < w * 0.5 or roi[3] < h * 0.5:
            rejected.append((name, "有效区不足 50%"))
            continue
        var = disparity_scale_variation(H1, H2, roi)
        if not np.isfinite(var) or var > MAX_SCALE_VARIATION:
            rejected.append((name, f"视差尺度变化 {var * 100:.1f}% > "
                                   f"{MAX_SCALE_VARIATION * 100:.0f}%（单应退化）"))
            continue
        wL = cv2.warpPerspective(gL, H1, (w, h))
        wR = cv2.warpPerspective(gR, H2, (w, h))
        err = epipolar_error(wL, wR)
        r = {"method": name, "image_size": [w, h],
             "H_left": H1.tolist(), "H_right": H2.tolist(),
             "valid_roi": list(roi),
             "scale_variation": round(var, 5),
             "rotation_deg": extra.get(name),
             "epipolar": err}
        # 几何健康度优先：尺度变化越小越好，其次才是极线误差。
        # 只追求 dy 小可能选到退化单应（dy 好看但视差被非均匀缩放）。
        key = (round(var, 4), err["dy_median"])
        if best_key is None or key < best_key:
            best, best_key = r, key

    if best is None:
        raise RuntimeError("所有候选都被拒：" + "; ".join(f"{n}:{m}" for n, m in rejected))
    if rejected:
        best["rejected"] = [{"method": n, "reason": m} for n, m in rejected]
    return best


_depth_calib_cache: Optional[dict] = None


def load_depth_calib(path: str = DEPTH_CALIB_PATH) -> Optional[dict]:
    """加载两点标定得到的深度尺度参数。缺失时返回 None。"""
    global _depth_calib_cache
    if _depth_calib_cache is not None:
        return _depth_calib_cache
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        _depth_calib_cache = json.load(f)
    return _depth_calib_cache


def save_rectification(result: dict, path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = dict(result)
    result["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- 校正器

class Rectifier:
    """加载校正变换，把原始左右目映射到行对齐的像对。"""

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

    def disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        """绝对视差 → 米制深度。

            Z = K / (d - d_inf),   K = fx * B

        `d_inf` 是相机固有的无穷远视差偏移（本机约 -156 px），**不能省略**：
        它来自两目主点不一致，数值上比物体视差还大，漏掉会让深度差出数倍。

        参数来自 config/depth_calib.json（两点标定产物）。缺失时返回全 0。
        """
        c = load_depth_calib()
        out = np.zeros_like(disp, dtype=np.float32)
        if not c:
            return out
        denom = disp.astype(np.float32) - float(c["disparity_inf_px"])
        # disp == 0 是"无效"哨兵，必须显式排除：
        # 否则 0 - d_inf ≈ 156 会算出 0.4 m 的伪深度（0 也落在有效区间内时尤其危险）
        ok = (denom > 1e-3) & (disp != 0)
        out[ok] = float(c["k_px_m"]) / denom[ok]
        return out

    def summary(self) -> dict:
        e = self.meta.get("epipolar", {})
        return {
            "method": self.meta.get("method"),
            "image_size": self.image_size,
            "valid_roi": self.valid_roi,
            "scale_variation": self.meta.get("scale_variation"),
            "epipolar_dy_median": round(e.get("dy_median", float("nan")), 3),
            "epipolar_within_1px": round(e.get("within_1px", 0) * 100, 1),
            "created": self.meta.get("created"),
        }
