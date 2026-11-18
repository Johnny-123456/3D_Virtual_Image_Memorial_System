"""Step 0 交付物：双目相机 I/O。

职责边界（只做三件事，深度/渲染不在这里）：
  1. 用 by-id 稳定路径打开相机，协商 MJPG 2560x720
  2. 拆成左右目
  3. 提供带序号与时间戳的取帧接口

设计要点
--------
- **不用 ``cv2.VideoCapture(21)``**：设备号会漂移，换 USB 口或多插一个 UVC 就错位。
- **必须显式设 FOURCC=MJPG**：该相机只有 MJPG，不设可能协商失败或退到 5fps。
- **开局丢弃 WARMUP 帧**：AE/AWB 收敛需要数帧，头几帧直接用于深度会跳变。
- 左右目在同一个 UVC buffer 内，天然同帧同曝光，**无需硬件同步**——这是本方案
  相对拼两台相机的最大优势。

实测坑记录（2026-11-18，2560x720 模式）
---------------------------------------
=========  ==========  ==========  ==============
BUFFERSIZE  fps         中位间隔     说明
=========  ==========  ==========  ==============
1          14.8        67.7 ms     **帧率减半**（驱动没有空闲 buffer 可填）
2          31.1        32.2 ms     30fps 所需的最小缓冲
4（默认）   31.1        32.2 ms     可用，但排队延迟更高
=========  ==========  ==========  ==============

结论：**``BUFFERSIZE=1`` 把帧率砍半**，是常见的错误直觉（以为越少越省延迟）。
取 2：既满帧，排队帧数又最少（最坏附加延迟约 1 帧 = 33 ms）。
另：该相机走 USB 2.0（480M），2560x720 MJPG@30 已接近总线实际上限，
再多插一路高带宽 USB 设备可能掉帧。

用法
----
    from capture.camera_io import StereoCamera
    with StereoCamera() as cam:
        f = cam.grab()          # f.left / f.right 均 (720,1280,3) BGR

    # 命令行自检
    python3 -m capture.camera_io --probe
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# 相机稳定路径。换 USB 口不会变，换相机才会变。
CAMERA_BY_ID = "/dev/v4l/by-id/usb-CCB_Camera_CCB_Camera-video-index0"
FALLBACK_DEVICE = "/dev/video21"

FOURCC = "MJPG"
FRAME_W = 2560          # 左右目并排：各 1280x720
FRAME_H = 720
HALF_W = FRAME_W // 2
DEFAULT_FPS = 30
WARMUP_FRAMES = 10      # AE/AWB 收敛
BUFFERSIZE = 2          # 实测：=1 掉到 15fps；=2 是 30fps 的最小值（见模块 docstring）


@dataclass(frozen=True)
class StereoFrame:
    """一帧双目图像。左右目来自同一 UVC buffer，同帧同曝光。"""

    left: np.ndarray        # (720, 1280, 3) BGR, C-contiguous
    right: np.ndarray
    index: int
    t_capture: float        # time.perf_counter() 时间戳

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.left.shape[:2]
        return w, h


class StereoCamera:
    """双目 UVC 相机。上下文管理器，异常安全。"""

    def __init__(
        self,
        device: Optional[str] = None,
        width: int = FRAME_W,
        height: int = FRAME_H,
        fps: int = DEFAULT_FPS,
        warmup: int = WARMUP_FRAMES,
        bufsize: int = BUFFERSIZE,
    ) -> None:
        self.device = device or resolve_device()
        self.req_width = width
        self.req_height = height
        self.req_fps = fps
        self.warmup = warmup
        self.bufsize = bufsize
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0

    # ---------------- 生命周期 ----------------

    def open(self) -> "StereoCamera":
        if not os.path.exists(self.device):
            raise RuntimeError(
                f"相机设备不存在: {self.device}\n"
                f"  确认相机已插好，或执行: ls -l /dev/v4l/by-id/"
            )

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"打开失败: {self.device}\n"
                f"  常见原因: 被其它进程占用（fuser -v {FALLBACK_DEVICE}）"
            )

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.bufsize)

        # 首帧校验：协商出来的分辨率必须等于请求值，否则后续拆分会错位
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError("相机已打开但读不到帧")

        h, w = frame.shape[:2]
        if w != self.req_width or h != self.req_height:
            cap.release()
            raise RuntimeError(
                f"协商分辨率不符: 得到 {w}x{h}, 需要 {self.req_width}x{self.req_height}。\n"
                f"  该相机 2560x720 仅支持 MJPG，确认 FOURCC 设置生效。"
            )

        self._cap = cap
        for _ in range(self.warmup):            # 丢弃前 N 帧，等 AE/AWB 收敛
            cap.read()
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "StereoCamera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- 取帧 ----------------

    def grab(self) -> Optional[StereoFrame]:
        """取一帧并拆分。读失败返回 None（不抛异常，便于循环内重试）。"""
        if self._cap is None:
            raise RuntimeError("相机未打开，先调用 open()")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None

        # 切片是视图（非连续），显式 copy 保证下游 cv2 调用安全
        left = frame[:, :HALF_W].copy()
        right = frame[:, HALF_W:].copy()
        self._index += 1
        return StereoFrame(left=left, right=right,
                           index=self._index, t_capture=time.perf_counter())

    # ---------------- 属性 ----------------

    @property
    def negotiated_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    @property
    def negotiated_size(self) -> tuple[int, int]:
        if not self._cap:
            return (0, 0)
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def real_bufsize(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE)) if self._cap else 0


def resolve_device() -> str:
    """优先 by-id 稳定路径，退回到固定节点。"""
    if os.path.exists(CAMERA_BY_ID):
        return CAMERA_BY_ID
    return FALLBACK_DEVICE


# ---------------- 命令行自检 ----------------

def _probe(frames: int) -> int:
    print(f"设备: {resolve_device()}")
    t0 = time.perf_counter()
    with StereoCamera() as cam:
        t_open = time.perf_counter() - t0
        nw, nh = cam.negotiated_size
        print(f"打开耗时: {t_open * 1000:.0f} ms  (含 {WARMUP_FRAMES} 帧预热)")
        print(f"协商格式: {FOURCC} {nw}x{nh} @ {cam.negotiated_fps:.0f} fps  "
              f"bufsize={cam.real_bufsize}")

        ts = []
        first = None
        for _ in range(frames):
            f = cam.grab()
            if f is None:
                print("读帧失败")
                return 1
            if first is None:
                first = f
            ts.append(f.t_capture)
        neg_fps = cam.negotiated_fps

    if len(ts) < 2:
        print(f"采样帧数不足: {len(ts)}")
        return 1

    dt = np.diff(np.asarray(ts))
    fps = 1.0 / np.median(dt) if np.median(dt) > 0 else 0.0
    print(f"实测帧率: {fps:.1f} fps (中位间隔 {np.median(dt) * 1000:.1f} ms, "
          f"抖动 p95 {np.percentile(dt, 95) * 1000:.1f} ms)")
    if neg_fps > 0 and fps < 0.8 * neg_fps:
        print(f"⚠ 实测帧率显著低于协商值 {neg_fps:.0f} fps —— 先查 BUFFERSIZE 是否被设成 1")

    L, R = first.left, first.right
    print(f"左目 {L.shape}  右目 {R.shape}")
    if L.shape != (FRAME_H, HALF_W, 3):
        print("✗ 拆分尺寸异常")
        return 1

    # 左右不同 = 真双目；如果完全相同说明拿到的是同一路的复制
    mad = float(np.abs(L.astype(np.int16) - R.astype(np.int16)).mean())
    print(f"左右目平均绝对差: {mad:.2f} (>1 说明是两路不同图像)")

    for name, img in (("左目", L), ("右目", R)):
        print(f"  {name}: 均值 {img.mean():5.1f}  标准差 {img.std():5.1f}  "
              f"过曝像素 {(img.max(axis=2) >= 250).mean() * 100:4.1f}%")

    print("✓ Step 0 通过")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="双目相机 I/O 自检")
    p.add_argument("--probe", action="store_true", help="协商格式/帧率/拆分自检")
    p.add_argument("--frames", type=int, default=60, help="帧率采样帧数")
    a = p.parse_args(argv)
    if a.probe:
        return _probe(a.frames)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
