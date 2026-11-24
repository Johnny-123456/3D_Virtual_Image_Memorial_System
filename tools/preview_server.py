#!/usr/bin/env python3
"""相机预览服务：把左目画面缩到 640×360 写成 JPEG，供 Qt 界面读取。

为什么不直接在 Qt 里用 OpenCV
------------------------------
板子上没有 OpenCV 的 C++ 头文件（只有 pip 版，无 dev headers）。
为了不在板端装大包，这里用 Python 出帧、Qt 读文件的方式做预览。

用法（由 ui/studio 自动拉起，通常不需要手动运行）：
    python3 tools/preview_server.py --out /tmp/memorial_preview.jpg
按 Ctrl-C 或收到 SIGTERM 退出。

另外：父进程（ui/studio）如果被 kill -9 掉，本进程会被 init 收养而继续占着
/dev/video21 —— 相机就再也不放了，下次进采集页直接"相机被占用"。所以循环里
检查 getppid()，一旦变成 1（被收养）就自己退出。
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capture.camera_io import CAMERA_BY_ID, FALLBACK_DEVICE  # noqa: E402

_running = True


def _stop(*_):
    global _running
    _running = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/memorial_preview.jpg")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    dev = CAMERA_BY_ID if os.path.exists(CAMERA_BY_ID) else FALLBACK_DEVICE
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print("相机打开失败", flush=True)
        return 1
    for _ in range(8):
        cap.read()

    h = int(a.width * 720 / 1280)
    period = 1.0 / max(a.fps, 1.0)
    tmp = a.out + ".tmp"
    print(f"preview -> {a.out} ({a.width}x{h} @ {a.fps:.0f}fps)", flush=True)

    while _running:
        # 父进程没了（被 kill -9 / 崩溃）就别撑着不放相机
        if os.getppid() == 1:
            print("父进程已退出，释放相机", flush=True)
            break
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        left = frame[:, :1280]
        small = cv2.resize(left, (a.width, h), interpolation=cv2.INTER_AREA)
        # 不能用 cv2.imwrite(tmp)：它靠文件名后缀推断编码器，".tmp" 会直接抛
        # "could not find a writer for the specified extension"，预览一直是黑的。
        ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok2:
            continue
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, a.out)          # 原子替换，避免 Qt 读到半个文件
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)

    cap.release()
    try:
        os.remove(a.out)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
