#!/usr/bin/env python3
"""Step 5：观众头/眼位追踪，通过 UDP 把归一化视点发给渲染器。

原理
----
复用 rehab-coach 的 YOLOv8n-Pose RKNN（17 关键点，含 nose / left_eye / right_eye），
在 NPU 上约 10-15 ms/帧。

眼点归一化：

    eye_x = (nose_x - cx) / (sens * half_w)
    eye_y = (nose_y - cy) / (sens * half_h)

`--sens` 控制灵敏度：sens=1 时，头部移到画面边缘即达到满偏。

为什么不做真正的 3D 反投影
--------------------------
单目没有绝对深度；用瞳距反推 Z 会引入更多噪声，而 DIBR 的 eye 只需要
"相对偏移"，与距离无关（视差量本身已含 1/z）。所以直接用归一化像素偏移，
配合平滑滤波即可。

输出协议
--------
UDP -> 127.0.0.1:9999，一行 ASCII：`ex,ey\\n`（各 -1..1）
检测不到人时发送 `nan,nan`，渲染端据此保持上一次位置。

用法
----
    python3 track/eye_tracker.py                 # 实时推流
    python3 track/eye_tracker.py --image a.png   # 离线验证一张图
    python3 track/eye_tracker.py --print         # 同时打印到终端
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REHAB = os.path.expanduser("~/rehab-coach/src/backend")
REHAB = os.path.join(_REHAB, "vision")
MODEL = os.path.expanduser("~/rehab-coach/models/yolov8n-pose.rknn")
DEV = "/dev/v4l/by-id/usb-CCB_Camera_CCB_Camera-video-index0"

if _REHAB not in sys.path:
    sys.path.insert(0, _REHAB)
if REHAB not in sys.path:
    sys.path.insert(0, REHAB)


class EyeTracker:
    def __init__(self, model: str = MODEL, sens: float = 1.0, alpha: float = 0.35):
        from pose_rknn import RknnPoseBackend
        self.pose = RknnPoseBackend(model, imgsz=640, conf_threshold=0.4)
        self.sens = sens
        self.alpha = alpha
        self.ex = 0.0
        self.ey = 0.0

    def close(self):
        try:
            self.pose.release()
        except Exception:
            pass

    def process(self, bgr: np.ndarray):
        """返回 (ex, ey, info)；检测失败返回 (None, None, info)。"""
        h, w = bgr.shape[:2]
        det = self.pose.detect(bgr)
        if not det:
            return None, None, {"ms": 0.0, "conf": 0.0, "nose": None}

        kp = det["keypoints_2d"]
        if "nose" in kp and kp["nose"]["conf"] > 0.3:
            px, py = kp["nose"]["x"], kp["nose"]["y"]
        elif "left_eye" in kp and "right_eye" in kp:
            px = (kp["left_eye"]["x"] + kp["right_eye"]["x"]) / 2
            py = (kp["left_eye"]["y"] + kp["right_eye"]["y"]) / 2
        else:
            return None, None, {"ms": det.get("inference_ms", 0), "conf": 0.0,
                                "nose": None}

        cx, cy = w / 2.0, h / 2.0
        raw_x = (px - cx) / (self.sens * (w / 2.0))
        raw_y = (py - cy) / (self.sens * (h / 2.0))

        self.ex += self.alpha * (float(np.clip(raw_x, -1, 1)) - self.ex)
        self.ey += self.alpha * (float(np.clip(raw_y, -1, 1)) - self.ey)
        return self.ex, self.ey, {
            "ms": det.get("inference_ms", 0),
            "conf": det.get("person_conf", 0),
            "nose": (px, py),
        }


def grab_left():
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    ok = False
    for _ in range(8):
        ok, f = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("相机读帧失败")
    return cap, f[:, :1280].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="眼位追踪 -> UDP")
    ap.add_argument("--image", default=None, help="离线验证单张图")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--sens", type=float, default=1.0)
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(MODEL):
        print(f"✗ 缺少模型: {MODEL}")
        return 1

    tr = EyeTracker(sens=a.sens)

    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            print("读图失败:", a.image)
            tr.close()
            return 1
        ex, ey, info = tr.process(img)
        tr.close()
        if ex is None:
            print("✗ 未检测到人体关键点")
            return 2
        print(f"图像 {img.shape[1]}x{img.shape[0]}  推理 {info['ms']} ms  "
              f"人体置信度 {info['conf']:.3f}")
        print(f"鼻尖像素 {info['nose']}")
        print(f"归一化眼点 ex={ex:+.3f} ey={ey:+.3f}")
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", a.port)
    cap, left = grab_left()
    print(f"追踪中 -> udp {addr[0]}:{addr[1]}   (Ctrl-C 退出)")
    n = 0
    t0 = time.time()
    try:
        while True:
            ex, ey, info = tr.process(left)
            if ex is None:
                sock.sendto(b"nan,nan\n", addr)
            else:
                sock.sendto(("%.4f,%.4f\n" % (ex, ey)).encode(), addr)
            n += 1
            if a.print and n % 15 == 0:
                fps = n / max(time.time() - t0, 1e-6)
                print(f"  {fps:4.1f} fps  ex={ex if ex is None else round(ex, 3)} "
                      f"ey={ey if ey is None else round(ey, 3)}  infer {info['ms']}ms")
            ok, f = cap.read()
            if not ok:
                break
            left = f[:, :1280]
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tr.close()
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
