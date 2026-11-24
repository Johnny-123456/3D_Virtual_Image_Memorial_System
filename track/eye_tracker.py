#!/usr/bin/env python3
"""Step 5：观众头/眼位追踪，通过 UDP 把归一化视点发给渲染器。

原理
----
复用 rehab-coach 的 YOLOv8n-Pose RKNN（17 关键点，含 nose / left_eye / right_eye），
在 NPU 上约 10-15 ms/帧。

眼点归一化：

    eye_x = (nose_x - cx) / (sens * half_w)
    eye_y = (nose_y - cy) / (sens * half_h)

`--sens` 控制灵敏度：eye 满偏对应鼻尖偏离中心 sens·(w/2) 像素。
主语 1.2 m 处 fx≈1198.7，所以 sens=0.3 ↔ 横移约 19 cm 满偏；sens=1 要移到
画面边缘（约 64 cm）才满偏——那样 §8 要求的"头动 15 cm 看得出视差"根本不够。

为什么不做真正的 3D 反投影
--------------------------
单目没有绝对深度；用瞳距反推 Z 会引入更多噪声，而 DIBR 的 eye 只需要
"相对偏移"，与距离无关（视差量本身已含 1/z）。所以直接用归一化像素偏移，
配合平滑滤波即可。

输出协议
--------
UDP -> 127.0.0.1:9999，一行 ASCII：`ex,ey\n`（各 -1..1）
检测不到人时发送 `nan,nan`，渲染端据此保持上一次位置。
加 `--stamp` 时末尾多一个抓帧时刻：`ex,ey,<epoch 秒>`，渲染端据此显示
端到端延迟（老渲染端会忽略第三个字段，协议向后兼容）。

用法
----
    python3 track/eye_tracker.py                 # 实时推流
    python3 track/eye_tracker.py --image a.png   # 离线验证一张图
    python3 track/eye_tracker.py --print         # 同时打印到终端
    python3 track/eye_tracker.py --replay a.png --stamp   # 无人时测延迟
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
MODEL = None   # 由 capture.pose 统一解析（models/ 优先，~/rehab-coach 兜底）
DEV = "/dev/v4l/by-id/usb-CCB_Camera_CCB_Camera-video-index0"

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class EyeTracker:
    def __init__(self, model: str | None = MODEL, sens: float = 0.3, alpha: float = 0.35):
        from capture.pose import load_pose
        self.pose = load_pose(model, imgsz=640, conf_threshold=0.4)
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
        from capture.landmarks import detect_main_person

        h, w = bgr.shape[:2]
        # 只在画面中央检测：直接整帧 detect 会被画面边缘的类人干扰抢走目标
        # （实测：portrait3 那张图上，整帧检测把鼻尖定位到右上角挂着的夹克上，
        #  x=1252 而真人鼻尖在 567）。与 finalize_asset / mono_calib 同一套判定。
        det = detect_main_person(self.pose, bgr)
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


class FrameGrabber:
    """专职抓帧线程：永远只保留最新一帧。

    为什么不在主循环里串行 cap.read()：推理要 50 ms，一次循环 85 ms，而相机
    出帧只要 33 ms。串行读到的永远是 2~3 帧前的画面（实测帧龄 70~210 ms）。
    开个线程一直读、只留最新一帧，主循环拿到的就是 33 ms 内的画面。
    """

    def __init__(self, buffers: int = 2):
        import threading

        self.cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffers)
        ok = False
        for _ in range(8):
            ok, f = self.cap.read()
        if not ok:
            self.cap.release()
            raise SystemExit("相机读帧失败")

        self._lock = threading.Lock()
        self._frame = f[:, :1280].copy()
        self._age = 0.0
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                break
            # POS_MSEC 是 monotonic 毫秒：这一帧"过期"了多久
            age = time.monotonic() * 1000.0 - self.cap.get(cv2.CAP_PROP_POS_MSEC)
            left = f[:, :1280].copy()
            with self._lock:
                self._frame = left
                self._age = age

    def latest(self):
        with self._lock:
            return self._frame, self._age

    def release(self):
        self._stop = True
        self._th.join(timeout=1.5)
        self.cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="眼位追踪 -> UDP")
    ap.add_argument("--image", default=None, help="离线验证单张图")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--sens", type=float, default=0.3)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--replay", default=None, metavar="IMG",
                    help="循环回放一张图代替相机（无真人在场时测延迟用）")
    ap.add_argument("--stamp", action="store_true",
                    help="UDP 包末尾附带抓帧时刻（epoch 秒），供渲染端量端到端延迟")
    ap.add_argument("--buffers", type=int, default=2,
                    help="V4L2 缓冲深度（默认 1：追踪只要最新帧，越深延迟越大）")
    a = ap.parse_args()

    from capture.pose import model_file
    if model_file() is None:
        print("✗ 缺少姿态模型：把 yolov8n-pose.rknn 放到 models/ 下")
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

    def payload(ex, ey, t_cap):
        body = "nan,nan" if ex is None else ("%.4f,%.4f" % (ex, ey))
        return (body + (",%.6f" % t_cap if a.stamp else "") + "\n").encode()

    grab = None
    if a.replay:
        left = cv2.imread(a.replay)
        if left is None:
            print("读图失败:", a.replay)
            tr.close()
            return 1
        print(f"回放 {a.replay} -> udp {addr[0]}:{addr[1]}   (Ctrl-C 退出)")
    else:
        grab = FrameGrabber(a.buffers)
        print(f"追踪中 -> udp {addr[0]}:{addr[1]}   (Ctrl-C 退出)")

    n = 0
    t0 = time.time()
    frame_age = 0.0
    try:
        while True:
            if grab is not None:
                left, frame_age = grab.latest()
            t_cap = time.time()
            ex, ey, info = tr.process(left)
            sock.sendto(payload(ex, ey, t_cap), addr)
            n += 1
            if a.print and n % 15 == 0:
                fps = n / max(time.time() - t0, 1e-6)
                print(f"  {fps:4.1f} fps  ex={ex if ex is None else round(ex, 3)} "
                      f"ey={ey if ey is None else round(ey, 3)}  infer {info['ms']}ms"
                      + ("" if grab is None else f"  帧龄 {frame_age:.0f}ms"), flush=True)
            if a.replay:
                left = cv2.imread(a.replay)      # 重新读，模拟下一帧
                time.sleep(max(0.0, 0.033 - (time.time() - t_cap)))
    except KeyboardInterrupt:
        pass
    finally:
        if grab is not None:
            grab.release()
        tr.close()
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
