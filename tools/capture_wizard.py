#!/usr/bin/env python3
"""采集向导：实时质检 + 引导站位 + 一键出资产。

流程
----
    预览（实时）→ 质检（人物/距离/居中/亮度/逆光/清晰度）
    → 全部通过（或手动空格）→ 连拍 3 帧 → 标定 → 深度 → matte → 背景板 → 资产

按键
----
    SPACE  立即采集
    A      切换自动（通过后倒计时 3 秒自动采集）
    Q/ESC  退出

为什么分「硬性」和「建议」
--------------------------
门限太严会导致永远拍不了。所以只把“检测到人”和“亮度够”设为硬性门槛，
距离/居中/清晰度作为建议项提示但不阻止。

用法
----
    python3 tools/capture_wizard.py --asset portrait
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_REHAB = os.path.expanduser("~/rehab-coach/src/backend")
sys.path.insert(0, os.path.join(_REHAB, "vision"))
sys.path.insert(0, _REHAB)   # pose_rknn 多目标时会 import core.*

from capture.camera_io import CAMERA_BY_ID, FALLBACK_DEVICE          # noqa: E402
from capture.pipeline import (MIN_DISPARITY, NUM_DISPARITIES,        # noqa: E402
                              evaluate, fit_affine, process_frame,
                              write_asset, write_background)

POSE_MODEL = os.path.expanduser("~/rehab-coach/models/yolov8n-pose.rknn")
DA_MODEL = os.path.join(ROOT, "models", "depth_anything_v2_small.rknn")
MATTE_MODEL = os.path.join(ROOT, "models", "modnet.rknn")

WIN = "capture wizard"

# ---- 质量门限 ----
DIST_MIN_H, DIST_MAX_H = 0.42, 0.97      # 人体框高度占画面比例
CENTER_MAX = 0.20                        # 鼻尖水平偏移（归一化）
BRIGHT_MIN = 48.0                        # 主体区域平均灰度
BACKLIT_MIN = 0.42                       # 主体/背景 亮度比
SHARP_MIN = 55.0                         # 拉普拉斯方差


def _font(size: int):
    from PIL import ImageFont
    for pat in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc"):
        if os.path.isfile(pat):
            try:
                return ImageFont.truetype(pat, size)
            except Exception:
                pass
    for f in glob.glob("/usr/share/fonts/**/*CJK*.tt[cf]", recursive=True):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_cn(img, lines, origin=(14, 12), size=20):
    """在 BGR 图上画中文。lines = [(text, (b,g,r)), ...]"""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = _font(size)
    y = origin[1]
    for text, color in lines:
        d.text((origin[0] + 1, y + 1), text, font=f, fill=(0, 0, 0))
        d.text((origin[0], y), text, font=f, fill=(color[2], color[1], color[0]))
        y += size + 7
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def check_quality(bgr: np.ndarray, det) -> dict:
    """返回质检结果：hard_ok / soft_ok / issues / metrics。"""
    h, w = bgr.shape[:2]
    r = {"hard_ok": False, "soft_ok": False, "issues": [], "m": {}}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if not det or "bbox" not in det:
        r["issues"].append(("未检测到人，请站到画面中", 0))
        return r

    bx0, by0, bx1, by1 = [int(v) for v in det["bbox"]]
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(w - 1, bx1), min(h - 1, by1)
    if bx1 - bx0 < 20 or by1 - by0 < 20:
        r["issues"].append(("检测框异常，请调整站位", 0))
        return r

    roi = gray[by0:by1, bx0:bx1]
    bright = float(roi.mean())
    sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    # 背景亮度（画面外围）
    mask = np.ones_like(gray, bool)
    mask[by0:by1, bx0:bx1] = False
    bg_bright = float(gray[mask].mean()) if mask.any() else bright
    backlit = bright / max(bg_bright, 1e-3)

    h_ratio = (by1 - by0) / h
    nose = det["keypoints_2d"].get("nose")
    dx = abs((nose["x"] - w / 2) / (w / 2)) if nose else 1.0

    r["m"] = {"bright": bright, "sharp": sharp, "backlit": backlit,
              "h_ratio": h_ratio, "dx": dx, "bbox": (bx0, by0, bx1, by1)}

    # 建议项（不阻止）
    if h_ratio < DIST_MIN_H:
        r["issues"].append(("离得有点远，可以再靠近一些", 1))
    elif h_ratio > DIST_MAX_H:
        r["issues"].append(("太近了，可能会被裁到，稍微后退", 1))
    if dx > CENTER_MAX:
        r["issues"].append(("请站到画面中间", 1))
    if backlit < BACKLIT_MIN:
        r["issues"].append(("逆光了，请面向光源或补光", 1))
    if sharp < SHARP_MIN:
        r["issues"].append(("画面偏模糊，请保持不动", 1))

    r["soft_ok"] = not r["issues"]

    # 硬性门槛：有人 + 够亮
    if bright < BRIGHT_MIN:
        r["issues"].insert(0, ("太暗了，请面向光源或开灯", 0))
    else:
        r["hard_ok"] = True
    return r


def overlay(bgr, q, state, extra=""):
    v = bgr.copy()
    m = q["m"]
    if "bbox" in m:
        bx0, by0, bx1, by1 = m["bbox"]
        col = (80, 220, 80) if q["soft_ok"] else ((60, 200, 240) if q["hard_ok"] else (60, 60, 230))
        cv2.rectangle(v, (bx0, by0), (bx1, by1), col, 2)

    lines = []
    for text, sev in q["issues"][:4]:
        lines.append((("必须：" if sev == 0 else "建议：") + text,
                      (90, 90, 255) if sev == 0 else (60, 200, 240)))
    if not lines:
        lines.append(("状态良好，可以拍摄", (100, 255, 120)))
    if extra:
        lines.append((extra, (230, 230, 230)))

    if "bright" in m:
        lines.append((f"亮度 {m['bright']:.0f}  逆光比 {m['backlit']:.2f}  "
                      f"清晰度 {m['sharp']:.0f}  占幅 {m['h_ratio']:.2f}",
                      (200, 200, 200)))
    lines.append((f"[空格]采集  [A]{'自动开' if state.get('auto') else '自动关'}  "
                  f"[Q]退出", (170, 170, 170)))
    return draw_cn(v, lines, size=20)


def open_camera():
    dev = CAMERA_BY_ID if os.path.exists(CAMERA_BY_ID) else FALLBACK_DEVICE
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise SystemExit("相机打开失败")
    for _ in range(8):
        cap.read()
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description="采集向导")
    ap.add_argument("--asset", default="portrait")
    ap.add_argument("--frames", type=int, default=3, help="采集帧数")
    a = ap.parse_args()

    for p in (POSE_MODEL, DA_MODEL, MATTE_MODEL):
        if not os.path.isfile(p):
            print("缺少模型:", p)
            return 1

    from pose_rknn import RknnPoseBackend
    from capture.mono_depth import MonocularDepth
    from capture.matte import PortraitMatte
    from capture.rectify import Rectifier
    from capture.sgbm_truth import SGBMTruth

    print("[1/3] 加载姿态模型 ...", flush=True)
    pose = RknnPoseBackend(POSE_MODEL, imgsz=640, conf_threshold=0.4)
    cap = open_camera()
    state = {"auto": False, "good_since": None, "countdown": None}

    disp_w, disp_h = 1200, 660
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)

    print("[2/3] 进入预览：站到镜头前，按空格采集", flush=True)
    captured = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            left = frame[:, :1280]
            det = pose.detect(left)
            q = check_quality(left, det)

            now = time.time()
            if q["hard_ok"] and q["soft_ok"]:
                if state["good_since"] is None:
                    state["good_since"] = now
                if state["auto"] and now - state["good_since"] > 0.8:
                    state["countdown"] = state["countdown"] or (now + 3.0)
            else:
                state["good_since"] = None
                state["countdown"] = None

            extra = ""
            if state["countdown"]:
                rem = state["countdown"] - now
                extra = f"自动拍摄倒计时 {max(1, int(np.ceil(rem)))}"
                if rem <= 0:
                    captured = "auto"
            elif state["good_since"] and state["auto"]:
                extra = "状态良好，即将自动拍摄…"

            vis = overlay(left, q, state, extra)
            cv2.imshow(WIN, cv2.resize(vis, (disp_w, disp_h)))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                captured = "manual"
            if k in (ord("a"), ord("A")):
                state["auto"] = not state["auto"]
                state["countdown"] = None
            if captured:
                break
    finally:
        cap.release()
        pose.release()
        cv2.destroyAllWindows()

    if not captured:
        print("已取消")
        return 0

    # ---------- 采集 ----------
    print(f"\n[3/3] 采集 {a.frames} 帧并处理 ...", flush=True)
    rect = Rectifier()
    sgbm = SGBMTruth(num_disparities=NUM_DISPARITIES, min_disparity=MIN_DISPARITY)
    mono = MonocularDepth(DA_MODEL)
    matte_net = PortraitMatte(MATTE_MODEL)

    cap = open_camera()
    samples = []
    t0 = time.time()
    try:
        for i in range(a.frames):
            ok, frame = cap.read()
            if not ok:
                continue
            s = process_frame(rect, sgbm, mono, frame[:, :1280], frame[:, 1280:])
            samples.append(s)
            print(f"    帧{i}: 双目有效 {s['valid_ratio'] * 100:.1f}%", flush=True)
    finally:
        cap.release()
    print(f"    采集用时 {time.time() - t0:.1f} s", flush=True)

    if not samples:
        print("✗ 采集失败")
        return 1

    rel = np.concatenate([s["rel_sq"][s["ok"]] for s in samples])
    inv = np.concatenate([1.0 / s["z_sq"][s["ok"]] for s in samples])
    z_st = np.concatenate([s["z_sq"][s["ok"]] for s in samples])
    if rel.size < 5000:
        print("✗ 共同有效像素不足，请调整站位后重试")
        return 1

    A, B, inl = fit_affine(rel, inv)
    metrics = evaluate(rel, inv, z_st, A, B)
    print(f"\n标定  1/Z = {A:.6f}·rel + {B:.6f}   内点 {inl.mean() * 100:.1f}%")
    print(f"留出集: 中位 {metrics['median_abs_err_mm']:.0f} mm  "
          f"相对 {metrics['median_rel_err'] * 100:.1f}%   R²={metrics['r2']:.3f}")
    for b in metrics["bands"]:
        if "note" in b:
            print(f"    {b['range']:>9} m  {b['n']:6d}  样本不足")
        else:
            print(f"    {b['range']:>9} m  {b['n']:6d}  "
                  f"{b['median_abs_mm']:7.0f} mm  {b['median_rel'] * 100:5.1f}%")

    best = max(samples, key=lambda s: s["valid_ratio"])
    out_dir = os.path.join(ROOT, "assets", a.asset)
    matte = matte_net.infer(best["rgb"])
    meta = write_asset(out_dir, best["rgb"], best["rel_full"], A, B, matte=matte)
    depth = 1.0 / np.clip(A * best["rel_full"] + B, 1e-4, None)
    write_background(out_dir, best["rgb"], depth.astype(np.float32),
                     meta["z_split"], matte)
    # 眨眼贴图：用姿态模型定位双眼后合成
    try:
        from capture.blink import make_blink
        pb = RknnPoseBackend(POSE_MODEL, imgsz=640, conf_threshold=0.4)
        det2 = pb.detect(best["rgb"])
        pb.release()
        kp2 = det2["keypoints_2d"] if det2 else {}
        if "left_eye" in kp2 and "right_eye" in kp2:
            le, re_ = kp2["left_eye"], kp2["right_eye"]
            inter = float(np.hypot(le["x"] - re_["x"], le["y"] - re_["y"]))
            bl = make_blink(best["rgb"], [(le["x"], le["y"]), (re_["x"], re_["y"])],
                            inter)
            cv2.imwrite(os.path.join(out_dir, "blink.png"), bl)
            print(f"  眨眼贴图已生成（瞳距 {inter:.0f}px）")
        else:
            print("  未检测到双眼，跳过眨眼贴图")
    except Exception as e:
        print("  眨眼贴图生成失败（不影响主流程）:", e)

    mono.close()
    matte_net.close()

    cfg = {
        "method": "stereo_to_mono_affine",
        "calibrated_on": time.strftime("%Y-%m-%d"),
        "model": os.path.basename(DA_MODEL),
        "input_size": 518,
        "a": round(A, 8), "b": round(B, 8),
        "formula": "Z = 1 / (a * rel + b)",
        "n_pixels": int(rel.size),
        "inlier_ratio": round(float(inl.mean()), 4),
        "holdout": metrics,
    }
    with open(os.path.join(ROOT, "config", "mono_depth_calib.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 资产已写入 {out_dir}")
    print(f"  主体 {meta['z_subject']:.2f} m   零视差面 {meta['z_focus']:.2f} m   "
          f"分割 {meta['z_split']:.2f} m")
    print(f"  标定已更新 config/mono_depth_calib.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
