#!/usr/bin/env python3
"""资产后处理：为资产补齐 blink.png 与 head.png。

`mono_calib.py --save-asset` 只生成 rgb/depth/matte/bg/meta，
眨眼与头部遮罩需要姿态模型，单独放在这里，由界面或脚本串起来调用。

用法：
    python3 tools/finalize_asset.py assets/portrait2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capture.blink import make_blink                       # noqa: E402
from capture.headmask import make_head_mask                # noqa: E402
from capture.landmarks import detect_main_person, eye_pair  # noqa: E402

POSE_MODEL = None   # 由 capture.pose 统一解析（models/ 优先，~/rehab-coach 兜底）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("--species", default="human", choices=["human", "animal"],
                    help="animal 时跳过眨眼/头部遮罩（姿态模型只认人）")
    a = ap.parse_args()

    if a.species == "animal":
        # 动物没有可用的关键点模型：不生成 blink/head。渲染端本来就支持
        # hasBlink/hasHead 为假时降级（不眨眼、不回望），视差照常。
        print("动物模式：跳过眨眼与头部遮罩，资产保持 5 个文件")
        return 0

    img = cv2.imread(os.path.join(a.asset_dir, "rgb.png"))
    if img is None:
        print("读不到 rgb.png")
        return 1
    matte = None
    mp = os.path.join(a.asset_dir, "matte.png")
    if os.path.isfile(mp):
        matte = cv2.imread(mp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    from capture.pose import load_pose
    from capture.quiet import silence_stdout
    with silence_stdout():
        pose = load_pose(POSE_MODEL, imgsz=640, conf_threshold=0.4)
    det = detect_main_person(pose, img)
    pose.release()

    pair = eye_pair(det)
    if not pair:
        print("✗ 未检测到双眼，跳过 blink/head")
        return 2

    le, re_, iod, mid = pair
    print(f"瞳距 {iod:.1f}px  中点 ({mid[0]:.0f},{mid[1]:.0f})")

    # 守卫：瞳距过小 = 检测到的是远处的人（或背景里的同事），不是受采者。
    # 实测踩过：镜头前没人时会选中画面深处 2.8px 瞳距的目标，
    # 流程照样产出资产，但 head.png 覆盖 0%，渲染完全无效。
    if iod < 45.0:
        print(f"✗ 受采者离得太远或不在画面中央（瞳距仅 {iod:.1f}px，至少 45px）")
        print("  请站到镜头前 1.2~1.5 m 处、位于画面中央后重试")
        return 3

    cv2.imwrite(os.path.join(a.asset_dir, "blink.png"),
                make_blink(img, [(le["x"], le["y"]), (re_["x"], re_["y"])], iod))
    print("  blink.png")

    m = make_head_mask(img.shape[:2], mid, iod, matte)
    cv2.imwrite(os.path.join(a.asset_dir, "head.png"),
                (m * 255).round().astype(np.uint8))
    print(f"  head.png（覆盖 {float((m > 0.5).mean()) * 100:.1f}%）")

    # 把眼睛位置写进 meta.json：渲染端据此做"眼神微动"（在两眼小椭圆内
    # 轻微挪动采样点，看起来是眼球在转）。
    mp = os.path.join(a.asset_dir, "meta.json")
    try:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        meta["eyes"] = {
            "left": [round(float(le["x"]), 1), round(float(le["y"]), 1)],
            "right": [round(float(re_["x"]), 1), round(float(re_["y"]), 1)],
            "iod": round(float(iod), 1),
        }
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("  meta.json 已记录眼睛位置")
    except Exception as e:
        print("  ⚠ 写入眼睛位置失败:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
