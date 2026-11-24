#!/usr/bin/env python3
"""把一张普通照片（不是相机拍的）导入成一个可渲染的 2.5D 资产。

用在没有受采者的场合：猫狗、老照片、朋友发来的图。和 `mono_calib.py
--save-asset` 产出同一套契约（rgb/depth.f32/matte.png/bg.png/meta.json），
渲染端不需要任何改动。

深度：DepthAnything V2 Small 出相对深度，再按 `config/mono_depth_calib.json`
的仿射关系转米制；因为照片的场景尺度与相机现场不同，这里会把主体重新
归一到 `--subject-dist`（默认 1.2 m，和现场拍摄时受采者的距离一致），
这样视差强度与真实采集的资产可比。

抠图：`--species animal` 用 U²-Netp（通用显著目标），
       `--species human` 用 MODNet。

眼睛：人可以走姿态模型；动物没有可用的关键点模型，这里用"上半脸最暗的
两块近似圆形区域"自动配对，配不上就要求显式 `--eyes x1,y1,x2,y2`。
眼睛只影响眼神微动，检测不到时资产照常可用。

用法：
    python3 tools/import_photo.py --photo dog.jpg --name dog --species animal
    python3 tools/import_photo.py --photo p.jpg --name p1 --eyes 120,80,210,84
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

from capture.matte import load_matte                     # noqa: E402
from capture.mono_depth import MonocularDepth            # noqa: E402
from capture.pipeline import make_background, write_asset  # noqa: E402

DA_MODEL = os.path.join(ROOT, "models", "depth_anything_v2_small.rknn")
MODNET = os.path.join(ROOT, "models", "modnet.rknn")
U2NET = os.path.join(ROOT, "models", "u2netp.rknn")
CALIB = os.path.join(ROOT, "config", "mono_depth_calib.json")


# ------------------------------------------------------------------ 眼睛

def _blobs_at(gray, mask, pct, bw, bh):
    """单一阈值下的暗斑候选。"""
    vals = gray[mask > 0]
    if vals.size == 0:
        return []
    t = float(np.percentile(vals, pct))
    dark = ((gray <= t) & (mask > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    total = float((mask > 0).sum())
    out = []
    for i in range(1, n):
        area = float(st[i, cv2.CC_STAT_AREA])
        w = float(st[i, cv2.CC_STAT_WIDTH])
        h = float(st[i, cv2.CC_STAT_HEIGHT])
        if not (0.0012 * total <= area <= 0.10 * total):
            continue
        if w > 0.34 * bw or h > 0.34 * bh:       # 太大 = 眼睛和耳根糊成一块
            continue
        ar = w / max(h, 1.0)
        if not (0.35 <= ar <= 2.8):
            continue
        fill = area / max(w * h, 1.0)
        if fill < 0.30:                          # 细长的毛发/阴影不要
            continue
        out.append(dict(x=float(cen[i][0]), y=float(cen[i][1]), area=area,
                        fill=fill, ar=ar, w=w, h=h))
    return out


def _round_blobs(gray, mask, bw, bh):
    """多个阈值各扫一遍再合并去重。

    单阈值不够用：阴影里的那只眼睛对比度低，阈值紧一点它就没了、松一点
    又和耳朵粘成一整块。多档阈值能保证两只眼都进候选池。
    """
    cand = []
    for pct in (6, 8, 10, 12, 15, 18, 22):
        cand += _blobs_at(gray, mask, pct, bw, bh)
    merged = []
    for b in sorted(cand, key=lambda d: -d["fill"]):
        r = max(b["w"], b["h"]) * 0.6
        if any(abs(b["x"] - o["x"]) < r and abs(b["y"] - o["y"]) < r for o in merged):
            continue
        merged.append(b)
    return merged


def detect_eyes(rgb: np.ndarray, matte: np.ndarray):
    """在 matte 上半脸找一对左右对称的暗斑。返回 (left, right, iod) 或 None。"""
    m = (matte > 0.5).astype(np.uint8)
    if m.sum() < 200:
        return None
    ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = float(x1 - x0 + 1), float(y1 - y0 + 1)
    cx = (x0 + x1) / 2.0

    # 只看脸的上部：眼睛一定在鼻子上面。留 0.42 是因为 bbox 常包含脖子和
    # 前胸（狗的挂耳、人的肩），放宽了就会把鼻头、耳根阴影当成眼睛。
    upper = np.zeros_like(m)
    upper[y0:y0 + max(int(round(0.42 * bh)), 1), x0:x1 + 1] = 1
    face = (m & upper).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blobs = _round_blobs(gray.astype(np.uint8), face, bw, bh)
    if len(blobs) < 2:
        return None

    best, bestscore = None, 1e9
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            p, q = blobs[i], blobs[j]
            left, right = (p, q) if p["x"] < q["x"] else (q, p)
            iod = right["x"] - left["x"]
            dy = abs(right["y"] - left["y"])
            # 瞳距要占脸宽的合理比例；左右大致等高；总体居中
            if not (0.14 * bw <= iod <= 0.62 * bw):
                continue
            if dy > 0.16 * bh:
                continue
            mid = (left["x"] + right["x"]) / 2.0
            if abs(mid - cx) > 0.20 * bw:
                continue
            if min(left["area"], right["area"]) / max(left["area"], right["area"]) < 0.35:
                continue
            # 配对打分：等高 > 居中 > 瞳距接近平均脸宽
            score = (dy / bh * 3.0 + abs(mid - cx) / bw * 2.0
                     + abs(iod / bw - 0.38) * 0.8)
            if score < bestscore:
                bestscore, best = score, (left, right, iod)
    return best


def drawover(rgb: np.ndarray, pair) -> np.ndarray:
    vis = rgb.copy()
    if not pair:
        return vis
    left, right, iod = pair
    for p, col, name in ((left, (80, 200, 255), "L"), (right, (80, 255, 160), "R")):
        cv2.circle(vis, (int(p["x"]), int(p["y"])), max(4, int(iod * 0.09)), col, 2)
        cv2.putText(vis, name, (int(p["x"]) - 4, int(p["y"]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.line(vis, (int(left["x"]), int(left["y"])), (int(right["x"]), int(right["y"])),
             (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ------------------------------------------------------------------ 主流程

def harden_matte(a: np.ndarray, q: float = 0.35):
    """把主体内部的半透明拉满，只保留边缘的软过渡。

    为什么需要：U²-Netp 对小动物的身体（尤其是贴边、被裁掉的胸口）只给
    0.4~0.6 的 alpha，渲染出来身体是半透明的，背景从狗肚子里透出来 —— 看起来
    就是"素材没匹配好"。人像的 MODNet 不会这样，所以这个修补只放在照片导入里。

    做法：低阈值取主体 → 保留面积相当的连通域 → 填洞 → 用它的模糊版当"实心区"，
    与原 alpha 取 max。内部变实、边缘仍是 2~3 px 的软过渡。
    """
    h, w = a.shape[:2]
    m = (a > q).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return a
    areas = st[1:, cv2.CC_STAT_AREA]
    big = float(areas.max())
    keep = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 0.04 * big:      # 同一主体可能被拆成几块
            keep |= (lab == i).astype(np.uint8)
    # 填洞：从边界 flood 反向填
    inv = (1 - keep).astype(np.uint8)
    ff = inv.copy()
    mk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 0)
    keep = keep | ff
    solid = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.2)
    return np.clip(np.maximum(a, solid), 0.0, 1.0).astype(np.float32)


def matte_is_soft(a: np.ndarray, q: float = 0.35) -> float:
    """主体内部"半透明"像素的比例：>0.06 就值得加固。"""
    m = (a > q).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return 0.0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    inner = cv2.erode((lab == i).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    if inner.sum() < 50:
        return 0.0
    v = a[inner]
    return float(((v > 0.2) & (v < 0.75)).mean())


def read_photo(path: str) -> np.ndarray:
    import base64
    data = np.fromfile(path, dtype=np.uint8)     # 支持中文路径
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:                              # webp 走 imdecode 一般没问题
        raise SystemExit("读不到图片：%s" % path)
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="把一张照片导入成 2.5D 渲染资产")
    ap.add_argument("--photo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--species", default="animal", choices=["human", "animal"])
    ap.add_argument("--subject-dist", type=float, default=1.2,
                    help="把主体归一到这个距离（米），决定视差强度")
    ap.add_argument("--matte-model", default=None)
    ap.add_argument("--frame", default="auto", choices=["auto", "mirror", "black"],
                    help="画面比窗口窄时两侧怎么补：auto（补边>18%% 就留黑）/ mirror / black")
    ap.add_argument("--matte-harden", default="auto", choices=["auto", "on", "off"],
                    help="把主体内部的半透明补实（U²-Netp 对动物身体常只给 0.5 的 alpha）")
    ap.add_argument("--eyes", default=None,
                    help='手动指定两眼中心 "x1,y1,x2,y2"（像素）')
    ap.add_argument("--no-eyes", action="store_true", help="不做眼睛检测")
    ap.add_argument("--preview", default=None,
                    help="把眼睛检测结果画到这张图上，方便用眼睛确认")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="过大照片先等比缩小到长边不超过这个值")
    a = ap.parse_args()

    rgb = read_photo(a.photo)
    h, w = rgb.shape[:2]
    if max(h, w) > a.max_side:
        s = a.max_side / float(max(h, w))
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
        print("  图片缩到 %dx%d" % (rgb.shape[1], rgb.shape[0]))
    print("照片 %dx%d  %s" % (rgb.shape[1], rgb.shape[0], os.path.basename(a.photo)))
    if max(rgb.shape[:2]) < 300:
        print("  ⚠ 分辨率偏低（长边 %d px），放大到全屏会发虚；"
              "用于验证功能没问题，正式展示建议换更大的原图" % max(rgb.shape[:2]))

    out_dir = os.path.join(ROOT, "assets", a.name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 相对深度 -> 米制（按标定仿射），再把主体归一到 --subject-dist
    print("  深度推理 ...")
    mono = MonocularDepth(DA_MODEL)
    try:
        rel = mono.infer(rgb)
    finally:
        mono.close()

    if os.path.isfile(CALIB):
        c = json.load(open(CALIB, encoding="utf-8"))
        ca, cb = float(c["a"]), float(c["b"])
        depth0 = 1.0 / np.clip(ca * rel + cb, 1e-4, None)
        ch, cw = depth0.shape
        x0, x1 = int(cw * 0.25), int(cw * 0.75)
        y0, y1 = int(ch * 0.10), int(ch * 0.75)
        z0 = float(np.percentile(depth0[y0:y1, x0:x1], 15))
        s = z0 / a.subject_dist          # inv 乘 s 等于距离乘 1/s
        ca, cb = ca * s, cb * s
        print("  标定主体距离 %.2f m，归一到 %.2f m（系数 %.3f）" % (z0, a.subject_dist, s))
    else:
        lo, hi = np.percentile(rel, [2, 98])
        n = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, 1)
        z = 0.5 + (1.0 - n) * 7.5
        ca, cb = 1.0 / np.clip(z, 0.3, 20), 0.0
        print("  ⚠ 无标定，退化为相对深度映射")

    # 2) 抠图
    mp = a.matte_model or (U2NET if a.species == "animal" else MODNET)
    if not os.path.isfile(mp):
        print("✗ 缺抠图模型 %s" % mp)
        return 1
    print("  抠图 %s ..." % os.path.basename(mp))
    with load_matte(mp) as mm:
        matte = mm.infer(rgb)
    cov = float((matte > 0.5).mean()) * 100
    print("  matte 覆盖 %.1f%%" % cov)
    soft = matte_is_soft(matte)
    if a.matte_harden == "on" or (a.matte_harden == "auto" and soft > 0.06):
        before = soft * 100
        matte = harden_matte(matte)
        print("  抠图加固：内部半透明像素 %.0f%% → %.0f%%（身体不再透明）"
              % (before, matte_is_soft(matte) * 100))
    if cov < 3 or cov > 92:
        print("✗ 抠图覆盖 %.1f%% 不合理：主体太小或几乎占满画面，检查照片" % cov)
        return 2

    # 3) 写 rgb/depth/matte 与 meta
    meta = write_asset(out_dir, rgb, rel, ca, cb, matte=matte,
                       depth_source="photo_import")
    meta["species"] = a.species
    meta["source_photo"] = os.path.basename(a.photo)
    meta["frame"] = a.frame        # 渲染端据此决定四周补边是镜像还是留黑

    # 4) 背景板
    depth = np.fromfile(os.path.join(out_dir, "depth.f32"), dtype=np.float32)
    depth = depth.reshape(meta["height"], meta["width"])
    try:
        bg = make_background(rgb, depth,
                             max(float(np.percentile(
                                 depth[matte > 0.5], 85) if np.any(matte > 0.5) else 2.0) * 1.6,
                                 1.5), matte=matte)
        cv2.imwrite(os.path.join(out_dir, "bg.png"), bg)
    except Exception as e:
        print("  背景板生成失败（渲染仍可用，只是背景层有残影）:", e)

    # 5) 眼睛
    pair = None
    if not a.no_eyes:
        if a.eyes:
            v = [float(t) for t in a.eyes.replace(" ", "").split(",")]
            if len(v) != 4:
                print("✗ --eyes 需要 4 个数字：x1,y1,x2,y2")
                return 3
            lx, ly, rx, ry = v
            if lx > rx:
                lx, ly, rx, ry = rx, ry, lx, ly
            pair = (dict(x=lx, y=ly, area=1.0), dict(x=rx, y=ry, area=1.0),
                    float(np.hypot(rx - lx, ry - ly)))
            print("  使用手动眼睛位置")
        else:
            pair = detect_eyes(rgb, matte)
            if pair:
                print("  眼睛自动定位：L(%.0f,%.0f) R(%.0f,%.0f) 瞳距 %.0f px"
                      % (pair[0]["x"], pair[0]["y"], pair[1]["x"], pair[1]["y"], pair[2]))
            else:
                print("  ⚠ 没找到成对的眼睛：眼神微动关闭（视差与观感不受影响）")

    if a.preview:
        vis = drawover(rgb, pair)
        cv2.imwrite(a.preview, vis)
        print("  预览图 ->", a.preview)

    if pair:
        l, r, iod = pair
        meta["eyes"] = {"left": [round(l["x"], 1), round(l["y"], 1)],
                        "right": [round(r["x"], 1), round(r["y"], 1)],
                        "iod": round(iod, 1)}

        # 头部遮罩：只用到双眼中点和瞳距的几何公式，人和动物通用。
        # 有了它渲染端才会做"轻微点头 + 回望"（Blink 仍然只有人像有，
        # 动物的睫毛线没法用肤色填补，硬做出眨眼会糊成一团）。
        from capture.headmask import make_head_mask
        mid = ((l["x"] + r["x"]) / 2.0, (l["y"] + r["y"]) / 2.0)
        hm = make_head_mask(rgb.shape[:2], mid, iod, matte)
        cv2.imwrite(os.path.join(out_dir, "head.png"),
                    (hm * 255).round().astype(np.uint8))
        print("  head.png（覆盖 %.0f%%，开启点头/回望）"
              % (float((hm > 0.5).mean()) * 100))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("资产已生成:", out_dir)
    print("  焦点 %.2f m  主体 %.2f m  范围 %.2f ~ %.2f m"
          % (meta["z_focus"], meta["z_subject"], meta["z_min"], meta["z_max"]))
    print("  文件:", ", ".join(sorted(os.listdir(out_dir))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
