"""姿态后端（YOLOv8n-Pose on RKNN）的唯一定位点。

为什么要有这个文件
------------------
这份后端最早写在另一个项目 `rehab-coach` 里。本项目的眨眼贴图、头部遮罩、
采集向导、眼动追踪、深度标定全都要用它，于是这些文件各自写了一遍

    _REHAB = os.path.expanduser("~/rehab-coach/src/backend")
    sys.path.insert(0, os.path.join(_REHAB, "vision"))
    POSE_MODEL = os.path.expanduser("~/rehab-coach/models/yolov8n-pose.rknn")

七份拷贝，任何一处换位置就一起崩，而且**项目本身不自包含**——把仓库拷到
另一台机器上，眨眼/遮罩/追踪全部静默降级。这份集中到一处，并且把后端代码
（`third_party/pose/`，MIT 之外的自家代码，来源 rehab-coach）和模型都收进仓库。

查找顺序
--------
    后端代码   third_party/pose/                     ← 项目自带，首选
               ~/rehab-coach/src/backend/{vision,}    ← 历史位置，已装这套旧项目时兜底
    模型       models/yolov8n-pose.rknn              ← 项目自带，首选
               ~/rehab-coach/models/yolov8n-pose.rknn ← 历史位置，兜底

模型是 8.8 MB 的 .rknn，被 .gitignore 排除，所以新克隆的仓库里没有——
此时 `models/` 找不到会自动退到 `~/rehab-coach`，两者都没有就抛
`PoseUnavailable`，由调用方决定是降级还是报错（眨眼/遮罩都是可降级项）。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL_BACKEND = os.path.join(ROOT, "third_party", "pose")
LEGACY_BACKEND = os.path.expanduser("~/rehab-coach/src/backend")

LOCAL_MODEL = os.path.join(ROOT, "models", "yolov8n-pose.rknn")
LEGACY_MODEL = os.path.expanduser("~/rehab-coach/models/yolov8n-pose.rknn")


class PoseUnavailable(RuntimeError):
    """姿态模型或后端代码缺失。"""


def backend_dirs() -> list[str]:
    """按优先级返回存活的 backend 目录（已展开为 sys.path 项）。"""
    out: list[str] = []
    if os.path.isfile(os.path.join(LOCAL_BACKEND, "pose_rknn.py")):
        out.append(LOCAL_BACKEND)
    if os.path.isfile(os.path.join(LEGACY_BACKEND, "vision", "pose_rknn.py")):
        out.append(os.path.join(LEGACY_BACKEND, "vision"))
        # pose_rknn 多人时会 import core.*（历史版本），需要父目录
        out.append(LEGACY_BACKEND)
    return out


def model_file() -> str | None:
    """按优先级返回存活模型的绝对路径。"""
    for p in (LOCAL_MODEL, LEGACY_MODEL):
        if os.path.isfile(p):
            return p
    return None


def available() -> bool:
    return model_file() is not None and bool(backend_dirs())


def missing_reason() -> str:
    """给人和给日志看的一句话，说明缺什么、怎么补。"""
    parts = []
    if model_file() is None:
        parts.append(
            f"模型缺失（找过 {LOCAL_MODEL} 与 {LEGACY_MODEL}；"
            "把 yolov8n-pose.rknn 放到 models/ 即可）")
    if not backend_dirs():
        parts.append(
            f"后端缺失（找过 {LOCAL_BACKEND} 与 {LEGACY_BACKEND}/vision；"
            "third_party/pose/ 应当随仓库提供）")
    return "；".join(parts) or "未知"


def _install_path() -> None:
    for d in reversed(backend_dirs()):
        if d not in sys.path:
            sys.path.insert(0, d)


def load_pose(model: str | None = None, imgsz: int = 640,
              conf_threshold: float = 0.4, pick_mode: str | None = None):
    """加载并返回 RknnPoseBackend。缺依赖时抛 PoseUnavailable。"""
    if model is None:
        model = model_file()
        if model is None:
            raise PoseUnavailable(f"姿态后端不可用：{missing_reason()}")
    _install_path()
    try:
        from pose_rknn import RknnPoseBackend
    except Exception as e:  # rknnlite 缺失、numpy 版本不兼容等
        raise PoseUnavailable(f"姿态后端导入失败：{e}") from e
    pose = RknnPoseBackend(model, imgsz=imgsz, conf_threshold=conf_threshold)
    if pick_mode:
        try:
            pose._pick_mode = pick_mode
        except Exception:
            pass
    return pose
