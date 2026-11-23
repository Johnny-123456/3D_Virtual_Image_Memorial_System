"""把 librknnrt 的加载横幅从 stdout 里摘出去。

RKNN 运行时初始化时会往 **fd 1** 打一段 `I RKNN: ...` 横幅（C 层 printf），
Python 的 print / sys.stdout 重定向拦不住，界面的采集日志被它刷屏——真正的
结果（"✗ 受采者离得太远…"）被埋在几十行噪音里，用户根本看不到。

两个坑（都实测踩过）：
1. 只 dup2 不够：librknnrt 在模块加载时就缓存了 fd 1，所以必须连 import
   一起放进静音区。
2. 还不够：C 的 stdio 有用户态缓冲，静音期间 printf 只是进了 buffer，等
   进程退出（fd 1 已还原）才 flush 出来。所以还原前要先 fflush(NULL)，
   把 buffer 里的字节真正冲进 /dev/null。
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import sys

try:
    _libc = ctypes.CDLL(None)
    _fflush = _libc.fflush
    _fflush.argtypes = [ctypes.c_void_p]      # fflush(NULL) = 刷所有流
    _fflush.restype = ctypes.c_int
except Exception:                              # pragma: no cover
    _fflush = None


def _flush_c_streams() -> None:
    if _fflush is None:
        return
    try:
        _fflush(None)
    except Exception:                          # pragma: no cover
        pass


@contextlib.contextmanager
def silence_stdout():
    """在 with 块内静音 C 层 stdout（fd 1），退出前把缓冲区一并丢掉。"""
    try:
        sys.stdout.flush()
        saved = os.dup(1)
    except (OSError, ValueError):
        yield                       # 没有可用 fd 1（极少数嵌入场景），直接放行
        return
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        yield
    finally:
        _flush_c_streams()          # 关键：在还原 fd 之前把 C buffer 冲进 /dev/null
        os.dup2(saved, 1)
        os.close(saved)
        os.close(null)
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
