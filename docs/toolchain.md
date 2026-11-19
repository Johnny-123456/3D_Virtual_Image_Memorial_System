# RKNN 转换工具链（Docker / WSL2）

把 ONNX 转成 Rockchip NPU 可用的 `.rknn`。

板端只有 `rknn-toolkit-lite2`（纯推理运行时），**转换必须在 x86 Linux 上做**——
所以放在开发机（Windows）的 WSL2 里，用 Docker 隔离。

---

## 1. 为什么是 WSL2 + Docker，而不是别的方式

| 方案 | 结论 |
|---|---|
| 装在板子上 | ❌ 不可能：`rknn-toolkit2` 只提供 x86_64 wheel |
| 装 Docker Desktop | ❌ 需要 Windows 管理员权限（本机非管理员） |
| **WSL2 里原生 Docker Engine** | ✅ 采用。本机 WSL 已自带 Docker 29.8.2，且用户已在 docker 组 |
| 直接装在 WSL 宿主上 | ❌ 依赖太重（torch + CUDA 运行库，镜像 9.94 GB），会污染环境 |

**板子上一个包都没装**，完全保持原样。

---

## 2. 版本锁定（都是实测撞出来的坑）

| 包 | 锁定值 | 原因 |
|---|---|---|
| `rknn-toolkit2` | **2.3.2** | 必须与板端 `rknn-toolkit-lite2 2.3.2` 一致，否则 `.rknn` 加载失败 |
| `numpy` | **<=1.26.4** | rknn-toolkit2 2.3.2 的硬性要求 |
| `opencv-python-headless` | **<5** | opencv 5.x 会把 numpy 强推到 2.x，直接破坏 rknn |
| `onnx` | **==1.16.1** | 1.17+ 移除了 rknn 内部依赖的 `onnx.mapping`，报 `AttributeError` |

> **wheel 必须从 tag 取。** master 分支上只剩 2.5.0，与板端不匹配：
>
> ```
> https://github.com/airockchip/rknn-toolkit2/raw/v2.3.2/rknn-toolkit2/packages/x86_64/
>     rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
> ```

实测报错对照：

| 症状 | 根因 |
|---|---|
| `AttributeError: module 'onnx' has no attribute 'mapping'` | onnx 版本过高 |
| `rknn-toolkit2 2.3.2 requires numpy<=1.26.4, but you have numpy 2.2.6` | 装了 opencv 5.x 连带升级 |

---

## 3. 位置与目录

工具链在**开发机**（Windows / WSL2），不在板子上：

```
WSL2 (Ubuntu-22.04)  ~/rknn-toolkit-docker/
├── Dockerfile                                    版本锁都在这里
├── rknn_toolkit2-2.3.2-cp310-...whl              转换工具 wheel（38 MB）
├── convert.sh                                    转换入口
├── scripts/convert.py                            容器内实际执行逻辑
├── work/                                         临时工作区（挂载进容器）
└── images/                                       量化校准图

Windows  C:\Users\29678\rknn_handoff\              产物中转（传给板子用）
```

镜像：`rknn-toolkit2:2.3.2`，**9.94 GB**

---

## 4. 用法

```bash
cd ~/rknn-toolkit-docker

./convert.sh model.onnx model.rknn                 # 不量化（fp16）
./convert.sh model.onnx model.rknn dataset.txt     # int8 量化
./convert.sh model.onnx model.rknn dataset.txt --mean 0,0,0 --std 255,255,255
```

`convert.sh` 自己处理卷挂载，输入输出路径不用迁就容器。产物落在 `<out.rknn>` 同目录。

**int8 校准集格式**：一个 txt，每行一张图片的绝对路径。板上的
`rehab-coach/models/quant_images/` 可直接拿来用。

---

## 5. 工作流（PC → 板）

```
1. 板子 / PC 上准备 ONNX
2. WSL:  ./convert.sh in.onnx out.rknn dataset.txt
3. 产物拷到 Windows 中转目录 C:\Users\29678\rknn_handoff\
4. ssh_upload 到板子（例如 /home/elf/rknn_handoff/）
5. 板上用 rknnlite 加载验证
```

板端加载样板：

```python
import numpy as np
from rknnlite.api import RKNNLite

rk = RKNNLite()
rk.load_rknn("model.rknn")          # 0 = 成功
rk.init_runtime()                   # 0 = 成功
out = rk.inference(inputs=[x])      # x: float32, NCHW
rk.release()
```

---

## 6. 验证状态

| 项 | 状态 |
|---|---|
| 镜像构建 | ✅ |
| `onnx.mapping` 兼容性修复 | ✅ 锁 onnx 1.16.1 |
| 端到端转换（ONNX → RKNN） | ✅ `tiny.onnx` → `tiny.rknn` (24 KB) |
| **板端加载 + 推理** | ✅ `lite2 2.3.2` 加载成功，输出 `(1,3,32,32)` |
| `convert.sh` 封装 | ✅ |

板端运行时信息（实测）：

```
librknnrt version: 2.3.2 (429f97ae6b@2025-04-09)
RKNN Driver version: 0.9.8
toolkit version: 2.3.2, target: RKNPU v2, target platform: rk3588
```

---

## 7. 环境事实

| 项 | 值 |
|---|---|
| WSL 发行版 | Ubuntu-22.04（x86_64，20 核，15 GB） |
| Docker | 29.8.2，**WSL 内原生 Docker Engine**（非 Docker Desktop） |
| Docker 镜像源 | `docker.m.daocloud.io`、`dockerproxy.net` |
| pip 源 | 清华 TUNA |
| apt 源 | 阿里云 |
| 代理 | Windows 侧 `127.0.0.1:7897`，**WSL 需走网关 IP**（NAT 模式下访问不到 localhost） |

> WSL 的 default gateway 可用 `ip route | awk '/^default/{print $3}'` 取，
> 本机为 `172.20.80.1`。代理只在下载 GitHub wheel 时用了一次，
> 其余 apt / pip / docker pull 都走国内镜像。
