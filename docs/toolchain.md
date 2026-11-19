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
  ↑ 注意：WSL 虚拟磁盘在 D 盘 D:\WSL\Ubuntu-22.04\ext4.vhdx（见 §8）
├── Dockerfile                                    版本锁都在这里
├── rknn_toolkit2-2.3.2-cp310-...whl              转换工具 wheel（38 MB）
├── convert.sh                                    转换入口
├── scripts/convert.py                            容器内实际执行逻辑
├── work/                                         临时工作区（挂载进容器）
└── images/                                       量化校准图

Windows  C:\Users\29678\rknn_handoff\              产物中转（传给板子用）
```

镜像：`rknn-toolkit2:2.3.2`，**2.42 GB**（精简版，见 §8）

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
| WSL 虚拟磁盘位置 | **`D:\WSL\Ubuntu-22.04\ext4.vhdx`**（5.89 GB） |
| Docker 镜像源 | `docker.m.daocloud.io`、`dockerproxy.net` |
| pip 源 | 清华 TUNA |
| apt 源 | 阿里云 |
| 代理 | Windows 侧 `127.0.0.1:7897`，**WSL 需走网关 IP**（NAT 模式下访问不到 localhost） |

> WSL 的 default gateway 可用 `ip route | awk '/^default/{print $3}'` 取，
> 本机为 `172.20.80.1`。代理只在下载 GitHub wheel 时用了一次，
> 其余 apt / pip / docker pull 都走国内镜像。

---

## 8. 磁盘占用与迁移（2026-11-19）

### 8.1 发现的问题

Docker 全部落在 WSL2 的虚拟磁盘里，而 WSL 默认把虚拟磁盘放在 **C 盘**：

```
C:\Users\<user>\AppData\Local\wsl\{GUID}\ext4.vhdx
```

首次搭建后实测占用：

| 项目 | 占用 |
|---|---|
| Docker 镜像（含 torch + CUDA 运行库） | 9.94 GB |
| Docker 构建缓存 | 19.92 GB |
| **vhdx 合计** | **29.60 GB** |

⚠️ **关键坑：清缓存不会释放 C 盘空间。** WSL 虚拟磁盘只增不减，
`docker builder prune` 之后 vhdx 依然 29.6 GB，必须压缩或迁移才能真正归还。

`wsl --manage <distro> --set-sparse true` 会因 vhdx 格式被拒（需 `--allow-unsafe`），
故采用导出/导入的方式，同时完成**迁移**和**压缩**。

### 8.2 镜像瘦身：9.94 GB → 2.42 GB

| 措施 | 效果 |
|---|---|
| `pip install --no-deps` 装 rknn wheel，跳过它声明的 torch | 去掉 797 MB 的 torch 主包 |
| 改从 `download.pytorch.org/whl/cpu` 装 **CPU 版 torch** | 去掉 2–3 GB 的 nvidia-* CUDA 运行库 |

> 不能完全去掉 torch：`rknn/api/rknn_utils.py` 在 **import 阶段** 就 `import torch`，
> 缺了会直接 `ModuleNotFoundError`。CPU 版（约 180 MB）是性价比最高的做法。

实测：精简镜像对 **DepthAnything-V2-Small（823 节点）** 的转换结果与完整镜像完全一致
（均产出 57.5 MB 的 .rknn），ONNX→RKNN 转换不依赖 torch 的 CUDA 能力。

### 8.3 WSL 迁移到 D 盘

```powershell
wsl --shutdown
wsl --export  Ubuntu-22.04 D:\WSL\ubuntu-22.04-backup.tar
wsl --unregister Ubuntu-22.04
wsl --import  Ubuntu-22.04 D:\WSL\Ubuntu-22.04 D:\WSL\ubuntu-22.04-backup.tar --version 2
```

结果：

| | 迁移前 | 迁移后 |
|---|---|---|
| vhdx 位置 | `C:\Users\...\wsl\{GUID}\ext4.vhdx` | **`D:\WSL\Ubuntu-22.04\ext4.vhdx`** |
| vhdx 大小 | 29.60 GB | **5.89 GB** |
| C 盘可用 | 48.3 GB | **77.9 GB**（回收 29.6 GB） |
| 导出 tar | — | 5.51 GB（保留在 D: 作回滚） |

迁移后验证：`/etc/wsl.conf`（`systemd=true`、`default=johnny`）随文件系统保留，
默认用户、systemd、Docker 守护进程、镜像与工具链文件全部正常。

> 回滚方法：`wsl --unregister Ubuntu-22.04` 后从 `D:\WSL\ubuntu-22.04-backup.tar`
> 重新 import 即可。

### 8.4 日常注意事项

- 重建镜像会重新产生构建缓存（最多再涨约 10 GB），用完及时
  `docker builder prune -a -f`；由于 vhdx 只增不减，长期需再次导出/导入来压缩
- 若不愿每次迁移，可考虑把 Docker 的 `data-root` 指到 D 盘挂载点，
  但 WSL 访问 `/mnt/d` 走 9p 协议，I/O 明显更慢，**不推荐**
