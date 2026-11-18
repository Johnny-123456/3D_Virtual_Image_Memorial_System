# 3D 虚拟影像怀念系统 — 路线 D（2.5D 分层视差）实现方案

> ⚠️ **本文档已按 Step 0–1 实测修订，请先读 §12。**
> §2.3–§2.5 与 §5 Step 1 中"复用 rehab-coach 标定"的内容**已被实测推翻**：
> 该标定文件在本相机上不可用，Step 1 已改为数据驱动自校正。冲突时以 §12 为准。

| 项 | 内容 |
|---|---|
| 版本 | v1.0 |
| 日期 | 2026-11-18 |
| 硬件 | ELF 2 板（RK3588）+ 双目 UVC 相机 |
| 目标平台 | Ubuntu 22.04 / Linux 5.10.209 / Python 3.10.12 |
| 状态 | 方案定稿，待进入 Step 0 实施 |
| 前置变更 | 已完成：rehab-coach 三条开机自启全部禁用（见附录 C） |

---

## 1. 目标与边界

### 1.1 一句话目标

在一台 RK3588 上，把一张平面的 2D 人像变成**有真实纵深的立体影像**——观众左右移动头部时，画面里的人物与背景产生反向视差，看起来像"隔着一层玻璃在看一个人"，并带有呼吸、眨眼等微动，产生"在场"的错觉。

### 1.2 三条设计原则

1. **不做全 3D 重建**。不追求多边形数量，追求"看不出是假的"。
2. **主动规避恐怖谷**。半透明、辉光、降饱和是设计语言，不是事后补丁。
3. **全部本地化**。影像与声音不出设备，不上云。

### 1.3 明确不做（Non-Goals）

| 不做 | 原因 |
|---|---|
| 全身骨骼动画 | 已确定不使用陀螺仪，2.5D 方案不承载全身运动 |
| 实时 3DGS / NeRF 重建 | RK3588 无 CUDA，训练不可能；渲染方案另立路线 B |
| 语音对话 / 人格模拟 | 属演示层，本方案只负责"影像呈现出立体感"这一件事 |
| 多视角环绕观看 | 单屏视差方案没有侧后方视角，超出范围 |

> 语音、对话、人格记忆是独立的后续模块，可复用 `rehab-coach` 已有的 `sherpa-onnx` + `piper-tts` + `Qwen2.5-1.5B RKLLM`，不在本文档范围。

---

## 2. 硬件与软件基线（实测数据）

### 2.1 板卡

| 项目 | 实测值 |
|---|---|
| SoC | RK3588（ELF 2 板） |
| CPU | 4×A76 @2.256GHz + 4×A55 @1.8GHz（8 核） |
| GPU | Mali-G610（`fb000000.gpu`） |
| NPU | `fdab0000.npu`，调频上限 1GHz，**6 TOPS INT8** |
| 内存 | 7.7 GiB（空闲时可用 6.5 GiB） |
| 存储 | 56 GB，已用 19 GB，可用 36 GB |
| 系统 | Ubuntu 22.04.5 LTS，kernel 5.10.209 aarch64 |

### 2.2 相机（关键实测）

| 项目 | 实测值 |
|---|---|
| 设备 | UVC `CCB Camera`（VID:PID `26e0:ccb1`） |
| 节点 | `/dev/video21`（index0，采集）、`/dev/video22`（index1，格式为空，疑为 metadata） |
| 稳定路径 | `/dev/v4l/by-id/usb-CCB_Camera_CCB_Camera-video-index0` |
| 最大分辨率 | **2560×720（左右目并排，各 1280×720）** |
| 帧率 | 60 / 30 / 20 / 15 / 10 / 5 fps（**实测 31.0 fps**，前提 `BUFFERSIZE>=2`；设 1 会掉到 14.8 fps，见 §12.1） |
| 像素格式 | **仅 MJPG**（无 YUYV/NV12，必须做 JPEG 解码） |
| 传感器 | AR0144（**全局快门**，运动无果冻效应） |
| 其他模式 | 1280×720、1280×712、800×600、640×480、640×360 |

> **结论**：`2560×720` 实测可用，返回 `(2560, 720, 3)`。左右目在**同一个 UVC buffer** 内，天然同帧同曝光——这是本方案相对于拼两台相机最大的优势，无需硬件同步。

### 2.3 已标定的双目参数（`rehab-coach/src/backend/calibration/stereo_calib.json`）

| 参数 | 左目 | 右目 |
|---|---|---|
| fx | 1198.70 | 1154.13 |
| fy | 1201.16 | 1156.83 |
| cx | **597.65** | **737.16** |
| cy | 308.45 | 323.69 |
| 畸变 | k1=0.179, k2=1.384, k3=-6.82 | k1=0.230, k2=-0.017, k3=-0.62 |

| 立体参数 | 值 |
|---|---|
| 重投影误差 | **0.708 px**（优秀） |
| 标定图 | 40 张，9×6 棋盘格，格距 28 mm |
| 旋转 R | 接近单位阵（最大分量 ~0.016） |
| 平移 T | **(-47.23, 2.34, -39.54) mm** |
| \|T\| | **61.64 mm**（与 `measured_baseline_mm` 一致） |
| `baseline_mm` 字段 | **52.0**（⚠️ 与实测矛盾，见 §2.5） |
| `rectify_alpha` | 0.0 |
| 旧管线是否校正 | **否**（`calibration_meta.json` 中 `"rectify": false`） |

### 2.4 关键几何推论（**必须理解**）

1. **有效水平基线只有 47.23 mm，不是 61.64 mm**。深度计算必须用 `T_x`，用 `|T|` 会把距离高估约 30%。
2. **两目存在 39.5 mm 的轴向错位**（`T_z`），意味着左右视图的**尺度不同**（3m 处约差 1.3%）。这是原始拆半直接做模板匹配失败的根因。
3. **cx 相差 139.5 px 是真实的**，不是标定错误。实测原始拆半后存在稳定 `dx≈144 px` 的静态错位，与之一致。
4. 因此：**拆半 ≠ 可用立体像对，必须 `stereoRectify` + `remap` 之后才能算视差。**

### 2.5 三个待确认项（Step 1 需解决）

| # | 待确认 | 影响 | 处置 |
|---|---|---|---|
| 1 | `baseline_mm=52.0` 与 `measured_baseline_mm=61.64` 矛盾 | 深度绝对尺度会差 18% | 用现有标定重算，或用已知尺寸物体实测复核 |
| 2 | `T_z=-39.54 mm` 是否真实 | 若标定有误，remap 会引入错误形变 | 用棋盘格重新标定一次比对 |
| 3 | `/dev/video22` 的作用 | 可能可拿到硬件时间戳 | 低优先，`v4l2-ctl --all` 查看 |

### 2.6 软件环境

| 组件 | 版本 / 状态 |
|---|---|
| Python | 3.10.12 |
| OpenCV | 4.9.0（`opencv-python`）/ 4.10.0（contrib） |
| numpy | 1.26.4 |
| **RKNN 运行时** | `rknn-toolkit-lite2` **2.3.2**（仅推理） |
| **RKNN 转换工具** | ❌ 未安装（`rknn-toolkit2` 仅支持 x86 PC） |
| 语音 | `sherpa_onnx` 1.13.2、`piper-tts`（`third_party/`） |
| 大模型 | `Qwen2.5-1.5B-Instruct` W8A8 RKLLM（2.0 GB，已就位） |
| 姿态模型 | `yolov8n-pose-int8.rknn`（5.9 MB，已就位） |
| 硬件编解码 | `librockchip_mpp` 已安装，GStreamer 1.20.3 |
| ROS2 | ❌ 未安装（本方案不依赖） |

### 2.7 可直接复用的既有资产

| 资产 | 路径 | 用途 |
|---|---|---|
| 双目标定参数 | `rehab-coach/src/backend/calibration/stereo_calib.json` | Step 1 |
| 拆半 + stereoRectify 实现 | `rehab-coach/src/backend/vision/camera_manager.py:540-780` | Step 1 |
| 立体三角化 | `rehab-coach/src/backend/vision/stereo_3d.py` | Step 1 |
| int8 量化校准集 | `rehab-coach/models/quant_images/` | Step 2 |
| Qt 界面工程 | `rehab-coach/src/qt_gui/` | Step 4 |
| RKNN 推理封装 | `rehab-coach/src/backend/vision/pose_rknn.py` | Step 5 |

---

## 3. 为什么选路线 D

| 路线 | 内容 | 工作量 | RK3588 可行 | 恐怖谷风险 | 结论 |
|---|---|---|---|---|---|
| A | 实时双目点云"魂影" | 2 周 | 中（SGBM 吃 CPU） | 低 | 备选，作为演示模式保留 |
| B | 3DGS 数字人 + 骨骼驱动 | 1–2 月 | 渲染勉强（需 GPU 优化） | 中高 | 长期目标 |
| C | 多视角阵列 + 表情驱动 | 3 月+ | ❌ 需加硬件 | 高 | 不做 |
| **D** | **2.5D 分层视差（本文档）** | **9 天** | **✅ 轻松** | **最低** | **✅ 采用** |

**决策依据：**

1. **算力非对称**——重型计算（分割、深度）只在采集期跑一次，呈现期每帧只有追踪 + 顶点着色器，30fps 毫无压力。这与"一次采集、永久呈现"的怀念产品形态天然吻合。
2. **规避恐怖谷最彻底**——不做 3D 几何，只做"有厚度的影像"，大脑不会去挑建模瑕疵。
3. **失败成本低**——每一步都能出可见成果，不像重建路线要等一整套管线跑通才知道行不行。

---

## 4. 系统架构

### 4.1 总体管线

```
┌─ 采集期（重活，每人只跑一次）────────────────────────────────┐
│                                                              │
│  /dev/video21  2560×720@30 MJPG                              │
│        │                                                     │
│        ├─ 拆半 ─→ L 1280×720 ─┐                              │
│        │                   R 1280×720 ─┐                     │
│        │                                │                     │
│        │   ┌────────────────────────────┴──┐                 │
│        │   │ stereoRectify + remap         │ → 校正像对       │
│        │   └────────────┬──────────────────┘                 │
│        │                ├─→ SGBM ──────────→ 稀疏深度真值     │
│        │                └─→ 单目深度(NPU) ──→ 稠密相对深度     │
│        │                              │                       │
│        │                     ┌────────┴────────┐              │
│        └──→ 人像分割(NPU) ───→│  尺度拟合(§5.3)  │→ 米制深度     │
│                               └────────┬────────┘              │
│                                        ↓                       │
│                        assets/<person>/{rgb.png, depth.npy,    │
│                                          matte.png, meta.json} │
└──────────────────────────────────────────────────────────────┘

┌─ 呈现期（极轻，60fps）──────────────────────────────────────┐
│                                                              │
│  同一台相机 /dev/video21 ──→ 人脸 + 虹膜检测(NPU)              │
│                                    │                          │
│                          鼻尖 3D 位置(双目深度)                │
│                                    │                          │
│                          One-Euro 平滑                         │
│                                    │                          │
│                    camera↔screen 外参(PnP 标定)                │
│                                    ↓                          │
│          assets 加载 ──→ DIBR 网格位移(GLSL) ──→ 屏幕 / 魂盒   │
│                              ↑                                 │
│                    程序化微动（呼吸/眨眼/微摆）                 │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 两种模式的资源占用

| 阶段 | 频率 | 主要负载 | 目标 |
|---|---|---|---|
| 采集期 | 一次性，60 秒 | SGBM(CPU) + 单目深度(NPU) + 分割(NPU) | 不要求实时，能跑完即可 |
| 呈现期 | 持续，60fps | 人脸检测(NPU) + DIBR(GPU) | 端到端延迟 <100ms |

### 4.3 数据产物

每个被怀念的人一个目录，**自包含、可离线分发**：

```
assets/<person_id>/
├── rgb.png      主图，1920×1080 或 2560×1440（采集时多帧择优）
├── depth.npy    float32，单位米，与 rgb 同尺寸
├── matte.png    8-bit alpha，头发边缘需保留半透明
├── meta.json    尺度参数、焦点深度、眨眼帧索引、采集时间
└── frames/      （可选）3–5 帧眼睛开合图，用于眨眼
```

> 体积估算：1920×1080 的 PNG + float32 深度 ≈ 6–8 MB/人。加密存储，本地保管。

---

## 5. 详细实现步骤

### Step 0 · 相机接管与分辨率确认（0.5 天）

**目标**：独占相机，确认实际输出与拆分正确。

**前置**：`rehab-coach` 的三个进程已停止（已完成，见附录 C）。

**做法**：

```python
import cv2
cap = cv2.VideoCapture('/dev/v4l/by-id/usb-CCB_Camera_CCB_Camera-video-index0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

for _ in range(10):          # 丢弃前 10 帧，等 AE/AWB 收敛
    cap.read()
ok, frame = cap.read()       # 期望 (720, 2560, 3)
L, R = frame[:, :1280], frame[:, 1280:]
```

**踩过的坑**：

- 用索引 `cv2.VideoCapture(21)` 不抗设备重编号，**必须用 `/dev/v4l/by-id/` 稳定路径**。
- MJPG 必须显式设置，否则可能协商到不支持的模式。
- 前 10 帧有曝光/白平衡震荡，直接用于深度会产生跳变。

**验收**：`frame.shape == (720, 2560, 3)`，左右半帧画面内容连贯。

---

### Step 1 · 极线校正与深度真值（1.5 天）★ 最关键

**目标**：把原始拆分变成**可直接算视差的校正像对**，并得到稀疏但正确的双目深度真值。

> ⚠️ **这是本方案最容易出错的一步。** 原始拆半后的左右图存在约 144 px 静态错位和 1.3% 尺度差，**不校正直接跑 SGBM 会得到完全错误的结果**（实测出现"中位视差 44 px 但相关系数仅 0.01"的自相矛盾现象）。

**做法**：

```python
import cv2, numpy as np, json

calib = json.load(open('calibration/stereo_calib.json'))
M1, d1 = np.array(calib['camera_matrix_left']), np.array(calib['dist_coeffs_left'])
M2, d2 = np.array(calib['camera_matrix_right']), np.array(calib['dist_coeffs_right'])
R, T = np.array(calib['R']), np.array(calib['T'])
h, w = 720, 1280

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    M1, d1, M2, d2, (w, h), R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

mapL = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (w, h), cv2.CV_32FC1)
mapR = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (w, h), cv2.CV_32FC1)

gL = cv2.remap(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), *mapL, cv2.INTER_LINEAR)
gR = cv2.remap(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), *mapR, cv2.INTER_LINEAR)
```

**SGBM 参数（建议初值）**：

```python
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=7,
    P1=8 * 3 * 7**2, P2=32 * 3 * 7**2,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_HH)
disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0
xyz  = cv2.reprojectImageTo3D(disp, Q)
```

**深度精度（用有效水平基线 47.23 mm 计算）**：

| 距离 Z | 视差 d = f·B/Z | 深度精度 δ Z = Z²·δd/(f·B)，δd=1/16 px |
|---|---|---|
| 1.0 m | 56.6 px | **1.1 mm** |
| 1.5 m | 37.7 px | 2.5 mm |
| 2.0 m | 28.3 px | **4.4 mm** |
| 3.0 m | 18.9 px | 9.9 mm |

> **结论**：1–2 m 肖像距离上深度精度 1–4 mm，**完全够用**。真正的短板不是精度，而是**皮肤/白墙等弱纹理区域的空洞率**——这正是 Step 3 引入单目深度的理由。

**验收标准**：

1. 极线校正后，同一棋盘格角点在左右图的 **y 差 < 1 px**；
2. 3 m 外参考物视差接近 0（验证 `CALIB_ZERO_DISPARITY` 生效）；
3. 1.2 m 处实测深度误差 < 2 cm；
4. SGBM 在人脸区域的有效像素率记录基线值（预期 30–60%）。

**若极线 y 差 > 2 px**：说明 `T` 或 `R` 有问题，用 20–30 张棋盘格重标定（28 mm 格距，9×6）。

---

### Step 2 · 单目深度模型上 NPU（1 天）

**目标**：得到稠密、边缘平滑、无空洞的相对深度图，用来补双目的空洞。

**模型选型**：

| 模型 | 输入 | int8 体积 | 说明 |
|---|---|---|---|
| **DepthAnything-V2-Small** | 518×518 | ~25 MB | **首选**，边缘质量最好 |
| MiDaS v2 small | 256×256 | ~10 MB | 备选，速度更快 |
| FastDepth | 224×224 | ~5 MB | 室内场景够用，边缘一般 |

**转换流程（必须在 x86 PC 上完成）**：

```bash
# PC 侧（x86 Linux，板子上只有 lite 运行时，不能转换）
pip install rknn-toolkit2
python convert.py                      # ONNX → RKNN int8
# 量化校准集直接用 rehab-coach/models/quant_images/ 里的现成图片
```

**板端推理**：

```python
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn('depth_anything_v2_s.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
out = rknn.inference(inputs=[resized_float])   # 返回相对深度（无量纲）
```

**性能目标**：518×518 单帧 **< 40 ms**。

**坑**：

- DepthAnything 输出是**相对视差**（大 = 近），不是深度，也不是米。**直接当深度用会得到错误结果**，必须过 Step 3。
- int8 量化对深度模型的远端精度损伤明显，量化后要与 fp32 输出做一次逐像素对比，`abs diff` 的 p95 应 < 满量程的 5%。

---

### Step 3 · 把相对深度变成米（0.5 天）★ 不可跳过

**目标**：把无量纲的相对深度映射为米制深度，使视差幅度物理正确。

**方法：用双目做标尺，最小二乘拟合**

```python
# 同一帧上：depth_stereo 稀疏但正确，rel 稠密但无尺度
z_stereo = xyz[:, :, 2]
mask = (z_stereo > 0.3) & (z_stereo < 5.0) & np.isfinite(z_stereo)

inv_z = 1.0 / z_stereo[mask]                    # 1/Z 与视差成正比，几何上更合理
A = np.stack([rel[mask], np.ones(mask.sum())], 1)
a, b = np.linalg.lstsq(A, inv_z, rcond=None)[0]

depth_metric = 1.0 / np.clip(a * rel + b, 1e-3, None)
```

**更稳的交叉验证（推荐同时做）**：利用人脸先验——瞳距 ≈ 63 mm、脸宽 ≈ 140 mm。检测 106 点取瞳心，双目三角化出距离，反解尺度。人像场景下这个先验比稀疏 SGBM 更鲁棒。

**验收标准**：屏幕上量出的深度范围（如 0.9–1.8 m）与卷尺实测相差 **< 10%**。

**坑**：`a` 的符号要检查——DepthAnything 输出"越大越近"，所以 `a` 应为正。若拟合出的 `a` 近 0 或为负，说明 SGBM 真值有误，回到 Step 1。

---

### Step 4 · DIBR 网格位移渲染（2 天）★ 核心

**目标**：视点移动时，画面按深度分层产生正确的反向位移。

**不要做硬分层。** 把深度切成 8–12 层会出现层间割裂感。正确做法是**网格变形**：

把 (RGB, depth) 建成一张 `160×90` 顶点网格，每顶点 `z` 取该点深度。

**渲染数学（重要：不要直接套用采集焦距）**

直觉上应该用 `parallax = f·t_x/Z`，但代入采集内参会得到灾难性结果：`f=1198, t_x=0.15m, Z=0.9m → 200 px` 位移，画面会崩。

正确做法是把深度**重映射到一个虚拟显示空间**，再按显示参数计算位移：

```glsl
// vertex shader 核心
attribute float aDepthNorm;   // 0.0=最近, 1.0=最远
uniform vec3  uEye;           // 眼点偏移，归一化到 [-0.5, 0.5]
uniform float uFocus;         // 焦点深度（人物主体所在层）
uniform float uGain;          // 视差增益，单位 px

void main() {
    // 1) 映射到虚拟显示深度：焦点处位移为 0
    float z = mix(0.45, 1.75, aDepthNorm);       // 虚拟深度 0.45~1.75 m（设定值）
    float parallax = uGain * (1.0/uFocus - 1.0/z) * 2.0;

    // 2) 前层反向、后层同向，形成相对位移
    vec3 p = aPos;
    p.xy += uEye.xy * parallax;
    gl_Position = uProjection * uModelView * vec4(p, 1.0);
}
```

**参数建议初值**：`uGain = 80 px`（对应眼动 ±0.15 m，画面位移约 ±12 px，肉眼清晰可辨且不晕）。`uFocus` 取人脸所在深度。

**空洞处理（决定成败）——前层遮挡法**

网格拉伸后人物边缘会拉出空洞。分两遍画：

1. **背景层**：同一张图放大 2% + 高斯模糊（σ≈8），先画满屏
2. **人物层**：用 `matte.png` 做 alpha，后画

视觉上自然呈现为"主体实、背景虚"，拉丝被背景层填掉。

**实现选型**：直接用 **Qt `QOpenGLWidget`**，复用 `rehab-coach/src/qt_gui/` 的编译脚本（`scripts/build_linux.sh`，产物 `prograss_copy`），改造成本最低。

**验收标准**：

1. 头左右移动 ±15 cm，人物与背景产生**方向相反**的位移，幅度 ≥ 8 px；
2. 头发/肩线边缘无 ≥ 3 mm 的拉丝或破洞；
3. 呈现帧率 ≥ 30 fps。

**坑**：

- 纹理配准必须按 `depth` 而不是 `matte` 做，否则边缘错位。
- 网格分辨率不要过高（>320×180）否则顶点着色器成瓶颈；也不要过低否则出现折面。160×90 是实测甜点。
- GLSL 在板子上调试痛苦，**先在 PC 上用 OpenGL 调通再搬**。

---

### Step 5 · 观众眼位追踪（2 天）

**目标**：实时、低延迟地拿到观众头部/眼睛在相机坐标系中的 3D 位置。

**做法**：

1. **人脸关键点**：`yolov8n-pose.rknn` 是人体姿态模型，不适合脸部细节。建议换成 106 点 face landmark（PIPNet / 2d106det 轻量版，同样可转 RKNN）；先跑通可用 OpenCV `FacemarkLBF`。
2. **取哪一点**：优先虹膜中心。简化版取**鼻尖 + 两瞳中点**，误差可接受。
3. **转 3D**：用双目 SGBM 深度在鼻尖邻域取中值，得 `(x, y, z)` 米。
4. **平滑**：**One-Euro Filter**（低延迟场景显著优于卡尔曼）：

```python
# 建议参数：mincutoff=1.5 Hz, beta=0.01, dcutoff=1.0
alpha = 1.0 / (1.0 + (2*math.pi*cutoff*dt))
x_filtered = alpha*x_prev + (1-alpha)*x
```

**必须做的一步：camera ↔ screen 外参标定**

漏掉这步，视差会"不跟手"且方向错误。

```
屏幕上显示已知尺寸的棋盘格 → 用相机拍下来
→ cv2.solvePnP(棋盘格角点3D, 图像点2D, 相机内参, 畸变)
→ 得到屏幕平面在相机坐标系中的位姿
→ 建立「屏幕像素坐标 ↔ 相机 3D 坐标」双向映射
```

**延迟预算**：

| 环节 | 耗时 |
|---|---|
| 采集（30fps） | 33 ms |
| 人脸检测 + 关键点 (NPU) | 15 ms |
| 平滑 | 10 ms |
| 渲染 | 16 ms |
| **合计** | **≈ 75 ms** |

> **硬指标：端到端 < 100 ms。** 超过会产生眩晕感，不可接受。

**验收标准**：头部快速左右移动，画面视差跟随流畅、无抖动、无可见滞后。

---

### Step 6 · 微动：制造"活着"的感觉（1 天）★ 情感核心

**目标**：让静止影像具有生命体征。

> 这一步对最终效果的影响，**远大于渲染精度**。人脑对"活着"的判定主要来自微动，而不是分辨率。

| 微动 | 实现方式 | 参数 |
|---|---|---|
| 呼吸 | 整体纵向正弦缩放 | ±0.3%，周期 3.5–5 s，带 15% 随机抖动 |
| 眨眼 | 3–5 帧眼部开合图交叉淡化 | 3–7 s 随机触发，过渡 80 ms |
| 微摆 | 头部区域（按深度选上层）低频摆动 | ±0.5°，叠加极轻微水平漂移 |
| 亮度呼吸 | 画面整体亮度极缓起伏 | ±0.2% |

**核心原则**：所有幅度都要小到**"说不清哪里在动"**。一旦观众能明确指出"它在动"，恐怖感立刻产生。

**加分项（可选，效果显著）**：呼吸节奏可以从**真实视频**中提取，而不是程序化正弦——用相位相关做亚像素级帧间位移估计，胸廓起伏可达亚毫米精度。这比正弦波有说服力得多，值得做。

---

### Step 7 · 美术与规避恐怖谷（1 天）

| 手法 | 实现 | 效果 |
|---|---|---|
| 轮廓辉光 | matte 外扩 2 px + 模糊，叠加为边缘光 | 把瑕疵变成风格 |
| **半透明魂影** | 人物层 alpha 降至 0.85 + 辉光 | **规避恐怖谷最有效的一招** |
| 色调 | 偏暖、降饱和、轻微褪色 | 符合"回忆"的视觉语言 |
| 颗粒 | 极淡 film grain | 掩盖深度边缘的锯齿 |
| 背景 | 最远层景深虚化 + 反向视差 | 强化纵深 |

---

### Step 8 · 呈现形态升级（可选，1 天）

| 方案 | 效果 | 成本 |
|---|---|---|
| 普通显示屏 | "窗口感"：能看出纵深，但人物出不了屏 | 0 |
| **45° 半透镜"魂盒"（推荐）** | **人物真正悬浮在盒中 + 真实视差**，效果质变 | < 200 元 |
| 光栅/柱镜裸眼 3D 屏 | 视角窄、分辨率减半、价格高 | 不建议 |

**魂盒要点**：

- 屏幕横置朝上，45° 半透玻璃反射，人像浮在盒中央
- 玻璃会吃掉约 50% 亮度，**屏幕亮度必须打满**
- 玻璃选低反射率（否则出现重影）
- 环境光要暗，背景必须是纯黑

---

## 6. 性能预算

### 6.1 采集期（一次性，不要求实时）

| 环节 | 耗时 | 硬件 |
|---|---|---|
| 采集 + MJPG 解码 | 25 ms | A76 |
| 极线校正 remap | 12 ms | A76 |
| SGBM（1280×720） | **510 ms** | A76 ×8（实测，见 §12.4） |
| 单目深度（518×518） | 35 ms | NPU |
| 人像分割 | 15 ms | NPU |
| **合计** | **≈ 600 ms/帧**（实测） | — |

> 采集 60 秒约 1800 帧；按 600 ms/帧算需 18 分钟——**因此 SGBM 只用于标定抽样**，
> 不做逐帧处理。若将来需要稠密视频深度，降到半分辨率可换 8 倍速度（见 §12.4）。

### 6.2 呈现期（持续 60 fps）

| 环节 | 耗时 | 硬件 |
|---|---|---|
| 采集（30fps） | 33 ms | A76 |
| 人脸检测 + 关键点 | 15 ms | NPU |
| 平滑 | 10 ms | A76 |
| DIBR 渲染 | 16 ms | Mali-G610 |
| **合计** | **≈ 75 ms** | — |

### 6.3 优化手段（若达不到 30 fps）

1. **采集降到 30 → 20 fps**，渲染插值，观感几乎无损
2. **人脸检测降频**：每 2–3 帧跑一次，中间用光流/位移预测插值
3. **渲染只更新变化区域**：眼位变化小于 1 px 时跳过重绘
4. **CPU 亲和性**：A76 前 4 核专给视觉，A55 给音频/IO/网络
5. **不要同时加载 LLM**（Qwen2.5-1.5B 占 2 GB 内存），演示模式下按需加载

---

## 7. 工程结构

```
~/3D_Virtual_Image_Memorial_System/
├── docs/
│   └── route-D-implementation.md     # 本文档
├── capture/                          # 采集期（重活）
│   ├── camera_io.py                  # 取帧 + 拆半 + 稳定设备路径
│   ├── rectify.py                    # stereoRectify + remap
│   ├── sgbm_truth.py                 # 双目稀疏真值
│   ├── depth_mono.py                 # 单目深度 RKNN 推理
│   ├── matte.py                      # 人像分割
│   └── scale_calib.py                # 相对深度 → 米制
├── assets/
│   └── <person_id>/
│       ├── rgb.png  depth.npy  matte.png  meta.json
│       └── frames/                   # 眨眼帧
├── render/
│   ├── shaders/dibr.vert  dibr.frag
│   ├── dibr_widget.py                # Qt QOpenGLWidget
│   ├── micro_motion.py               # 呼吸/眨眼/微摆
│   └── art_grade.py                  # 辉光/色调/颗粒
├── track/
│   ├── face_landmark.py              # 106 点（RKNN）
│   ├── eye_position.py               # 3D 眼点
│   ├── one_euro.py                   # 平滑滤波
│   └── cam2screen.py                 # 外参标定（PnP）
├── config/
│   ├── display.yaml                  # 屏幕尺寸/分辨率/相机-屏幕外参
│   └── assets.yaml                   # 当前加载的人物
├── models/                           # 从 rehab-coach 迁入
│   ├── depth_anything_v2_s.rknn
│   ├── face_landmark_106.rknn
│   └── human_seg.rknn
├── tools/
│   ├── depth_eval.py                 # 深度质量评测（Step 1 验收）
│   ├── rectify_check.py              # 极线校验
│   └── cam2screen_calib.py           # 屏幕标定辅助
└── README.md
```

---

## 8. 验收标准

| # | 指标 | 阈值 | 测量方法 |
|---|---|---|---|
| 1 | 视差效果 | 头动 ±15 cm → 人物相对背景反向位移 ≥ 8 px | 录屏逐帧比对 |
| 2 | 端到端延迟 | **< 100 ms** | 高速摄影 / 屏幕-实物同步打点 |
| 3 | 呈现帧率 | ≥ 30 fps | `QOpenGLWidget` 帧计时 |
| 4 | 边缘质量 | 头发/肩线无 ≥ 3 mm 拉丝 | 100% 放大目视 |
| 5 | 深度绝对精度 | 与卷尺误差 < 10% | Step 3 验收 |
| 6 | 观感 | 连续观看 5 分钟不晕、不恐怖 | 主观评测 ≥ 3 人 |

---

## 9. 排期（9 个工作日）

| 天 | 内容 | 关键交付 |
|---|---|---|
| D1 | Step 0 + Step 1 上半（校正、极线校验） | 一张正确的视差图 |
| D2 | Step 1 下半（SGBM 调参 + 深度精度实测） | `tools/depth_eval.py` 报告 |
| D3 | Step 2（PC 转换单目深度模型）+ Step 3 | 米制深度，误差 < 10% |
| D4–D5 | Step 4（PC 侧先调通 DIBR，再搬板子） | 可交互的 2.5D 视差 demo |
| D6–D7 | Step 5（人脸追踪 + 屏幕标定 + 平滑） | 头动跟手，延迟 < 100 ms |
| D8 | Step 6 + Step 7（微动 + 美术） | 完整观感 |
| D9 | Step 8（魂盒）+ 整机联调 | 可演示样机 |

**关键决策点：D2 结束。** 如果那时 SGBM 有效像素率 < 25%，说明双目弱纹理问题严重，应调整为"单目深度为主、双目仅做尺度标尺"，Step 3 权重加大。

---

## 10. 风险与对策

| # | 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|---|
| 1 | 皮肤/白墙弱纹理导致深度空洞 | **高** | 中 | 主路径用单目深度，双目仅提供稀疏真值（Step 3 已设计） |
| 2 | 校正后仍残留尺度差（T_z=39.5 mm） | 中 | **高** | 用 1.2 m 处棋盘格实测复核；必要时按行做尺度补偿 |
| 3 | `baseline_mm` 与实测 61.64 mm 矛盾 | 中 | 高 | Step 1 用已知尺寸物体实测标定，不信任配置字段 |
| 4 | 端侧渲染达不到 30 fps | 中 | 中 | §6.3 五级降级策略 |
| 5 | 端到端延迟 > 100 ms 导致眩晕 | 中 | **高** | 人脸检测降频 + 光流预测；必要时降低采集帧率 |
| 6 | 恐怖谷（越像越怕） | 中 | 中 | Step 7 半透明魂影模式兜底 |
| 7 | C++/Qt 渲染开发慢 | 中 | 低 | 先在 PC 用 Python + moderngl 验证算法 |
| 8 | 单目深度 int8 量化掉点 | 中 | 中 | 量化后与 fp32 逐像素对比，p95 差异 < 5% |

---

## 11. 隐私与伦理红线（不可妥协）

> 这是"怀念"类产品，做错了会真实伤害人。

1. **全部本地化**：影像、深度、声音不出设备，不联网，不上云。
2. **静态加密**：`assets/` 目录整体加密存储，密钥不落盘明文。
3. **明确标注"这是模拟"**：界面上必须有可见标识，并提供防伪开关。此类技术最大的滥用方向是"复活式诈骗"。
4. **不做 24 小时陪伴**：持续使用会加重**延长哀伤障碍（Prolonged Grief Disorder）**。必须设计：
   - 「纪念日模式」与「日常模式」分离
   - 单次使用时长提示
   - 有仪式感的"关闭/告别"过渡动画
5. **知情同意**：被采集者生前须明确同意其影像被用于此用途，且可随时撤回。
6. **设计文档第一行**：**目标不是让他回来，是让你能好好说再见。**

---

## 附录

### A. 环境与命令速查

```bash
# 相机枚举
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video21 --list-formats-ext

# 稳定设备路径
ls -l /dev/v4l/by-id/

# 抓一帧测试
python3 -c "
import cv2
cap=cv2.VideoCapture('/dev/v4l/by-id/usb-CCB_Camera_CCB_Camera-video-index0')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
for _ in range(10): cap.read()
ok,f=cap.read(); print(ok, None if not f else f.shape)
"

# NPU 负载（需 root）
sudo cat /sys/kernel/debug/rknpu/load

# 温度
cat /sys/class/thermal/thermal_zone0/temp
```

### B. 标定参数全量

```json
{
  "camera_matrix_left":  [[1198.699, 0, 597.653], [0, 1201.158, 308.449], [0, 0, 1]],
  "camera_matrix_right": [[1154.127, 0, 737.159], [0, 1156.831, 323.686], [0, 0, 1]],
  "dist_coeffs_left":    [[0.17940, 1.38448, -0.00601, 0.00769, -6.81561]],
  "dist_coeffs_right":   [[0.23032, -0.01657, 0.00269, -0.00139, -0.61999]],
  "R":  [[0.999959, -0.00075, 0.00903], [0.0009, 0.99987, -0.01614], [-0.00902, 0.01615, 0.99983]],
  "T":  [[-47.228], [2.342], [-39.536]],
  "reprojection_error": 0.708,
  "baseline_mm": 52.0,
  "measured_baseline_mm": 61.636,
  "pattern_size": [9, 6],
  "square_size_mm": 28.0,
  "camera_model": "AR0144"
}
```

**派生量**：

- `|T|` = 61.64 mm（两目中心距）
- `T_x` = **47.23 mm**（有效水平基线，深度计算用这个）
- `cx_right - cx_left` = **139.5 px**（原始拆分静态错位的来源）

### C. 已完成的前置变更：rehab-coach 自启禁用

| 项 | 处理 | 当前状态 |
|---|---|---|
| `rehab_backend.service` | `systemctl disable --now` | disabled / inactive |
| `bt_rehab_gateway.service` | `systemctl disable --now` | disabled / inactive |
| `~/.config/autostart/rehab-qt.desktop` | 改名为 `.desktop.disabled` | 已禁用 |

**恢复方法**：

```bash
sudo systemctl enable --now rehab_backend.service bt_rehab_gateway.service
mv ~/.config/autostart/rehab-qt.desktop.disabled ~/.config/autostart/rehab-qt.desktop
```

### D. 术语表

| 术语 | 说明 |
|---|---|
| DIBR | Depth-Image-Based Rendering，基于深度图的渲染 |
| 视差 | 同一物体在左右目成像中的水平位置差，与深度成反比 |
| 极线校正 | 把双目图像重投影到同一平面，使对应点位于同一水平行 |
| SGBM | Semi-Global Block Matching，半全局块匹配立体算法 |
| One-Euro Filter | 低延迟自适应低通滤波，适合交互式追踪 |
| 恐怖谷 | 拟真度接近真人但未完全相同时引发的反感 |
| 魂盒 | Pepper's Ghost 半透镜悬浮成像装置 |
| 延长哀伤障碍 | Prolonged Grief Disorder，无法走出丧亲之痛的精神障碍 |

---

*文档结束。下一步：执行 Step 0 与 Step 1，用 `tools/depth_eval.py` 量化这套双目在 1–2 m 的实际可用性，作为 D2 决策点的输入。*

---

## 12. 实测修正记录（Step 0–1 回填）

> 本节记录实施过程中被实测推翻的假设。**与前面章节冲突时以本节为准。**

### 12.1 相机帧率：`BUFFERSIZE=1` 会把帧率砍半

实测（2560×720，同一场景连续采 40–50 帧）：

| BUFFERSIZE | 实测 fps | 中位间隔 |
|---|---|---|
| 1 | **14.8** | 67.7 ms |
| 2 | **31.1** | 32.2 ms |
| 4（驱动默认） | 31.1 | 32.2 ms |

仅 `grab()` 不解码时同样是 14.8 fps，证明**不是解码瓶颈，是驱动没有空闲 buffer 可填**。
结论取 `BUFFERSIZE=2`（满帧且排队帧数最少）。已写入 `capture/camera_io.py`。

### 12.2 标定文件不可用（推翻 §2.3–§2.5）

`rehab-coach/src/backend/calibration/stereo_calib.json` **在本相机上不可用**。证据：

| 现象 | 数值 |
|---|---|
| 基线与 x 轴夹角 | **40.0°**（T = −47.23, 2.34, −39.54 mm） |
| `stereoRectify` 得到的 cx | **−885.5**（负数，物理不可能） |
| `alpha=0` 时的 fx | **5579.2**（原始 1198.7，等于 4.66 倍变焦） |
| 公共有效区 | 仅 56.9% |
| 校正后特征匹配 | 完全失效（SIFT 匹配不到对） |

同时实测证明了两件事：

1. **是真立体，不是伪立体**：纯水平平移模型残差 RMSE=82.4、相关系数 0.55
   （排除"右图是左图副本"）；分块匹配 27/28 块 NCC > 0.7（多块 0.99+），
   视差在 124–162 px 内有 38 px 离散——这是真实的深度变化。
2. **原始拆半图本身已接近行对齐**：校正前中位 |Δy| 仅 0.49 px，81.9% < 1px。
   说明该模块出厂已基本对准，只是存在约 143 px 水平偏移 + 0.36% 尺度差。

因此 §2.4 中"有效基线必须用 T_x = 47.23 mm"的推论**不再适用**——它建立在不可用的
外参之上。（几何上，若走标定路线，校正后的有效基线应等于 |T|；但本方案已改用
自校正，不依赖该外参。）

### 12.3 Step 1 改为数据驱动自校正（推翻 §5 Step 1 的做法）

实际实现（`capture/rectify.py`）：

1. SIFT 匹配 → `findFundamentalMat`（RANSAC）→ `stereoRectifyUncalibrated` 得 H1/H2
2. 写入 `config/rectify.json`（相机属性，入库，之后固定不变）
3. 备选路径：`estimateAffinePartial2D` 相似变换（只变换右目，保留左目为参考视图）
4. 用 SIFT 匹配点的 |Δy| 衡量极线对齐度（不需要棋盘格）

实测结果：

| 指标 | 校正前 | 校正后 | 阈值 |
|---|---|---|---|
| 中位 \|Δy\| | 0.49 px | **0.23 px** | < 1.0 |
| \|Δy\| < 1 px 占比 | 81.9% | **93.0%** | > 80% |
| 公共有效区 | — | 964×712（74.5%） | — |

`fx` 仍取标定文件的 1198.7 作为量纲估计；**基线绝对尺度留到 Step 3 标定**。

### 12.4 SGBM 实测性能（修正 §6.1 里 40 ms 的估计）

964×712 有效区、8 核：

| 参数 | 耗时 | 有效视差率 |
|---|---|---|
| HH nd=128 bs=7 全分辨率 | **510 ms** | **54.5%** |
| SGBM nd=128 bs=7 全分辨率 | 512 ms | 36.4% |
| SGBM nd=64 bs=7 全分辨率 | 276 ms | 39.7% |
| SGBM nd=64 bs=5 半分辨率 | 60 ms | 30.9% |
| SGBM nd=48 bs=5 半分辨率 | 49 ms | 32.2% |

§6.1 的 40 ms 严重低估，实际 510 ms。但 **SGBM 只在 Step 3 标定时用几次**，
不需要逐帧跑，故不影响排期。默认参数取 HH + nd=128（有效率最高）。

### 12.5 深度评测第一轮（场景不在肖像距离，结论仅作参考）

当前相机朝向 3–8 m 的办公室，不是目标场景：

| 指标 | 实测 |
|---|---|
| 有效视差率 | 41.7% |
| 中位视差 | 4.7 px |
| 中央肖像区有效率 | 52.0% |
| 左侧（白墙） | 14.4% ← 弱纹理空洞 |
| SGBM 耗时 | 618 ms |

**待办：需在 1.2 m 肖像距离复测。** 预期视差升到 30–70 px，有效率和深度精度都会显著改善。

### 12.6 已完成的提交

| commit | 内容 |
|---|---|
| `cd2f02b` | 仓库骨架 + Step 0 相机 I/O |
| `c6e1971` | Step 1 极线校正 + SGBM 深度真值 |
