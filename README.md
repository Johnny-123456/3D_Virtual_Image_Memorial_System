# 3D 虚拟影像怀念系统

在 ELF 2（RK3588）上，用双目 UVC 相机把 2D 人像变成**有真实纵深的 2.5D 立体影像**：
观众移动头部时，画面中人物与背景产生反向视差，并带呼吸/眨眼等微动。

完整方案见 **[docs/route-D-implementation.md](docs/route-D-implementation.md)**
（实施过程中的实测修正见该文档 **§12**）。

---

## 项目状态（2026-11-24）

> **给下一个对话的定位**：功能主线（Step 0–7）全部可用，采集→出资产→观看闭环已跑通；
> 另外新增 Step 9/10：**一张照片就能出资产**（动物已实测跑通）、首页可删素材。
> 最近一轮改动：素材删除键（`f295c92`）、`tools/import_photo.py` 照片导入、
> 视差/微动按资产分辨率归一（180 px 的小图不再被放大成夸张晃动）。
> §8 验收 6 条里 **3 条达标、1 条不达标、2 条未测**（下节有实测数字）。
> 当前短板是**端到端延迟 138 ms（要求 100 ms）**，瓶颈在姿态模型推理 54 ms。
> 所有已修的问题都写在 `docs/route-D-implementation.md` §12，改动看 `git log`。

### §8 验收实测

| # | 指标 | 阈值 | 实测 | 结论 |
|---|---|---|---|---|
| 1 | 视差（头动 ±15 cm） | ≥ 8 px | 1.0 m 16.2 / 1.2 m 13.5 / 1.5 m 10.8 / 2.0 m 8.1 px | ✅ |
| 2 | 端到端延迟 | < 100 ms | **138 ms**（相机 64 + 推理+传输 66 + 呈现 8） | ❌ |
| 3 | 呈现帧率 | ≥ 30 fps | 57.9 fps | ✅ |
| 4 | 边缘质量（无 ≥3 mm 拉丝） | — | 轮廓梯度 78~80（干净基准 36），gain 75 / eye 1.0 目视无拉丝 | ✅ |
| 5 | 深度绝对精度 | < 10 % | 1–2 m 段中位 108 mm / 9.9 %（R²=0.791） | ✅ |
| 6 | 观感（3 人 × 5 分钟） | — | 未做 | ⬜ |

微动（"别太死板"）：呼吸（纵向 0.26% + 横向 0.7 px）、点头（头部 1.8 px / 8.6 s）、
眨眼（2.5–6 s）、眼神（两眼小椭圆内 1–2 px 游移）、背景 25% 视差、回望（轻微前倾）。

复跑方式（两条命令，不占相机/不要求真人在场）：

```bash
python3 tools/acceptance.py parallax        # §8-1，渲染快照 + 光流测位移
python3 tools/acceptance.py latency         # §8-2，起 tracker + viewer 量端到端
```

### 改渲染之后必跑的回归检查

给某个素材/某种比例做的改动，很容易顺手把别的素材改坏（本项目踩过：为了
让狗铺满而把背景层改成 cover，结果所有人像都变成"背景被放大、人物没放大"）。
所以动 `dibrwidget.*` / `studio.cpp` 之后按这个清单走一遍：

```bash
# 1) 取改动前的提交编一个基线二进制
git worktree add /tmp/base 8fe12eb
cd /tmp/base/ui && mkdir -p build && cd build && cmake .. && make -j4 memorial_viewer

# 2) 逐资产 × 逐参数（8 种：正视点/左右/上下/回望/眨眼/两种风格）逐像素对比
python3 tools/regress_render.py --baseline /tmp/base/ui/build/memorial_viewer

# 3) §8 视差不能被改弱
for a in portrait portrait2 portrait3; do python3 tools/acceptance.py parallax --asset $a; done

# 4) 界面三条路径（删素材 / 采集页 / 观看页）各看一次
cp -r assets/portrait2 assets/_tmp && ./ui/build/memorial_studio --autodelete _tmp
./ui/build/memorial_studio --capture
./ui/build/memorial_studio --asset portrait3
```

判据：中部平均差 < 1 灰阶、p99 ≤ 24（同一版本连跑两次本来就有 p99=1 的动画相位抖动）。
**2026-11-24 这一轮的结果**：

| 检查 | 结果 |
|---|---|
| portrait / portrait2 / portrait3 × 8 种参数 | 24/24 通过，中部 mean 0.35–0.57、p99 ≤ 5，差异只落在轮廓边缘那 1 个像素上 |
| §8-1 视差（1.0 m 处） | portrait 22.3 / portrait2 44.7 / portrait3 16.6 px，全部 ≥ 8 |
| 删除键（`--autodelete`） | OK，目录真的删掉、列表刷新 |
| 采集页 | 相机预览正常，无"相机不可用" |
| 首页 / 观看页（最大化） | 正常，无黑边、无接缝 |

### 已知短板 / 下一步

1. **延迟 138 ms**：模型推理 54 ms 是硬瓶颈，相机帧龄 64 ms 已接近 Python 端
   解码 2560×720 MJPEG 的极限。要进 100 ms 得换更小的姿态模型或降输入尺寸
   （需 x86 侧重新导出 .rknn）。理论下限按 33+54+13+8 ≈ 108 ms。
2. **真正的转头**：单张照片 + 2.5D 网格做不到，需要 LivePortrait 那类神经重演
   （隐式关键点 + 生成式解码器），属于换技术路线，以周计。
3. **观感验收没做**：需要在板子上让人连续看 5 分钟，判断晕不晕、像不像。
4. **动物模式**：抠图已换 U²-Netp（通用显著目标），眼睛用"多阈值暗斑 + 对称配对"
   自动定位，头部遮罩用双眼中点/瞳距的几何公式生成 → 动物也有点头与回望。
   已用一张 180×198 的小狗照片端到端跑通（视差折算 ≈11 px @ ±15 cm），
   细节与限制见 **[docs/animal-mode.md](docs/animal-mode.md)** §7。
   **仍缺**：一张 ≥800 px 的真实猫狗照片做观感验证；动物眨眼故意不做。

### 最近的关键修正（2026-11-22 ~ 23）

| 提交 | 内容 |
|---|---|
| `fc9ebb7` | 追踪抓帧线程，11.3 → 16.8 fps，延迟 146 → 138 ms |
| `e5f3e9d` | 主体压成刚性平面 → 视差四条距离全达标，轮廓不再拉丝 |
| `0efd9ab` | sens 0.3 + gain 70 + 深度轮廓扩边（第一版视差达标） |
| `3ac7c8c` | 追踪选错人（选中画面边缘的夹克）+ 缓冲太深 |
| `a44d397` | `tools/acceptance.py`：§8 可复跑的验收测量 |
| `d2b2e80` | 采集守卫贯通到界面：站太远不再产出废资产 |
| `26a89ec` | 采集页预览一直黑屏（`cv2.imwrite(".tmp")` 不支持） |
| `c54e572` | 背景板 inpaint 放射状白扇 → 金字塔填充 |
| `17a1e9f` | 眨眼从"黑矩形"改成"填皮肤 + 睫毛线" |
| `f295c92` | 首页删除键 + `tools/import_photo.py` 照片导入 + 视差按分辨率归一 |
| `3a5a616` | 去掉两侧黑边；studio 默认最大化 + `--fullscreen` |
| `25fa045` | 补边改成"背景镜像延伸"，且背景/主体同比例（修"素材没匹配好"）|
| `next` | 抠图加固：U²-Netp 给动物身体的 0.5 alpha 补实，身体不再半透明 |
| `8fe12eb` | 进采集页前停掉眼动追踪（相机占用导致"相机无法加载"） |

## 当前进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 0 | 相机 I/O（by-id 路径、MJPG 2560x720、拆半） | ✅ 31.0 fps |
| Step 1 | 极线校正 + SGBM 深度真值 | ✅ 中位\|Δy\| 0.389 px，有效视差 65.2% |
| Step 3a | 深度尺度标定（两点法） | ✅ 误差 +0.8%，1.2m 处分辨率 1.44 mm |
| Step 2 | 单目深度模型上 NPU | ✅ DA-V2-Small fp16，763 ms/帧（离线用，精度合格） |
| Step 3b | 单目相对深度 → 米制（用双目做标尺） | ✅ 肖像 1–2m 中位误差 108 mm / 9.9%，R²=0.791 |
| Step 4 | DIBR 网格位移渲染 + Qt 界面 | ✅ 板端 57.9 fps |
| Step 5 | 观众眼位追踪 | ✅ YOLOv8-pose 16.8 fps + UDP 驱动视点 |
| Step 6 | 人像 matte（MODNet）| ✅ 286ms/帧，离线一次性 |
| Step 7 | 美术风格化（清晰/回忆/魂影）| ✅ `S` 键切换 |
| Step 7b | 程序化眨眼 | ✅ 随机间隔 2.5–6s，单次 160ms |
| Step 7c | 眼神回望 | ✅ 头部反向局部位移，追踪到人时生效 |
| Step 7d | 采集质量守卫（瞳距 / 标定 / 资产） | ✅ 站太远直接拒绝，不产出废资产 |
| Step 8 | 魂盒（45° 半透镜）| ⬜ 需硬件 |
| Step 9 | 照片导入（不经相机，一张图出资产）| ✅ `tools/import_photo.py` |
| Step 10 | 素材管理（首页删除）| ✅ ✕ + 二次确认，`--autodelete` 可自测 |

## 目录结构

```
capture/     采集期：相机 I/O、极线校正、SGBM 深度真值
tools/       自检与评测脚本
config/      rectify.json（校正变换）、depth_calib.json（深度尺度），均入库
docs/        方案文档
assets/      采集产物（用户数据，不入库）
```

## 完整系统界面

三页面 Qt 应用（首页 / 采集 / 观看）：

```bash
# 构建（产出 memorial_studio + memorial_viewer 两个可执行文件）
cd ui && mkdir -p build && cd build && cmake .. && make -j4

# 运行完整系统（首页）
./ui/build/memorial_studio

# 直接进入观看页 / 采集页
./ui/build/memorial_studio --asset portrait2
./ui/build/memorial_studio --capture

# 全屏（kiosk / 展台上用；默认是最大化）
./ui/build/memorial_studio --fullscreen
```

> **画面比例**：背景层和主体层用同一个装配方式（contain，资产完整可见、不裁脸），
> 窗口比资产"更宽/更高"时多出来的部分由 pass 3 补。
> 关键点：两层必须同比例，否则背景被放大、主体没放大，边缘会露出"贴上去"的台阶
> （踩过一次：背景 cover + 主体 contain，狗像贴在巨大灰墙上）。
>
> 补边方式写在 `meta.json` 的 `"frame"` 里：
> `"mirror"` 把背景沿边缘镜像铺满（16:9 素材默认，两侧 9% 基本看不出接缝）、
> `"black"` 直接留黑框（竖版素材用，比如狗那张 180×198，镜像面积比原图还大，
> 看着像"复制了一份背景"）、`"auto"`/不写 = 单侧补边 > 18% 就自动留黑。
> 照片导入时用 `--frame black|mirror|auto` 指定。

观看页自动运行眨眼 + 眼神回望，可一键开关眼动追踪、切换三档风格。
采集页一键完成：标定 → 深度 → 抠图 → 背景板 → 眨眼 → 头部遮罩 → 资产。

**首页素材卡片右上角的 ✕ 用来删除**：二次确认（默认按钮是"取消"），
删的是整个 `assets/<名字>/` 目录（图像/深度/标定一起走），不可恢复。

### 一张照片直接出资产（不经相机）

```bash
# 动物：U²-Netp 抠图 + 自动找眼睛 + 生成头部遮罩（有点头/回望，不眨眼）
python3 tools/import_photo.py --photo dog.jpg --name dog --species animal

# 人像：MODNet 抠图，其余同上；也可以手动指定眼睛
python3 tools/import_photo.py --photo mom.jpg --name mom --species human
python3 tools/import_photo.py --photo dog.jpg --name dog --eyes 58,62,115,52

# 想先确认眼睛找得准不准：把检测结果画到一张图上
python3 tools/import_photo.py --photo dog.jpg --name dog --preview /tmp/eyes.png
```

产出的资产与现场采集的完全同构（rgb/depth.f32/matte.png/bg.png/head.png/meta.json），
渲染端不需要改动。主体距离默认归一到 1.2 m（`--subject-dist`），
所以视差强度与真实采集的资产可比。

> 分辨率：源照片长边建议 ≥ 800 px。低于 300 px 会提示"放大到全屏会发虚"——
> 链路能跑通，但别拿去做正式展示。

## 界面（Step 4）

板端 Qt5 + OpenGL 查看器：网格位移 DIBR + 双层合成 + 呼吸微动 + HUD。

```bash
# 【推荐】采集向导：实时质检 + 引导站位 + 一键出资产
python3 tools/capture_wizard.py --asset portrait

# 或分步做：单目尺度标定 + 出资产
python3 tools/mono_calib.py --frames 4 --write --save-asset portrait

# 生成渲染资产（截一帧 + 单目稠密深度 + matte + 背景板）
python3 tools/make_asset.py --name demo

# 给已有资产单独补 matte / 背景板
python3 tools/make_matte.py assets/demo
python3 tools/make_bg.py assets/demo

# 构建
cd ui && mkdir -p build && cd build && cmake .. && make -j4

# 运行（自动处理 DISPLAY / XAUTHORITY）
./ui/run.sh assets/demo

# 离屏截图（回归测试用，不需要人看屏）
./ui/build/memorial_viewer --asset assets/demo --snapshot out.png --eye 0.6,0 --focus 1.8 --gain 70
```

交互：鼠标=视点、空格=自动摆动、**`S`=风格（清晰/回忆/魂影）**、`H`=HUD、
`F`=全屏、`+/-`=视差、`[/]`=焦点、`Q`=退出。

风格也可用命令行指定：`--look 0|1|2`（旧名 `--style` 会和 Qt 内置参数冲突，已改名）。

### 眼动追踪（Step 5）

两个进程，UDP 通信（`127.0.0.1:9999`，一行 `ex,ey`）：

```bash
# 终端 1：追踪器（占相机）
python3 -u track/eye_tracker.py --print

# 终端 2：渲染器（不占相机）
./ui/build/memorial_viewer --asset assets/portrait
```

收不到追踪数据时渲染器自动退回鼠标/自动摆动模式，可单独演示。
注意用 **fp16** 的 `yolov8n-pose.rknn`，int8 版置信度全 0 检不出人。

追踪器要点（都是踩过的坑）：
- 只在**画面中央 62%** 内找目标（`capture/landmarks.py`）。整帧 detect 会被画面
  边缘的类人干扰抢走——实测选中过右上角挂着的夹克，鼻尖定到 x=1252。
- `--sens 0.3`：满偏对应头横移约 19 cm（1.2 m 处）。sens=1 要移 64 cm 才满偏，
  视差根本出不来。
- 有专职抓帧线程，只保留最新一帧；串行 `cap.read()` 会读到 2~3 帧前的画面。
- `--stamp` 让 UDP 包带上抓帧时刻，渲染端据此打印端到端延迟。

## 常用命令

```bash
python3 -m capture.camera_io --probe        # Step 0 自检：格式/帧率/拆分
python3 tools/rectify_selfcal.py            # 估计并写入校正变换 config/rectify.json
python3 tools/rectify_check.py              # Step 1 验收：极线对齐
python3 tools/depth_eval.py --frames 6      # 深度质量评测（含米制深度）
python3 tools/depth_eval.py --visual /tmp/x # 另存可视化
python3 tools/acceptance.py parallax        # §8-1 视差验收（离屏渲染，不占相机）
python3 tools/acceptance.py latency         # §8-2 延迟验收（起 tracker + viewer）
```

## 关键结论（踩过的坑）

| 坑 | 结论 |
|---|---|
| `BUFFERSIZE=1` | 采集路径帧率从 31 掉到 **14.8 fps**，标定要多等一倍；追踪路径相反，缓冲越浅延迟越小 |
| 复用 rehab-coach 的标定 | **不可用**：40° 基线角 → fx 被逼到 5579、cx 为负 |
| `stereoRectifyUncalibrated` | **单应退化**：局部缩放 0.76→13.6 倍，视差被非均匀缩放 |
| 相似变换拟合平移 | 会吸收场景中位视差，破坏绝对尺度；必须**不平移** |
| 深度公式 | 不能用 `Z=fB/d`，必须 `Z = K/(d - d_inf)`，本机 d_inf ≈ **-156 px** |
| 相机必须用 by-id 路径 | `/dev/video21` 的编号会随插拔漂移 |
| 板端只有 rknn-toolkit-lite2 | 仅推理；ONNX→RKNN 转换需 x86 Linux（Win 上走 WSL2） |
| 单目深度很慢 | DA-V2-Small fp16 实测 **763 ms/帧**，int8 更慢且更差；只在采集期离线跑 |

## 硬件

- 板卡：ELF 2（RK3588，8GB，Ubuntu 22.04，kernel 5.10.209）
- 相机：UVC `CCB Camera`，AR0144 全局快门，双目并排 2560x720@30（各 1280x720），仅 MJPG
- 深度标定：fx 1198.7 px，B 52.0 mm，d_inf −156.22 px，K 62.3324 px·m

## 仓库约定

- 只提交源码与文档；`assets/`、模型权重、评测产物不入库
- 每次功能性改动单独提交，提交信息用 `type(scope): 描述` 格式
