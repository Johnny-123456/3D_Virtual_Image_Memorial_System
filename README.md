# 3D 虚拟影像怀念系统

在 ELF 2（RK3588）上，用双目 UVC 相机把 2D 人像变成**有真实纵深的 2.5D 立体影像**：
观众移动头部时，画面中人物与背景产生反向视差，并带呼吸/眨眼等微动。

完整方案见 **[docs/route-D-implementation.md](docs/route-D-implementation.md)**
（实施过程中的实测修正见该文档 **§12**）。

---

## 当前进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 0 | 相机 I/O（by-id 路径、MJPG 2560x720、拆半） | ✅ 31.0 fps |
| Step 1 | 极线校正 + SGBM 深度真值 | ✅ 中位\|Δy\| 0.389 px，有效视差 65.2% |
| Step 3a | 深度尺度标定（两点法） | ✅ 误差 +0.8%，1.2m 处分辨率 1.44 mm |
| Step 2 | 单目深度模型上 NPU | ✅ DA-V2-Small fp16，763 ms/帧（离线用，精度合格） |
| Step 3b | 单目相对深度 → 米制（用双目做标尺） | ✅ 肖像 1–2m 中位误差 143 mm / 11.9% |
| Step 4 | DIBR 网格位移渲染 + Qt 界面 | ✅ 板端 57.9 fps |
| Step 5 | 观众眼位追踪 | ⬜ |
| Step 6–8 | 微动 / 美术 / 魂盒 | ⬜ |

## 目录结构

```
capture/     采集期：相机 I/O、极线校正、SGBM 深度真值
tools/       自检与评测脚本
config/      rectify.json（校正变换）、depth_calib.json（深度尺度），均入库
docs/        方案文档
assets/      采集产物（用户数据，不入库）
```

## 界面（Step 4）

板端 Qt5 + OpenGL 查看器：网格位移 DIBR + 双层合成 + 呼吸微动 + HUD。

```bash
# 生成渲染资产（截一帧 + 单目稠密深度）
python3 tools/make_asset.py --name demo

# 构建
cd ui && mkdir -p build && cd build && cmake .. && make -j4

# 运行（自动处理 DISPLAY / XAUTHORITY）
./ui/run.sh assets/demo

# 离屏截图（回归测试用，不需要人看屏）
./ui/build/memorial_viewer --asset assets/demo --snapshot out.png --eye 0.6,0 --focus 1.8 --gain 70
```

交互：鼠标=视点、空格=自动摆动、`H`=HUD、`F`=全屏、`+/-`=视差、`[/]`=焦点、`Q`=退出。

## 常用命令

```bash
python3 -m capture.camera_io --probe        # Step 0 自检：格式/帧率/拆分
python3 tools/rectify_selfcal.py            # 估计并写入校正变换 config/rectify.json
python3 tools/rectify_check.py              # Step 1 验收：极线对齐
python3 tools/depth_eval.py --frames 6      # 深度质量评测（含米制深度）
python3 tools/depth_eval.py --visual /tmp/x # 另存可视化
```

## 关键结论（踩过的坑）

| 坑 | 结论 |
|---|---|
| `BUFFERSIZE=1` | 帧率从 31 掉到 **14.8 fps**；必须 ≥2 |
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
