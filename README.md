# 3D 虚拟影像怀念系统

在 ELF 2（RK3588）上，用双目 UVC 相机把 2D 人像变成**有真实纵深的 2.5D 立体影像**：
观众移动头部时，画面中人物与背景产生反向视差，并带呼吸/眨眼等微动。

完整方案见 **[docs/route-D-implementation.md](docs/route-D-implementation.md)**。

---

## 当前进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 0 | 相机 I/O（by-id 路径、MJPG 2560x720、拆半） | ✅ 完成，实测 31.0 fps |
| Step 1 | 极线校正 + SGBM 深度真值 + 深度评测 | 🚧 进行中 |
| Step 2 | 单目深度模型上 NPU | ⬜ 待开始 |
| Step 3 | 相对深度 → 米制尺度标定 | ⬜ |
| Step 4 | DIBR 网格位移渲染 | ⬜ |
| Step 5 | 观众眼位追踪 | ⬜ |
| Step 6–8 | 微动 / 美术 / 魂盒 | ⬜ |

## 目录结构

```
capture/     采集期：相机 I/O、极线校正、深度真值
tools/       自检与评测脚本
config/      运行时配置
docs/        方案文档
assets/      采集产物（用户数据，不入库）
```

## 运行

```bash
# Step 0 自检（协商格式 / 帧率 / 左右拆分）
python3 -m capture.camera_io --probe
```

## 环境约束（重要）

| 约束 | 说明 |
|---|---|
| 相机必须用 by-id 路径 | `/dev/video21` 的编号会随插拔漂移 |
| `BUFFERSIZE` 不能设 1 | 实测会把帧率砍半（15fps）；取 2 才满 30fps |
| 板端只有 `rknn-toolkit-lite2` | 仅推理。ONNX→RKNN 转换需在 x86 PC（Linux / WSL2）上做 |
| 相机走 USB 2.0 | 2560x720 MJPG@30 已接近总线上限，避免再插高带宽 USB 设备 |
| 无 ROS2 | 本方案不依赖，纯 Python + Qt |

## 硬件

- 板卡：ELF 2（RK3588，8GB，Ubuntu 22.04，kernel 5.10.209）
- 相机：UVC `CCB Camera`，AR0144 全局快门，双目并排 2560x720@30（各 1280x720），仅 MJPG
- 标定：`rehab-coach/src/backend/calibration/stereo_calib.json`（重投影误差 0.708 px）

## 仓库约定

- 只提交源码与文档；`assets/`、模型权重、评测产物不入库
- 每次功能性改动单独提交，提交信息用 `type(scope): 描述` 格式
