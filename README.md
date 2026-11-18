# 3D 虚拟影像怀念系统

在 ELF 2（RK3588）上，用双目 UVC 相机把 2D 人像变成**有真实纵深的 2.5D 立体影像**：
观众移动头部时，画面中人物与背景产生反向视差，并带呼吸/眨眼等微动。

完整方案见 **[docs/route-D-implementation.md](docs/route-D-implementation.md)**
（实施过程中的实测修正见该文档 **§12**）。

---

## 当前进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| Step 0 | 相机 I/O（by-id 路径、MJPG 2560x720、拆半） | ✅ 实测 31.0 fps |
| Step 1 | 极线校正 + SGBM 深度真值 + 深度评测 | ✅ 中位\|Δy\| 0.23 px，有效视差 41.7% |
| Step 2 | 单目深度模型上 NPU | ⬜ 待开始 |
| Step 3 | 相对深度 → 米制尺度标定 | ⬜ |
| Step 4 | DIBR 网格位移渲染 | ⬜ |
| Step 5 | 观众眼位追踪 | ⬜ |
| Step 6–8 | 微动 / 美术 / 魂盒 | ⬜ |

## 目录结构

```
capture/     采集期：相机 I/O、极线校正、SGBM 深度真值
tools/       自检与评测脚本
config/      运行时配置（rectify.json = 相机校正变换，入库）
docs/        方案文档
assets/      采集产物（用户数据，不入库）
```

## 常用命令

```bash
python3 -m capture.camera_io --probe        # Step 0 自检：格式/帧率/拆分
python3 tools/rectify_selfcal.py            # 估计并写入校正变换 config/rectify.json
python3 tools/rectify_check.py              # Step 1 验收：极线对齐
python3 tools/depth_eval.py --frames 8      # SGBM 深度质量评测
python3 tools/depth_eval.py --visual /tmp/x # 另存可视化三联图
```

## 环境约束（重要）

| 约束 | 说明 |
|---|---|
| 相机必须用 by-id 路径 | `/dev/video21` 的编号会随插拔漂移 |
| `BUFFERSIZE` 不能设 1 | 实测帧率从 31 掉到 14.8 fps；取 2 才满帧 |
| **不要用 rehab-coach 的标定做 stereoRectify** | 该标定在本相机上不可用（40° 基线角 → fx 5579、cx 负值） |
| 板端只有 `rknn-toolkit-lite2` | 仅推理。ONNX→RKNN 转换需在 x86 PC（Linux / WSL2）上做 |
| 相机走 USB 2.0 | 2560x720 MJPG@30 已接近总线上限，避免再插高带宽 USB 设备 |
| 无 ROS2 | 本方案不依赖，纯 Python + Qt |

## 硬件

- 板卡：ELF 2（RK3588，8GB，Ubuntu 22.04，kernel 5.10.209）
- 相机：UVC `CCB Camera`，AR0144 全局快门，双目并排 2560x720@30（各 1280x720），仅 MJPG
- 校正：数据驱动自标定 → `config/rectify.json`（中位极线误差 0.23 px）

## 仓库约定

- 只提交源码与文档；`assets/`、模型权重、评测产物不入库
- 每次功能性改动单独提交，提交信息用 `type(scope): 描述` 格式
