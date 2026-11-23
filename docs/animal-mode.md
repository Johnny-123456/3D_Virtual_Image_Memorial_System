# 动物模式：可行性、卡点与落地顺序

> 目标：把现在的人像 2.5D 影像管线，扩到猫 / 狗等宠物。
> 结论先说：**视差渲染部分几乎不用改，卡在两处"只认人"的模型**——抠图和关键点。

---

## 1. 现在哪些环节是"只认人"的

| 环节 | 用的东西 | 对动物是否可用 |
|---|---|---|
| 深度（相对 + 米制） | DepthAnything-V2-Small | ✅ 与主体无关，直接可用 |
| 人像抠图 matte | MODNet（人像专用） | ❌ 换成动物会抠成一团 |
| 头部遮罩 head.png | YOLOv8n-pose 的双眼 | ❌ 人体关键点网 |
| 眨眼 blink.png | 双眼位置合成 | ❌ 同上 |
| 眼神微动 | meta.eyes | ❌ 同上 |
| 视差渲染 / 背景板 / 风格 | 只依赖 depth + matte | ✅ 直接可用 |
| 采集守卫（瞳距 ≥45px） | 双眼 | ❌ 动物没有"瞳距"概念 |

也就是说：**渲染管线本身是物种无关的**，问题全在"产资产"这一步。

## 2. 需要替换的两个件

### 2.1 抠图（必须换）

MODNet 在动物上会漏毛、把耳朵吃掉。可选：

| 方案 | 体积 | RK3588 可行性 | 备注 |
|---|---|---|---|
| U²-Net / U²-Netp（通用显著目标） | 4.7 MB / 1.1 MB | ✅ NPU 可跑，512² 约 100–300 ms | 通用，人和动物都行，发丝一般 |
| ISNet（DIS） | 176 MB | ◐ 太大，需要裁剪 | 发丝最好，但板端吃力 |
| SAM / MobileSAM | 40 MB+ | ❌ 算力不够 | 不适合端侧 |
| 自训轻量分割（用我们自己的采集数据） | ~10 MB | ✅ | 最贴合场景，但要标注 |

**建议先用 U²-Netp**：体积小、通用、pth→onnx→rknn 转换链成熟，和人像 MODNet 并存（按物种选模型）。

### 2.2 关键点（可以先不做）

眼神 / 眨眼 / 头部遮罩都依赖双眼坐标。动物关键点可选 X-Pose（LivePortrait 的动物模式就用它），但它的 `MultiScaleDeformableAttention` 自定义算子**在 RKNPU 上没有现成实现**，移植成本高。

**务实做法：先做"无关键点降级模式"。**
渲染端本来就已经支持 `hasHead` / `hasBlink` 缺失时静默降级（不眨眼、不回望，但视差和风格照常），所以动物资产可以先只有 `rgb / depth / matte / bg / meta` 五个文件。

## 3. 落地顺序（建议）

1. **matte 模型配置化**（0.5 天）
   `capture/matte.py` 现在把模型路径写死在调用方；改成从 `config/` 或命令行读，
   允许 `--matte-model modnet.rknn | u2netp.rknn`。
2. **采集守卫按物种分支**（0.5 天）
   `mono_calib` / `finalize_asset` 加 `--species animal`：跳过瞳距守卫，
   不生成 head/blink，资产少三个文件但不影响渲染。
3. **换 U²-Netp 并转换**（1 天，需要 x86 侧 onnx→rknn）
   量化校准集可以直接用 `rehab-coach/models/quant_images/`。
4. **验证**（0.5 天）
   用 `tools/acceptance.py parallax` 跑一遍，确认视差和边缘指标不退化；
   再拍两只猫狗实拍资产。
5. （可选，1–2 周）**动物关键点**：找现成的轻量动物面部 landmark 模型，
   替代 YOLO-pose 只为拿"眼睛位置"。拿到眼睛位置后，head/blink/眼神三样可原样复用。

## 4. 不做的事

- 不做"动物表情重演"（那同样要 LivePortrait 级别的生成模型）。
- 不做多只动物同框的实例分割——一张资产只处理一个主体（现有管线就是单主体假设）。

## 5. 现在就能做的验证（不需要新模型）

```bash
# 手工准备一张动物照片的 rgb.png，跑现有管线看视差是否正常
python3 tools/make_asset.py --name cat_demo
./ui/build/memorial_viewer --asset assets/cat_demo --snapshot /tmp/cat.png --eye 0.5 0
```

（`make_asset.py` 会走 MODNet，抠图质量会差；这一步只看"视差渲染是否成立"。
真正的动物资产需要先把 2.1 的抠图模型换掉。）

---

## 6. 已落地（2026-11-23）

| 项 | 状态 |
|---|---|
| U²-Netp 转 rknn | ✅ `models/u2netp.rknn`，7.2 MB fp16（输入 1×3×320×320，7 个输出） |
| 板端推理 | ✅ 190–212 ms/帧；与人像模型同图的掩码覆盖率一致（28.0% vs 28.0%） |
| `capture/matte.py` | ✅ 新增 `SalientMatte` + `load_matte()`：文件名含 `u2net` 走通用抠图，否则 MODNet |
| `tools/mono_calib.py` | ✅ `--matte-model`、`--species human\|animal`（animal 跳过瞳距守卫） |
| `tools/finalize_asset.py` | ✅ `--species animal` 跳过眨眼/头部遮罩 |
| 五文件资产渲染 | ✅ 无 `blink.png`/`head.png` 时正常出画（已实测） |

**用法**（采集一只猫/狗）：

```bash
python3 tools/mono_calib.py --frames 3 --write --save-asset mycat \
        --matte-model models/u2netp.rknn --species animal
python3 tools/finalize_asset.py assets/mycat --species animal   # 只打印"跳过"
```

**还差一步**：还没有一张真实的猫/狗照片做观感验证（板子上没有素材）。
拿到一张（正脸、0.8–1.5 m、背景别太乱）就能跑完整链路。

**已知限制**：U²-Netp 的边缘比 MODNet 硬（发丝/胡须会略"切"），
且没有眼睛坐标 → 不眨眼、不回望、没有眼神微动。

