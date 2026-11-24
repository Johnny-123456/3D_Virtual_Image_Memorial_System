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


---

## 7. 实测：一张狗的照片跑通了（2026-11-24）

素材：网上随手找的小狗正面照 `dog.webp`，180×198 px（**分辨率很低，只是功能验证**）。

```bash
python3 tools/import_photo.py --photo _verify/dog.webp --name dog --species animal
```

| 环节 | 结果 |
|---|---|
| 深度 | DA-V2-Small + `mono_depth_calib`，标定主体 0.86 m → 归一到 1.2 m；z_focus 2.04 m，深度范围 1.13–14.1 m |
| 抠图 | U²-Netp，覆盖率 69.9%（狗几乎占满画面，合理） |
| 眼睛 | 自动定位 L(58,62) R(115,52)，瞳距 56 px——多阈值扫暗斑 + 左右对称配对 |
| 头部遮罩 | head.png 覆盖 34%，**动物也有点头/回望**（几何公式只用到双眼中点与瞳距） |
| 眨眼 | 无（动物的睫毛线不能用肤色填补，硬做会糊成一团） |
| 视差 | ±1 视点：主体位移 5.44 px、背景 2.2 px（180 px 宽资产，相对位移 1.8% 画宽）；折合 1272 px 参照资产约 11 px @ ±15 cm/1.2 m，仍在 §8 的 8 px 之上 |

渲染端零改动：5+1 个文件（rgb/depth/matte/bg/head/meta）就能出画。

**这一轮顺带修掉的坑**

1. **视差按"资产像素"写死**：增益 75 是在 1272 px 宽的现场资产上调出来的，
   直接用绝对像素位移，换到 180 px 的照片时同样的位移占满画面、边缘露洞。
   现在 `uGain`/`uMaxShift`/点头/漂移/回望全部乘 `m_w/1272` 归一因子
   （1272 px 资产系数正好 1.0，已有素材观感不变）。
2. **眼睛配对**：单阈值要么漏掉阴影里的那只眼、要么把它和耳朵粘成一整块。
   改成 6/8/10/12/15/18/22 多档阈值各扫一遍再按位置去重。
3. **只看"上半脸"不够**：bbox 常含脖子和前胸（狗挂耳、人的肩），留到 55% 就会把
   鼻头当眼睛；收紧到 0.42 并对"合并成一大块"的候选做尺寸上限。

4. **身体是半透明的**：U²-Netp 对小狗胸口/身体只给 0.4–0.6 的 alpha（实测内部
   半透明像素占 19%），渲染出来背景从肚子里透过去，像"贴上去没对好"。
   `import_photo.py` 加了 `--matte-harden auto`：低阈值取主体 → 面积相当的连通域
   → 填洞 → 用它的模糊版当实心区、与原 alpha 取 max。加固后内部半透明降到 0%，
   边缘仍是 2–3 px 软过渡。

**还差的**：一张够大的真实照片（≥ 800 px 长边，正脸、0.8–1.5 m、背景干净）。
180 px 放大到全屏必然发虚，只能证明链路成立。
