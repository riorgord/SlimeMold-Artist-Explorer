# SlimeMold Artist Explorer —— 基于交互式进化的 AI 绘画风格探索系统

> **黏菌风格探索器**：用 AI 帮你找到属于自己的画师风格，而不是别人告诉你的风格。

> ⚠️ **免责声明**：这是一个个人兴趣探索项目，**并不是一个成熟的产品**。作者并非专业程序员，代码质量和项目结构可能存在不足。如果在使用中遇到问题，欢迎提 Issue，但请友善交流。

---

## 项目定位

这是一个**交互式风格搜索引擎**，通过进化算法和用户反馈，从数万个画师标签中自动发现、筛选和培育符合个人审美的画风组合。

---

## 核心问题

二次元 AI 绘画社区面临一个普遍困境：**风格同质化**。默认输出的"美少女"风格高度雷同，而现有的风格探索工具都是静态浏览模式，用户只能被动翻看，无法主动定义自己的审美方向。

---

## 核心机制

### 黏菌模型（Slime Mold Model）

灵感来自黏菌的觅食行为。系统维护一群"触角"（Tentacles），每个触角代表一个画师组合。用户通过打分（-10 ~ +10）引导触角在风格空间中移动、生长或萎缩。

- **正反馈**：高分的触角获得更多"营养"，其代表的方向被强化
- **负反馈**：低分的触角被淘汰，新触角从高分区域附近萌发
- **信息交换**：相似触角之间共享画师基因
- **长跳机制**：当种群多样性过低时，强制部分触角跳跃到随机位置

### 辅助机制

| 机制 | 说明 |
|------|------|
| **Ban 区** | 封禁不喜欢的风格区域，触角自动避开，支持边缘衰减 |
| **保护区** | 圈定有潜力的风格区域，用进化算法精细培育 |
| **探测点** | 在 Ban 区边缘放置探索点，寻找"伴生甜点" |

---

## 技术架构

| 架构 | 编码器 | 向量维度 | 画师串格式 |
|------|--------|---------|-----------|
| **SDXL (Illustrious)** | CLIP Text Encode | 2048 | `(by artist:0.90)` |
| **Anima (DiT)** | Qwen-3 | 1024 | `(@artist:0.90)` |

> Anima **turbo** 模式需要额外安装加速 LoRA：[anima-turbo-lora-v0.1.safetensors](https://civitai.com/models/2560840/anima-turbo-lora)  
> 放入 ComfyUI `models/loras/` 目录。缺少 LoRA 时系统会自动切换为 base 模式（速度约慢 4 倍）。

---

## 项目结构

```
engines/
  comfy_clip/                         # ComfyUI 原生 CLIP/Qwen3 模型（本地编码，无需 ComfyUI 运行时）
  interactive_style_explorer8.py     # SDXL (Illustrious) 版引擎
  interactive_style_explorer_anima.py # Anima (DiT) 版引擎
  clip_encoder_sdxl.py               # SDXL 本地 CLIP 编码器
  clip_encoder_anima.py              # Anima 本地 Qwen3 编码器

webui/            # Web 界面入口
  webui.py              # SDXL 风格探索 (端口 17324)
  anima_webui.py        # Anima 风格探索 (端口 17326)
  build_webui.py        # SDXL 向量库构建 (端口 7862)
  build_webui_anima.py  # Anima 向量库构建 (端口 17327)

scripts/          # 命令行工具
  build_sdxl_local.py      # SDXL 一步本地建库（读 MD → 编码 → FAISS）
  build_anima_local.py     # Anima 一步本地建库
  visualize_artists_2048.py / visualize_anima.py  # PCA 可视化

_debug/           # 调试/验证脚本 (.gitignore)

workflows/        # ComfyUI 工作流模板（仅用于生图，不用于编码）
artists/          # 画师标签数据
data/             # 运行时数据 (.gitignore)
outputs_2048/     # SDXL 向量库 (.gitignore)
outputs_anima/    # Anima 向量库 (.gitignore)
```

---

## 快速开始

### 环境要求

- Python 3.10+
- ComfyUI（仅用于生图，需安装并启动 API 模式。）
- PyTorch（`pip install torch`）
- 预构建的向量库（见下方"构建向量库"）

### 一键安装 + 启动

**Windows**：双击 `setup_cn.bat`（中国大陆）或 `setup.bat`（其他地区）

**Linux / Mac**：`bash setup_cn.sh`（中国大陆）或 `bash setup.sh`（其他地区）

脚本会自动创建虚拟环境、检测 GPU、安装依赖，安装完成后弹出启动菜单：

```
[1] SDXL 风格探索    [2] Anima 风格探索
[3] SDXL 向量库构建  [4] Anima 向量库构建
```

### 手动启动

```bash
# 激活虚拟环境后
python webui/webui.py           # SDXL 风格探索
python webui/anima_webui.py     # Anima 风格探索
```

### 构建向量库

如果 `outputs_2048/` 或 `outputs_anima/` 为空，启动探索器时会提示构建。

**WebUI 方式：** 启动构建助手 → 填好 Markdown 路径、模型路径和前缀 → 点"开始构建"→ 一步完成。

**命令行方式：**

```bash
# SDXL
python scripts/build_sdxl_local.py --md ./artists/artists40k.md --ckpt "path/to/checkpoint.safetensors"

# Anima
python scripts/build_anima_local.py --md ./artists/artists40k.md --encoder "path/to/qwen_3_06b_base.safetensors"
```

> 不依赖 ComfyUI。编码完全在本地完成，不需要 ComfyUI 安装任何额外节点。每个库背后都自动生成 `library.json` 声明文件，支持多库共存（如 `outputs_anima_1girl/`、`outputs_anima_1boy/`）。

### 使用流程

1. **初始化**：在 WebUI 中点击「初始化新会话」
2. **探索**：点击「探索 & 提交」，系统生成一批图片
3. **评分**：对每张图打分（-10 ~ +10）
4. **迭代**：再次点击「探索 & 提交」，系统根据评分进化触角
5. **辅助操作**：封禁不喜欢的风格、收藏喜欢的画师、创建保护区进行精细培育

---

## 已知问题

- 单轮探索评估耗时较长（取决于触角数量 × 单张生图时间），已在进度条显示预估剩余时间
- Anima 版本在画师画风混合上不如 SDXL（模型架构的天然差异）
- 保护区持久化在极端情况下可能损坏（已有自动备份机制）

---

## 数据来源

画师标签数据来源于公开网站，详见 [DATA_SOURCES.md](DATA_SOURCES.md)。数据以 CC BY 4.0 许可发布。

---

## 开源协议

本项目整体以 **AGPL-3.0** 发布。

`engines/comfy_clip/` 目录下的模型代码改编自 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（GPL-3.0），详见该目录下的 `NOTICE` 文件。分词器数据来自 OpenAI/HuggingFace（MIT）和 ComfyUI（Apache 2.0）。

---

## 贡献

欢迎提交 Issue 和 PR。

