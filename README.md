# SlimeMold Artist Explorer —— 基于交互式进化的 AI 绘画风格探索系统

> **黏菌风格探索器**：用 AI 帮你找到属于自己的画师风格，而不是别人告诉你的风格。

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
| **Anima (DiT)** | Qwen-3 + CLIP | 1024 | `(@artist:0.90)` |

> Anima **turbo** 模式需要额外安装加速 LoRA：[anima-turbo-lora-v0.1.safetensors](https://civitai.com/models/2560840/anima-turbo-lora)  
> 放入 ComfyUI `models/loras/` 目录。缺少 LoRA 时系统会自动切换为 base 模式（速度约慢 4 倍）。

---

## 项目结构

```
engines/          # 核心算法引擎
  interactive_style_explorer8.py     # SDXL (Illustrious) 版
  interactive_style_explorer_anima.py # Anima (DiT) 版

webui/            # Web 界面入口
  webui.py              # SDXL 风格探索 (端口 17324)
  anima_webui.py        # Anima 风格探索 (端口 17326)
  build_webui.py        # SDXL 向量库构建 (端口 17325)
  build_webui_anima.py  # Anima 向量库构建 (端口 17327)

scripts/          # 独立工具（命令行）
  tag_to_tensor.py / *_anima.py      # 批量编码画师标签
  build_index_2048.py / build_artist_index_anima.py  # 构建 FAISS 索引
  visualize_artists_2048.py / visualize_anima.py     # PCA 可视化

workflows/        # ComfyUI 工作流模板
artists/          # 画师标签数据
data/             # 运行时数据 (.gitignore)
outputs_2048/     # SDXL 向量库 (.gitignore)
outputs_anima/    # Anima 向量库 (.gitignore)
```

---

## 快速开始

### 环境要求

- ComfyUI（需安装并启动，API 模式）
- Python 3.10+
- 预构建的向量库（见下方"构建向量库"）

### 一键安装 + 启动

**Windows**：双击 `setup_cn.bat`（中国大陆）或 `setup.bat`（其他地区）

**Linux / Mac**：`bash setup_cn.sh`（中国大陆）或 `bash setup.sh`

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

1. 准备好画师标签数据（`artists/` 下的 Markdown 文件）
2. 确保 ComfyUI 安装了 **SaveCondition** 节点（ComfyUI Manager 搜索 `Comfyui-Condition-Utils`）
3. 运行构建助手并依次执行三个步骤

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

AGPL-3.0

---

## 贡献

欢迎提交 Issue 和 PR。

---

## 免责声明

这是一个个人兴趣探索项目，**并不是一个成熟的产品**。作者并非专业程序员，代码质量和项目结构可能存在不足。如果在使用中遇到问题，欢迎提 Issue，但请友善交流。
