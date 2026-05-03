#!/usr/bin/env bash
set -euo pipefail

# 中国大陆用户可取消下行注释，使用清华镜像加速
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "============================================"
echo "  画師串小助手 - 一键环境安装 (Linux/Mac)"
echo "============================================"
echo ""

# ── 找 Python ──
PYTHON=""
for p in python3 python; do
    if command -v "$p" &>/dev/null && "$p" --version &>/dev/null; then
        PYTHON="$p"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "[ERROR] 未找到 Python，请安装 Python 3.10+"
    exit 1
fi

echo "[检测] Python:"
$PYTHON --version

# 检查版本 >= 3.10
PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo "$PYVER" | cut -d. -f1)
MINOR=$(echo "$PYVER" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "[ERROR] Python 版本过低 (需 >= 3.10)"
    exit 1
fi

# ── 创建 venv（已存在则跳过安装）──
SKIP_INSTALL=0
if [ -f venv/bin/activate ] || [ -f venv/Scripts/activate ]; then
    echo "[1/4] 虚拟环境已存在，跳过安装"
    SKIP_INSTALL=1
else
    echo ""
    echo "[1/4] 创建虚拟环境 (venv/) ..."
    $PYTHON -m venv venv --clear
fi

# ── 激活 venv (兼容 bash/zsh) ──
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv/Scripts/activate ]; then
    # Git Bash on Windows
    source venv/Scripts/activate
else
    echo "[ERROR] 无法激活虚拟环境"
    exit 1
fi

# ── 升级 pip ──
echo ""
echo "[2/4] 升级 pip ..."
pip install --upgrade pip

# ── 检测 GPU 并安装 torch ──
echo ""
echo "[3/4] 检测 GPU ..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "  [OK] NVIDIA GPU 已检测到"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs echo "  GPU:" || true
    echo "  正在安装 CUDA 版 torch..."
    set +e
    pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cu130
    if [ $? -ne 0 ]; then
        echo "  [WARN] CUDA torch 安装失败，回退 CPU 版..."
        pip install torch
    fi
    set -e
else
    echo "  [INFO] 未检测到 NVIDIA GPU，安装 CPU 版 torch"
    set +e; pip install torch; set -e
fi

# ── 安装其余依赖 ──
echo ""
echo "[4/4] 安装依赖 ..."
pip install -r requirements.txt

# ── 给脚本加执行权限 ──
chmod +x setup.sh 2>/dev/null || true

if [ "$SKIP_INSTALL" = "1" ]; then
    echo "  跳过依赖安装"
else
# ── 升级 pip + 装依赖 ──
echo ""
echo "[2/4] 升级 pip ..."
pip install --upgrade pip

echo ""
echo "[3/4] 检测 GPU ..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "  [OK] NVIDIA GPU"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs echo "  GPU:"
    PIP_INDEX_URL="" pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cu130 || {
        echo "  [WARN] 回退 CPU torch"; PIP_INDEX_URL="" pip install torch; }
else
    echo "  [INFO] CPU torch"
    PIP_INDEX_URL="" pip install torch
fi

echo ""
echo "[4/4] 安装依赖 ..."
pip install -r requirements.txt
fi

# ── 菜单 ──
echo ""
echo "============================================"
echo "  选择启动模式："
echo "============================================"
echo ""
echo "  [1] SDXL 风格探索   (webui/webui.py)"
echo "  [2] Anima 风格探索  (webui/anima_webui.py)"
echo "  [3] SDXL 向量库构建 (webui/build_webui.py)"
echo "  [4] Anima 向量库构建 (webui/build_webui_anima.py)"
echo "  [0] 退出"
echo ""
read -p "请输入数字 (0-4): " CHOICE

case "$CHOICE" in
    1) python webui/webui.py ;;
    2) python webui/anima_webui.py ;;
    3) python webui/build_webui.py ;;
    4) python webui/build_webui_anima.py ;;
    0) exit 0 ;;
esac

echo ""
echo "  以后直接运行本脚本即可重新启动。"
echo "  或手动: source venv/bin/activate"
