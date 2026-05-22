import numpy as np
import pandas as pd
import json
import random
import time
import requests
import faiss
import os
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import deque
import re
from skopt import gp_minimize
from skopt.space import Real

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===============================================
# 用户可配置参数（带中文注释）
# ===============================================

# ---- 项目根目录 ----
# ---- 向量库路径 ----
VECTOR_DIR = Path("./outputs_2048")              # 向量库文件夹
VECTORS_NPY = VECTOR_DIR / "artist_vectors_2048.npy"
IDS_JSON = VECTOR_DIR / "artist_ids_2048.json"
INDEX_PATH = VECTOR_DIR / "artist_index_2048.faiss"
METADATA_CSV = VECTOR_DIR / "artist_metadata_2048.csv"
CLUSTER_CSV = VECTOR_DIR / "artist_pca_clusters.csv"

# ---- ComfyUI 连接 ----
COMFYUI_SERVER = "127.0.0.1:8188"
WORKFLOW_TEMPLATE_PATH = Path("workflows/t2i.json")
COMFYUI_OUTPUT_DIR = Path(".")  # 终端版用来定位本地 ComfyUI 输出的基目录（WebUI 不依赖此项）

# ---- 生成参数 ----
BASE_POSITIVE_PROMPT = "1girl, masterpiece, best quality"  # 基础正面提示词
BASE_NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"  # 基础负面提示词
WIDTH = 1024                                            # 图像宽度
HEIGHT = 1024                                           # 图像高度
STEPS = 20                                              # 采样步数
CFG = 5                                                 # CFG引导强度
SAMPLER_NAME = "euler_ancestral"                        # 采样器名称
SCHEDULER = "simple"                                    # 调度器类型
CHECKPOINT_NAME = "waiIllustriousSDXL_v140.safetensors"   # 底模文件名

# ---- 黏菌探索器参数 ----
N_TENTACLES = 30                                        # 触角总数
EVAL_BUDGET = 30                                        # 每轮全局探索的生图数量（整数）
MAX_GENERATIONS = 50                                    # 最大迭代轮数（全局探索的总轮数上限）
SIMILARITY_THRESHOLD = 0.85                             # 触角相似度阈值（超过此值视为"拥挤"，触发排斥力）
MIXED_ARTISTS_COUNT = 4                                 # 生成的画师串默认包含的画师数量（可被保护区设置覆盖）
TEMPERATURE = 1.0                                       # 画师权重温度（<1 更均匀，>1 主画师更突出）
STEP_SIZE = 0.12                                        # 触角变异步长

# ---- Ban区阈值 ----
BAN_SELECT_THRESH = 0.8                                 # 选择惩罚阈值：相似度>此值时，触角被选中概率大幅降低
BAN_MUTATE_THRESH = 0.90                                # 变异回退阈值：相似度>此值时，触角变异会被强制回退
BAN_REBIRTH_THRESH = 0.80                               # 重生/长跳禁止阈值：新触角必须保证相似度<此值
BAN_PENALTY_COEFFICIENT = 2.0                           # Ban区在保护区内的软斥力系数（越高越排斥，建议1.0-3.0）
BAN_DECAY_RATE = 0.7                                    # Ban区衰减速率（边缘高分时，penalty乘以此值，越小衰减越快）

# ---- 保护区参数 ----
PROTECT_POP_SIZE = 4                                    # 保护区内菌丝数量（每轮生图数）
PROTECT_MAX_CANDIDATES = 15                             # 贝叶斯优化最大候选画师数（超过15会提示效率警告）
PROTECT_CONVERGE_ROUNDS = 3                             # 连续N轮评分无提升则判定为收敛

# ---- 弱触角与探测点参数 ----
WEAK_CONSECUTIVE_NEGATIVE = 1      # 连续负分多少轮触发弱触角
PROMOTE_SUGGEST_ROUNDS = 3         # 弱触角连续正分多少轮系统建议转正
PULL_STRENGTH = 0.2                # 绳子拉力系数（0~1，越大越拉向探测点）
MAX_WEAK_TENTACLES = 4             # 弱触角上限（探测点未满时可用）

# ---- Debug调试与可视化 ----
DEBUG_MODE = False                 # 调试模式（启动时 --debug 开启）
PLOT_STYLE = "lite"                # 绘图风格："full"=全库背景（较慢），"lite"=仅触角（快速）
TRACE_HISTORY_LEN = 5              # 触角路径连线显示最近几步

# ---- 文件存储 ----
FAVORITES_FILE = Path("./data/favorites.json")        # 收藏夹文件
BAN_LIST_FILE = Path("./data/ban_list.json")          # Ban区黑名单文件
PROTECT_ZONES_DIR = Path("./data/protect_zones")      # 保护区数据文件夹
SCOUT_POINTS_FILE = Path("./data/scout_points.json")  # 探测点数据文件

# ---- Tag辅助 (从外部通过 WebUI 设置) ----
ENCODE_CLIP_PATH = ""  # checkpoint 完整路径
TAG_POS = ""           # 正面 Tag 文本
TAG_NEG = ""           # 排除 Tag 文本
TAG_GUIDANCE = 0.6     # 引导强度 (0=无引导, 1=最强)
TAG_POOL_SPREAD = 0.5   # 扩散度 (0=集中, 1=扩散)

_CACHED_TAG_DIR = None
_LAST_TAG_POS = ""
_LAST_TAG_NEG = ""
_LAST_CLIP_PATH = ""
_LIB_PREFIX = None
FIXED_SEED = -1  # -1=随机(ComfyUI管理), 其他值=固定种子

def get_lib_prefix():
    """从库目录的 library.json 读取建库前缀。缓存，只读一次。"""
    global _LIB_PREFIX
    if _LIB_PREFIX is None:
        lib_json = VECTOR_DIR / "library.json"
        if lib_json.exists():
            _LIB_PREFIX = json.loads(lib_json.read_text(encoding='utf-8')).get("prefix", "")
        else:
            _LIB_PREFIX = ""
    return _LIB_PREFIX

def compute_tag_direction(force=False):
    """返回 Tag 方向向量 (2048-dim, L2归一化) 或 None。
    force=True: 强制重新编码。否则 Tag 没变时复用缓存。"""
    global _CACHED_TAG_DIR, _LAST_TAG_POS, _LAST_TAG_NEG, _LAST_CLIP_PATH
    if not ENCODE_CLIP_PATH:
        return None
    pos = TAG_POS.strip() if TAG_POS else ""
    neg = TAG_NEG.strip() if TAG_NEG else ""
    if not pos and not neg:
        return None
    if not force and _CACHED_TAG_DIR is not None:
        if (pos == _LAST_TAG_POS and neg == _LAST_TAG_NEG
                and ENCODE_CLIP_PATH == _LAST_CLIP_PATH):
            return _CACHED_TAG_DIR
    try:
        from engines.clip_encoder_sdxl import encode_text_local
        prefix = get_lib_prefix()
        if pos and neg:
            pv = encode_text_local(f"{prefix}{pos}", ENCODE_CLIP_PATH)
            nv = encode_text_local(f"{prefix}{neg}", ENCODE_CLIP_PATH)
            combined = pv - nv
        elif pos:
            combined = encode_text_local(f"{prefix}{pos}", ENCODE_CLIP_PATH)
        else:
            combined = -encode_text_local(f"{prefix}{neg}", ENCODE_CLIP_PATH)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        combined = combined.astype(np.float32)
        _CACHED_TAG_DIR = combined
        _LAST_TAG_POS = pos
        _LAST_TAG_NEG = neg
        _LAST_CLIP_PATH = ENCODE_CLIP_PATH
        return combined
    except Exception as e:
        print(f"[WARN] Tag方向计算失败: {e}")
        return None

# ---- 图片展示方式 ----
SHOW_IMAGES_IN_BROWSER = True
print("📂 加载画师向量库...")
vectors_all = np.load(VECTORS_NPY).astype(np.float32)
with open(IDS_JSON, 'r') as f:
    artist_ids_all = json.load(f)
index = faiss.read_index(str(INDEX_PATH))
metadata_df = pd.read_csv(METADATA_CSV)
id_to_name = dict(zip(metadata_df['artist_id'], metadata_df['artist_name']))
name_to_id = {v: k for k, v in id_to_name.items()}
N_TOTAL_ARTISTS = len(artist_ids_all)
print(f"✅ 全库共有 {N_TOTAL_ARTISTS} 位画师")

# 聚类信息
cluster_dict = {}
if CLUSTER_CSV.exists():
    df_cluster = pd.read_csv(CLUSTER_CSV)
    for _, row in df_cluster.iterrows():
        cluster_dict[row['artist_id']] = row['cluster']
    print("✅ 聚类信息已加载")
id_to_cluster = {aid: cluster_dict.get(aid, -1) for aid in artist_ids_all}
unique_clusters = list(set(id_to_cluster.values()))
if -1 in unique_clusters:
    unique_clusters.remove(-1)

# PCA模型
pca_model = None
pca_vectors_2d = None
if DEBUG_MODE:
    print("🧮 准备 PCA 可视化数据...")
    pca_model = PCA(n_components=2, random_state=42)
    pca_vectors_2d = pca_model.fit_transform(vectors_all)
    print("✅ PCA 模型已就绪")

# 输出文件夹
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = Path(f"slime_{RUN_TIMESTAMP}")
os.makedirs("./data", exist_ok=True)  # 保证运行时数据目录存在
PROTECT_ZONES_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================
# 工具函数
# ===============================================
def blend_from_indices(indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    selected = vectors_all[indices]
    blended = np.sum(weights[:, np.newaxis] * selected, axis=0)
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended.astype(np.float32)

def vector_to_artist_string(blended_vec: np.ndarray, top_k: int = None, skip_ids: set = None) -> str:
    if top_k is None:
        top_k = MIXED_ARTISTS_COUNT
    blended_vec = np.asarray(blended_vec, dtype=np.float32).flatten()
    search_k = max(top_k * 3, top_k + (len(skip_ids) if skip_ids else 0))
    distances, idxs = index.search(blended_vec.reshape(1, -1), search_k * 2)
    ds = distances[0]
    parts = []
    used_ids = set()
    if skip_ids:
        used_ids.update(skip_ids)
    local_rank = 0
    for idx, dist in zip(idxs[0], ds):
        artist_id = artist_ids_all[idx]
        if artist_id in used_ids:
            continue
        weight = max(0.01, 1.0 - (local_rank / top_k) * TEMPERATURE)
        name = id_to_name.get(artist_id, str(artist_id))
        parts.append(f"(by {name}:{weight:.2f})")
        local_rank += 1
        used_ids.add(artist_id)
        if len(parts) >= top_k:
            break
    return ", ".join(parts)

def encode_text_to_vector(text: str) -> np.ndarray:
    if DEBUG_MODE:
        print(f"[DEBUG] encode_text_to_vector: encoding '{text[:40]}'...")
    wf = {
        "4": {"inputs": {"ckpt_name": CHECKPOINT_NAME}, "class_type": "CheckpointLoaderSimple"},
        "6": {"inputs": {"text": text, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "14": {"inputs": {"previewMode": None, "source": ["6", 0]}, "class_type": "PreviewAny"}
    }
    r = requests.post(f"http://{COMFYUI_SERVER}/prompt", json={"prompt": wf})
    pid = r.json()['prompt_id']
    start = time.time()
    while time.time() - start < 60:
        hist = requests.get(f"http://{COMFYUI_SERVER}/history/{pid}").json()
        if pid in hist:
            out = hist[pid]['outputs'].get('14', {}).get('text', [''])
            m = re.search(r"'pooled_output':\s*tensor\(\[\[(.*?)\]\]\)", out[0], re.DOTALL)
            if m:
                nums = np.fromstring(m.group(1), sep=',')
                vec = nums.astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                if DEBUG_MODE:
                    print(f"[DEBUG] encode_text_to_vector: 完成 ({time.time()-start:.1f}s)")
                return vec
        time.sleep(0.5)
    if DEBUG_MODE:
        print(f"[DEBUG] encode_text_to_vector: 超时 (60s)")
    raise TimeoutError("文本编码超时")

# ===============================================
# ComfyUI API
# ===============================================
def load_workflow_template() -> Dict:
    with open(WORKFLOW_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_workflow(positive_prompt: str, negative_prompt: str, seed: int, filename_prefix: str) -> Dict:
    if DEBUG_MODE:
        print(f"[DEBUG] build_workflow: loading template from {WORKFLOW_TEMPLATE_PATH}")
    wf = load_workflow_template()
    for node_id, node in wf.items():
        class_type = node.get('class_type')
        if class_type == 'CLIPTextEncode':
            if node_id == '6':
                node['inputs']['text'] = positive_prompt
            elif node_id == '7':
                node['inputs']['text'] = negative_prompt
        elif class_type == 'KSampler':
            node['inputs']['seed'] = seed
            node['inputs']['steps'] = STEPS
            node['inputs']['cfg'] = CFG
            node['inputs']['sampler_name'] = SAMPLER_NAME
            node['inputs']['scheduler'] = SCHEDULER
        elif class_type == 'EmptyLatentImage':
            node['inputs']['width'] = WIDTH
            node['inputs']['height'] = HEIGHT
            node['inputs']['batch_size'] = 1
        elif class_type == 'CheckpointLoaderSimple':
            node['inputs']['ckpt_name'] = CHECKPOINT_NAME
        elif class_type == 'SaveImage':
            node['inputs']['filename_prefix'] = str(RUN_OUTPUT_DIR / filename_prefix)
    if DEBUG_MODE:
        print(f"[DEBUG] build_workflow: seed={seed}, steps={STEPS}, cfg={CFG}, prefix={filename_prefix}")
        print(f"[DEBUG] build_workflow: ckpt={CHECKPOINT_NAME}, pos_prompt={positive_prompt[:80]}...")
    return wf

def queue_prompt(workflow: Dict) -> str:
    r = requests.post(f"http://{COMFYUI_SERVER}/prompt", json={"prompt": workflow})
    r.raise_for_status()
    pid = r.json()['prompt_id']
    if DEBUG_MODE:
        print(f"[DEBUG] queue_prompt: prompt_id={pid}")
    return pid

def wait_for_prompt(prompt_id: str, timeout: int = 120,
                    progress_callback: Optional[Callable[[int], None]] = None) -> Optional[Dict]:
    if DEBUG_MODE:
        print(f"[DEBUG] wait_for_prompt: waiting for {prompt_id} (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"http://{COMFYUI_SERVER}/history/{prompt_id}")
        hist = resp.json()
        if prompt_id in hist:
            elapsed = time.time() - start
            if DEBUG_MODE:
                print(f"[DEBUG] wait_for_prompt: {prompt_id} completed in {elapsed:.1f}s")
            return hist[prompt_id]
        if progress_callback:
            progress_callback(int(time.time() - start))
        time.sleep(1)
    if DEBUG_MODE:
        print(f"[DEBUG] wait_for_prompt: {prompt_id} TIMEOUT after {timeout}s")
    return None

def generate_single_image(vector: np.ndarray, gen: int, tag: str) -> Optional[str]:
    prompt = f"{BASE_POSITIVE_PROMPT}, {vector_to_artist_string(vector)}"
    seed = FIXED_SEED if FIXED_SEED != -1 else (gen * 1000 + hash(tag) % 1000)
    fname = f"gen{gen}_{tag}"
    if DEBUG_MODE:
        print(f"[DEBUG] generate_single_image: gen={gen}, tag={tag}, seed={seed}, fname={fname}")
    wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
    pid = queue_prompt(wf)
    res = wait_for_prompt(pid)
    if res:
        for out in res.get('outputs', {}).values():
            if 'images' in out:
                path = str(RUN_OUTPUT_DIR / out['images'][0]['filename'])
                if DEBUG_MODE:
                    print(f"[DEBUG] generate_single_image: done -> {path}")
                return path
    if DEBUG_MODE:
        print(f"[DEBUG] generate_single_image: FAILED (no image output)")
    return None

# ===============================================
# HTML 画廊
# ===============================================
def display_images(image_data: List[Tuple[str, str]], gen: int, title: str = "黏菌探索器"):
    html_path = RUN_OUTPUT_DIR / f"gen_{gen:02d}_gallery.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title} - 第{gen}代</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#eee;margin:20px}}
h2{{color:#e0e0ff}}
.gallery{{display:flex;flex-wrap:wrap;gap:20px;justify-content:center}}
.card{{background:#16213e;border-radius:12px;padding:15px;width:340px}}
.card img{{width:100%;border-radius:8px;border:2px solid #0f3460}}
.card h3{{margin:10px 0 5px;color:#e94560}}
.card p{{margin:5px 0;font-size:13px;background:#0f3460;padding:8px;border-radius:6px;word-break:break-all}}
.filename{{color:#aaa;font-size:12px}}
</style></head><body>
<h2>🧫 {title} - 第{gen}代</h2>
<div class='gallery'>""")
        for i, (path, astr) in enumerate(image_data):
            f.write('<div class="card">')
            f.write(f'<h3>📌 #{i+1}</h3>')
            if path and os.path.exists(path):
                f.write(f'<a href="file://{path}" target="_blank"><img src="file://{path}"></a>')
                f.write(f'<p class="filename">{os.path.basename(path)}</p>')
            else:
                f.write('<div style="height:200px;background:#333;display:flex;align-items:center;justify-content:center;">❌ 缺失</div>')
            f.write(f'<p><b>🎭 画师串:</b><br><span style="color:#ffd700;">{astr}</span></p>')
            f.write('</div>')
        f.write("</div></body></html>")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    print(f"📁 画廊已生成: {html_path}")

# ===============================================
# Ban区管理
# ===============================================
_ban_list_cache = None
_ban_list_mtime = 0.0

def load_ban_list() -> List[Dict]:
    global _ban_list_cache, _ban_list_mtime
    if BAN_LIST_FILE.exists():
        mtime = BAN_LIST_FILE.stat().st_mtime
        if _ban_list_cache is not None and mtime == _ban_list_mtime:
            return _ban_list_cache  # 文件未变，走缓存
        with open(BAN_LIST_FILE, 'r', encoding='utf-8') as f:
            _ban_list_cache = json.load(f)
        _ban_list_mtime = mtime
        if DEBUG_MODE:
            active = sum(1 for b in _ban_list_cache if b.get('active', True))
            print(f"[DEBUG] load_ban_list: {len(_ban_list_cache)} total, {active} active")
        return _ban_list_cache
    _ban_list_cache = []
    if DEBUG_MODE:
        print("[DEBUG] load_ban_list: no file")
    return []

def save_ban_list(ban_list: List[Dict]):
    global _ban_list_cache, _ban_list_mtime
    with open(BAN_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(ban_list, f, indent=2, ensure_ascii=False)
    _ban_list_cache = ban_list
    _ban_list_mtime = BAN_LIST_FILE.stat().st_mtime
    if DEBUG_MODE:
        active_count = sum(1 for b in ban_list if b.get('active', True))
        print(f"[DEBUG] save_ban_list: {len(ban_list)} total, {active_count} active")

def add_ban(description: str, vector: np.ndarray, source: str, gen: int) -> int:
    ban_list = load_ban_list()
    ban_id = 1
    if ban_list:
        ban_id = max(b['id'] for b in ban_list) + 1
    ban_list.append({
        "id": ban_id,
        "description": description,
        "vector": vector.tolist(),
        "source": source,
        "created_gen": gen,
        "active": True,
        "penalty": 1.0
    })
    save_ban_list(ban_list)
    if DEBUG_MODE:
        print(f"[DEBUG] add_ban: id={ban_id}, desc={description[:40]}..., source={source}, gen={gen}")
    print(f"🚫 已封禁风格区域: {description[:60]}... (Ban ID: {ban_id}, 当前禁区数: {len(ban_list)})")
    return ban_id

def get_active_bans() -> Tuple[List[np.ndarray], List[Dict]]:
    ban_list = load_ban_list()
    vectors = []
    active_bans = []
    for b in ban_list:
        if b.get('active', True):
            vectors.append(np.array(b['vector'], dtype=np.float32))
            active_bans.append(b)
    return vectors, active_bans

def is_near_ban(vector: np.ndarray, threshold: float) -> Tuple[bool, Optional[float], Optional[Dict]]:
    ban_vecs, active_bans = get_active_bans()
    if not ban_vecs:
        return False, None, None
    vector = np.asarray(vector, dtype=np.float32).flatten()
    for i, ban_vec in enumerate(ban_vecs):
        sim = np.dot(vector, ban_vec)
        if sim > threshold:
            return True, sim, active_bans[i]
    return False, None, None

def decay_ban_penalty(ban_id: int):
    ban_list = load_ban_list()
    for b in ban_list:
        if b['id'] == ban_id and b.get('active', True):
            old_penalty = b.get('penalty', 1.0)
            b['penalty'] = old_penalty * BAN_DECAY_RATE
            if b['penalty'] < 0.2:
                b['active'] = False
                if DEBUG_MODE:
                    print(f"[DEBUG] decay_ban: ban_id={ban_id} 完全衰减，已解封")
                print(f"🔓 Ban区 ID {ban_id} ({b['description'][:40]}) 已被完全照亮，自动解封！")
            else:
                if DEBUG_MODE:
                    print(f"[DEBUG] decay_ban: ban_id={ban_id}, {old_penalty:.2f} -> {b['penalty']:.2f}")
                print(f"💡 Ban区 ID {ban_id} ({b['description'][:40]}) 衰减至 {b['penalty']:.2f}")
            save_ban_list(ban_list)
            break

# ===============================================
# 收藏夹管理
# ===============================================
def load_favorites() -> List[Dict]:
    if FAVORITES_FILE.exists():
        with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if DEBUG_MODE:
            print(f"[DEBUG] load_favorites: {len(data)} entries")
        return data
    return []

def save_favorite(artist_str: str, vector: np.ndarray):
    favs = load_favorites()
    favs.append({
        "timestamp": datetime.now().isoformat(),
        "artist_string": artist_str,
        "vector": vector.tolist()
    })
    with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
        json.dump(favs, f, indent=2, ensure_ascii=False)
    if DEBUG_MODE:
        print(f"[DEBUG] save_favorite: 总数={len(favs)}, str={artist_str[:40]}...")
    print(f"⭐ 已收藏: {artist_str[:60]}...")

# ===============================================
# 保护区持久化
# ===============================================
def save_protect_zones(protect_zones: Dict[int, 'ProtectZone']):
    data = {}
    for zid, zone in protect_zones.items():
        data[str(zid)] = {
            "zone_id": int(zone.zone_id),
            "center_vector": zone.center_vector.astype(float).tolist(),
            "radius": float(zone.radius),
            "mixed_count": int(zone.mixed_count),
            "generation": int(zone.generation),
            "best_score": float(zone.best_score),
            "best_artist_str": zone.best_artist_str,
            "best_vector": zone.best_vector.astype(float).tolist() if zone.best_vector is not None else None,
            "score_history": [float(s) for s in zone.score_history],
            "no_improve_rounds": int(zone.no_improve_rounds),
        }
    filepath = PROTECT_ZONES_DIR / "protect_zones.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if DEBUG_MODE:
        print(f"[DEBUG] save_protect_zones: {len(protect_zones)} zones saved")

def load_protect_zones() -> Dict[int, 'ProtectZone']:
    filepath = PROTECT_ZONES_DIR / "protect_zones.json"
    if not filepath.exists():
        if DEBUG_MODE:
            print("[DEBUG] load_protect_zones: no file")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception):
        backup_path = filepath.with_suffix('.json.bak')
        filepath.rename(backup_path)
        print(f"⚠️ 保护区文件损坏，已备份至 {backup_path}")
        return {}
    if DEBUG_MODE:
        print(f"[DEBUG] load_protect_zones: loaded {len(data)} zones")
    protect_zones = {}
    for zid_str, zone_data in data.items():
        try:
            zid = int(zid_str)
            center_vec = np.array(zone_data["center_vector"], dtype=np.float32)
            mixed_count = zone_data.get("mixed_count", MIXED_ARTISTS_COUNT)
            zone = ProtectZone(zid, center_vec, zone_data["radius"], mixed_count)
            zone.generation = zone_data.get("generation", 0)
            zone.best_score = zone_data.get("best_score", -999.0)
            zone.best_artist_str = zone_data.get("best_artist_str", "")
            zone.score_history = zone_data.get("score_history", [])
            zone.no_improve_rounds = zone_data.get("no_improve_rounds", 0)
            if zone_data.get("best_vector") is not None:
                zone.best_vector = np.array(zone_data["best_vector"], dtype=np.float32)
            protect_zones[zid] = zone
        except Exception as e:
            print(f"⚠️ 跳过损坏的保护区_{zid_str}: {e}")
    return protect_zones

# ===============================================
# 探测点与弱触角管理
# ===============================================
def load_scout_points() -> List[Dict]:
    if SCOUT_POINTS_FILE.exists():
        with open(SCOUT_POINTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if DEBUG_MODE:
            print(f"[DEBUG] load_scout_points: {len(data)} scouts loaded")
        return data
    if DEBUG_MODE:
        print("[DEBUG] load_scout_points: no file")
    return []

def save_scout_points(scouts: List[Dict]):
    with open(SCOUT_POINTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(scouts, f, indent=2, ensure_ascii=False)
    if DEBUG_MODE:
        print(f"[DEBUG] save_scout_points: {len(scouts)} scouts saved, active={sum(1 for s in scouts if s.get('active', True))}")

# ===============================================
# 触角类（增加弱触角标记）
# ===============================================
class Tentacle:
    def __init__(self, indices: np.ndarray, weights: np.ndarray, birth_gen: int = 0):
        self.indices = np.array(indices, dtype=np.int64)
        self.weights = np.array(weights, dtype=np.float32)
        self.birth_gen = birth_gen
        self.last_eval_gen = 0
        self.score_history = []         # (gen, score)
        self.vector = blend_from_indices(self.indices, self.weights)
        self.step_size = STEP_SIZE
        self.active = True
        self.trace = deque(maxlen=TRACE_HISTORY_LEN)
        # 弱触角相关
        self.is_weak = False
        self.scout_id = None            # 绑定的探测点ID，None表示无绑定

    def update_trace(self, gen: int):
        self.trace.append((gen, self.vector.copy()))

    def recent_avg_score(self, window: int = 3) -> float:
        if not self.score_history:
            return 0.0
        recent = self.score_history[-window:]
        return np.mean([s for _, s in recent])

    def consecutive_negative_count(self) -> int:
        """最近连续负分轮数"""
        count = 0
        for _, s in reversed(self.score_history):
            if s < 0:
                count += 1
            else:
                break
        return count

    def consecutive_positive_count(self) -> int:
        """最近连续正分轮数"""
        count = 0
        for _, s in reversed(self.score_history):
            if s > 0:
                count += 1
            else:
                break
        return count

# ===============================================
# 全局探索器（增加弱触角与探测点逻辑）
# ===============================================
class SlimeMoldExplorerV6:
    def __init__(self, n_tentacles=30, eval_budget=6, spread=0.5):
        self.n_tentacles = n_tentacles
        self.eval_budget = eval_budget
        self.generation = 0
        self.tentacles: List[Tentacle] = []
        self.global_best = None
        self.recent_evaluated_vectors = []
        self.scout_points = load_scout_points()  # 探测点列表
        self.init_tentacles(spread)

    def init_tentacles(self, spread: float):
        if not unique_clusters:
            clusters_available = [0]
        else:
            clusters_available = unique_clusters.copy()
        n_clusters_to_use = max(1, int(len(clusters_available) * spread + 0.5))
        selected_clusters = np.random.choice(clusters_available, size=n_clusters_to_use, replace=False)
        if DEBUG_MODE:
            print(f"[DEBUG] init_tentacles: n={self.n_tentacles}, spread={spread:.2f}, clusters={n_clusters_to_use}/{len(clusters_available)}")
        self.tentacles = []
        for i in range(self.n_tentacles):
            target_cluster = selected_clusters[i % n_clusters_to_use] if i < n_clusters_to_use else random.choice(clusters_available)
            candidates = [aid for aid, c in id_to_cluster.items() if c == target_cluster]
            if not candidates:
                start_id = random.choice(artist_ids_all)
            else:
                start_id = random.choice(candidates)
            start_idx = artist_ids_all.index(start_id)
            t = Tentacle([start_idx], np.array([1.0]), birth_gen=0)
            t.update_trace(0)
            self.tentacles.append(t)
        print(f"🌱 初始扩散度 {spread:.2f}，触角分布在 {n_clusters_to_use}/{len(clusters_available)} 个风格簇中")

    def init_from_vector(self, seed_vector: np.ndarray, spread: float):
        """从编码向量初始化触角。spread=0 集中在最匹配画师，1=扩散。"""
        seed_vector = np.asarray(seed_vector, dtype=np.float32).flatten()
        k_search = min(N_TOTAL_ARTISTS, max(self.n_tentacles * 5, 120))
        _, idxs = index.search(seed_vector.reshape(1, -1), k_search)
        pool_size = max(self.n_tentacles, int(k_search * (0.03 + spread * 0.97)))
        candidate_pool = idxs[0][:pool_size]
        self.tentacles = []
        for i in range(self.n_tentacles):
            sidx = random.choice(candidate_pool)
            t = Tentacle([sidx], np.array([1.0]), birth_gen=0)
            t.update_trace(0)
            self.tentacles.append(t)
        self.recent_evaluated_vectors = []
        self.generation = 0
        self.global_best = None
        n_unique = len(set(t.indices[0] for t in self.tentacles))
        print(f"🌱 Tag初始化: {self.n_tentacles}个触角 → {n_unique}个不同画师 (扩散度{spread:.2f})")

    def redistribute_from_vector(self, seed_vector: np.ndarray, spread: float):
        """热切换扩散度：重新分配不在新候选池内的触角，保留评分历史。"""
        seed_vector = np.asarray(seed_vector, dtype=np.float32).flatten()
        k_search = min(N_TOTAL_ARTISTS, max(self.n_tentacles * 5, 120))
        _, idxs = index.search(seed_vector.reshape(1, -1), k_search)
        pool_size = max(self.n_tentacles, int(k_search * (0.03 + spread * 0.97)))
        candidate_pool = idxs[0][:pool_size]
        changed = 0
        for t in self.tentacles:
            if not t.active:
                continue
            if t.indices[0] in candidate_pool:
                continue
            t.indices = np.array([random.choice(candidate_pool)], dtype=np.int64)
            t.weights = np.array([1.0], dtype=np.float32)
            changed += 1
        if changed:
            print(f"🔄 扩散度热切换: {changed}个触角重新分配画师 (扩散度{spread:.2f})")

    def _novelty_bonus(self, vector: np.ndarray) -> float:
        if not self.recent_evaluated_vectors:
            return 1.0
        dists = [1 - np.dot(vector, v) for v in self.recent_evaluated_vectors]
        return np.mean(dists)

    def _assign_weak_tentacles(self):
        """自动将符合条件的触角标记为弱，并分配探测点"""
        weak_count = sum(1 for t in self.tentacles if t.is_weak)
        for t in self.tentacles:
            if not t.active or t.is_weak:
                continue
            if t.consecutive_negative_count() >= WEAK_CONSECUTIVE_NEGATIVE and weak_count < MAX_WEAK_TENTACLES:
                for sp in self.scout_points:
                    if sp.get('active', True) and len(sp.get('assigned', [])) < sp['capacity']:
                        t.is_weak = True
                        t.scout_id = sp['id']
                        sp.setdefault('assigned', []).append(self.tentacles.index(t))
                        weak_count += 1
                        if DEBUG_MODE:
                            print(f"[DEBUG] 触角 {self.tentacles.index(t)} 标记为弱触角，绑定探测点 {sp['id']}")
                        break
                if not t.is_weak:
                    t.is_weak = True
                    t.scout_id = None
                    if DEBUG_MODE:
                        print(f"[DEBUG] 触角 {self.tentacles.index(t)} 标记为弱触角（无探测点绑定）")

    def _check_promote_suggestions(self):
        for i, t in enumerate(self.tentacles):
            if t.is_weak and t.consecutive_positive_count() >= PROMOTE_SUGGEST_ROUNDS:
                if DEBUG_MODE:
                    print(f"[DEBUG] 触角 {i} 已连续 {PROMOTE_SUGGEST_ROUNDS} 轮正分，建议 !promote {i} 转正")

    def select_tentacles_to_evaluate(self) -> List[int]:
        self._assign_weak_tentacles()
        active_before = sum(1 for t in self.tentacles if t.active)
        weak_before = sum(1 for t in self.tentacles if t.is_weak)
        tag_dir = compute_tag_direction()  # 只算一次
        candidates = []
        for i, t in enumerate(self.tentacles):
            if not t.active:
                continue
            time_since = self.generation - t.last_eval_gen
            time_w = 1.0 + 0.3 * min(time_since, 5)
            nov = self._novelty_bonus(t.vector)
            nov_w = 1.0 + nov * 2.0
            max_s = max([s for _, s in t.score_history]) if t.score_history else 0.0
            score_w = 1.0 + max(0, max_s) * 0.15
            weight = time_w * nov_w * score_w
            near, sim, ban_entry = is_near_ban(t.vector, BAN_SELECT_THRESH)
            if near and ban_entry:
                pen = 0.2 * ban_entry.get('penalty', 1.0)
                if t.is_weak and t.scout_id is not None:
                    pen *= 0.3
                weight *= pen
            if t.is_weak:
                weight *= 1.5
            # Tag引导偏置：靠近Tag方向的触角更容易被选中
            if tag_dir is not None:
                sim = np.dot(t.vector, tag_dir)
                weight *= 1.0 + sim * TAG_GUIDANCE
            candidates.append((i, weight))
        if not candidates:
            if DEBUG_MODE:
                print(f"[DEBUG] select_tentacles: 无候选触角（活跃={active_before}）")
            return []
        idxs, wgts = zip(*candidates)
        probs = np.array(wgts) / np.sum(wgts)
        n_sample = min(EVAL_BUDGET, len(candidates))
        selected = np.random.choice(len(candidates), size=n_sample, p=probs, replace=False)
        selected_idx = [candidates[s][0] for s in selected]
        if DEBUG_MODE:
            print(f"[DEBUG] select_tentacles: 活跃={active_before}, 弱={weak_before}, 候选={len(candidates)}, 选中={selected_idx}")
        return selected_idx

    def evaluate_selected(self, selected_indices: List[int]) -> List[Tuple[str, str, int]]:
        if DEBUG_MODE:
            print(f"[DEBUG] evaluate_selected: gen={self.generation}, ids={selected_indices}")
        results = []
        used_artist_ids = set()
        for idx in selected_indices:
            t = self.tentacles[idx]
            artist_str = vector_to_artist_string(t.vector, skip_ids=used_artist_ids)
            # 记录已用画师，下一根触角跳过
            for token in artist_str.split(", "):
                name = token.split(":")[0].replace("(by ", "").strip()
                aid = name_to_id.get(name)
                if aid is not None:
                    used_artist_ids.add(aid)
            prompt = f"{BASE_POSITIVE_PROMPT}, {artist_str}"
            fname = f"gen{self.generation:02d}_t{idx:03d}"
            seed = FIXED_SEED if FIXED_SEED != -1 else (self.generation * 100 + idx)
            if DEBUG_MODE:
                print(f"[DEBUG]   tentacle #{idx}: FIXED_SEED={FIXED_SEED}, computed seed={seed}")
                marker_parts = []
                if t.is_weak:
                    marker_parts.append("弱")
                if t.scout_id is not None:
                    marker_parts.append(f"探测S{t.scout_id}")
                marker_str = "[" + ",".join(marker_parts) + "]" if marker_parts else ""
                print(f"[DEBUG]   tentacle #{idx}: seed={seed}, file={fname} {marker_str}")
                print(f"[DEBUG]     artist_str={artist_str[:80]}...")
            wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
            pid = queue_prompt(wf)
            t0 = time.time()
            res = wait_for_prompt(pid)
            elapsed = time.time() - t0
            img_path = None
            if res:
                for out in res.get('outputs', {}).values():
                    if 'images' in out:
                        img_path = str(RUN_OUTPUT_DIR / out['images'][0]['filename'])
                        break
            if DEBUG_MODE:
                print(f"[DEBUG]   tentacle #{idx}: elapsed={elapsed:.1f}s, img={'OK' if img_path else 'FAIL'}")
            results.append((img_path, artist_str, idx))
            time.sleep(0.3)
        return results

    def update_tentacles(self, selected_indices: List[int], scores: List[float]):
        if DEBUG_MODE:
            score_pairs = [f"#{idx}={s:.1f}" for idx, s in zip(selected_indices, scores)]
            print(f"[DEBUG] update_tentacles: gen={self.generation}, scores=[{', '.join(score_pairs)}]")
        for idx, score in zip(selected_indices, scores):
            t = self.tentacles[idx]
            t.last_eval_gen = self.generation
            t.score_history.append((self.generation, score))
            if self.global_best is None or score > self.global_best[2]:
                self.global_best = (t.indices.copy(), t.weights.copy(), score)
            # 弱触角若得分转正，保持弱身份但记录（等用户手动 !promote）
            # 得分处理
            if score >= 0:
                self._grow(t, score)
            else:
                self._shrink(t, score)
            t.update_trace(self.generation)
            self.recent_evaluated_vectors.append(t.vector.copy())
            if len(self.recent_evaluated_vectors) > 20:
                self.recent_evaluated_vectors.pop(0)
            if score >= 5:
                near, sim, ban_entry = is_near_ban(t.vector, BAN_SELECT_THRESH)
                if near and sim < BAN_MUTATE_THRESH and ban_entry:
                    decay_ban_penalty(ban_entry['id'])
        # 弱触角建议转正
        self._check_promote_suggestions()
        self._prune_rebirth()
        self._apply_weak_repulsion()
        self._auto_long_jump_if_needed()

    def _grow(self, t: Tentacle, score: float):
        # 弱触角移动限制：步长减半，且加入绳子拉力
        if t.is_weak:
            effective_step = t.step_size * 0.5
            # 如果有绑定探测点，应用绳子拉力
            if t.scout_id is not None:
                scout = next((sp for sp in self.scout_points if sp['id'] == t.scout_id), None)
                if scout:
                    scout_vec = np.array(scout['vector'], dtype=np.float32)
                    # 随机变异方向
                    random_dir = np.random.randn(2048).astype(np.float32)
                    random_dir /= np.linalg.norm(random_dir)
                    # 朝向探测点的方向
                    toward_scout = scout_vec - t.vector
                    toward_norm = np.linalg.norm(toward_scout)
                    if toward_norm > 0:
                        toward_scout /= toward_norm
                    else:
                        toward_scout = random_dir
                    # 合成方向：拉力系数 PULL_STRENGTH
                    move_dir = (1 - PULL_STRENGTH) * random_dir + PULL_STRENGTH * toward_scout
                    move_dir /= np.linalg.norm(move_dir)
                    # 在权重空间上施加这个方向（将方向转回画师权重空间，这里简化：直接对向量施加偏移）
                    noise = move_dir * effective_step
                    new_vec = t.vector + noise
                    new_vec /= np.linalg.norm(new_vec)
                    # Ban变异回退检查
                    near, _, _ = is_near_ban(new_vec, BAN_MUTATE_THRESH)
                    if near:
                        # 回退且偏移
                        new_vec = t.vector - noise * 0.5
                        new_vec /= np.linalg.norm(new_vec)
                    t.vector = new_vec.astype(np.float32)
                    # 同时微调权重（通过反解？这里不复杂化，保持向量更新，权重复用原有）
                    return
        if DEBUG_MODE:
            print(f"[DEBUG]   grow: idx={self.tentacles.index(t)}, score={score:.1f}, weak={t.is_weak}, scout={t.scout_id}")
        # 普通触角生长逻辑（原版）
        noise = np.random.normal(0, t.step_size, t.weights.shape)
        new_w = t.weights + noise
        new_w = np.clip(new_w, 0.05, None)
        new_w /= np.sum(new_w)
        t.weights = new_w
        new_vec = blend_from_indices(t.indices, t.weights)
        near, _, _ = is_near_ban(new_vec, BAN_MUTATE_THRESH)
        if near:
            t.weights = t.weights - noise * 0.5
            t.weights = np.clip(t.weights, 0.05, None)
            t.weights /= np.sum(t.weights)
            new_vec = blend_from_indices(t.indices, t.weights)
        t.vector = new_vec
        if score > 7 and np.random.rand() < 0.15:
            new_idx = np.random.choice(N_TOTAL_ARTISTS)
            if new_idx not in t.indices:
                t.indices = np.append(t.indices, new_idx)
                t.weights = np.append(t.weights, 0.2)
                t.weights /= np.sum(t.weights)
                t.vector = blend_from_indices(t.indices, t.weights)

    def _shrink(self, t: Tentacle, score: float):
        if DEBUG_MODE:
            print(f"[DEBUG]   shrink: idx={self.tentacles.index(t)}, score={score:.1f}, n_indices={len(t.indices)}")
        if len(t.indices) > 1:
            min_pos = np.argmin(t.weights)
            t.indices = np.delete(t.indices, min_pos)
            t.weights = np.delete(t.weights, min_pos)
            t.weights /= np.sum(t.weights)
            t.vector = blend_from_indices(t.indices, t.weights)
        elif score < -5:
            if DEBUG_MODE:
                print(f"[DEBUG]   shrink: score={score:.1f} < -5, 触角失活")
            t.active = False

    def _prune_rebirth(self):
        for t in self.tentacles:
            if not t.active:
                continue
            max_s = max([s for _, s in t.score_history]) if t.score_history else -999
            since_eval = self.generation - t.last_eval_gen
            if t.is_weak and t.scout_id is None:
                if t.consecutive_negative_count() >= WEAK_CONSECUTIVE_NEGATIVE + 2:
                    t.active = False
            elif (max_s < 0 and since_eval > 3) or max_s < -5:
                t.active = False
        inactive_indices = [i for i, t in enumerate(self.tentacles) if not t.active]
        if DEBUG_MODE and inactive_indices:
            print(f"[DEBUG] prune: {len(inactive_indices)} 触角将重生, ids={inactive_indices}")
        for i in inactive_indices:
            # 重生时避开Ban
            attempts = 0
            while attempts < 50:
                if self.global_best and np.random.rand() < 0.6:
                    base_idx, base_w, _ = self.global_best
                    new_idx = base_idx.copy()
                    new_w = base_w.copy()
                    noise = np.random.normal(0, 0.1, new_w.shape)
                    new_w = np.clip(new_w + noise, 0.05, None)
                    new_w /= np.sum(new_w)
                else:
                    n = np.random.randint(1, 4)
                    new_idx = np.random.choice(N_TOTAL_ARTISTS, n, replace=False)
                    new_w = np.random.rand(n); new_w /= np.sum(new_w)
                new_vec = blend_from_indices(new_idx, new_w)
                near, _, _ = is_near_ban(new_vec, BAN_REBIRTH_THRESH)
                if not near:
                    break
                attempts += 1
            self.tentacles[i] = Tentacle(new_idx, new_w, birth_gen=self.generation)
            self.tentacles[i].update_trace(self.generation)

    def _apply_weak_repulsion(self):
        active_indices = [i for i, t in enumerate(self.tentacles) if t.active and not t.is_weak]  # 弱触角不参与排斥
        active_objects = [self.tentacles[i] for i in active_indices]
        if len(active_objects) < 2:
            return
        vecs = np.array([t.vector for t in active_objects])
        sim = np.dot(vecs, vecs.T)
        for i in range(len(active_objects)):
            for j in range(i + 1, len(active_objects)):
                if sim[i, j] > SIMILARITY_THRESHOLD + 0.05:
                    target = active_objects[i] if np.random.rand() < 0.5 else active_objects[j]
                    noise = np.random.normal(0, 0.15, target.weights.shape)
                    new_w = target.weights + noise
                    new_w = np.clip(new_w, 0.05, None)
                    target.weights = new_w / np.sum(new_w)
                    target.vector = blend_from_indices(target.indices, target.weights)

    def _auto_long_jump_if_needed(self):
        active_indices = [i for i, t in enumerate(self.tentacles) if t.active and not t.is_weak]  # 弱触角不参与长跳
        active_objects = [self.tentacles[i] for i in active_indices]
        if len(active_objects) < 3:
            return
        vecs = np.array([t.vector for t in active_objects])
        sim_matrix = np.dot(vecs, vecs.T)
        avg_sim = (np.sum(sim_matrix) - len(active_objects)) / (len(active_objects) * (len(active_objects) - 1))
        if avg_sim > 0.92:
            n_jump = min(3, len(active_objects))
            jump_selected = np.random.choice(len(active_objects), size=n_jump, replace=False)
            for idx_in_active in jump_selected:
                original_idx = active_indices[idx_in_active]
                attempts = 0
                while attempts < 50:
                    n = np.random.randint(1, 4)
                    new_idx = np.random.choice(N_TOTAL_ARTISTS, n, replace=False)
                    new_w = np.random.rand(n); new_w /= np.sum(new_w)
                    new_vec = blend_from_indices(new_idx, new_w)
                    near, _, _ = is_near_ban(new_vec, BAN_REBIRTH_THRESH)
                    if not near:
                        break
                    attempts += 1
                self.tentacles[original_idx] = Tentacle(new_idx, new_w, birth_gen=self.generation)
                self.tentacles[original_idx].update_trace(self.generation)
            if DEBUG_MODE:
                print(f"[DEBUG] 多样性过低 (平均相似度 {avg_sim:.3f})，已触发 {n_jump} 次长跳（不含弱触角）")

    def exchange_info(self):
        active_ts = [t for t in self.tentacles if t.active and not t.is_weak]
        if len(active_ts) < 2:
            return
        if DEBUG_MODE:
            print(f"[DEBUG] exchange_info: gen={self.generation}, {len(active_ts)} active (weak excluded)")
        vecs = np.array([t.vector for t in active_ts])
        tmp_idx = faiss.IndexFlatIP(2048)
        tmp_idx.add(vecs)
        exchange_count = 0
        for i, t in enumerate(active_ts):
            _, neigh = tmp_idx.search(vecs[i:i+1], 3)
            for j in neigh[0]:
                if j != i and np.dot(vecs[i], vecs[j]) > SIMILARITY_THRESHOLD:
                    other = active_ts[j]
                    combined = list(set(t.indices) | set(other.indices))[:MIXED_ARTISTS_COUNT]
                    t.indices = np.array(combined, dtype=np.int64)
                    t.weights = np.ones(len(combined), dtype=np.float32) / len(combined)
                    t.vector = blend_from_indices(t.indices, t.weights)
                    exchange_count += 1
        if DEBUG_MODE and exchange_count:
            print(f"[DEBUG] exchange_info: {exchange_count} 个触角信息交换完成")

    def ban_current_tentacle(self, tentacle_idx: int):
        t = self.tentacles[tentacle_idx]
        artist_str = vector_to_artist_string(t.vector)
        add_ban(artist_str, t.vector, "current_tentacle", self.generation)
        t.active = False
        if DEBUG_MODE:
            print(f"[DEBUG] ban_current_tentacle: idx={tentacle_idx}, artist={artist_str[:60]}...")
        print(f"🧹 触角 {tentacle_idx} 已被标记为重生")

    def spread_weakest_tentacles(self, x: int, y: float):
        active = [t for t in self.tentacles if t.active and not t.is_weak]  # 只对普通触角扩散
        if len(active) < 3:
            if DEBUG_MODE:
                print("[DEBUG] spread_weakest: 活跃触角不足")
            print("⚠️ 活跃触角不足，无法扩散")
            return
        sorted_active = sorted(active, key=lambda t: t.recent_avg_score())
        victims = sorted_active[:min(x, len(sorted_active))]
        if DEBUG_MODE:
            victim_ids = [self.tentacles.index(v) for v in victims]
            print(f"[DEBUG] spread_weakest: x={x}, y={y:.2f}, victims={victim_ids}")
        vecs = np.array([t.vector for t in active])
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        max_dist = max(1 - np.dot(t.vector, centroid) for t in active)
        if max_dist < 0.01:
            max_dist = 0.5
        target_dist = max_dist * y
        for victim in victims:
            attempts = 0
            while attempts < 30:
                random_dir = np.random.randn(2048).astype(np.float32)
                random_dir -= np.dot(random_dir, centroid) * centroid
                dir_norm = np.linalg.norm(random_dir)
                if dir_norm > 0:
                    random_dir /= dir_norm
                new_vec = centroid + target_dist * random_dir
                new_norm = np.linalg.norm(new_vec)
                if new_norm > 0:
                    new_vec /= new_norm
                near, _, _ = is_near_ban(new_vec, BAN_REBIRTH_THRESH)
                if not near:
                    break
                attempts += 1
            _, idxs = index.search(new_vec.reshape(1, -1), MIXED_ARTISTS_COUNT)
            new_indices = idxs[0]
            new_weights = np.ones(len(new_indices), dtype=np.float32) / len(new_indices)
            idx_orig = self.tentacles.index(victim)
            self.tentacles[idx_orig] = Tentacle(new_indices, new_weights, birth_gen=self.generation)
            self.tentacles[idx_orig].update_trace(self.generation)
        print(f"🦘 已扩散 {len(victims)} 个触角到边缘 (边缘强度 {y})")

    def replace_weakest_with_vector(self, target_vector: np.ndarray) -> bool:
        active = [t for t in self.tentacles if t.active and not t.is_weak]
        if not active:
            return False
        never_eval = [t for t in active if not t.score_history]
        victim = never_eval[0] if never_eval else min(active, key=lambda t: t.recent_avg_score())
        _, idxs = index.search(target_vector.reshape(1, -1), MIXED_ARTISTS_COUNT)
        new_indices = idxs[0]
        new_weights = np.ones(len(new_indices), dtype=np.float32) / len(new_indices)
        idx_orig = self.tentacles.index(victim)
        self.tentacles[idx_orig] = Tentacle(new_indices, new_weights, birth_gen=self.generation)
        self.tentacles[idx_orig].update_trace(self.generation)
        if DEBUG_MODE:
            print(f"[DEBUG] replace_weakest: 替换触角 {idx_orig} (avg_score={victim.recent_avg_score():.1f}) -> 新向量")
        return True

    def stats(self) -> Tuple[int, int, int]:
        active = [t for t in self.tentacles if t.active]
        artists = set()
        clusters = set()
        for t in active:
            for idx in t.indices:
                artists.add(artist_ids_all[idx])
                clusters.add(id_to_cluster.get(artist_ids_all[idx], -1))
        return len(active), len(artists), len(clusters)

    def promote_tentacle(self, idx: int):
        if idx < 0 or idx >= len(self.tentacles):
            if DEBUG_MODE:
                print(f"[DEBUG] promote: 无效触角ID {idx}")
            print("❌ 无效触角ID")
            return
        t = self.tentacles[idx]
        if not t.is_weak:
            if DEBUG_MODE:
                print(f"[DEBUG] promote: 触角 {idx} 不是弱触角")
            print("❌ 该触角不是弱触角")
            return
        scout_id_before = t.scout_id
        if t.scout_id is not None:
            for sp in self.scout_points:
                if sp['id'] == t.scout_id:
                    if idx in sp.get('assigned', []):
                        sp['assigned'].remove(idx)
                    break
        t.is_weak = False
        t.scout_id = None
        save_scout_points(self.scout_points)
        if DEBUG_MODE:
            print(f"[DEBUG] promote: 触角 {idx} 转正 (原探测点 {scout_id_before})")
        print(f"✅ 触角 {idx} 已转正")

# ===============================================
# 保护区类（保持不变）
# ===============================================
class ProtectZone:
    def __init__(self, zone_id: int, center_vector: np.ndarray, radius: float, mixed_count: int):
        self.zone_id = zone_id
        self.center_vector = np.asarray(center_vector, dtype=np.float32).flatten()
        self.radius = radius
        self.mixed_count = mixed_count
        self.generation = 0
        self.candidate_artists = []
        self.candidate_vectors = None
        self.best_score = -999.0
        self.best_artist_str = ""
        self.best_vector = None
        self.score_history = []
        self.no_improve_rounds = 0
        self.optimizer_result = None
        _, idxs = index.search(self.center_vector.reshape(1, -1), PROTECT_MAX_CANDIDATES * 3)
        for idx in idxs[0]:
            artist_vec = vectors_all[idx]
            sim = np.dot(self.center_vector, artist_vec)
            if sim >= 1 - radius and idx not in self.candidate_artists:
                self.candidate_artists.append(idx)
            if len(self.candidate_artists) >= PROTECT_MAX_CANDIDATES:
                break
        self.candidate_vectors = vectors_all[self.candidate_artists]
        n_candidates = len(self.candidate_artists)
        print(f"🛡️ 保护区_{zone_id} 初始化完成：中心画师={vector_to_artist_string(self.center_vector, top_k=1)[:40]}，候选画师={n_candidates}个")
        if n_candidates > PROTECT_MAX_CANDIDATES:
            print(f"⚠️ 候选画师数量({n_candidates})超过贝叶斯优化建议上限({PROTECT_MAX_CANDIDATES})，可能影响搜索效率")

    def ban_penalty(self, vector: np.ndarray) -> float:
        ban_vecs, active_bans = get_active_bans()
        if not ban_vecs:
            return 0.0
        vector = np.asarray(vector, dtype=np.float32).flatten()
        total_penalty = 0.0
        for i, ban_vec in enumerate(ban_vecs):
            sim = np.dot(vector, ban_vec)
            if sim > BAN_SELECT_THRESH:
                penalty = active_bans[i].get('penalty', 1.0)
                total_penalty += penalty * (sim - BAN_SELECT_THRESH) * BAN_PENALTY_COEFFICIENT
        total_penalty = min(total_penalty, 10.0)
        return total_penalty

    def objective_function(self, weights) -> float:
        weights = np.asarray(weights, dtype=np.float32).flatten()
        top_indices = np.argsort(weights)[-self.mixed_count:]
        selected_indices = [self.candidate_artists[int(i)] for i in top_indices]
        selected_weights = weights[top_indices]
        selected_weights = selected_weights / np.sum(selected_weights)
        blended = blend_from_indices(np.array(selected_indices), selected_weights)
        artist_str = vector_to_artist_string(blended, top_k=self.mixed_count)
        prompt = f"{BASE_POSITIVE_PROMPT}, {artist_str}"
        fname = f"protect_{self.zone_id}_gen{self.generation}"
        seed = FIXED_SEED if FIXED_SEED != -1 else (self.generation * 1000 + self.zone_id)
        wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
        pid = queue_prompt(wf)
        res = wait_for_prompt(pid)
        if not res:
            return 10.0
        img_path = None
        for out in res.get('outputs', {}).values():
            if 'images' in out:
                img_path = str(RUN_OUTPUT_DIR / out['images'][0]['filename'])
                break
        if img_path:
            display_images([(img_path, artist_str)], self.generation, f"保护区_{self.zone_id}")
        ban_pen = self.ban_penalty(blended)
        while True:
            s = input(f"菌丝评分 (-10~10，当前Ban区软斥力: {ban_pen:.2f}): ").strip()
            try:
                score = float(s)
                if -10 <= score <= 10:
                    break
            except ValueError:
                print("请输入有效数字")
        effective_score = score - ban_pen
        self.generation += 1
        if effective_score > self.best_score:
            self.best_score = effective_score
            self.best_artist_str = artist_str
            self.best_vector = blended.copy()
            self.no_improve_rounds = 0
        else:
            self.no_improve_rounds += 1
        self.score_history.append(effective_score)
        if score >= 5:
            near, sim, ban_entry = is_near_ban(blended, BAN_SELECT_THRESH)
            if near and sim < BAN_MUTATE_THRESH and ban_entry:
                decay_ban_penalty(ban_entry['id'])
        return -effective_score

    def optimize(self, n_calls: int = 10) -> bool:
        n_candidates = len(self.candidate_artists)
        space = [Real(0.0, 1.0, name=f'w_{i}') for i in range(n_candidates)]
        print(f"🧬 保护区_{self.zone_id} 开始贝叶斯优化（{n_candidates}维，计划{n_calls}轮）")
        for _ in range(n_calls):
            if self.no_improve_rounds >= PROTECT_CONVERGE_ROUNDS:
                print(f"✅ 保护区_{self.zone_id} 连续{PROTECT_CONVERGE_ROUNDS}轮无提升，已收敛")
                return True
            try:
                if self.optimizer_result is None:
                    result = gp_minimize(
                        func=self.objective_function,
                        dimensions=space,
                        n_calls=1,
                        n_initial_points=1,
                        random_state=self.generation
                    )
                else:
                    result = gp_minimize(
                        func=self.objective_function,
                        dimensions=space,
                        n_calls=1,
                        n_initial_points=0,
                        x0=self.optimizer_result.x,
                        y0=[-self.best_score],
                        random_state=self.generation
                    )
                self.optimizer_result = result
            except Exception as e:
                print(f"❌ 贝叶斯优化出错: {e}")
                return False
        return False

    # ------------------------------------------------------------
    # 进化培育 API（用于 WebUI 逐轮培育）
    # ------------------------------------------------------------
    def _weights_to_candidate(self, weights: np.ndarray) -> Dict:
        """将权重向量转为候选画师串 + 混合向量"""
        top_k = min(self.mixed_count, len(weights))
        top_i = np.argsort(weights)[-top_k:]
        sel_indices = [self.candidate_artists[int(i)] for i in top_i]
        sel_weights = weights[top_i]
        sel_weights = sel_weights / np.sum(sel_weights)
        blended = blend_from_indices(np.array(sel_indices), sel_weights)
        artist_str = vector_to_artist_string(blended, top_k=self.mixed_count)
        ban_pen = self.ban_penalty(blended)
        return {
            "artist_str": artist_str,
            "blended": blended,
            "ban_penalty": float(ban_pen),
        }

    def _mutate_weights(self, weights: np.ndarray, strength: float = 0.15) -> np.ndarray:
        """高斯噪声突变权值，归一化"""
        w = weights + np.random.normal(0, strength, len(weights))
        w = np.clip(w, 0.01, 1.0)
        return (w / np.sum(w)).astype(np.float32)

    def _crossover_weights(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """两父代权重按随机比例混合"""
        alpha = np.random.random()
        w = alpha * w1 + (1 - alpha) * w2
        return w.astype(np.float32)

    def evo_init(self, pop_size: int) -> List[Dict]:
        """初始化随机种群，返回 pop_size 个候选"""
        population = []
        for i in range(pop_size):
            w = np.random.rand(len(self.candidate_artists))
            w = w / np.sum(w)
            c = self._weights_to_candidate(w)
            c['weights'] = w
            c['id'] = i
            population.append(c)
        self._evo_population = population
        self.generation = 0
        return population

    def evo_submit_scores(self, scores: Dict[int, float], n_parents: int) -> Dict:
        """评分提交 → 选父代 → 突变+杂交 → 生成下一代种群"""
        pop_size = len(self._evo_population)
        n_parents = min(n_parents, pop_size)
        # 计算有效分，记录
        evals = []
        for i, score in scores.items():
            if i >= pop_size:
                continue
            pen = self.ban_penalty(self._evo_population[i]['blended'])
            eff = score - pen
            evals.append((i, eff, score))
            self.score_history.append(eff)
        # 最佳更新
        evals.sort(key=lambda x: x[1], reverse=True)
        best_i, best_eff, _ = evals[0]
        if best_eff > self.best_score:
            self.best_score = best_eff
            self.best_artist_str = self._evo_population[best_i]['artist_str']
            self.best_vector = self._evo_population[best_i]['blended'].copy()
            self.no_improve_rounds = 0
            # 高分衰减Ban
            if scores.get(best_i, 0) >= 5:
                near, sim, ban_entry = is_near_ban(self._evo_population[best_i]['blended'], BAN_SELECT_THRESH)
                if near and sim < BAN_MUTATE_THRESH and ban_entry:
                    decay_ban_penalty(ban_entry['id'])
        else:
            self.no_improve_rounds += 1
        # 选父代，标记⭐
        parent_ids = {p[0] for p in evals[:n_parents]}
        for i in range(pop_size):
            self._evo_population[i]['is_parent'] = i in parent_ids
        # 生成下一代
        parent_weights = [self._evo_population[p[0]]['weights'] for p in evals[:n_parents]]
        n_crossover = max(1, pop_size - n_parents * max(1, pop_size // n_parents))
        new_pop = []
        kid_id = 0
        # 突变子代
        for pw in parent_weights:
            for _ in range(max(1, (pop_size - n_crossover) // len(parent_weights))):
                if kid_id >= pop_size - n_crossover:
                    break
                m = self._mutate_weights(pw)
                c = self._weights_to_candidate(m)
                c['weights'] = m
                c['id'] = kid_id
                new_pop.append(c)
                kid_id += 1
        # 杂交子代
        for _ in range(n_crossover):
            if len(parent_weights) >= 2:
                i1, i2 = np.random.choice(len(parent_weights), 2, replace=False)
                x = self._crossover_weights(parent_weights[i1], parent_weights[i2])
                c = self._weights_to_candidate(x)
                c['weights'] = x
                c['id'] = kid_id
                new_pop.append(c)
                kid_id += 1
        # 补齐（如果上述未填满）
        while len(new_pop) < pop_size:
            pw = parent_weights[np.random.randint(len(parent_weights))]
            m = self._mutate_weights(pw)
            c = self._weights_to_candidate(m)
            c['weights'] = m
            c['id'] = kid_id
            new_pop.append(c)
            kid_id += 1
        self._evo_population = new_pop
        self.generation += 1
        converged = self.no_improve_rounds >= PROTECT_CONVERGE_ROUNDS
        return {
            "parents": [p[0] for p in evals[:n_parents]],
            "best_score": float(self.best_score),
            "converged": converged,
            "generation": self.generation,
            "no_improve_rounds": self.no_improve_rounds,
            "best_artist_str": self.best_artist_str,
            "scores": {str(p[0]): round(p[1], 2) for p in evals[:n_parents]},
        }

# ===============================================
# 可视化（增加弱触角/探测点）
# ===============================================
def plot_tentacles(explorer: SlimeMoldExplorerV6, protect_zones: Dict[int, ProtectZone] = None):
    if not DEBUG_MODE or pca_model is None:
        return
    active = [t for t in explorer.tentacles if t.active]
    if not active:
        return
    plt.figure(figsize=(10, 8))
    if PLOT_STYLE == "full" and pca_vectors_2d is not None:
        plt.scatter(pca_vectors_2d[:, 0], pca_vectors_2d[:, 1], c='lightgray', alpha=0.3, s=5)
    # 轨迹
    for t in active:
        if len(t.trace) >= 2:
            pts = [pca_model.transform(vec.reshape(1, -1))[0] for _, vec in t.trace]
            xs, ys = zip(*pts)
            alphas = np.linspace(0.2, 1.0, len(xs))
            for i in range(len(xs) - 1):
                plt.plot(xs[i:i+2], ys[i:i+2], color='gray', alpha=alphas[i], linewidth=1)
    # 触角点
    scores = [t.recent_avg_score() for t in active]
    norm_scores = (np.array(scores) + 10) / 20
    colors = plt.cm.RdYlGn(norm_scores)[:, :3]
    t_vecs = np.array([t.vector for t in active])
    t_2d = pca_model.transform(t_vecs)
    # 区分弱触角
    weak_mask = [t.is_weak for t in active]
    if any(weak_mask):
        weak_idx = np.where(weak_mask)[0]
        plt.scatter(t_2d[weak_idx, 0], t_2d[weak_idx, 1], c='yellow', s=120, marker='D', edgecolors='black', linewidth=1.5, label='弱触角')
    normal_mask = [not t.is_weak for t in active]
    if any(normal_mask):
        norm_idx = np.where(normal_mask)[0]
        plt.scatter(t_2d[norm_idx, 0], t_2d[norm_idx, 1], c=colors[norm_idx], s=100, marker='X', edgecolors='white', linewidth=1.5)
    # Ban区
    ban_vecs, _ = get_active_bans()
    if ban_vecs:
        ban_2d = pca_model.transform(ban_vecs)
        plt.scatter(ban_2d[:, 0], ban_2d[:, 1], c='red', s=150, marker='o', alpha=0.6, label='Ban区')
    # 保护区
    if protect_zones:
        for zid, zone in protect_zones.items():
            zone_2d = pca_model.transform(zone.center_vector.reshape(1, -1))[0]
            plt.scatter(zone_2d[0], zone_2d[1], c='cyan', s=200, marker='*', alpha=0.8, label=f'保护区_{zid}')
    # 探测点
    for sp in explorer.scout_points:
        sp_vec = np.array(sp['vector'], dtype=np.float32)
        sp_2d = pca_model.transform(sp_vec.reshape(1, -1))[0]
        plt.scatter(sp_2d[0], sp_2d[1], c='green', s=200, marker='o', alpha=0.5, label='探测点' if '探测点' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.legend()
    plt.title(f"触角分布 (第{explorer.generation}代)")
    plot_path = RUN_OUTPUT_DIR / f"tentacles_gen{explorer.generation:02d}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"📊 触角分布图已保存: {plot_path}")

# ===============================================
# 主菜单（增加探测点管理、转正指令）
# ===============================================
def list_bans_command():
    _, active_bans = get_active_bans()
    if active_bans:
        print("📋 当前Ban区列表:")
        for b in active_bans:
            penalty = b.get('penalty', 1.0)
            print(f"  🟢 ID {b['id']}: {b['description'][:50]} (惩罚强度: {penalty:.2f}, 代数: {b['created_gen']})")
        if DEBUG_MODE:
            print(f"[DEBUG] list_bans: {len(active_bans)} 个激活Ban区")
    else:
        if DEBUG_MODE:
            print("[DEBUG] list_bans: 暂无")
        print("📋 暂无Ban区")

def list_protect_zones_command(protect_zones: Dict[int, ProtectZone]):
    if protect_zones:
        print("🛡️ 当前保护区列表:")
        for zid, zone in protect_zones.items():
            center_str = vector_to_artist_string(zone.center_vector, top_k=1)[:40] if zone.center_vector is not None else "未知"
            print(f"  ID {zid}: 中心≈{center_str}，候选={len(zone.candidate_artists)}画师，最佳分={zone.best_score:.2f}")
    else:
        print("🛡️ 暂无保护区")

def list_scouts_command(explorer: SlimeMoldExplorerV6):
    if explorer.scout_points:
        print("🔭 当前探测点列表:")
        for sp in explorer.scout_points:
            assigned = len(sp.get('assigned', []))
            print(f"  ID {sp['id']}: 容量{sp['capacity']}，已分配{assigned}，位置≈{vector_to_artist_string(np.array(sp['vector']), top_k=1)[:40]}")
    else:
        print("🔭 暂无探测点")

def main_menu(explorer: SlimeMoldExplorerV6, protect_zones: Dict[int, ProtectZone]):
    active_zone_id = None
    while True:
        print("\n" + "=" * 60)
        print("🧫 黏菌风格探索器 V6 - 主菜单")
        print("=" * 60)
        act_cnt, art_cnt, clus_cnt = explorer.stats()
        print(f"📊 全局触角: 活跃 {act_cnt}/{explorer.n_tentacles} | 覆盖画师: {art_cnt} | 覆盖簇: {clus_cnt}")
        weak_cnt = sum(1 for t in explorer.tentacles if t.is_weak)
        print(f"🔗 弱触角: {weak_cnt} (上限{MAX_WEAK_TENTACLES}) | 探测点: {len(explorer.scout_points)}个")
        print(f"🛡️ 保护区: {len(protect_zones)}个")
        print("-" * 60)
        print("  1. 继续全局扩散探索 (一轮)")
        print("  2. 激活/切换保护区进行培育")
        print("  3. 查看/管理保护区")
        print("  4. 查看/管理Ban区")
        print("  5. 查看/管理探测点")
        print("  6. 转正弱触角 (!promote)")
        print("  0. 退出程序")
        print("-" * 60)
        choice = input("请选择操作: ").strip()
        if choice == '1':
            explorer.generation += 1
            print(f"\n🧪 第 {explorer.generation} 代 全局探索")
            selected = explorer.select_tentacles_to_evaluate()
            if not selected:
                print("⚠️ 无可用触角")
                continue
            print(f"🎲 激活 {len(selected)} 个触角进行评估...")
            img_data = explorer.evaluate_selected(selected)
            display_images([(p, a) for p, a, _ in img_data], explorer.generation, "全局探索")
            print("\n📝 评分 (-10~10)，可用指令: !save !ban !ban 描述 !list_bans !unban id !protect !scout 容量 !promote id !spread x y")
            scores = []
            i = 0
            while i < len(img_data):
                path, astr, idx = img_data[i]
                s = input(f"触角 {i+1}: ").strip()
                if s.startswith('!'):
                    parts = s.split()
                    cmd = parts[0].lower()
                    if cmd == '!save':
                        t = explorer.tentacles[idx]
                        save_favorite(astr, t.vector)
                    elif cmd == '!ban' and len(parts) == 1:
                        explorer.ban_current_tentacle(idx)
                        scores.append(0.0)
                        i += 1
                    elif cmd == '!ban' and len(parts) > 1:
                        desc = s[5:].strip()
                        if desc:
                            try:
                                print("🧬 编码描述并封禁...")
                                vec = encode_text_to_vector(desc)
                                add_ban(desc, vec, "manual_text", explorer.generation)
                            except Exception as e:
                                print(f"❌ 封禁失败: {e}")
                    elif cmd == '!list_bans':
                        list_bans_command()
                    elif cmd == '!unban':
                        try:
                            unban_id = int(parts[1])
                            ban_list = load_ban_list()
                            found = False
                            for b in ban_list:
                                if b['id'] == unban_id:
                                    b['active'] = False
                                    save_ban_list(ban_list)
                                    print(f"🔓 已解封 ID {unban_id}")
                                    found = True
                                    break
                            if not found:
                                print(f"❌ 未找到 ID {unban_id}")
                        except:
                            print("用法: !unban <id>")
                    elif cmd == '!protect':
                        t = explorer.tentacles[idx]
                        new_id = len(protect_zones)
                        zone = ProtectZone(new_id, t.vector.copy(), 0.15, MIXED_ARTISTS_COUNT)
                        protect_zones[new_id] = zone
                        save_protect_zones(protect_zones)
                        print(f"🛡️ 已创建并保存保护区_{new_id}")
                    elif cmd == '!scout':
                        if len(parts) < 2:
                            print("❌ 用法: !scout <容量>，如 !scout 3")
                        else:
                            try:
                                capacity = int(parts[1])
                                t = explorer.tentacles[idx]
                                new_id = max([sp['id'] for sp in explorer.scout_points], default=-1) + 1
                                explorer.scout_points.append({
                                    "id": new_id,
                                    "vector": t.vector.tolist(),
                                    "capacity": capacity,
                                    "assigned": [],
                                    "active": True
                                })
                                save_scout_points(explorer.scout_points)
                                print(f"🔭 已创建探测点 {new_id}，容量 {capacity}")
                            except:
                                print("容量需为整数")
                    elif cmd == '!promote':
                        try:
                            promote_idx = int(parts[1])
                            explorer.promote_tentacle(promote_idx)
                        except:
                            print("用法: !promote <触角ID>")
                    elif cmd == '!spread':
                        try:
                            x = int(parts[1])
                            y = float(parts[2])
                            explorer.spread_weakest_tentacles(x, y)
                        except:
                            print("用法: !spread <触角数> <边缘强度>")
                    else:
                        print("未知指令")
                    continue
                try:
                    sc = float(s)
                    if -10 <= sc <= 10:
                        scores.append(sc)
                        i += 1
                except ValueError:
                    print("无效输入")
            explorer.update_tentacles(selected, scores)
            if explorer.generation % 3 == 0:
                explorer.exchange_info()
            plot_tentacles(explorer, protect_zones)
        elif choice == '2':
            if not protect_zones:
                print("⚠️ 尚无保护区，请先在全局探索中使用 !protect 创建")
                continue
            list_protect_zones_command(protect_zones)
            zid_input = input("选择要激活的保护区ID: ").strip()
            try:
                zid = int(zid_input)
                if zid in protect_zones:
                    mixed_input = input(f"混合画师数量 (默认{MIXED_ARTISTS_COUNT}, 1-5): ").strip()
                    mixed_count = MIXED_ARTISTS_COUNT
                    if mixed_input:
                        try:
                            mixed_count = int(mixed_input)
                            mixed_count = max(1, min(5, mixed_count))
                        except:
                            pass
                    old_zone = protect_zones[zid]
                    zone = ProtectZone(zid, old_zone.center_vector, old_zone.radius, mixed_count)
                    protect_zones[zid] = zone
                    save_protect_zones(protect_zones)
                    active_zone_id = zid
                    print(f"🛡️ 保护区_{zid} 已激活，开始培育...")
                    protect_cultivate_menu(explorer, zone, protect_zones)
                else:
                    print("❌ 无效的保护区ID")
            except ValueError:
                print("❌ 请输入数字ID")
        elif choice == '3':
            list_protect_zones_command(protect_zones)
        elif choice == '4':
            list_bans_command()
            action = input("输入 !unban <id> 解封，或按回车返回: ").strip()
            if action.startswith('!unban'):
                try:
                    unban_id = int(action.split()[1])
                    ban_list = load_ban_list()
                    for b in ban_list:
                        if b['id'] == unban_id:
                            b['active'] = False
                            save_ban_list(ban_list)
                            print(f"🔓 已解封 ID {unban_id}")
                            break
                except:
                    print("用法: !unban <id>")
        elif choice == '5':
            list_scouts_command(explorer)
            action = input("输入 !remove_scout <id> 删除，或按回车返回: ").strip()
            if action.startswith('!remove_scout'):
                try:
                    sid = int(action.split()[1])
                    for sp in explorer.scout_points:
                        if sp['id'] == sid:
                            sp['active'] = False
                            save_scout_points(explorer.scout_points)
                            print(f"🗑️ 探测点 {sid} 已标记删除")
                            break
                except:
                    print("用法: !remove_scout <id>")
        elif choice == '6':
            # 手动转正
            idx_input = input("输入要转正的触角ID: ").strip()
            try:
                idx = int(idx_input)
                explorer.promote_tentacle(idx)
            except:
                print("❌ 无效ID")
        elif choice == '0':
            save_protect_zones(protect_zones)
            save_scout_points(explorer.scout_points)
            print("💾 保护区和探测点已保存")
            print("👋 感谢使用，再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")

def protect_cultivate_menu(explorer: SlimeMoldExplorerV6, zone: ProtectZone, all_zones: Dict[int, ProtectZone]):
    while True:
        print(f"\n🛡️ 保护区_{zone.zone_id} 培育菜单")
        print(f"   最佳评分: {zone.best_score:.2f} | 连续无提升: {zone.no_improve_rounds}轮")
        if zone.best_artist_str:
            print(f"   最佳画师串: {zone.best_artist_str[:80]}...")
        print("  1. 继续培育 (一轮)")
        print("  2. 查看最佳菌丝并预览")
        print("  3. 将最佳菌丝释放为全局触角")
        print("  4. 返回主菜单")
        choice = input("请选择: ").strip()
        if choice == '1':
            zone.no_improve_rounds = 0
            converged = zone.optimize(n_calls=1)
            save_protect_zones(all_zones)
            if converged:
                print(f"✅ 保护区_{zone.zone_id} 已收敛")
                if zone.best_artist_str:
                    print(f"🏆 最优菌丝画师串: {zone.best_artist_str}")
                release = input("是否将最优菌丝释放为全局触角？(y/n): ").strip().lower()
                if release == 'y' and zone.best_vector is not None:
                    explorer.replace_weakest_with_vector(zone.best_vector)
                    print("✅ 最优菌丝已释放到全局探索池")
                return
        elif choice == '2':
            if zone.best_artist_str:
                print(f"🏆 保护区_{zone.zone_id} 最优菌丝:")
                print(f"   画师串: {zone.best_artist_str}")
                print(f"   评分: {zone.best_score:.2f}")
                if zone.best_vector is not None:
                    print("🖼️ 正在生成预览图...")
                    img_path = generate_single_image(zone.best_vector, zone.generation, f"best_zone{zone.zone_id}")
                    if img_path:
                        print(f"   预览图: {img_path}")
            else:
                print("⚠️ 尚未进行培育")
        elif choice == '3':
            if zone.best_vector is not None:
                explorer.replace_weakest_with_vector(zone.best_vector)
                print("✅ 最优菌丝已释放到全局探索池")
            else:
                print("⚠️ 尚无最优菌丝可释放")
        elif choice == '4':
            save_protect_zones(all_zones)
            print("💾 保护区已保存")
            return
        else:
            print("❌ 无效选择")

# ===============================================
# 程序入口
# ===============================================
def main():
    print("\n🧫 黏菌风格探索器 V6 - 弱触角与探测点")
    print("   新功能: 弱触角自动分配探测点 | 绳子拉力 | 手动转正")
    print("=" * 60)
    spread_input = input("请设定初始触角扩散程度 (0=集中一个簇, 1=完全分散, 默认0.5): ").strip()
    try:
        spread = float(spread_input) if spread_input else 0.5
        spread = max(0.0, min(1.0, spread))
    except ValueError:
        spread = 0.5
    print(f"🌱 初始扩散度 = {spread:.2f}")
    protect_zones = load_protect_zones()
    if protect_zones:
        print(f"📂 已加载 {len(protect_zones)} 个历史保护区")
    explorer = SlimeMoldExplorerV6(n_tentacles=N_TENTACLES, eval_budget=EVAL_BUDGET, spread=spread)
    main_menu(explorer, protect_zones)

if __name__ == "__main__":
    main()