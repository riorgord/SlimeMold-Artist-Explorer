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
from typing import Dict, List, Tuple, Optional
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
# 用户可配置参数
# ===============================================

# ---- 向量库路径 (Anima 专用) ----
VECTOR_DIR = Path("./outputs_anima")
VECTORS_NPY = VECTOR_DIR / "artist_vectors_anima.npy"
IDS_JSON = VECTOR_DIR / "artist_ids_anima.json"
INDEX_PATH = VECTOR_DIR / "artist_index_anima.faiss"
METADATA_CSV = VECTOR_DIR / "artist_metadata_anima.csv"
CLUSTER_CSV = VECTOR_DIR / "artist_pca_clusters_anima.csv"

# ---- ComfyUI 连接 ----
COMFYUI_SERVER = "127.0.0.1:8188"
COMFYUI_OUTPUT_DIR = Path(".")

# ---- 生成参数 ----
BASE_POSITIVE_PROMPT = "1girl"
BASE_NEGATIVE_PROMPT = "nsfw,uncolored"
WIDTH = 1024
HEIGHT = 1024

# ---- 模式选择 ----
MODE = "turbo"  # "turbo" 或 "base"，启动时会询问确认
# 加速版 (turbo) 推荐参数
TURBO_STEPS = 12
TURBO_CFG = 1
TURBO_SAMPLER = "er_sde"
TURBO_SCHEDULER = "beta57"
# 标准版 (base) 推荐参数
BASE_STEPS = 30
BASE_CFG = 5
BASE_SAMPLER = "er_sde"
BASE_SCHEDULER = "beta57"
# 运行时使用的实际参数 (根据模式赋值)
STEPS = TURBO_STEPS if MODE == "turbo" else BASE_STEPS
CFG = TURBO_CFG if MODE == "turbo" else BASE_CFG
SAMPLER_NAME = TURBO_SAMPLER if MODE == "turbo" else BASE_SAMPLER
SCHEDULER = TURBO_SCHEDULER if MODE == "turbo" else BASE_SCHEDULER

# ---- 工作流模板文件（根据模式自动选择） ----
WORKFLOW_TEMPLATE_PATH = Path("workflows/t2i_anima_turbo.json") if MODE == "turbo" else Path("workflows/t2i_anima_base.json")

# ---- Anima UNet 模型 ----
UNET_NAME = "anima-base-v1.0.safetensors"

# ---- Anima 加速 LoRA（turbo 模式需要，从 Civitai 下载后放入 ComfyUI models/loras/） ----
TURBO_LORA_NAME = "anima-turbo-lora-v0.1.safetensors"
TURBO_LORA_URL = "https://civitai.com/models/2560840/anima-turbo-lora"

# ---- 黏菌探索器参数 ----
N_TENTACLES = 30
EVAL_BUDGET = 30
MAX_GENERATIONS = 50
SIMILARITY_THRESHOLD = 0.85
MIXED_ARTISTS_COUNT = 4
TEMPERATURE = 1.0                                       # 画师权重温度（>1 主画师更突出，<1 更均匀）

# ---- Ban区阈值 ----
BAN_SELECT_THRESH = 0.85
BAN_MUTATE_THRESH = 0.90
BAN_REBIRTH_THRESH = 0.80
BAN_PENALTY_COEFFICIENT = 2.0
BAN_DECAY_RATE = 0.7

# ---- 保护区参数 ----
PROTECT_POP_SIZE = 4
PROTECT_MAX_CANDIDATES = 15
PROTECT_CONVERGE_ROUNDS = 3

# ---- 弱触角与探测点参数 ----
WEAK_CONSECUTIVE_NEGATIVE = 4
PROMOTE_SUGGEST_ROUNDS = 3
PULL_STRENGTH = 0.2
MAX_WEAK_TENTACLES = 4

# ---- 触角步长 (1024维优化) ----
TENTACLE_STEP_SIZE = 0.085

# ---- Debug调试与可视化 ----
DEBUG_MODE = False                 # 调试模式（启动时 --debug 开启）
PLOT_STYLE = "lite"
TRACE_HISTORY_LEN = 5

# ---- 文件存储 ----
FAVORITES_FILE = Path("./data/favorites_anima.json")
BAN_LIST_FILE = Path("./data/ban_list_anima.json")
PROTECT_ZONES_DIR = Path("./data/protect_zones_anima")
SCOUT_POINTS_FILE = Path("./data/scout_points_anima.json")

# ---- Tag辅助 (从外部通过 WebUI 设置) ----
ENCODE_CLIP_PATH = ""
TAG_POS = ""
TAG_NEG = ""
TAG_GUIDANCE = 0.6  # 引导强度 (0=无引导, 1=最强)
TAG_POOL_SPREAD = 0.5
_LIB_PREFIX = None
FIXED_SEED = -1  # -1=随机(ComfyUI管理), 其他=固定种子

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
    """返回 Tag 方向向量 (1024-dim, L2归一化) 或 None。
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
        from engines.clip_encoder_anima import encode_text_local
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

_CACHED_TAG_DIR = None
_LAST_TAG_POS = ""
_LAST_TAG_NEG = ""
_LAST_CLIP_PATH = ""

# ---- 图片展示方式 ----
SHOW_IMAGES_IN_BROWSER = True

# ---- 模式提示（turbo 需手动下载 LoRA） ----
if MODE == "turbo":
    print(f"⚡ Anima 加速模式 (turbo)")
    print(f"   依赖 LoRA: {TURBO_LORA_NAME}")
    print(f"   下载: {TURBO_LORA_URL}")
    print(f"   如未安装 LoRA，ComfyUI 生成时会报错")
else:
    print(f"📌 Anima 标准模式 (base)")

# ===============================================
# 加载全库数据
# ===============================================
print("📂 加载 Anima 画师向量库...")
vectors_all = np.load(VECTORS_NPY).astype(np.float32)
with open(IDS_JSON, 'r') as f:
    artist_ids_all = json.load(f)
index = faiss.read_index(str(INDEX_PATH))
metadata_df = pd.read_csv(METADATA_CSV)
id_to_name = dict(zip(metadata_df['artist_id'], metadata_df['artist_name']))
name_to_id = {v: k for k, v in id_to_name.items()}
N_TOTAL_ARTISTS = len(artist_ids_all)
print(f"✅ 全库共有 {N_TOTAL_ARTISTS} 位画师，向量维度 1024")

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
os.makedirs("./data", exist_ok=True)
RUN_OUTPUT_DIR = Path(f"anima_slime_{RUN_TIMESTAMP}")
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
    """Anima 格式：@artist_name:weight"""
    if top_k is None:
        top_k = MIXED_ARTISTS_COUNT
    blended_vec = np.asarray(blended_vec, dtype=np.float32).flatten()
    search_k = max(top_k * 3, top_k + (len(skip_ids) if skip_ids else 0))
    distances, idxs = index.search(blended_vec.reshape(1, -1), search_k * 2)
    parts = []
    used_ids = set()
    if skip_ids:
        used_ids.update(skip_ids)
    ds = distances[0]
    d_min, d_max = ds.min(), ds.max()
    d_range = d_max - d_min
    for rank, (idx, dist) in enumerate(zip(idxs[0], ds)):
        artist_id = artist_ids_all[idx]
        if artist_id in used_ids:
            continue
        norm = (dist - d_min) / d_range if d_range > 0 else 1.0
        weight = max(0.01, norm ** TEMPERATURE)
        name = id_to_name.get(artist_id, str(artist_id))
        parts.append(f"(@{name}:{weight:.2f})")
        used_ids.add(artist_id)
        if len(parts) >= top_k:
            break
    return ", ".join(parts)

def encode_text_to_vector(text: str) -> np.ndarray:
    """通过 ComfyUI 编码文本，适用于 Anima"""
    wf = {
        "13": {
            "inputs": {"clip_name": CLIP_NAME if 'CLIP_NAME' in dir() else "qwen_3_06b_base.safetensors", "type": "qwen_image", "device": "default"},
            "class_type": "CLIPLoader"
        },
        "11": {
            "inputs": {"text": text, "clip": ["13", 0]},
            "class_type": "CLIPTextEncode"
        },
        "14": {
            "inputs": {"previewMode": None, "source": ["11", 0]},
            "class_type": "PreviewAny"
        }
    }
    r = requests.post(f"http://{COMFYUI_SERVER}/prompt", json={"prompt": wf})
    pid = r.json()['prompt_id']
    start = time.time()
    while time.time() - start < 60:
        hist = requests.get(f"http://{COMFYUI_SERVER}/history/{pid}").json()
        if pid in hist:
            out = hist[pid]['outputs'].get('14', {}).get('text', [''])
            m = re.search(r'tensor\(\[\[(.*?)\]\]\)', out[0], re.DOTALL)
            if m:
                nums = np.fromstring(m.group(1).replace('\n', '').replace(' ', ',').replace(',,', ','), sep=',')
                vec = nums.astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec
        time.sleep(0.5)
    raise TimeoutError("文本编码超时")

# ===============================================
# ComfyUI API
# ===============================================
def load_workflow_template() -> Dict:
    with open(WORKFLOW_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_workflow(positive_prompt: str, negative_prompt: str, seed: int, filename_prefix: str) -> Dict:
    wf = load_workflow_template()
    for node_id, node in wf.items():
        class_type = node.get('class_type', '')
        
        # UNet 模型
        if class_type == 'UNETLoader':
            node['inputs']['unet_name'] = UNET_NAME
        # 正面提示词（节点38）和负面提示词（节点39）
        elif class_type == 'CLIPTextEncode':
            if node_id == '38':
                node['inputs']['text'] = positive_prompt
            elif node_id == '39':
                node['inputs']['text'] = negative_prompt
        
        # 采样器（节点65）
        elif class_type == 'KSampler' and node_id == '65':
            node['inputs']['seed'] = seed
            node['inputs']['steps'] = STEPS
            node['inputs']['cfg'] = CFG
            node['inputs']['sampler_name'] = SAMPLER_NAME
            node['inputs']['scheduler'] = SCHEDULER

        # 空Latent图像（节点32）
        elif class_type == 'EmptyLatentImage':
            node['inputs']['width'] = WIDTH
            node['inputs']['height'] = HEIGHT
            node['inputs']['batch_size'] = 1
        
        # 保存图像（节点64）
        elif class_type == 'SaveImage':
            node['inputs']['filename_prefix'] = str(RUN_OUTPUT_DIR / filename_prefix)
    
    if DEBUG_MODE:
        print(f"[DEBUG] build_workflow: seed={seed}, steps={STEPS}, cfg={CFG}, prefix={filename_prefix}")
    return wf

def queue_prompt(workflow: Dict) -> str:
    r = requests.post(f"http://{COMFYUI_SERVER}/prompt", json={"prompt": workflow})
    r.raise_for_status()
    return r.json()['prompt_id']

def wait_for_prompt(prompt_id: str, timeout: int = 120) -> Optional[Dict]:
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"http://{COMFYUI_SERVER}/history/{prompt_id}")
        hist = resp.json()
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(1)
    return None

def generate_single_image(vector: np.ndarray, gen: int, tag: str) -> Optional[str]:
    prompt = f"{BASE_POSITIVE_PROMPT}, {vector_to_artist_string(vector)}"
    seed = FIXED_SEED if FIXED_SEED != -1 else (gen * 1000 + hash(tag) % 1000)
    if DEBUG_MODE:
        print(f"[DEBUG] generate_single_image: gen={gen}, tag={tag}, seed={seed}")
    fname = f"gen{gen}_{tag}"
    wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
    pid = queue_prompt(wf)
    res = wait_for_prompt(pid)
    if res:
        for out in res.get('outputs', {}).values():
            if 'images' in out:
                return str(RUN_OUTPUT_DIR / out['images'][0]['filename'])
    return None

# ===============================================
# HTML 画廊
# ===============================================
def display_images(image_data: List[Tuple[str, str]], gen: int, title: str = "黏菌 Anima 探索器"):
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
def load_ban_list() -> List[Dict]:
    if BAN_LIST_FILE.exists():
        with open(BAN_LIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_ban_list(ban_list: List[Dict]):
    with open(BAN_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(ban_list, f, indent=2, ensure_ascii=False)

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
    print(f"🚫 已封禁风格区域: {description[:60]}... (Ban ID: {ban_id})")
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
            b['penalty'] = b.get('penalty', 1.0) * BAN_DECAY_RATE
            if b['penalty'] < 0.2:
                b['active'] = False
                print(f"🔓 Ban区 ID {ban_id} 已被完全照亮，自动解封！")
            else:
                print(f"💡 Ban区 ID {ban_id} 衰减至 {b['penalty']:.2f}")
            save_ban_list(ban_list)
            break

# ===============================================
# 收藏夹管理
# ===============================================
def load_favorites() -> List[Dict]:
    if FAVORITES_FILE.exists():
        with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
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

def load_protect_zones() -> Dict[int, 'ProtectZone']:
    filepath = PROTECT_ZONES_DIR / "protect_zones.json"
    if not filepath.exists():
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return {}
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
        except:
            pass
    return protect_zones

# ===============================================
# 探测点管理
# ===============================================
def load_scout_points() -> List[Dict]:
    if SCOUT_POINTS_FILE.exists():
        with open(SCOUT_POINTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_scout_points(scouts: List[Dict]):
    with open(SCOUT_POINTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(scouts, f, indent=2, ensure_ascii=False)

# ===============================================
# 触角类
# ===============================================
class Tentacle:
    def __init__(self, indices: np.ndarray, weights: np.ndarray, birth_gen: int = 0):
        self.indices = np.array(indices, dtype=np.int64)
        self.weights = np.array(weights, dtype=np.float32)
        self.birth_gen = birth_gen
        self.last_eval_gen = 0
        self.score_history = []
        self.vector = blend_from_indices(self.indices, self.weights)
        self.step_size = TENTACLE_STEP_SIZE
        self.active = True
        self.trace = deque(maxlen=TRACE_HISTORY_LEN)
        self.is_weak = False
        self.scout_id = None

    def update_trace(self, gen: int):
        self.trace.append((gen, self.vector.copy()))

    def recent_avg_score(self, window: int = 3) -> float:
        if not self.score_history:
            return 0.0
        return np.mean([s for _, s in self.score_history[-window:]])

    def consecutive_negative_count(self) -> int:
        count = 0
        for _, s in reversed(self.score_history):
            if s < 0:
                count += 1
            else:
                break
        return count

    def consecutive_positive_count(self) -> int:
        count = 0
        for _, s in reversed(self.score_history):
            if s > 0:
                count += 1
            else:
                break
        return count

# ===============================================
# 全局探索器
# ===============================================
class SlimeMoldExplorerAnima:
    def __init__(self, n_tentacles=30, eval_budget=6, spread=0.5):
        self.n_tentacles = n_tentacles
        self.eval_budget = eval_budget
        self.generation = 0
        self.tentacles: List[Tentacle] = []
        self.global_best = None
        self.recent_evaluated_vectors = []
        self.scout_points = load_scout_points()
        self.init_tentacles(spread)

    def init_tentacles(self, spread: float):
        if not unique_clusters:
            clusters_available = [0]
        else:
            clusters_available = unique_clusters.copy()
        n_clusters_to_use = max(1, int(len(clusters_available) * spread + 0.5))
        selected_clusters = np.random.choice(clusters_available, size=n_clusters_to_use, replace=False)
        self.tentacles = []
        for i in range(self.n_tentacles):
            target_cluster = selected_clusters[i % n_clusters_to_use] if i < n_clusters_to_use else random.choice(clusters_available)
            candidates = [aid for aid, c in id_to_cluster.items() if c == target_cluster]
            if not candidates:
                start_id = random.choice(artist_ids_all)
            else:
                start_id = random.choice(candidates)
            start_idx = artist_ids_all.index(start_id)
            n_artists = np.random.randint(2, 4)  # 至少 2 个画师
            weights = np.ones(n_artists, dtype=np.float32) / n_artists
            indices = [start_idx] * n_artists if n_artists > 1 else [start_idx]
            # 注意：上面的写法有误，应该生成不同的画师，修正如下：
            # 重新生成：随机选取 n_artists 个不同的画师
            actual_indices = [start_idx]
            for _ in range(n_artists - 1):
                new_id = random.choice(artist_ids_all)
                while new_id in actual_indices:
                    new_id = random.choice(artist_ids_all)
                actual_indices.append(artist_ids_all.index(new_id))
            t = Tentacle(actual_indices, weights, birth_gen=0)
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
                        print(f"🔗 触角 {self.tentacles.index(t)} 标记为弱触角，绑定探测点 {sp['id']}")
                        break
                if not t.is_weak:
                    t.is_weak = True
                    t.scout_id = None
                    print(f"🔗 触角 {self.tentacles.index(t)} 标记为弱触角（无探测点绑定）")

    def _check_promote_suggestions(self):
        for i, t in enumerate(self.tentacles):
            if t.is_weak and t.consecutive_positive_count() >= PROMOTE_SUGGEST_ROUNDS:
                print(f"💡 触角 {i} 已连续 {PROMOTE_SUGGEST_ROUNDS} 轮正分，建议 !promote {i} 转正")

    def select_tentacles_to_evaluate(self) -> List[int]:
        self._assign_weak_tentacles()
        tag_dir = compute_tag_direction()
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
            if tag_dir is not None:
                sim = np.dot(t.vector, tag_dir)
                weight *= 1.0 + sim * TAG_GUIDANCE
            candidates.append((i, weight))
        if not candidates:
            return []
        idxs, wgts = zip(*candidates)
        probs = np.array(wgts) / np.sum(wgts)
        n_sample = min(EVAL_BUDGET, len(candidates))
        selected = np.random.choice(len(candidates), size=n_sample, p=probs, replace=False)
        return [candidates[s][0] for s in selected]

    def evaluate_selected(self, selected_indices: List[int]) -> List[Tuple[str, str, int]]:
        results = []
        for idx in selected_indices:
            t = self.tentacles[idx]
            prompt = f"{BASE_POSITIVE_PROMPT}, {vector_to_artist_string(t.vector)}"
            fname = f"gen{self.generation:02d}_t{idx:03d}"
            seed = FIXED_SEED if FIXED_SEED != -1 else (self.generation * 100 + idx)
            if DEBUG_MODE:
                print(f"[DEBUG]   tentacle #{idx}: seed={seed}, file={fname}")
            wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
            pid = queue_prompt(wf)
            res = wait_for_prompt(pid)
            img_path = None
            if res:
                for out in res.get('outputs', {}).values():
                    if 'images' in out:
                        img_path = str(RUN_OUTPUT_DIR / out['images'][0]['filename'])
                        break
            results.append((img_path, vector_to_artist_string(t.vector), idx))
            time.sleep(0.3)
        return results

    def update_tentacles(self, selected_indices: List[int], scores: List[float]):
        for idx, score in zip(selected_indices, scores):
            t = self.tentacles[idx]
            t.last_eval_gen = self.generation
            t.score_history.append((self.generation, score))
            if self.global_best is None or score > self.global_best[2]:
                self.global_best = (t.indices.copy(), t.weights.copy(), score)
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
        self._check_promote_suggestions()
        self._prune_rebirth()
        self._apply_weak_repulsion()
        self._auto_long_jump_if_needed()

        unique_artists = set()
        for t in self.tentacles:
            if t.active:
                for idx in t.indices:
                    unique_artists.add(idx)
        print(f"  [多样性] 活跃触角画师数: {len(unique_artists)}")

    def _grow(self, t: Tentacle, score: float):
        if t.is_weak:
            effective_step = t.step_size * 0.3
            if t.scout_id is not None:
                scout = next((sp for sp in self.scout_points if sp['id'] == t.scout_id), None)
                if scout:
                    scout_vec = np.array(scout['vector'], dtype=np.float32)
                    random_dir = np.random.randn(1024).astype(np.float32)
                    random_dir /= np.linalg.norm(random_dir)
                    toward_scout = scout_vec - t.vector
                    toward_norm = np.linalg.norm(toward_scout)
                    if toward_norm > 0:
                        toward_scout /= toward_norm
                    else:
                        toward_scout = random_dir
                    move_dir = (1 - PULL_STRENGTH) * random_dir + PULL_STRENGTH * toward_scout
                    move_dir /= np.linalg.norm(move_dir)
                    noise_val = move_dir * effective_step
                    new_vec = t.vector + noise_val
                    new_vec /= np.linalg.norm(new_vec)
                    near, _, _ = is_near_ban(new_vec, BAN_MUTATE_THRESH)
                    if near:
                        new_vec = t.vector - noise_val * 0.5
                        new_vec /= np.linalg.norm(new_vec)
                    t.vector = new_vec.astype(np.float32)
        # 正常权重更新
        noise_w = np.random.normal(0, t.step_size * 0.5, t.weights.shape)
        new_w = t.weights + noise_w
        new_w = np.clip(new_w, 0.05, None)
        new_w /= np.sum(new_w)
        t.weights = new_w
        new_vec = blend_from_indices(t.indices, t.weights)
        near, _, _ = is_near_ban(new_vec, BAN_MUTATE_THRESH)
        if near:
            t.weights -= noise_w * 0.5
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
                print(f"  🌟 触角引入新画师: {id_to_name.get(artist_ids_all[new_idx], '?')}")
        # 低分强制换血
        if score <= 2 and np.random.rand() < 0.35 and len(t.indices) > 0:
            remove_idx = np.random.randint(len(t.indices))
            new_artist = np.random.choice(N_TOTAL_ARTISTS)
            while new_artist in t.indices:
                new_artist = np.random.choice(N_TOTAL_ARTISTS)
            t.indices[remove_idx] = new_artist
            t.weights = np.ones(len(t.indices), dtype=np.float32) / len(t.indices)
            t.vector = blend_from_indices(t.indices, t.weights)

    def _shrink(self, t: Tentacle, score: float):
        if len(t.indices) > 1:
            min_pos = np.argmin(t.weights)
            t.indices = np.delete(t.indices, min_pos)
            t.weights = np.delete(t.weights, min_pos)
            t.weights /= np.sum(t.weights)
            t.vector = blend_from_indices(t.indices, t.weights)
        elif score < -5:
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
        for i in inactive_indices:
            attempts = 0
            while attempts < 50:
                n = np.random.randint(2, 4)
                new_idx = np.random.choice(N_TOTAL_ARTISTS, n, replace=False)
                new_w = np.random.rand(n)
                new_w /= np.sum(new_w)
                new_vec = blend_from_indices(new_idx, new_w)
                near, _, _ = is_near_ban(new_vec, BAN_REBIRTH_THRESH)
                if not near:
                    break
                attempts += 1
            self.tentacles[i] = Tentacle(new_idx, new_w, birth_gen=self.generation)
            self.tentacles[i].update_trace(self.generation)

    def _apply_weak_repulsion(self):
        active_indices = [i for i, t in enumerate(self.tentacles) if t.active and not t.is_weak]
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
        active_indices = [i for i, t in enumerate(self.tentacles) if t.active and not t.is_weak]
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
                    n = np.random.randint(2, 4)
                    new_idx = np.random.choice(N_TOTAL_ARTISTS, n, replace=False)
                    new_w = np.random.rand(n)
                    new_w /= np.sum(new_w)
                    new_vec = blend_from_indices(new_idx, new_w)
                    near, _, _ = is_near_ban(new_vec, BAN_REBIRTH_THRESH)
                    if not near:
                        break
                    attempts += 1
                self.tentacles[original_idx] = Tentacle(new_idx, new_w, birth_gen=self.generation)
                self.tentacles[original_idx].update_trace(self.generation)
            print(f"  🦘 多样性过低，已触发 {n_jump} 次长跳")

    def exchange_info(self):
        active_ts = [t for t in self.tentacles if t.active and not t.is_weak]
        if len(active_ts) < 2:
            return
        vecs = np.array([t.vector for t in active_ts])
        tmp_idx = faiss.IndexFlatIP(1024)
        tmp_idx.add(vecs)
        for i, t in enumerate(active_ts):
            _, neigh = tmp_idx.search(vecs[i:i+1], 3)
            for j in neigh[0]:
                if j != i and np.dot(vecs[i], vecs[j]) > SIMILARITY_THRESHOLD:
                    other = active_ts[j]
                    combined = list(set(t.indices) | set(other.indices))[:MIXED_ARTISTS_COUNT]
                    t.indices = np.array(combined, dtype=np.int64)
                    t.weights = np.ones(len(combined), dtype=np.float32) / len(combined)
                    t.vector = blend_from_indices(t.indices, t.weights)

    def ban_current_tentacle(self, tentacle_idx: int):
        t = self.tentacles[tentacle_idx]
        add_ban(vector_to_artist_string(t.vector), t.vector, "current_tentacle", self.generation)
        t.active = False

    def spread_weakest_tentacles(self, x: int, y: float):
        active = [t for t in self.tentacles if t.active and not t.is_weak]
        if len(active) < 3:
            print("⚠️ 活跃触角不足")
            return
        sorted_active = sorted(active, key=lambda t: t.recent_avg_score())
        victims = sorted_active[:min(x, len(sorted_active))]
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
                random_dir = np.random.randn(1024).astype(np.float32)
                random_dir -= np.dot(random_dir, centroid) * centroid
                dir_norm = np.linalg.norm(random_dir)
                if dir_norm > 0:
                    random_dir /= dir_norm
                new_vec = centroid + target_dist * random_dir
                new_vec /= np.linalg.norm(new_vec)
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
        print(f"🦘 已扩散 {len(victims)} 个触角到边缘")

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
        return True

    def promote_tentacle(self, idx: int):
        if idx < 0 or idx >= len(self.tentacles):
            print("❌ 无效触角ID")
            return
        t = self.tentacles[idx]
        if not t.is_weak:
            print("❌ 该触角不是弱触角")
            return
        if t.scout_id is not None:
            for sp in self.scout_points:
                if sp['id'] == t.scout_id:
                    if idx in sp.get('assigned', []):
                        sp['assigned'].remove(idx)
                    break
        t.is_weak = False
        t.scout_id = None
        save_scout_points(self.scout_points)
        print(f"✅ 触角 {idx} 已转正")

    def stats(self) -> Tuple[int, int, int]:
        active = [t for t in self.tentacles if t.active]
        artists = set()
        clusters = set()
        for t in active:
            for idx in t.indices:
                artists.add(artist_ids_all[idx])
                clusters.add(id_to_cluster.get(artist_ids_all[idx], -1))
        return len(active), len(artists), len(clusters)

# ===============================================
# 保护区类
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
        print(f"🛡️ 保护区_{zone_id} 初始化完成，候选画师={n_candidates}个")

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
        return min(total_penalty, 10.0)

    def objective_function(self, weights) -> float:
        weights = np.asarray(weights, dtype=np.float32).flatten()
        top_indices = np.argsort(weights)[-self.mixed_count:]
        selected_indices = [self.candidate_artists[int(i)] for i in top_indices]
        selected_weights = weights[top_indices]
        selected_weights /= np.sum(selected_weights)
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
            s = input(f"菌丝评分 (-10~10，Ban区软斥力: {ban_pen:.2f}): ").strip()
            try:
                score = float(s)
                if -10 <= score <= 10:
                    break
            except:
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
        space = [Real(0.0, 1.0) for _ in range(n_candidates)]
        for _ in range(n_calls):
            if self.no_improve_rounds >= PROTECT_CONVERGE_ROUNDS:
                print(f"✅ 保护区_{self.zone_id} 已收敛")
                return True
            try:
                if self.optimizer_result is None:
                    result = gp_minimize(self.objective_function, space, n_calls=1, n_initial_points=1, random_state=self.generation)
                else:
                    result = gp_minimize(self.objective_function, space, n_calls=1, n_initial_points=0, x0=self.optimizer_result.x, y0=[-self.best_score], random_state=self.generation)
                self.optimizer_result = result
            except Exception as e:
                print(f"❌ 贝叶斯优化出错: {e}")
                return False
        return False

    # ------------------------------------------------------------
    # 进化培育 API（用于 WebUI 逐轮培育）
    # ------------------------------------------------------------
    def _weights_to_candidate(self, weights: np.ndarray) -> Dict:
        top_k = min(self.mixed_count, len(weights))
        top_i = np.argsort(weights)[-top_k:]
        sel_indices = [self.candidate_artists[int(i)] for i in top_i]
        sel_weights = weights[top_i]; sel_weights = sel_weights / np.sum(sel_weights)
        blended = blend_from_indices(np.array(sel_indices), sel_weights)
        artist_str = vector_to_artist_string(blended, top_k=self.mixed_count)
        ban_pen = self.ban_penalty(blended)
        return {"artist_str": artist_str, "blended": blended, "ban_penalty": float(ban_pen)}

    def _mutate_weights(self, weights: np.ndarray, strength: float = 0.15) -> np.ndarray:
        w = weights + np.random.normal(0, strength, len(weights)); w = np.clip(w, 0.01, 1.0)
        return (w / np.sum(w)).astype(np.float32)

    def _crossover_weights(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        alpha = np.random.random(); w = alpha * w1 + (1 - alpha) * w2
        return w.astype(np.float32)

    def evo_init(self, pop_size: int) -> List[Dict]:
        population = []
        for i in range(pop_size):
            w = np.random.rand(len(self.candidate_artists)); w = w / np.sum(w)
            c = self._weights_to_candidate(w); c['weights'] = w; c['id'] = i; population.append(c)
        self._evo_population = population; self.generation = 0
        return population

    def evo_submit_scores(self, scores: Dict[int, float], n_parents: int) -> Dict:
        pop_size = len(self._evo_population); n_parents = min(n_parents, pop_size)
        evals = []
        for i, score in scores.items():
            if i >= pop_size: continue
            pen = self.ban_penalty(self._evo_population[i]['blended']); eff = score - pen
            evals.append((i, eff, score)); self.score_history.append(eff)
        evals.sort(key=lambda x: x[1], reverse=True)
        best_i, best_eff, _ = evals[0]
        if best_eff > self.best_score:
            self.best_score = best_eff; self.best_artist_str = self._evo_population[best_i]['artist_str']
            self.best_vector = self._evo_population[best_i]['blended'].copy(); self.no_improve_rounds = 0
            if scores.get(best_i, 0) >= 5:
                near, sim, ban_entry = is_near_ban(self._evo_population[best_i]['blended'], BAN_SELECT_THRESH)
                if near and sim < BAN_MUTATE_THRESH and ban_entry: decay_ban_penalty(ban_entry['id'])
        else: self.no_improve_rounds += 1
        parent_ids = {p[0] for p in evals[:n_parents]}
        for i in range(pop_size): self._evo_population[i]['is_parent'] = i in parent_ids
        parent_weights = [self._evo_population[p[0]]['weights'] for p in evals[:n_parents]]
        n_crossover = max(1, pop_size - n_parents * max(1, pop_size // n_parents))
        new_pop = []; kid_id = 0
        for pw in parent_weights:
            for _ in range(max(1, (pop_size - n_crossover) // len(parent_weights))):
                if kid_id >= pop_size - n_crossover: break
                m = self._mutate_weights(pw); c = self._weights_to_candidate(m)
                c['weights'] = m; c['id'] = kid_id; new_pop.append(c); kid_id += 1
        for _ in range(n_crossover):
            if len(parent_weights) >= 2:
                i1, i2 = np.random.choice(len(parent_weights), 2, replace=False)
                x = self._crossover_weights(parent_weights[i1], parent_weights[i2])
                c = self._weights_to_candidate(x); c['weights'] = x; c['id'] = kid_id; new_pop.append(c); kid_id += 1
        while len(new_pop) < pop_size:
            pw = parent_weights[np.random.randint(len(parent_weights))]
            m = self._mutate_weights(pw); c = self._weights_to_candidate(m)
            c['weights'] = m; c['id'] = kid_id; new_pop.append(c); kid_id += 1
        self._evo_population = new_pop; self.generation += 1
        converged = self.no_improve_rounds >= PROTECT_CONVERGE_ROUNDS
        return {"parents": [p[0] for p in evals[:n_parents]], "best_score": float(self.best_score),
                "converged": converged, "generation": self.generation, "no_improve_rounds": self.no_improve_rounds,
                "best_artist_str": self.best_artist_str}

# ===============================================
# 可视化
# ===============================================
def plot_tentacles(explorer: SlimeMoldExplorerAnima, protect_zones: Dict[int, ProtectZone] = None):
    if not DEBUG_MODE or pca_model is None:
        return
    active = [t for t in explorer.tentacles if t.active]
    if not active:
        return
    plt.figure(figsize=(10, 8))
    if PLOT_STYLE == "full" and pca_vectors_2d is not None:
        plt.scatter(pca_vectors_2d[:, 0], pca_vectors_2d[:, 1], c='lightgray', alpha=0.3, s=5)
    for t in active:
        if len(t.trace) >= 2:
            pts = [pca_model.transform(vec.reshape(1, -1))[0] for _, vec in t.trace]
            xs, ys = zip(*pts)
            alphas = np.linspace(0.2, 1.0, len(xs))
            for i in range(len(xs) - 1):
                plt.plot(xs[i:i+2], ys[i:i+2], color='gray', alpha=alphas[i], linewidth=1)
    scores = [t.recent_avg_score() for t in active]
    norm_scores = (np.array(scores) + 10) / 20
    colors = plt.cm.RdYlGn(norm_scores)[:, :3]
    t_vecs = np.array([t.vector for t in active])
    t_2d = pca_model.transform(t_vecs)
    weak_mask = [t.is_weak for t in active]
    if any(weak_mask):
        weak_idx = np.where(weak_mask)[0]
        plt.scatter(t_2d[weak_idx, 0], t_2d[weak_idx, 1], c='yellow', s=120, marker='D', edgecolors='black', linewidth=1.5, label='弱触角')
    normal_mask = [not t.is_weak for t in active]
    if any(normal_mask):
        norm_idx = np.where(normal_mask)[0]
        plt.scatter(t_2d[norm_idx, 0], t_2d[norm_idx, 1], c=colors[norm_idx], s=100, marker='X', edgecolors='white', linewidth=1.5)
    ban_vecs, _ = get_active_bans()
    if ban_vecs:
        ban_2d = pca_model.transform(ban_vecs)
        plt.scatter(ban_2d[:, 0], ban_2d[:, 1], c='red', s=150, marker='o', alpha=0.6, label='Ban区')
    if protect_zones:
        for zid, zone in protect_zones.items():
            zone_2d = pca_model.transform(zone.center_vector.reshape(1, -1))[0]
            plt.scatter(zone_2d[0], zone_2d[1], c='cyan', s=200, marker='*', alpha=0.8, label=f'保护区_{zid}')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    
    plt.title(f"Anima 触角分布 (第{explorer.generation}代)")
    plot_path = RUN_OUTPUT_DIR / f"tentacles_gen{explorer.generation:02d}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"📊 触角分布图已保存: {plot_path}")

# ===============================================
# 主菜单
# ===============================================
def main():
    print("\n🧫 黏菌风格探索器 - Anima 版")
    print("=" * 60)
    spread_input = input("初始扩散程度 (0=集中, 1=分散, 默认0.5): ").strip()
    try:
        spread = float(spread_input) if spread_input else 0.5
        spread = max(0.0, min(1.0, spread))
    except:
        spread = 0.5

    protect_zones = load_protect_zones()
    if protect_zones:
        print(f"📂 已加载 {len(protect_zones)} 个历史保护区")

    explorer = SlimeMoldExplorerAnima(n_tentacles=N_TENTACLES, eval_budget=EVAL_BUDGET, spread=spread)

    while True:
        print("\n" + "=" * 60)
        print("Anima 主菜单")
        act_cnt, art_cnt, clus_cnt = explorer.stats()
        print(f"活跃触角: {act_cnt} | 覆盖画师: {art_cnt} | 簇: {clus_cnt}")
        print("1. 全局探索一轮 | 2. 保护区培育 | 3. 管理保护区 | 4. Ban区 | 0. 退出")
        choice = input("选择: ").strip()

        if choice == '1':
            explorer.generation += 1
            selected = explorer.select_tentacles_to_evaluate()
            if not selected:
                continue
            img_data = explorer.evaluate_selected(selected)
            display_images([(p, a) for p, a, _ in img_data], explorer.generation)
            scores = []
            i = 0
            while i < len(img_data):
                s = input(f"触角 {i+1}: ").strip()
                if s.startswith('!'):
                    parts = s.split()
                    cmd = parts[0].lower()
                    if cmd == '!save':
                        t = explorer.tentacles[img_data[i][2]]
                        save_favorite(img_data[i][1], t.vector)
                    elif cmd == '!ban' and len(parts) == 1:
                        explorer.ban_current_tentacle(img_data[i][2])
                        scores.append(0.0)
                        i += 1
                    elif cmd == '!protect':
                        t = explorer.tentacles[img_data[i][2]]
                        new_id = len(protect_zones)
                        protect_zones[new_id] = ProtectZone(new_id, t.vector.copy(), 0.15, MIXED_ARTISTS_COUNT)
                        save_protect_zones(protect_zones)
                    elif cmd == '!scout':
                        try:
                            cap = int(parts[1])
                            t = explorer.tentacles[img_data[i][2]]
                            new_id = max([sp['id'] for sp in explorer.scout_points], default=-1) + 1
                            explorer.scout_points.append({"id": new_id, "vector": t.vector.tolist(), "capacity": cap, "assigned": [], "active": True})
                            save_scout_points(explorer.scout_points)
                            print(f"🔭 探测点 {new_id} 已创建")
                        except:
                            print("用法: !scout 容量")
                    elif cmd == '!spread':
                        try:
                            explorer.spread_weakest_tentacles(int(parts[1]), float(parts[2]))
                        except:
                            print("用法: !spread x y")
                    continue
                try:
                    sc = float(s)
                    if -10 <= sc <= 10:
                        scores.append(sc)
                        i += 1
                except:
                    print("无效输入")
            explorer.update_tentacles(selected, scores)
            plot_tentacles(explorer, protect_zones)

        elif choice == '2':
            if not protect_zones:
                print("暂无保护区")
                continue
            zid = int(input("保护区ID: "))
            if zid in protect_zones:
                zone = protect_zones[zid]
                zone.no_improve_rounds = 0
                zone.optimize(n_calls=1)
                save_protect_zones(protect_zones)

        elif choice == '3':
            if not protect_zones:
                print("🛡️ 暂无保护区")
            else:
                print("🛡️ 当前保护区列表:")
                for zid, zone in protect_zones.items():
                    center_str = vector_to_artist_string(zone.center_vector, top_k=1)[:40] if zone.center_vector is not None else "未知"
                    print(f"  ID {zid}: 中心≈{center_str}，候选={len(zone.candidate_artists)}画师，最佳分={zone.best_score:.2f}")
            input("按回车返回主菜单...")

        elif choice == '4':
            _, active_bans = get_active_bans()
            if active_bans:
                print("📋 当前Ban区列表:")
                for b in active_bans:
                    penalty = b.get('penalty', 1.0)
                    print(f"  🟢 ID {b['id']}: {b['description'][:50]} (惩罚强度: {penalty:.2f}, 代数: {b['created_gen']})")
            else:
                print("📋 暂无Ban区")
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

        elif choice == '0':
            save_protect_zones(protect_zones)
            save_scout_points(explorer.scout_points)
            print("已保存，再见！")
            break

if __name__ == "__main__":
    main()