import torch
import numpy as np
import pandas as pd
import faiss
import json
from pathlib import Path
from tqdm import tqdm
import time

# ========== 配置 ==========
CONDITIONS_DIR = Path("<你的ComfyUI目录>/models/conditions")   # ⚠️ 改成你的 ComfyUI conditions 目录
METADATA_CSV = "./data/anima_vectors_metadata.csv"           # 批量提取元数据
OUTPUT_DIR = Path("./outputs_anima")                          # Anima 专用输出目录
OUTPUT_VECTORS_NPY = OUTPUT_DIR / "artist_vectors_anima.npy"
OUTPUT_IDS_JSON = OUTPUT_DIR / "artist_ids_anima.json"
OUTPUT_INDEX_FILE = OUTPUT_DIR / "artist_index_anima.faiss"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "artist_metadata_anima.csv"
OUTPUT_PCA_CSV = OUTPUT_DIR / "artist_pca_clusters_anima.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 1. 获取所有 ckpt 文件列表 ==========
print("🔍 扫描 anima_artist_*.ckpt 文件...")
ckpt_files = list(CONDITIONS_DIR.glob("anima_artist_*.ckpt"))
print(f"✅ 找到 {len(ckpt_files)} 个文件")

# ========== 2. 从 ckpt 提取 1024 维向量 ==========
def extract_vector_from_ckpt(ckpt_path: Path) -> np.ndarray:
    """提取 last_hidden_state 并平均池化得到 1024 维向量，归一化"""
    data = torch.load(ckpt_path, map_location='cpu')
    # 结构: [ ( tensor(1, seq_len, 1024), meta_dict ) ]
    last_hidden = data[0][0]                        # (1, seq_len, 1024)
    pooled = last_hidden.mean(dim=1).squeeze(0)      # (1024,)
    vector = pooled.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector

# ========== 3. 批量提取并保存 ==========
print("🧬 提取向量...")
vectors = []
valid_ids = []
failed = []

for ckpt_path in tqdm(ckpt_files):
    # 从文件名解析 artist_id
    stem = ckpt_path.stem                # e.g. "anima_artist_12345"
    try:
        artist_id = int(stem.split("_")[-1])
    except:
        print(f"⚠️ 无法解析 ID: {ckpt_path.name}")
        continue

    try:
        vec = extract_vector_from_ckpt(ckpt_path)
        vectors.append(vec)
        valid_ids.append(artist_id)
    except Exception as e:
        print(f"❌ {artist_id} 失败: {e}")
        failed.append(artist_id)

vectors_matrix = np.stack(vectors, axis=0).astype(np.float32)
print(f"✅ 成功提取 {len(valid_ids)} 个向量，形状: {vectors_matrix.shape}")

# 保存向量矩阵和 ID 列表
np.save(OUTPUT_VECTORS_NPY, vectors_matrix)
with open(OUTPUT_IDS_JSON, 'w', encoding='utf-8') as f:
    json.dump(valid_ids, f, ensure_ascii=False, indent=2)
print(f"💾 向量矩阵: {OUTPUT_VECTORS_NPY}")
print(f"💾 ID 列表: {OUTPUT_IDS_JSON}")

# ========== 4. 构建 FAISS 索引 ==========
print("🔍 构建 FAISS 索引...")
dim = 1024
index = faiss.IndexFlatIP(dim)
index.add(vectors_matrix)
faiss.write_index(index, str(OUTPUT_INDEX_FILE))
print(f"💾 FAISS 索引: {OUTPUT_INDEX_FILE}")

# ========== 5. 整理元数据 ==========
if Path(METADATA_CSV).exists():
    df_meta = pd.read_csv(METADATA_CSV)
    # 只保留成功提取的画师
    df_out = df_meta[df_meta['artist_id'].isin(valid_ids)].copy()
    df_out.to_csv(OUTPUT_METADATA_CSV, index=False, encoding='utf-8')
    print(f"💾 元数据: {OUTPUT_METADATA_CSV}")

# ========== 6. 生成 PCA 聚类数据（供可视化） ==========
print("📊 生成 PCA 聚类数据...")
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(vectors_matrix)

kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(vectors_matrix)

df_pca = pd.DataFrame({
    'artist_id': valid_ids,
    'pc1': coords_2d[:, 0],
    'pc2': coords_2d[:, 1],
    'cluster': clusters
})
df_pca.to_csv(OUTPUT_PCA_CSV, index=False, encoding='utf-8')
print(f"💾 PCA 数据: {OUTPUT_PCA_CSV}")

if failed:
    print(f"⚠️ 失败画师 ID: {failed[:20]}...")

print("\n🎉 Anima 向量索引构建完成！")