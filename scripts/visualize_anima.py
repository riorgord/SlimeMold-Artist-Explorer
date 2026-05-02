import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

# ========== 配置 ==========
VECTORS_NPY = "./outputs_anima/artist_vectors_anima.npy"
IDS_JSON = "./outputs_anima/artist_ids_anima.json"
METADATA_CSV = "./outputs_anima/artist_metadata_anima.csv"
OUTPUT_DIR = Path("./outputs_anima")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载数据 ==========
print("📂 加载 Anima 向量库...")
vectors = np.load(VECTORS_NPY).astype(np.float32)
with open(IDS_JSON, 'r') as f:
    artist_ids = json.load(f)
metadata_df = pd.read_csv(METADATA_CSV)
id_to_name = dict(zip(metadata_df['artist_id'], metadata_df['artist_name']))

print(f"✅ 画师数量: {len(artist_ids)}，向量维度: {vectors.shape[1]}")

# ========== PCA 降维 ==========
print("🔧 执行 PCA...")
pca = PCA(n_components=2, random_state=42)
vectors_2d = pca.fit_transform(vectors)
print(f"   PCA 解释方差比: {pca.explained_variance_ratio_}")

# ========== KMeans 聚类 ==========
n_clusters = 20
print(f"🔧 执行 KMeans 聚类 (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(vectors)

# ========== 保存聚类数据 ==========
df_pca = pd.DataFrame({
    'artist_id': artist_ids,
    'pc1': vectors_2d[:, 0],
    'pc2': vectors_2d[:, 1],
    'cluster': labels
})
df_pca = df_pca.merge(metadata_df[['artist_id', 'artist_name']], on='artist_id', how='left')
df_pca.to_csv(OUTPUT_DIR / "artist_pca_clusters_anima.csv", index=False, encoding='utf-8')
print("💾 PCA 聚类数据已保存")

# ========== 绘图 ==========
print("🎨 绘制风格地图...")
plt.figure(figsize=(16, 12))
scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='tab20', alpha=0.6, s=12, edgecolors='none')

# 标注簇中心
centers_2d = pca.transform(kmeans.cluster_centers_)
for i, (x, y) in enumerate(centers_2d):
    plt.scatter(x, y, c='black', marker='X', s=200, edgecolors='white', linewidth=1.5)
    # 找最近的画师
    center_vec = kmeans.cluster_centers_[i]
    distances = np.linalg.norm(vectors - center_vec, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_id = artist_ids[nearest_idx]
    nearest_name = id_to_name.get(nearest_id, str(nearest_id))
    plt.annotate(nearest_name, (x, y), fontsize=9, fontweight='bold', color='navy',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray'), ha='center', va='bottom')

plt.colorbar(scatter, label='Cluster ID', ticks=range(n_clusters))
plt.title(f'Anima Artist Style Map (1024-dim, PCA 2D)\nn={len(artist_ids)} artists, {n_clusters} clusters', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "artist_style_map_anima.png", dpi=150, bbox_inches='tight')
print(f"💾 风格地图: {OUTPUT_DIR / 'artist_style_map_anima.png'}")
plt.show()