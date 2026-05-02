import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

# ===============================================
# 配置路径
# ===============================================
VECTORS_NPY = "./outputs_2048/artist_vectors_2048.npy"
IDS_JSON = "./outputs_2048/artist_ids_2048.json"
METADATA_CSV = "./outputs_2048/artist_metadata_2048.csv"
OUTPUT_DIR = Path("./outputs_2048")
OUTPUT_PLOT = OUTPUT_DIR / "artist_style_map_2048.png"

# ===============================================
# 加载数据
# ===============================================
print("📂 加载向量数据...")
vectors = np.load(VECTORS_NPY)  # shape (N, 2048)
with open(IDS_JSON, 'r') as f:
    artist_ids = json.load(f)
metadata_df = pd.read_csv(METADATA_CSV)

# 建立 id -> name 的映射
id_to_name = dict(zip(metadata_df['artist_id'], metadata_df['artist_name']))

print(f"✅ 向量矩阵形状: {vectors.shape}")
print(f"✅ 画师数量: {len(artist_ids)}")

# ===============================================
# PCA 降维到 2D
# ===============================================
print("🔧 正在执行 PCA 降维...")
pca = PCA(n_components=2, random_state=42)
vectors_2d = pca.fit_transform(vectors)
print(f"   PCA 解释方差比: {pca.explained_variance_ratio_}")

# ===============================================
# K-Means 聚类
# ===============================================
n_clusters = 20  # 你可以调整簇的数量
print(f"🔧 正在执行 K-Means 聚类 (k={n_clusters})...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(vectors)

# ===============================================
# 绘图
# ===============================================
print("🎨 绘制风格地图...")
plt.figure(figsize=(16, 12))

# 散点图，按簇着色
scatter = plt.scatter(
    vectors_2d[:, 0], vectors_2d[:, 1],
    c=labels, cmap='tab20', alpha=0.6, s=12, edgecolors='none'
)

# 添加簇中心标注
centers_2d = pca.transform(kmeans.cluster_centers_)
for i, (x, y) in enumerate(centers_2d):
    plt.scatter(x, y, c='black', marker='X', s=200, edgecolors='white', linewidth=1.5)
    
    # 找到离簇中心最近的画师作为代表
    center_vec = kmeans.cluster_centers_[i]
    # 计算所有画师到该中心的距离
    distances = np.linalg.norm(vectors - center_vec, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_id = artist_ids[nearest_idx]
    nearest_name = id_to_name.get(nearest_id, str(nearest_id))
    
    plt.annotate(
        nearest_name, (x, y),
        fontsize=9, fontweight='bold', color='navy',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray'),
        ha='center', va='bottom'
    )

plt.colorbar(scatter, label='Cluster ID', ticks=range(n_clusters))
plt.title(f'Artist Style Map (2048-dim, PCA 2D)\nn={len(artist_ids)} artists, {n_clusters} clusters', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(alpha=0.2)
plt.tight_layout()

# 保存图片
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"💾 图片已保存至: {OUTPUT_PLOT}")

# 可选：保存降维坐标和聚类标签供后续分析
df_out = pd.DataFrame({
    'artist_id': artist_ids,
    'pc1': vectors_2d[:, 0],
    'pc2': vectors_2d[:, 1],
    'cluster': labels
})
df_out = df_out.merge(metadata_df[['artist_id', 'artist_name']], on='artist_id', how='left')
df_out.to_csv(OUTPUT_DIR / "artist_pca_clusters.csv", index=False, encoding='utf-8')
print(f"💾 降维坐标与聚类标签已保存至: {OUTPUT_DIR / 'artist_pca_clusters.csv'}")

plt.show()