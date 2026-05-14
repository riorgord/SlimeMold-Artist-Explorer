"""临时脚本：测一下 FAISS 相似度分布。跑完把输出贴给我，然后删掉。"""
import numpy as np, faiss, json
from pathlib import Path

# 选你要测的向量库：SDXL 或 Anima
VEC_DIR = Path("./outputs_anima")  # 改成 ./outputs_anima 测 Anima

vectors = np.load(VEC_DIR / "artist_vectors_anima.npy").astype(np.float32)
index = faiss.read_index(str(VEC_DIR / "artist_index_anima.faiss"))

print(f"向量数: {len(vectors)}, 维度: {vectors.shape[1]}")
print("随机取 100 个点，各搜 36 个近邻...")

distances = []
for _ in range(100):
    v = vectors[np.random.randint(len(vectors))]
    d, _ = index.search(v.reshape(1, -1), 36)
    distances.extend(d[0].tolist())

d = np.array(distances)
print(f"\n=== FAISS 余弦相似度分布 ({len(d)} 个样本) ===")
print(f"最小值: {d.min():.4f}")
print(f"最大值: {d.max():.4f}")
print(f"平均值: {d.mean():.4f}")
print(f"中位数: {np.median(d):.4f}")
print(f"标准差: {d.std():.4f}")
print(f"P1  : {np.percentile(d, 1):.4f}")
print(f"P5  : {np.percentile(d, 5):.4f}")
print(f"P10 : {np.percentile(d, 10):.4f}")
print(f"P25 : {np.percentile(d, 25):.4f}")
print(f"P50 : {np.percentile(d, 50):.4f}")
print(f"P75 : {np.percentile(d, 75):.4f}")
print(f"P90 : {np.percentile(d, 90):.4f}")
print(f"P95 : {np.percentile(d, 95):.4f}")
print(f"P99 : {np.percentile(d, 99):.4f}")

# 如果 P1 ~ P99 区间 < 0.15 就算"高密度"
spread = np.percentile(d, 99) - np.percentile(d, 1)
print(f"\nP99-P1 跨度: {spread:.4f}")
if spread < 0.15:
    print("结论: 高密度分布，用排名权重合理")
else:
    print("结论: 分布较散，用真实距离权重可行")
