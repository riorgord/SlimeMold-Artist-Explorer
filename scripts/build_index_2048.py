import torch
import numpy as np
import pandas as pd
import faiss
import json
from pathlib import Path
from tqdm import tqdm

# ===============================================
# 画师向量索引构建器（步骤 2 / 3）
#
# 前置条件：
#   1. ComfyUI 已安装 SaveCondition 节点
#   2. 已通过 tag_to_tensor.py（或 WebUI 构建 Tab）将画师编码为 .ckpt 文件
#   3. 已通过 tag_to_tensor.py（或 build_webui.py）完成编码，.ckpt 文件在 ComfyUI models/conditions/
#   4. 将 CONDITIONS_DIR 改成你的 ComfyUI models/conditions/ 路径
#
# 产出：
#   outputs_2048/artist_vectors_2048.npy    向量矩阵
#   outputs_2048/artist_ids_2048.json       ID 列表
#   outputs_2048/artist_index_2048.faiss    FAISS 索引
#   outputs_2048/artist_metadata_2048.csv   元数据
# ===============================================
CONDITIONS_DIR = Path("./data/conditions")    # ⚠️ 改成你的 ComfyUI models/conditions/ 目录，避免拷贝
METADATA_CSV = "./data/artist_vectors_metadata.csv"  # tag_to_tensor 产出的映射表
OUTPUT_DIR = Path("./outputs_2048")                   # 向量库输出目录
OUTPUT_VECTORS_NPY = OUTPUT_DIR / "artist_vectors_2048.npy"
OUTPUT_IDS_JSON = OUTPUT_DIR / "artist_ids_2048.json"
OUTPUT_INDEX_FILE = OUTPUT_DIR / "artist_index_2048.faiss"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "artist_metadata_2048.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================
# 从 .ckpt 提取 2048 维风格向量
# ===============================================
def extract_2048_vector_from_ckpt(ckpt_path: Path, pool_method: str = "mean") -> np.ndarray:
    """
    从 .ckpt 文件中提取 last_hidden_state，池化为 2048 维向量。
    pool_method: "mean" (平均池化), "max" (最大池化), "cls" (取第一个token), 或 "attention" (需要额外注意力权重)
    推荐使用 "mean"，因为画师风格通常分布在整个序列中。
    """
    data = torch.load(ckpt_path, map_location='cpu')
    # 结构: [ ( tensor[1,77,2048], {'pooled_output': tensor[1,1280]} ) ]
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, (tuple, list)) and len(first_item) > 0:
            last_hidden_state = first_item[0]  # shape (1, 77, 2048)
            if pool_method == "mean":
                # 对序列维度（dim=1）求平均，忽略 padding 部分？CLIP 编码的序列通常无显式 padding mask，直接平均即可
                vector = last_hidden_state.mean(dim=1).squeeze(0)  # shape (2048,)
            elif pool_method == "max":
                vector = last_hidden_state.max(dim=1)[0].squeeze(0)
            elif pool_method == "cls":
                vector = last_hidden_state[:, 0, :].squeeze(0)  # 取第一个 token (类似 CLS)
            else:
                raise ValueError(f"不支持的池化方法: {pool_method}")
            
            vector = vector.cpu().numpy().astype(np.float32)
            # 归一化，便于余弦相似度
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
    raise ValueError(f"无法从 {ckpt_path} 提取 2048 向量")

# ===============================================
# 主流程
# ===============================================
def main():
    # 读取元数据，获取画师 ID 与文件名的对应关系
    df_meta = pd.read_csv(METADATA_CSV)
    print(f"📋 读取元数据，共 {len(df_meta)} 条记录")

    vectors = []
    valid_ids = []
    failed = []

    print("🧬 开始提取 2048 维向量...")
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        artist_id = row['artist_id']
        ckpt_filename = row['ckpt_filename']
        ckpt_path = CONDITIONS_DIR / ckpt_filename
        
        if not ckpt_path.exists():
            print(f"⚠️ 文件不存在: {ckpt_path}")
            failed.append(artist_id)
            continue
        
        try:
            vec = extract_2048_vector_from_ckpt(ckpt_path, pool_method="mean")
            vectors.append(vec)
            valid_ids.append(artist_id)
        except Exception as e:
            print(f"❌ 画师 {artist_id} 提取失败: {e}")
            failed.append(artist_id)

    if not vectors:
        print("❌ 没有成功提取任何向量")
        return

    vectors_matrix = np.stack(vectors, axis=0)
    print(f"✅ 成功提取 {len(valid_ids)} 个向量，形状: {vectors_matrix.shape}")

    # 保存
    np.save(OUTPUT_VECTORS_NPY, vectors_matrix)
    with open(OUTPUT_IDS_JSON, 'w', encoding='utf-8') as f:
        json.dump(valid_ids, f, ensure_ascii=False, indent=2)

    # 构建 FAISS 索引
    index = faiss.IndexFlatIP(2048)
    index.add(vectors_matrix)
    faiss.write_index(index, str(OUTPUT_INDEX_FILE))

    # 保存精简元数据
    meta_out = df_meta[df_meta['artist_id'].isin(valid_ids)].copy()
    meta_out.to_csv(OUTPUT_METADATA_CSV, index=False, encoding='utf-8')

    print(f"💾 已保存至: {OUTPUT_DIR}")
    if failed:
        print(f"⚠️ 失败画师 ID: {failed}")

if __name__ == "__main__":
    main()