"""
一键本地构建 SDXL 画师向量库。
替代原先 tag_to_tensor.py + build_index_2048.py 两步 ComfyUI 依赖流程。

输入：Markdown 画师表格 + SDXL checkpoint
输出：FAISS 索引 + 向量矩阵 + 元数据 + PCA 聚类 + library.json

用法：
  python scripts/build_sdxl_local.py \
    --md ./artists/artists40k.md \
    --ckpt E:/path/to/illustrious.safetensors \
    --output ./outputs_2048_local \
    --prefix "1girl, "
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import faiss
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def parse_artist_md_table(md_file_path: str) -> list[dict]:
    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    artists = []
    headers = []
    in_table = False
    passed_header = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_table and passed_header:
                break
            continue
        if line.startswith('|'):
            if not in_table:
                in_table = True
                headers = [h.strip() for h in line.split('|')[1:-1]]
                continue
            if not passed_header:
                if '---' in line.replace(' ', ''):
                    passed_header = True
                    continue
                else:
                    passed_header = True
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) == len(headers):
                artist = dict(zip(headers, parts))
                try:
                    artist['id'] = int(artist['id'])
                except Exception:
                    artist['id'] = None
                try:
                    artist['post_count'] = int(artist['post_count'])
                except Exception:
                    artist['post_count'] = 0
                try:
                    artist['uniqueness_score'] = float(artist['uniqueness_score'])
                except Exception:
                    artist['uniqueness_score'] = 0.0
                artists.append(artist)
        else:
            if in_table and passed_header:
                break

    return artists


def main():
    parser = argparse.ArgumentParser(description="一键本地构建 SDXL 画师向量库")
    parser.add_argument("--md", required=True, help="Markdown 画师表格路径")
    parser.add_argument("--ckpt", required=True, help="SDXL checkpoint .safetensors 路径")
    parser.add_argument("--output", default="./outputs_2048", help="输出目录")
    parser.add_argument("--prefix", default="1girl, ", help="Prompt 前缀，末尾接 'by 画师名'")
    parser.add_argument("--device", default=None, help="设备 (cuda/cpu，默认自动)")
    parser.add_argument("--save-vectors", action="store_true", help="保存完整向量矩阵 .npy")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析画师表格
    print(f"📋 解析画师表格: {args.md}")
    artists = parse_artist_md_table(args.md)
    print(f"   共 {len(artists)} 位画师")

    # 2. 加载编码器
    print(f"🔧 加载 SDXL CLIP 编码器: {args.ckpt}")
    from engines.clip_encoder_sdxl import LocalSDXLEncoder
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LocalSDXLEncoder(args.ckpt, device=device)
    encoder.load()
    print(f"   设备: {device}")

    # 3. 逐画师编码
    print(f"🧬 编码中 (prefix='{args.prefix}by ')...")
    vectors = []
    valid_ids = []
    valid_names = []
    failed = []

    for artist in tqdm(artists, desc="编码画师"):
        aid = artist['id']
        aname = artist['name']
        try:
            vec = encoder.encode(f"{args.prefix}by {aname}")
            vectors.append(vec)
            valid_ids.append(aid)
            valid_names.append(aname)
        except Exception as e:
            failed.append((aid, aname, str(e)))

    print(f"   ✅ {len(valid_ids)} 成功, ❌ {len(failed)} 失败")
    if failed:
        print(f"   前5个失败: {failed[:5]}")

    vectors_matrix = np.stack(vectors, axis=0).astype(np.float32)
    dim = vectors_matrix.shape[1]
    print(f"   向量矩阵: {vectors_matrix.shape}")

    # 4. FAISS 索引
    print("🔍 构建 FAISS 索引...")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_matrix)
    faiss.write_index(index, str(output_dir / "artist_index_2048.faiss"))
    print(f"   💾 {output_dir / 'artist_index_2048.faiss'}")

    # 5. ID 列表
    with open(output_dir / "artist_ids_2048.json", 'w', encoding='utf-8') as f:
        json.dump(valid_ids, f, ensure_ascii=False, indent=2)

    # 6. 向量矩阵
    if args.save_vectors:
        np.save(output_dir / "artist_vectors_2048.npy", vectors_matrix)

    # 7. 元数据
    df_meta = pd.DataFrame({"artist_id": valid_ids, "artist_name": valid_names})
    df_meta.to_csv(output_dir / "artist_metadata_2048.csv", index=False, encoding='utf-8')

    # 8. PCA 聚类
    print("📊 PCA 聚类...")
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors_matrix)
        kmeans = KMeans(n_clusters=min(20, len(valid_ids)), random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(vectors_matrix)
        df_pca = pd.DataFrame({"artist_id": valid_ids, "pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": clusters})
        df_pca.to_csv(output_dir / "artist_pca_clusters.csv", index=False, encoding='utf-8')
    except ImportError:
        print("   ⚠️ sklearn 未安装，跳过 PCA")

    # 9. library.json
    lib_json = {
        "name": f"SDXL {args.prefix.strip()}",
        "arch": "sdxl",
        "dim": int(dim),
        "prefix": args.prefix,
    }
    with open(output_dir / "library.json", 'w', encoding='utf-8') as f:
        json.dump(lib_json, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 完成！输出目录: {output_dir}")


if __name__ == "__main__":
    main()