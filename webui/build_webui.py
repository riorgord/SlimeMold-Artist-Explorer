#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDXL 向量库构建助手 — 本地编码，不依赖 ComfyUI。
调用项目内置的 CLIP 编码器，一步完成：读 MD → 编码 → FAISS 索引。
"""
import gradio as gr
import numpy as np
import json
import time
import threading
from pathlib import Path

_BUILD_PROJ = Path(__file__).parent.parent
os_module = __import__('os')
os_module.makedirs(_BUILD_PROJ / "data/log", exist_ok=True)

CSS = """footer {display: none !important;}"""


def run_build(md_path, ckpt_path, output_dir, prefix, progress=gr.Progress()):
    """生成器：逐步执行本地构建，yield 状态文本。"""
    md_path = str(md_path); ckpt_path = str(ckpt_path); output_dir = str(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导入
    yield "🔧 导入模块..."
    import sys
    sys.path.insert(0, str(_BUILD_PROJ))
    from scripts.build_sdxl_local import parse_artist_md_table
    from engines.clip_encoder_sdxl import LocalSDXLEncoder
    import torch, faiss, pandas as pd
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False

    # 1. 解析 MD
    yield "📋 解析画师表格..."
    artists = parse_artist_md_table(md_path)
    total = len(artists)
    yield f"   共 {total} 位画师"

    # 2. 加载编码器
    yield f"🔧 加载 SDXL CLIP 编码器..."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = LocalSDXLEncoder(ckpt_path, device=device)
    encoder.load()
    yield f"   设备: {device}"

    # 3. 编码
    vectors, valid_ids, valid_names, failed = [], [], [], []
    total = len(artists)
    t_start = time.time()
    last_report = 0
    for i, artist in enumerate(artists):
        aid = artist['id']; aname = artist['name']
        try:
            vec = encoder.encode(f"{prefix}by {aname}")
            vectors.append(vec); valid_ids.append(aid); valid_names.append(aname)
        except Exception as e:
            failed.append((aid, aname, str(e)))

        # 每 50 个画师或最后一个时更新进度
        done = i + 1
        if done % 50 == 0 or done == total:
            elapsed = time.time() - t_start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            m, s = divmod(int(elapsed), 60)
            em, es = divmod(int(eta), 60)
            status = f"🧬 编码中: {done}/{total} ({100*done/total:.1f}%) | ⏱ 已过 {m}:{s:02d} | ⚡ {speed:.0f}人/秒 | ⏳ 预计 {em}:{es:02d}"
            progress(done / total, desc=status)
            if done - last_report >= 500 or done == total:
                yield status
                last_report = done

    yield f"   ✅ {len(valid_ids)} 成功, ❌ {len(failed)} 失败"
    if failed:
        yield f"   前5个失败: {failed[:5]}"

    vectors_matrix = np.stack(vectors, axis=0).astype(np.float32)
    dim = vectors_matrix.shape[1]

    # 4. FAISS
    yield "🔍 构建 FAISS 索引..."
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_matrix)
    faiss.write_index(index, str(output_dir / "artist_index_2048.faiss"))

    # 5. 保存
    with open(output_dir / "artist_ids_2048.json", 'w', encoding='utf-8') as f:
        json.dump(valid_ids, f, ensure_ascii=False, indent=2)
    np.save(output_dir / "artist_vectors_2048.npy", vectors_matrix)
    pd.DataFrame({"artist_id": valid_ids, "artist_name": valid_names}).to_csv(
        output_dir / "artist_metadata_2048.csv", index=False, encoding='utf-8')

    # 6. PCA
    if HAS_SKLEARN:
        yield "📊 PCA 聚类..."
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors_matrix)
        kmeans = KMeans(n_clusters=min(20, len(valid_ids)), random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(vectors_matrix)
        pd.DataFrame({"artist_id": valid_ids, "pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": clusters}).to_csv(
            output_dir / "artist_pca_clusters.csv", index=False, encoding='utf-8')

    # 7. library.json
    with open(output_dir / "library.json", 'w', encoding='utf-8') as f:
        json.dump({"name": f"SDXL {prefix.strip()}", "arch": "sdxl", "dim": int(dim), "prefix": prefix}, f, ensure_ascii=False, indent=2)

    yield f"\n🎉 完成！输出目录: {output_dir}"


def build_app():
    with gr.Blocks(title="SDXL 向量库构建助手", css=CSS) as app:
        gr.Markdown("# 🔨 SDXL 向量库构建助手（本地编码）")
        gr.Markdown("不依赖 ComfyUI。读 Markdown → 本地 CLIP 编码 → 一步生成 FAISS 索引。")

        with gr.Row():
            build_md = gr.Textbox(value="./artists/artists40k.md", label="Markdown 画师表")
            build_ckpt = gr.Textbox(label="SDXL Checkpoint (.safetensors)", info="完整底模，含 CLIP 权重")
        with gr.Row():
            build_out = gr.Textbox(value="./outputs_2048", label="输出目录")
            build_prefix = gr.Textbox(value="1girl, ", label="Prompt 前缀", info="末尾自动接 'by 画师名'")
        with gr.Row():
            build_btn = gr.Button("▶ 开始构建", variant="primary")
        build_status = gr.Markdown("")

        build_btn.click(fn=run_build, inputs=[build_md, build_ckpt, build_out, build_prefix], outputs=[build_status])

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7862, css=CSS, theme=gr.themes.Soft(), inbrowser=True)