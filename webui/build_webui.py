#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量库构建助手 SDXL — 独立 WebUI
将画师标签编码为 2048 维风格向量，构建 FAISS 索引
"""
import gradio as gr
import numpy as np
import requests
import time
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict

COMFYUI_SERVER = "127.0.0.1:8188"
CHECKPOINT_NAME = "waiIllustriousSDXL_v140.safetensors"
_BUILD_PROJ = Path(__file__).parent.parent
VECTOR_DIR = _BUILD_PROJ / "outputs_2048"

os.makedirs(_BUILD_PROJ / "data/log", exist_ok=True)

_step1_stop_event = threading.Event()

CSS = """
footer {display: none !important;}
"""


class BuildLogger:
    """调试日志器：debug=True 时写 .log 文件 + print，否则静默"""

    def __init__(self, debug: bool, label: str = ""):
        self.debug = debug
        self.logfile = None
        if debug:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(f"data/log/build_{label}_{ts}.log")
            self.logfile = open(path, 'w', encoding='utf-8')
            self.logfile.write(f"# 构建日志 — {ts}\n\n")

    def log(self, msg: str):
        if self.debug and self.logfile:
            ts = time.strftime("%H:%M:%S")
            self.logfile.write(f"[{ts}] {msg}\n")
            self.logfile.flush()

    def close(self):
        if self.logfile:
            self.logfile.close()
            self.logfile = None


def build_app():
    with gr.Blocks(title="向量库构建助手 SDXL") as app:
        gr.Markdown("# 🔨 向量库构建助手 SDXL")
        gr.Markdown("为每个画师标签生成 2048 维风格向量，构建 FAISS 索引")

        # ---- 步骤 1: 编码画师标签 → .ckpt ----
        with gr.Accordion("① 编码画师标签 → .ckpt（调 ComfyUI API）", open=True):
            gr.Markdown(
                "> ⚠️ 需要 **SaveCondition** 节点。ComfyUI Manager 搜索 `Comfyui-Condition-Utils`，"
                "或 [GitHub](https://github.com/lrzjason/Comfyui-Condition-Utils) 手动下载放入 `custom_nodes/`。"
                "缺失会导致编码失败。"
            )
            with gr.Row():
                build_md_path = gr.Textbox(label="Markdown 画师表路径", value="./artists/artists40k.md",
                                           info="必含 | id | name | 两列")
                build_wf_template = gr.Textbox(label="编码工作流模板", value="workflows/artist_workflow_api.json",
                                               info="需含 CLIPTextEncode + SaveCondition 节点")
            with gr.Row():
                build_comfy_addr = gr.Textbox(value=COMFYUI_SERVER, label="ComfyUI 地址")
                build_ckpt_name = gr.Textbox(value=CHECKPOINT_NAME, label="底模文件名",
                                             info="ComfyUI models/checkpoints/ 目录下的 .safetensors 文件")
            with gr.Row():
                build_prompt_prefix = gr.Textbox(value="1girl, ", label="Prompt 前缀")
            with gr.Row():
                build_concurrency = gr.Number(value=150, label="并发请求数", precision=0,
                                             info="同时发送 N 个 API 请求")
                def on_concurrency_change(v):
                    if v is None or int(v) < 1:
                        gr.Warning("并发数不能小于 1")
                        return
                    if int(v) > 1000:
                        gr.Warning("并发数不能超过 1000")
                        return
                    if int(v) > 30:
                        gr.Warning("⚠️ 高并发时建议关闭 ComfyUI 浏览器界面，避免卡顿")
                build_concurrency.change(fn=on_concurrency_change, inputs=[build_concurrency], outputs=[])
                build_debug = gr.Checkbox(value=False, label="调试模式",
                                          info="启用后写详细日志到 data/log/")
            with gr.Row():
                build_step1_btn = gr.Button("▶ 开始编码", variant="primary")
                build_step1_stop_btn = gr.Button("⏹ 停止", variant="stop", size="sm")
            build_step1_progress = gr.Markdown("")
            build_step1_log = gr.Markdown("")

        # ---- 步骤 2: .ckpt → 向量 → FAISS ----
        with gr.Accordion("② .ckpt → 向量 → FAISS 索引（需 PyTorch）", open=False):
            gr.Markdown("*依赖 `torch` 库。如未安装请先 `pip install torch`*")
            with gr.Row():
                build_cond_dir = gr.Textbox(value="./path/to/your/ComfyUI/models/conditions", label="条件文件目录 (.ckpt)",
                                           info="⚠️ 请改为你的 ComfyUI models/conditions/ 目录，避免拷贝大文件")
                build_meta_csv = gr.Textbox(value="./data/artist_vectors_metadata.csv", label="元数据 CSV")
            with gr.Row():
                build_out_dir = gr.Textbox(value="./outputs_2048", label="输出目录")
            with gr.Row():
                build_step2_btn = gr.Button("▶ 构建索引", variant="primary")
            build_step2_progress = gr.Markdown("")
            build_step2_log = gr.Markdown("")

        # ---- 步骤 3: PCA 可视化 ----
        with gr.Accordion("③ PCA 风格地图可视化", open=False):
            gr.Markdown("生成全库画师的 PCA 降维散点图（含 K-Means 聚类标注）")
            with gr.Row():
                s_viz_vec_path = gr.Textbox(value=str(_BUILD_PROJ / "outputs_2048/artist_vectors_2048.npy"),
                                            label="向量文件 (.npy)")
                s_viz_out_dir = gr.Textbox(value=str(_BUILD_PROJ / "outputs_2048"), label="输出目录")
            with gr.Row():
                run_viz_btn = gr.Button("▶ 生成可视化", variant="primary")
            viz_result = gr.HTML("<p style='color:gray;text-align:center;'>尚未生成</p>")

        # ---- 说明 ----
        with gr.Accordion("📖 关于向量库", open=False):
            gr.Markdown(
                """
                ### 向量库基于以下配置构建：

                - **底模适配**: SDXL 系列（CLIPTextEncode, 2048 维 pooled_output）
                - **编码器**: ComfyUI `CLIPTextEncode` + `SaveCondition` 节点
                - **向量提取**: `torch.load` 读取 `.ckpt`，取 `last_hidden_state` mean-pool
                - **索引方式**: FAISS 余弦相似度（IndexFlatIP, 向量 L2 归一化）

                ### 换底模后重建流程：

                1. 准备好你的画师标签 Markdown 表（`| id | name |`）
                2. 准备好含 `CLIPTextEncode` + `SaveCondition` 的工作流模板
                3. 执行步骤 ① → 编码为 `.ckpt`（结果在 ComfyUI models/conditions/）
                4. 执行步骤 ② → 条件文件目录直接填 ComfyUI 路径，提取向量并构建 FAISS 索引
                5. 执行步骤 ③（可选）→ 生成 PCA 风格地图查看分布

                > 向量空间由编码器决定。换了底模或编码提示词后必须重建。
                """
            )

        # ============================================
        # 事件绑定
        # ============================================

        # -- 步骤 1: 编码画师标签 ----
        # -- 停止按钮 ----
        def on_step1_stop():
            _step1_stop_event.set()
            gr.Info("⏹ 已发送停止信号，正在收尾...")

        build_step1_stop_btn.click(fn=on_step1_stop)

        def on_build_step1(md_path, wf_path, comfy_srv, prefix, ckpt_name, concurrency, debug):
            """Generator: 读取 Markdown → 逐画师调 ComfyUI 编码 → 产出 .ckpt + 元数据 CSV"""
            _step1_stop_event.clear()
            concurrency = max(1, min(1000, int(concurrency or 10)))
            L = BuildLogger(debug, "step1")
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                L.log(f"Markdown 加载: {md_path} ({len(lines)} 行)")
            except Exception as e:
                yield f"### ❌ 读取失败", f"无法读取 Markdown: {e}"
                L.close()
                return
            artists = []
            headers = []
            in_table = False
            passed_header = False
            for line in lines:
                s = line.strip()
                if not s:
                    if in_table and passed_header:
                        break
                    continue
                if s.startswith('|'):
                    if not in_table:
                        in_table = True
                        headers = [h.strip() for h in s.split('|')[1:-1]]
                        continue
                    if not passed_header:
                        if '---' in s.replace(' ', ''):
                            passed_header = True
                            continue
                        else:
                            passed_header = True
                    parts = [p.strip() for p in s.split('|')[1:-1]]
                    if len(parts) == len(headers):
                        artists.append(dict(zip(headers, parts)))
                else:
                    if in_table and passed_header:
                        break
            if not artists:
                yield "### ❌ 解析失败", "未在 Markdown 中找到画师表格"
                L.close()
                return
            L.log(f"解析完成: {len(artists)} 位画师")

            try:
                with open(wf_path, 'r', encoding='utf-8') as f:
                    base_wf = json.load(f)
                node_types = [n.get('class_type', '?') for n in base_wf.values()]
                L.log(f"工作流模板加载: {wf_path} ({len(base_wf)} 节点: {node_types})")
            except Exception as e:
                yield "### ❌ 加载失败", f"无法加载工作流模板: {e}"
                L.close()
                return

            metadir = _BUILD_PROJ / "data"
            metadir.mkdir(parents=True, exist_ok=True)
            out_csv = metadir / "artist_vectors_metadata.csv"
            records = []
            total = len(artists)
            t_start = time.time()

            yield f"### ⏳ 开始编码 {total} 位画师（{concurrency} 并发）...", ""

            from concurrent.futures import ThreadPoolExecutor, as_completed

            # 过滤无效画师，预构建所有 workflow
            valid_artists = []
            for a in artists:
                try:
                    aid = int(a['id'])
                except (ValueError, KeyError):
                    yield f"### ⏳ 预处理", f"⚠️ 跳过: id 无效 — {a}"
                    continue
                aname = a.get('name', '')
                if not aname:
                    yield f"### ⏳ 预处理", f"⚠️ 跳过: id={aid} name 为空"
                    continue
                wf = json.loads(json.dumps(base_wf))
                filename = f"artist_{aid}"
                found_clip = found_save = False
                for node_id, node in wf.items():
                    if node.get('class_type') == 'CheckpointLoaderSimple':
                        node['inputs']['ckpt_name'] = ckpt_name
                    elif node.get('class_type') == 'CLIPTextEncode':
                        node['inputs']['text'] = f"{prefix}by {aname}"
                        found_clip = True
                    elif node.get('class_type') == 'SaveCondition':
                        node['inputs']['filename'] = filename
                        found_save = True
                if not found_clip or not found_save:
                    yield f"### ⏳ 预处理", f"⚠️ 工作流缺少必要节点"
                    L.close()
                    return
                valid_artists.append((a, aid, aname, wf, filename))

            total_valid = len(valid_artists)
            L.log(f"预过滤: 原始 {len(artists)}, 有效 {total_valid}, 跳过 {len(artists) - total_valid}")
            L.log(f"并发数: {concurrency}, 预计批次: {(total_valid + int(concurrency) - 1) // int(concurrency)}")
            L.log(f"Prompt 前缀: '{prefix}', 底模: {ckpt_name}")

            # 分批并发
            def _submit_one(wf, comfy_srv):
                """提交一个工作流，返回 prompt_id"""
                r = requests.post(f"http://{comfy_srv}/prompt", json={"prompt": wf}, timeout=10)
                return r.json()['prompt_id']

            # 循环：每次取并发数个任务，提交后统一轮询
            i = 0
            while i < total_valid:
                if _step1_stop_event.is_set():
                    L.log(f"用户手动停止于 {i}/{total_valid}")
                    if records:
                        import pandas as pd
                        df = pd.DataFrame(records)
                        df.to_csv(out_csv, index=False, encoding='utf-8')
                        L.log(f"部分结果已保存: {out_csv} ({len(records)} 行)")
                    L.close()
                    yield f"### ⏹ 已停止 ({i}/{total_valid})", f"手动停止，已编码 {i} 个画师（部分结果已保存）"
                    _step1_stop_event.clear()
                    return
                batch = valid_artists[i:i + int(concurrency)]
                batch_size = len(batch)

                # 并发提交
                pending = {}
                batch_idx = i // int(concurrency) + 1
                total_batches = (total_valid + int(concurrency) - 1) // int(concurrency)
                L.log(f"[批次 {batch_idx}/{total_batches}] 提交 {batch_size} 个任务:")
                with ThreadPoolExecutor(max_workers=batch_size) as exc:
                    futures = {}
                    for a_dict, aid, aname, wf, fname in batch:
                        prompt_text = f"{prefix}by {aname}"
                        L.log(f"  artist_id={aid}: \"{prompt_text}\" → {fname}.ckpt")
                        fut = exc.submit(_submit_one, wf, comfy_srv)
                        futures[fut] = (aid, aname, fname, a_dict)
                    for fut in as_completed(futures):
                        aid, aname, fname, a_dict = futures[fut]
                        try:
                            pid = fut.result()
                            pending[pid] = (aid, aname, fname, a_dict)
                        except Exception as e:
                            yield f"### ⏳ {i+1}/{total_valid}", f"❌ {aname} 排队失败: {e}"
                            L.log(f"  提交失败: {aname} - {e}")

                # 批量轮询
                batch_t0 = time.time()
                while pending:
                    pids = list(pending.keys())
                    try:
                        h = requests.get(f"http://{comfy_srv}/history", timeout=10).json()
                    except Exception:
                        time.sleep(1)
                        continue
                    for pid in pids:
                        if pid in h:
                            aid, aname, fname, a_dict = pending.pop(pid)
                            # 检查 ComfyUI 执行错误
                            status = h[pid].get('status', {})
                            if status.get('status_str') == 'error':
                                msgs = status.get('messages', [])
                                err_text = str(msgs[0]) if msgs else '未知错误'
                                gr.Warning(f"⚠️ 模型加载失败！请确认底模文件名是否正确: {err_text[:150]}")
                                yield (f"### ❌ {aname} 执行错误", f"⚠️ {err_text[:200]}")
                            try:
                                pc = int(a_dict.get('post_count', 0))
                            except (ValueError, TypeError):
                                pc = 0
                            try:
                                us = float(a_dict.get('uniqueness_score', 0.0))
                            except (ValueError, TypeError):
                                us = 0.0
                            records.append({
                                'artist_id': aid,
                                'artist_name': aname,
                                'post_count': pc,
                                'uniqueness_score': us,
                                'ckpt_filename': f"{fname}.ckpt",
                            })
                            i += 1
                            elapsed = time.time() - t_start
                            etc = (elapsed / i) * (total_valid - i) if i > 0 else 0
                            yield (
                                f"### ⏳ {i}/{total_valid} 完成 ({elapsed:.0f}s, 预计剩余 {etc:.0f}s)",
                                f"✅ {aname} → {fname}.ckpt"
                            )
                    if pending and time.time() - batch_t0 > 180:
                        L.log(f"  超时 {len(pending)} 个: {[(p, v[1]) for p, v in list(pending.items())[:10]]}")
                        for pid, (_, aname, _, _) in list(pending.items()):
                            yield f"### ⏳ {i}/{total_valid}", f"⚠️ {aname} 超时"
                            pending.pop(pid)
                            i += 1
                    if pending:
                        time.sleep(1)

                L.log(f"[批次 {batch_idx}/{total_batches}] 完成 ({time.time() - batch_t0:.1f}s)")

            if records:
                import pandas as pd
                df = pd.DataFrame(records)
                df.to_csv(out_csv, index=False, encoding='utf-8')
                L.log(f"步骤①完成: 成功 {len(records)}/{total_valid}, 耗时 {time.time() - t_start:.0f}s, 元数据: {out_csv}")
                L.close()
                yield (
                    f"### ✅ 完成！{len(records)}/{total} 位画师编码成功",
                    f"元数据已保存: {out_csv}\n⚠️ 请将路径改到 ComfyUI的 models/conditions/"
                )
            else:
                L.log("步骤①: 无成功记录")
                L.close()
                yield "### ❌ 无成功记录", ""

        build_step1_btn.click(
            fn=on_build_step1,
            inputs=[build_md_path, build_wf_template, build_comfy_addr, build_prompt_prefix, build_ckpt_name, build_concurrency, build_debug],
            outputs=[build_step1_progress, build_step1_log]
        )

        # -- 步骤 2: .ckpt → 向量 → FAISS ----
        def on_build_step2(cond_dir, meta_csv, out_dir, debug):
            """Generator: 读取 .ckpt → torch 提取向量 → 建 FAISS"""
            L = BuildLogger(debug, "step2")
            try:
                import torch
            except ImportError:
                yield "### ❌ 缺少 PyTorch", "请先 pip install torch"
                L.close()
                return
            try:
                import pandas as pd
                df = pd.read_csv(meta_csv)
                L.log(f"元数据 CSV 加载: {meta_csv} ({len(df)} 行, 列: {list(df.columns)})")
            except Exception as e:
                yield "### ❌ 读取失败", f"读取元数据 CSV 失败: {e}"
                L.close()
                return

            cond_path = Path(cond_dir)
            if not cond_path.exists():
                yield "### ❌ 目录不存在", f"{cond_dir} 不存在，请先执行步骤①并拷贝文件"
                L.close()
                return

            ckpt_files = list(cond_path.glob("artist_*.ckpt")) + list(cond_path.glob("anima_artist_*.ckpt"))
            L.log(f"Conditions 目录: {cond_dir}, 找到 {len(ckpt_files)} 个 .ckpt 文件")

            vectors = []
            valid_ids = []
            failed = []
            total = len(df)
            t_start = time.time()

            yield f"### ⏳ 开始提取 {total} 个向量...", ""

            for idx, (_, row) in enumerate(df.iterrows()):
                aid = row['artist_id']
                ckpt_fn = row.get('ckpt_filename', f"artist_{aid}.ckpt")
                ckpt_path = cond_path / ckpt_fn
                if not ckpt_path.exists():
                    failed.append(aid)
                    yield f"### ⏳ {idx+1}/{total}", f"⚠️ 文件不存在: {ckpt_fn}"
                    continue
                try:
                    data = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(data, list) and data:
                        item = data[0]
                        if isinstance(item, (tuple, list)) and item:
                            lh = item[0]
                            vec = lh.mean(dim=1).squeeze(0).numpy().astype(np.float32)
                            norm = np.linalg.norm(vec)
                            if norm > 0:
                                vec = vec / norm
                            vectors.append(vec)
                            valid_ids.append(aid)
                        else:
                            failed.append(aid)
                    else:
                        failed.append(aid)
                except Exception as e:
                    failed.append(aid)
                    yield f"### ⏳ {idx+1}/{total}", f"❌ {ckpt_fn}: {e}"
                    continue

                if (idx + 1) % 500 == 0:
                    elapsed = time.time() - t_start
                    L.log(f"进度: {idx+1}/{total} ({elapsed:.0f}s), 成功 {len(vectors)}, 失败 {len(failed)}")
                    yield f"### ⏳ {idx+1}/{total} ({elapsed:.0f}s)", f"✅ 已提取 {len(vectors)} 个向量..."

            if not vectors:
                yield "### ❌ 构建失败", "无向量可构建"
                L.close()
                return

            import faiss
            vectors_m = np.stack(vectors, axis=0)
            L.log(f"向量矩阵: {vectors_m.shape}")
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "artist_vectors_2048.npy", vectors_m)
            with open(out / "artist_ids_2048.json", 'w', encoding='utf-8') as f:
                json.dump(valid_ids, f, ensure_ascii=False)
            index = faiss.IndexFlatIP(2048)
            index.add(vectors_m)
            faiss.write_index(index, str(out / "artist_index_2048.faiss"))
            L.log(f"FAISS 索引: {index.ntotal} 条")
            import pandas as pd
            meta_out = df[df['artist_id'].isin(valid_ids)].copy()
            meta_out.to_csv(out / "artist_metadata_2048.csv", index=False, encoding='utf-8')
            L.log(f"元数据输出: {len(meta_out)} 行")

            elapsed = time.time() - t_start
            L.log(f"步骤②完成: 成功 {len(valid_ids)}/{total}, 失败 {len(failed)}, 耗时 {elapsed:.0f}s")
            L.close()
            yield (
                f"### ✅ 索引构建完成 ({elapsed:.0f}s)",
                f"成功 {len(valid_ids)}/{total} 个向量\n维度: 2048\n输出: {out_dir}\n失败: {len(failed)}"
            )

        build_step2_btn.click(
            fn=on_build_step2,
            inputs=[build_cond_dir, build_meta_csv, build_out_dir, build_debug],
            outputs=[build_step2_progress, build_step2_log]
        )

        # -- 步骤 3: PCA 可视化 --
        def on_run_viz(vec_path, out_dir):
            import numpy as np
            import json
            import pandas as pd
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            import matplotlib.pyplot as plt
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            try:
                vectors = np.load(vec_path)
                with open(out / "artist_ids_2048.json", 'r') as f:
                    artist_ids = json.load(f)
                meta_path = out / "artist_metadata_2048.csv"
                if meta_path.exists():
                    meta = pd.read_csv(meta_path)
                    id_to_name = dict(zip(meta['artist_id'], meta['artist_name']))
                else:
                    id_to_name = {}
                pca = PCA(n_components=2, random_state=42)
                vecs_2d = pca.fit_transform(vectors)
                kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(vectors)
                fig, ax = plt.subplots(figsize=(16, 12))
                ax.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=labels, cmap='tab20', alpha=0.6, s=12)
                centers_2d = pca.transform(kmeans.cluster_centers_)
                for i, (x, y) in enumerate(centers_2d):
                    ax.scatter(x, y, c='black', marker='X', s=200, edgecolors='white', linewidth=1.5)
                    dists = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                    nearest_id = artist_ids[np.argmin(dists)]
                    nearest_name = id_to_name.get(nearest_id, str(nearest_id))
                    ax.annotate(nearest_name, (x, y), fontsize=9, fontweight='bold', color='navy',
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='gray'),
                                ha='center', va='bottom')
                ax.set_title(f'Artist Style Map ({vectors.shape[1]}dim PCA 2D)', fontsize=14)
                plot_path = out / "artist_style_map_2048.png"
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                import base64, io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                df_out = pd.DataFrame({'artist_id': artist_ids, 'pc1': vecs_2d[:, 0], 'pc2': vecs_2d[:, 1], 'cluster': labels})
                if id_to_name:
                    df_out['artist_name'] = df_out['artist_id'].map(id_to_name)
                df_out.to_csv(out / "artist_pca_clusters.csv", index=False)
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;">'
            except Exception as e:
                return f"<p style='color:red'>❌ 可视化失败: {e}</p>"

        run_viz_btn.click(
            fn=on_run_viz,
            inputs=[s_viz_vec_path, s_viz_out_dir],
            outputs=[viz_result]
        )

    return app


if __name__ == "__main__":
    print("🔨 启动向量库构建助手 SDXL...")
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=17325,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(),
    )
