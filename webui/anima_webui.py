#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
画师串小助手 - Anima 版 WebUI
"""
import gradio as gr
import numpy as np
import requests
import time
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple, Dict

if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'gb2312'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ))

_VECTOR_DIR_A = _PROJ / "outputs_anima"
_VECTOR_DIR_A.mkdir(parents=True, exist_ok=True)
_VECTOR_REQUIRED_A = [
    _VECTOR_DIR_A / "artist_vectors_anima.npy",
    _VECTOR_DIR_A / "artist_ids_anima.json",
    _VECTOR_DIR_A / "artist_index_anima.faiss",
    _VECTOR_DIR_A / "artist_metadata_anima.csv",
]
_VECTOR_MISSING_A = [f for f in _VECTOR_REQUIRED_A if not f.exists()]
if _VECTOR_MISSING_A:
    print("=" * 60)
    print("❌ Anima 向量库文件缺失！")
    print("   目录 outputs_anima/ 已创建")
    print("   缺失文件：")
    for _m in _VECTOR_MISSING_A:
        print(f"   - {_m.name}")
    print("=" * 60)
    print("是否现在启动 Anima 向量库构建助手？(y/n): ", end="")
    try: ans = input().strip().lower()
    except (EOFError, KeyboardInterrupt): ans = "n"
    if ans in ("y", "yes"):
        import subprocess
        print("🚀 正在启动 build_webui_anima.py ...")
        subprocess.Popen([sys.executable, str(_PROJ / "webui/build_webui_anima.py")])
    sys.exit(1)

import engines.interactive_style_explorer_anima as v8

os.makedirs(_PROJ / "data", exist_ok=True)

SETTINGS_FILE = _PROJ / "data/webui_settings.json"
def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: return {}
    return {}
def save_settings(s: dict):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(s, f, indent=2, ensure_ascii=False)
def get_setting(key: str, default):
    return load_settings().get(key, default)

_score_buffer: Dict[str, float] = {}
SCORES_CACHE_PATH = _PROJ / "data/cache/scores_cache_anima.json"
def buffer_save_score(idx: int, value: float): _score_buffer[str(idx)] = value
def buffer_get_scores() -> dict: return _score_buffer
def buffer_clear(): _score_buffer.clear()
def buffer_flush_to_file():
    SCORES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CACHE_PATH, 'w', encoding='utf-8') as f: json.dump(_score_buffer, f, ensure_ascii=False)

def img_info_to_url(img_info: dict, server: str = None) -> str:
    srv = server or v8.COMFYUI_SERVER
    return f"http://{srv}/view?filename={img_info['filename']}&subfolder={img_info.get('subfolder','')}&type={img_info.get('type','output')}"

def create_explorer(spread: float):
    return v8.SlimeMoldExplorerAnima(n_tentacles=v8.N_TENTACLES, eval_budget=v8.EVAL_BUDGET, spread=spread)

def get_stats_text(exp, pzones: dict) -> str:
    if exp is None: return "尚未初始化"
    act, art, clus = exp.stats(); weak = sum(1 for t in exp.tentacles if t.is_weak)
    return (f"🟢 活跃 {act}/{exp.n_tentacles}　🎨 覆盖画师 {art}　📊 覆盖簇 {clus}　"
            f"🔗 弱触角 {weak}　🔭 探测点 {len(exp.scout_points)}　🛡️ 保护区 {len(pzones)}　🧬 第 {exp.generation} 代")

def get_ban_md() -> str:
    ban_list = v8.load_ban_list(); active = [b for b in ban_list if b.get('active', True)]
    if not active: return "*暂无Ban区*"
    lines = ["| ID | 描述 | 惩罚强度 | 创建代数 |", "|---|---|---|---|"]
    for b in active: lines.append(f"| {b['id']} | {b['description'][:50]} | {b.get('penalty',1.0):.2f} | {b.get('created_gen',0)} |")
    return "\n".join(lines)

def get_pz_md() -> str:
    pzones = v8.load_protect_zones()
    if not pzones: return "*暂无保护区*"
    lines = ["| ID | 中心画师 | 候选数 | 最佳分 | 无提升轮数 |", "|---|---|---|---|---|"]
    for zid, zone in pzones.items():
        bs = zone.best_artist_str or v8.vector_to_artist_string(zone.center_vector, top_k=zone.mixed_count)
        lines.append(f"| {zid} | {bs} | {len(zone.candidate_artists)} | {zone.best_score:.2f} | {zone.no_improve_rounds} |")
    return "\n".join(lines)

def get_scout_md(exp) -> str:
    if exp is None: return "*暂无探测点*"
    asp = [sp for sp in exp.scout_points if sp.get('active', True)]
    if not asp: return "*暂无探测点*"
    lines = ["| ID | 容量 | 已分配 | 位置 |", "|---|---|---|---|"]
    for sp in asp:
        pos = v8.vector_to_artist_string(np.array(sp['vector']), top_k=1)[:40]
        lines.append(f"| {sp['id']} | {sp['capacity']} | {len(sp.get('assigned',[]))} | {pos} |")
    return "\n".join(lines)

def get_fav_md() -> str:
    fp = _PROJ / "data/favorites_anima.json"
    if not fp.exists(): return "*暂无收藏*"
    with open(fp, 'r', encoding='utf-8') as f: favs = json.load(f)
    if not favs: return "*暂无收藏*"
    lines = ["| # | 画师串 | 收藏时间 |", "|---|---|---|"]
    for i, fe in enumerate(favs):
        ts = fe.get('timestamp','')[:16].replace('T',' ')
        lines.append(f"| {i+1} | {fe.get('artist_string','')[:60]} | {ts} |")
    return "\n".join(lines)

def _make_plot(exp, pzones=None):
    if v8.pca_model is None: return None
    active = [t for t in exp.tentacles if t.active]
    if not active: return None
    fig, ax = plt.subplots(figsize=(10, 8))
    if v8.PLOT_STYLE == "full" and v8.pca_vectors_2d is not None:
        ax.scatter(v8.pca_vectors_2d[:,0], v8.pca_vectors_2d[:,1], c='lightgray', alpha=0.3, s=5)
    for t in active:
        if len(t.trace) >= 2:
            pts = [v8.pca_model.transform(vec.reshape(1,-1))[0] for _, vec in t.trace]
            xs, ys = zip(*pts); alphas = np.linspace(0.2, 1.0, len(xs))
            for j in range(len(xs)-1): ax.plot(xs[j:j+2], ys[j:j+2], color='gray', alpha=alphas[j], linewidth=1)
    scores = [t.recent_avg_score() for t in active]; norm_scores = (np.array(scores)+10)/20
    colors = plt.cm.RdYlGn(norm_scores)[:,:3]
    t_vecs = np.array([t.vector for t in active]); t_2d = v8.pca_model.transform(t_vecs)
    weak_mask = [t.is_weak for t in active]
    if any(weak_mask):
        wi = np.where(weak_mask)[0]; ax.scatter(t_2d[wi,0], t_2d[wi,1], c='yellow', s=120, marker='D', edgecolors='black', linewidth=1.5, label='弱触角')
    normal_mask = [not t.is_weak for t in active]
    if any(normal_mask):
        ni = np.where(normal_mask)[0]; ax.scatter(t_2d[ni,0], t_2d[ni,1], c=colors[ni], s=100, marker='X', edgecolors='white', linewidth=1.5)
    ban_list = v8.load_ban_list(); active_bans = [b for b in ban_list if b.get('active',True)]
    if active_bans:
        try:
            ban_vecs = np.array([b['vector'] for b in active_bans], dtype=np.float32)
            ban_2d = v8.pca_model.transform(ban_vecs); ax.scatter(ban_2d[:,0], ban_2d[:,1], c='red', s=150, marker='o', alpha=0.6, label='Ban区')
        except: pass
    if pzones:
        for zid, zone in pzones.items():
            z2d = v8.pca_model.transform(zone.center_vector.reshape(1,-1))[0]; ax.scatter(z2d[0], z2d[1], c='cyan', s=200, marker='*', alpha=0.8, label=f'保护区_{zid}')
    for sp in exp.scout_points:
        sv = np.array(sp['vector'], dtype=np.float32)
        s2d = v8.pca_model.transform(sv.reshape(1,-1))[0]; ax.scatter(s2d[0], s2d[1], c='green', s=200, marker='o', alpha=0.5, label='探测点')
    ax.legend(fontsize=9); ax.set_title(f"触角分布 (第{exp.generation}代)", fontsize=14); fig.tight_layout(); return fig

def _check_comfyui_error(res) -> Optional[str]:
    """检查 ComfyUI 返回的 status，有错误返回提示文字，无错误返回 None"""
    status = res.get('status', {})
    if status.get('status_str') == 'error':
        msgs = status.get('messages', [])
        for m in msgs:
            msg_text = str(m) if isinstance(m, str) else str(m[1]) if isinstance(m, (list,tuple)) and len(m)>1 else str(m)
            msg_lower = msg_text.lower()
            if 'lora' in msg_lower:
                return (f"⚠️ LoRA 缺失！自动切换为 base 模式（速度慢约 4 倍）\n{msg_text[:200]}")
            if 'model' in msg_lower or 'checkpoint' in msg_lower or 'unet' in msg_lower or 'clip' in msg_lower or 'load' in msg_lower:
                return (f"⚠️ 模型加载失败！请确认设置里填的模型名是否正确，文件是否在 ComfyUI 对应目录下\n{msg_text[:200]}")
            return f"⚠️ ComfyUI 执行错误：{msg_text[:200]}"
    return None

def submit_scores(exp, score_data):
    if exp is None or score_data is None: return get_stats_text(exp, v8.load_protect_zones()), None
    if isinstance(score_data, dict):
        if not score_data: return get_stats_text(exp, v8.load_protect_zones()), None
        selected = []; scores = []
        for idx_str, s in score_data.items():
            if s is None: continue
            idx = int(idx_str); s = max(-10.0, min(10.0, float(s))); selected.append(idx); scores.append(s)
        if not selected: return get_stats_text(exp, v8.load_protect_zones()), None
        exp.update_tentacles(selected, scores)
        if exp.generation % 3 == 0: exp.exchange_info()
        return get_stats_text(exp, v8.load_protect_zones()), _make_plot(exp, v8.load_protect_zones())
    return get_stats_text(exp, v8.load_protect_zones()), None

def run_explore_round_gen(exp, server):
    if exp is None: yield "探索器未初始化", [], "请先初始化探索器", None; return
    exp.generation += 1; selected = exp.select_tentacles_to_evaluate()
    if not selected: yield "⚠️ 无可用触角", [], "⚠️ 无可用触角", None; return
    total = len(selected); batch_items = []; taco_times = []
    yield f"⏳ 第 {exp.generation} 代: 选中 {total} 个触角...", batch_items, "", None
    for i, idx in enumerate(selected):
        t = exp.tentacles[idx]; artist_str = v8.vector_to_artist_string(t.vector)
        prompt = f"{v8.BASE_POSITIVE_PROMPT}, {artist_str}"
        fname = f"gen{exp.generation:02d}_t{idx:03d}"; seed = exp.generation*100+idx
        wf = v8.build_workflow(prompt, v8.BASE_NEGATIVE_PROMPT, seed, fname); pid = v8.queue_prompt(wf)
        poll_start = time.time(); res = None
        while True:
            elapsed = time.time()-poll_start
            avg_t = sum(taco_times)/len(taco_times) if taco_times else 0
            eta = avg_t*(total-i-1)+max(0, avg_t-elapsed) if taco_times else 0
            eta_str = f" | 预计剩余 ~{eta:.0f}s" if taco_times else " | 剩余时间...（正在计算）"
            yield f"⏳ 触角 {i+1}/{total} (ID {idx}) 生图中... {elapsed:.0f}s{eta_str}", [], "", None
            try:
                resp = requests.get(f"http://{v8.COMFYUI_SERVER}/history/{pid}", timeout=10); hist = resp.json()
            except Exception as e:
                yield f"⏳ 触角 {i+1}/{total} (ID {idx}) 请求失败 ({e})，重试中...", [], "", None; time.sleep(2); continue
            if pid in hist:
                res = hist[pid]
                err = _check_comfyui_error(res)
                if err:
                    if 'lora' in err.lower() and v8.MODE == 'turbo':
                        v8.MODE = 'base'; v8.WORKFLOW_TEMPLATE_PATH = Path("workflows/t2i_anima_base.json")
                        v8.STEPS = 30; v8.CFG = 5; v8.SAMPLER_NAME = 'er_sde'; v8.SCHEDULER = 'beta57'
                        gr.Warning("Anima 已自动切换为 base 模式（速度慢约 4 倍）")
                        yield f"⏳ LoRA缺失，已切base，请重试", [], "", None; return
                    yield f"❌ 触角 {i+1}/{total} (ID {idx}) 错误: {err}", [], "", None; return
                elapsed_done = time.time()-poll_start; taco_times.append(elapsed_done)
                yield f"✅ 触角 {i+1}/{total} (ID {idx}) 完成 ({elapsed_done:.0f}s)", [], "", None; break
            if elapsed > 120: yield f"⏳ 触角 {i+1}/{total} (ID {idx}) 超时", [], "", None; break
            time.sleep(1)
        img_url = None
        if res:
            for out in res.get('outputs',{}).values():
                if 'images' in out: img_url = img_info_to_url(out['images'][0], server); break
        if img_url:
            marker = "";
            if t.is_weak: marker += "🔗弱"
            if t.scout_id is not None: marker += f"🔭S{t.scout_id}"
            batch_items.append({"idx":idx,"url":img_url,"artist_str":artist_str,"marker":marker,"seed":seed})
    fig = _make_plot(exp, v8.load_protect_zones()) if batch_items else None
    stats = get_stats_text(exp, v8.load_protect_zones())
    exp.last_batch = batch_items
    yield f"✅ 第 {exp.generation} 代完成！{len(batch_items)}/{total} 张成功", batch_items, stats, fig

def _rh_score(value, idx):
    val = max(-10.0, min(10.0, float(value))) if value is not None else 0.0; buffer_save_score(int(idx), val)
def _rh_ban(exp, idx):
    if exp is None: gr.Warning("请先初始化探索器"); return exp
    if 0 <= int(idx) < len(exp.tentacles): exp.ban_current_tentacle(int(idx)); gr.Info(f"🚫 已封禁触角 {idx}")
    return exp
def _rh_save(exp, idx):
    if exp is None: gr.Warning("请先初始化探索器"); return
    t = exp.tentacles[int(idx)]; astr = v8.vector_to_artist_string(t.vector); v8.save_favorite(astr, t.vector); gr.Info(f"⭐ 已收藏触角 {idx}")
def _rh_protect(exp, idx):
    if exp is None: gr.Warning("请先初始化探索器"); return exp
    t = exp.tentacles[int(idx)]; pzones = v8.load_protect_zones(); new_id = len(pzones)
    zone = v8.ProtectZone(new_id, t.vector.copy(), 0.15, v8.MIXED_ARTISTS_COUNT); pzones[new_id] = zone; v8.save_protect_zones(pzones)
    gr.Info(f"🛡️ 已创建保护区_{new_id}"); return exp
def _rh_scout(exp, idx):
    if exp is None: gr.Warning("请先初始化探索器"); return exp
    t = exp.tentacles[int(idx)]; new_id = max((sp['id'] for sp in exp.scout_points), default=-1)+1
    exp.scout_points.append({"id":new_id,"vector":t.vector.tolist(),"capacity":3,"assigned":[],"active":True})
    v8.save_scout_points(exp.scout_points); gr.Info(f"🔭 已创建探测点 {new_id}"); return exp
def _rh_save_image(url: str, idx: int):
    if not url or idx < 0: gr.Warning("无可保存图片"); return
    try:
        resp = requests.get(url); resp.raise_for_status()
        _PROJ / "data/output".mkdir(exist_ok=True)
        fname = f"data/output/tentacle_anima_{idx}.png"
        with open(fname, 'wb') as f: f.write(resp.content); gr.Info(f"已保存: {fname}")
    except Exception as e: gr.Error(f"保存失败: {e}")

CSS = """footer {display: none !important;}"""

def build_app():
    with gr.Blocks(title="画师串小助手 - Anima版") as app:
        gr.Markdown("# 🧫 画师串小助手 - Anima版")
        explorer_state = gr.State(None)

        with gr.Tab("🧫 风格探索 Anima"):
            # ── 状态栏 + 控制 ──
            stats_display = gr.Markdown("### 尚未初始化"); progress_status = gr.Markdown("### ")
            with gr.Row():
                spread_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="初始扩散度")
                init_btn = gr.Button("🆕 初始化", variant="secondary", scale=1)
                main_btn = gr.Button("🚀 探索 & 提交", variant="primary", scale=2)
            batch_state = gr.State([]); page_state = gr.State(0); page_size_state = gr.State(4)
            # ── 分页 ──
            with gr.Row():
                psize_input = gr.Number(value=4, label="每页显示", precision=0, minimum=1, maximum=16, scale=1)
            psize_input.change(fn=lambda v: (int(v), 0), inputs=[psize_input], outputs=[page_size_state, page_state])
            grid_css = gr.HTML("<style>" + " ".join([f".ga-{i}{{display:none!important}}" for i in range(16)]) + "</style>")
            with gr.Row():
                prev_btn = gr.Button("◀", size="sm", scale=1); page_info = gr.Markdown("**0/0** 共0条"); next_btn = gr.Button("▶", size="sm", scale=1)
            def on_prev_page(page, batch, psize): return max(0, page-1) if batch else 0
            def on_next_page(page, batch, psize): return min(((len(batch)-1)//max(1,psize)), page+1) if batch else 0
            def on_update_page_info(page, batch, psize):
                if not batch: return "**0/0** 共0条"
                return f"**{page+1}/{max(1,(len(batch)+psize-1)//psize)}** 共{len(batch)}条"
            prev_btn.click(fn=on_prev_page, inputs=[page_state,batch_state,page_size_state], outputs=[page_state])
            next_btn.click(fn=on_next_page, inputs=[page_state,batch_state,page_size_state], outputs=[page_state])
            page_state.change(fn=on_update_page_info, inputs=[page_state,batch_state,page_size_state], outputs=[page_info])
            batch_state.change(fn=on_update_page_info, inputs=[page_state,batch_state,page_size_state], outputs=[page_info])
            # ── 网格 ──
            gr.Markdown("🚫=封禁 ⭐=收藏 🛡️=保护区 🔭=探测点 💾=保存 | 评分 -10~10")
            cell_states = []
            for gi in range(16):
                if gi % 4 == 0: _rw = gr.Row().__enter__()
                with gr.Column(elem_classes=[f"ga-{gi}"]):
                    idx_s = gr.State(-1); img_h = gr.HTML(""); url_s = gr.State("")
                    with gr.Row():
                        sc_n = gr.Number(value=0, minimum=-10, maximum=10, step=1, show_label=False, scale=2)
                        sc_n.input(fn=_rh_score, inputs=[sc_n, idx_s], outputs=[])
                        gr.Button("🚫", scale=1).click(fn=_rh_ban, inputs=[explorer_state, idx_s], outputs=[explorer_state])
                        gr.Button("⭐", scale=1).click(fn=_rh_save, inputs=[explorer_state, idx_s], outputs=[])
                        gr.Button("🛡️", scale=1).click(fn=_rh_protect, inputs=[explorer_state, idx_s], outputs=[explorer_state])
                        gr.Button("🔭", scale=1).click(fn=_rh_scout, inputs=[explorer_state, idx_s], outputs=[explorer_state])
                        gr.Button("💾", scale=1).click(fn=_rh_save_image, inputs=[url_s, idx_s], outputs=[])
                    art_md = gr.Markdown("")
                if gi % 4 == 3 or gi == 15: _rw.__exit__(None,None,None)
                cell_states.extend([idx_s, img_h, sc_n, url_s, art_md])
            def refresh_grid(batch, page, psize):
                if not batch:
                    ret = []; [ret.extend([-1,"",0.0,"",""]) for _ in range(16)]
                    return tuple(ret + [f"<style>{' '.join([f'.ga-{i}{{display:none!important}}' for i in range(16)])}</style>"])
                start = page*psize; end = min(start+psize,len(batch)); page_items = batch[start:end]
                scores = buffer_get_scores(); ret = []; css_parts = []
                for i in range(16):
                    if i < len(page_items):
                        item = page_items[i]; idx = item["idx"]
                        mk = f" {item['marker']}" if item.get("marker") else ""
                        sv = scores.get(str(idx),0); img = f'<img src="{item["url"]}" style="width:100%;border-radius:6px;display:block">'
                        ret.extend([idx, img, float(sv), item["url"], f"`#{idx}{mk}` {item['artist_str']}"])
                        css_parts.append(f".ga-{i}{{display:block!important}}")
                    else: ret.extend([-1,"",0.0,"",""]); css_parts.append(f".ga-{i}{{display:none!important}}")
                ret.append(f"<style>{' '.join(css_parts)}</style>"); return tuple(ret)
            _ga_out = cell_states + [grid_css]
            batch_state.change(fn=refresh_grid, inputs=[batch_state,page_state,page_size_state], outputs=_ga_out)
            page_state.change(fn=refresh_grid, inputs=[batch_state,page_state,page_size_state], outputs=_ga_out)
            page_size_state.change(fn=refresh_grid, inputs=[batch_state,page_state,page_size_state], outputs=_ga_out)
            # ── PCA ──
            with gr.Accordion("📊 PCA 触角分布图", open=False): pca_plot = gr.Plot(label="触角分布")
            # ── 快捷指令 ──
            with gr.Accordion("⚡ 快捷指令", open=False):
                with gr.Row(): cmd_idx = gr.Number(value=0, label="触角ID", precision=0)
                with gr.Row():
                    ban_btn = gr.Button("🚫 Ban此触角", size="sm"); save_btn = gr.Button("⭐ 收藏此触角", size="sm")
                    protect_btn = gr.Button("🛡️ 创建保护区", size="sm"); promote_btn = gr.Button("🔗 转正弱触角", size="sm")
                with gr.Row():
                    scout_cap = gr.Number(value=3, label="探测点容量", precision=0); scout_btn = gr.Button("🔭 创建探测点", size="sm")
                with gr.Row():
                    spread_n = gr.Number(value=3, label="扩散触角数", precision=0); spread_strength = gr.Slider(0.0,1.0,0.3,label="扩散强度")
                    spread_btn = gr.Button("🌱 扩散弱触角", size="sm")
                with gr.Row(): ban_desc = gr.Textbox(label="文本封禁描述", placeholder="输入描述如：丑陋风格, bad hands"); ban_text_btn = gr.Button("📝 文本封禁", size="sm")
                cmd_log = gr.Textbox(label="反馈", interactive=False)
            # ── Ban 区 ──
            with gr.Accordion("🚫 Ban区管理", open=False):
                ban_md = gr.Markdown("*暂无Ban区*"); refresh_ban_btn = gr.Button("🔄 刷新Ban区", size="sm")
                clear_ban_btn = gr.Button("🗑️ 清空所有Ban区", size="sm", variant="secondary")
                with gr.Row(): unban_id = gr.Number(value=0, label="解封ID", precision=0); unban_btn = gr.Button("🔓 解封", size="sm")
            # ── 保护区 ──
            with gr.Accordion("🛡️ 保护区管理", open=False):
                pz_md = gr.Markdown("*暂无保护区*"); refresh_pz_btn = gr.Button("🔄 刷新保护区", size="sm")
                clear_pz_btn = gr.Button("🗑️ 清空所有保护区", size="sm", variant="secondary")
                with gr.Row():
                    release_zid = gr.Number(value=0, label="保护区ID", precision=0)
                    release_btn = gr.Button("📤 释放最优到全局", size="sm"); release_del_btn = gr.Button("📤 释放并删除", size="sm")
                    pz_delete_btn = gr.Button("🗑️ 删除保护区", size="sm")
            # ── 保护区培育 ──
            with gr.Accordion("🧬 保护区培育（进化模式）", open=False):
                gr.Markdown("**规则**: 每轮生成N张图 → 评分 → 高分父代⭐ → 突变+杂交 → 循环至收敛")
                with gr.Row():
                    cultivate_zone_input = gr.Number(value=0, label="保护区ID", precision=0, minimum=0)
                    cultivate_activate_btn = gr.Button("🎯 加载保护区", variant="secondary", size="sm")
                    cultivate_pop_size = gr.Number(value=get_setting('anima_cpop', 4), visible=False)
                    cultivate_n_parents = gr.Number(value=get_setting('anima_cpar', 2), visible=False)
                with gr.Row(): cultivate_round_btn = gr.Button("🌱 培育一轮", variant="primary", scale=2)
                cultivate_artist_info = gr.Markdown("### 尚未开始培育"); cultivate_status = gr.Markdown("")
                cultivate_zone_state = gr.State(None); cultivate_batch_state = gr.State([])
                cultivate_page_state = gr.State(0); cultivate_page_size = gr.State(4)
                with gr.Row():
                    cultivate_psize = gr.Number(value=4, label="每页显示", precision=0, minimum=1, maximum=16)
                cultivate_psize.change(fn=lambda v: (int(v), 0), inputs=[cultivate_psize], outputs=[cultivate_page_size, cultivate_page_state])
                cultivate_grid_css = gr.HTML("<style>" + " ".join([f".cza-{i}{{display:none!important}}" for i in range(16)]) + "</style>")
                with gr.Row():
                    cultivate_prev_btn = gr.Button("◀", size="sm", scale=1)
                    cultivate_page_info = gr.Markdown("**0/0** 共0图"); cultivate_next_btn = gr.Button("▶", size="sm", scale=1)
                cultivate_cell_images, cultivate_cell_infos, cultivate_scores = [], [], []
                for czi in range(16):
                    if czi % 4 == 0: _czr = gr.Row().__enter__()
                    with gr.Column(elem_classes=[f"cza-{czi}"]):
                        cultivate_cell_images.append(gr.HTML("")); cultivate_cell_infos.append(gr.Markdown(""))
                        cultivate_scores.append(gr.Number(value=0, minimum=-10, maximum=10, step=1, precision=0, show_label=False, scale=2))
                    if czi % 4 == 3 or czi == 15: _czr.__exit__(None,None,None)
                with gr.Row(): cultivate_submit_btn = gr.Button("📊 提交全部评分", variant="primary")
            # ── 探测点 ──
            with gr.Accordion("🔭 探测点管理", open=False):
                scout_md = gr.Markdown("*暂无探测点*"); refresh_scout_btn = gr.Button("🔄 刷新探测点", size="sm")
                with gr.Row(): rm_scout_id = gr.Number(value=0, label="删除探测点ID", precision=0); rm_scout_btn = gr.Button("🗑️ 删除", size="sm")
            # ── 收藏 ──
            with gr.Accordion("⭐ 收藏", open=False):
                fav_md = gr.Markdown("*暂无收藏*"); refresh_fav_btn = gr.Button("🔄 刷新收藏", size="sm")
                with gr.Row(): fav_rm_idx = gr.Number(value=1, label="移除编号", precision=0, minimum=1); fav_rm_btn = gr.Button("🗑️ 移除", size="sm")

        # ========== 设置 Anima ==========
        with gr.Tab("⚙️ 设置 Anima"):
            gr.Markdown("## 设置 Anima")
            with gr.Row():
                comfy_test_addr = gr.Textbox(value=v8.COMFYUI_SERVER, label="ComfyUI 地址", info="ip:port")
                test_btn = gr.Button("🔌 测试连接", variant="secondary", scale=1)
                conn_result = gr.Textbox(label="连接状态", interactive=False)
            gr.Markdown("---"); gr.Markdown("### ⚙️ 设置")
            with gr.Accordion("🔌 ComfyUI 连接", open=False):
                s_srv = gr.Textbox(value=get_setting('anima_comfy', v8.COMFYUI_SERVER), label="ComfyUI 地址")
                s_wf = gr.Textbox(value=get_setting('anima_wf', str(v8.WORKFLOW_TEMPLATE_PATH)), label="工作流模板")
            with gr.Accordion("🎨 生图参数", open=False):
                s_mode = gr.Dropdown(choices=["turbo","base"], value=get_setting('anima_mode', v8.MODE), label="模式")
                with gr.Row():
                    s_w = gr.Number(value=get_setting('anima_w', v8.WIDTH), label="宽", precision=0)
                    s_h = gr.Number(value=get_setting('anima_h', v8.HEIGHT), label="高", precision=0)
                    s_st = gr.Number(value=get_setting('anima_st', v8.STEPS), label="步数", precision=0)
                    s_cf = gr.Number(value=get_setting('anima_cf', v8.CFG), label="CFG", precision=1)
                s_samp = gr.Dropdown(choices=["euler","euler_ancestral","dpmpp_2m","dpmpp_2s_ancestral","ddim","uni_pc","er_sde","ser_sde"], value=get_setting('anima_samp', v8.SAMPLER_NAME), label="采样器")
                s_sch = gr.Dropdown(choices=["simple","karras","exponential","sgm_uniform","beta27","beta57"], value=get_setting('anima_sch', v8.SCHEDULER), label="调度器")
                s_unet = gr.Textbox(value=get_setting('anima_unet', v8.UNET_NAME), label="UNet 模型名", info="ComfyUI models/unet/")
                s_pos = gr.Textbox(value=get_setting('anima_pos', v8.BASE_POSITIVE_PROMPT), label="正面提示词", lines=2)
                s_neg = gr.Textbox(value=get_setting('anima_neg', v8.BASE_NEGATIVE_PROMPT), label="负面提示词", lines=3)
            with gr.Accordion("🧬 黏菌探索器", open=False):
                with gr.Row():
                    s_nt = gr.Number(value=get_setting('anima_nt', v8.N_TENTACLES), label="触角总数", precision=0)
                    s_ev = gr.Number(value=get_setting('anima_ev', v8.EVAL_BUDGET), label="每轮生图数", precision=0)
                    s_mg = gr.Number(value=get_setting('anima_mg', v8.MAX_GENERATIONS), label="最大轮数", precision=0)
                s_sth = gr.Slider(0.5,1.0,value=get_setting('anima_sth', v8.SIMILARITY_THRESHOLD),step=0.01,label="相似度阈值")
                with gr.Row():
                    s_mc = gr.Number(value=get_setting('anima_mc', v8.MIXED_ARTISTS_COUNT), label="画师串长度", precision=0)
                    s_ss = gr.Slider(0.01,0.5,value=get_setting('anima_ss', v8.TENTACLE_STEP_SIZE),step=0.01,label="变异步长")
                    s_tmp = gr.Slider(0.1,5.0,value=get_setting('anima_temp', getattr(v8,'TEMPERATURE',1.0)),step=0.1,label="权重温度",info=">1主画师突出，<1更均匀")
            with gr.Accordion("🚫 Ban区参数", open=False):
                with gr.Row():
                    s_bs = gr.Slider(0.5,1.0,value=get_setting('anima_bs', v8.BAN_SELECT_THRESH),step=0.01,label="选择阈值")
                    s_bm = gr.Slider(0.5,1.0,value=get_setting('anima_bm', v8.BAN_MUTATE_THRESH),step=0.01,label="变异阈值")
                    s_br = gr.Slider(0.5,1.0,value=get_setting('anima_br', v8.BAN_REBIRTH_THRESH),step=0.01,label="重生阈值")
                with gr.Row():
                    s_bp = gr.Slider(0.5,5.0,value=get_setting('anima_bp', v8.BAN_PENALTY_COEFFICIENT),step=0.1,label="软斥力")
                    s_bd = gr.Slider(0.1,1.0,value=get_setting('anima_bd', v8.BAN_DECAY_RATE),step=0.05,label="衰减速率")
            with gr.Accordion("🛡️ 保护区参数", open=False):
                with gr.Row():
                    s_pp = gr.Number(value=get_setting('anima_pp', v8.PROTECT_POP_SIZE), label="菌丝数", precision=0)
                    s_pm = gr.Number(value=get_setting('anima_pm', v8.PROTECT_MAX_CANDIDATES), label="候选上限", precision=0)
                    s_pc = gr.Number(value=get_setting('anima_pc', v8.PROTECT_CONVERGE_ROUNDS), label="收敛轮数", precision=0)
                with gr.Row():
                    s_cp = gr.Number(value=get_setting('anima_cpop', 4), label="培育每轮生图数", precision=0, minimum=2, maximum=12)
                    s_cpr = gr.Number(value=get_setting('anima_cpar', 2), label="培育父代数", precision=0, minimum=1, maximum=4)
            with gr.Accordion("🔗 弱触角 & 探测点", open=False):
                with gr.Row():
                    s_wn = gr.Number(value=get_setting('anima_wn', v8.WEAK_CONSECUTIVE_NEGATIVE), label="弱触角触发", precision=0)
                    s_pr = gr.Number(value=get_setting('anima_pr', v8.PROMOTE_SUGGEST_ROUNDS), label="转正建议", precision=0)
                    s_pl = gr.Slider(0.0,1.0,value=get_setting('anima_pl', v8.PULL_STRENGTH),step=0.05,label="拉力")
                    s_mw = gr.Number(value=get_setting('anima_mw', v8.MAX_WEAK_TENTACLES), label="弱上限", precision=0)
            with gr.Accordion("📊 调试 & 显示", open=False):
                s_debug = gr.Checkbox(value=get_setting('anima_debug', v8.DEBUG_MODE), label="调试模式")
                with gr.Row():
                    s_plot = gr.Dropdown(choices=["full","lite"], value=get_setting('anima_plot', v8.PLOT_STYLE), label="绘图风格")
                    s_trace = gr.Number(value=get_setting('anima_trace', v8.TRACE_HISTORY_LEN), label="轨迹步数", precision=0)
            save_log = gr.Textbox(label="设置反馈", interactive=False)
            save_btn = gr.Button("💾 保存全部设置", variant="primary")
            # 关于向量库
            with gr.Accordion("📖 关于向量库", open=False):
                gr.Markdown(
                    """
                    ### Anima 向量库基于以下配置构建：
                    - **CLIP 模型**: `CLIPLoader` (Qwen)
                    - **编码器**: `CLIPTextEncode` + `SaveCondition` 节点
                    - **向量提取**: `torch.load` 读取 `.ckpt`，取 `last_hidden_state` mean-pool
                    - **索引方式**: FAISS 余弦相似度（IndexFlatIP, 向量 L2 归一化）
                    ### 换 CLIP 模型后重建
                    运行 `python build_webui_anima.py` 打开构建助手，按步骤操作。
                    > 向量空间由 CLIP 模型决定。换模型后必须重建。
                    """
                )
                build_launch_btn = gr.Button("🚀 启动向量库构建助手", variant="secondary")
                build_launch_msg = gr.Markdown("")
            # 导入导出
            with gr.Accordion("📦 导入导出进度", open=False):
                gr.Markdown("导出当前 Anima 探索进度到本地备份，或从备份文件恢复进度。")
                with gr.Row():
                    export_btn = gr.Button("📥 导出进度", variant="secondary")
                    export_file = gr.File(label="下载", interactive=False, file_count="single", file_types=[".json"])
                gr.Markdown("---")
                with gr.Row():
                    import_file = gr.File(label="选择备份文件", file_count="single", file_types=[".json"])
                    import_btn = gr.Button("📤 导入进度", variant="secondary")
                import_msg = gr.Markdown("")

        # ============================================================
        # 事件绑定
        # ============================================================
        def on_init(spread):
            buffer_clear(); buffer_flush_to_file()
            exp = create_explorer(spread); save_explorer_state(exp); gr.Info("🆕 新会话已初始化")
            return (exp, f"### {get_stats_text(exp, v8.load_protect_zones())}", [], 0, get_ban_md(), get_pz_md(), get_scout_md(exp))
        init_btn.click(fn=on_init, inputs=[spread_slider], outputs=[explorer_state, stats_display, batch_state, page_state, ban_md, pz_md, scout_md])

        def on_main_action(exp):
            import traceback
            try:
                if exp is None: gr.Warning("请先初始化探索器"); yield [], "### ❌ 请先初始化", None, exp, 0, ""; return
                gr.Info("开始探索...")
                server = get_setting('anima_comfy', v8.COMFYUI_SERVER)
                sd = buffer_get_scores()
                if sd: yield [], "### 探索中...", None, exp, 0, "⏳ 提交评分中..."; submit_scores(exp, sd); buffer_clear(); buffer_flush_to_file()
                for pt, bi, st, fig in run_explore_round_gen(exp, server):
                    sm = f"### {st}" if st else ""; pv = fig if fig is not None else gr.update()
                    yield bi, sm, pv, exp, 0, pt
                if bi: save_explorer_state(exp)
            except Exception as e:
                tb = traceback.format_exc(); print(f"[Anima] {e}\n{tb}"); gr.Error(str(e))
                yield [], f"### ❌ {e}", None, exp, 0, f"❌ {e}"
        main_btn.click(fn=on_main_action, inputs=[explorer_state], outputs=[batch_state, stats_display, pca_plot, explorer_state, page_state, progress_status])

        # -- 快捷指令 --
        def cmd_ban(exp, idx):
            if exp is None or idx<0 or idx>=len(exp.tentacles): return "❌ 无效ID"
            exp.ban_current_tentacle(int(idx)); return f"🚫 已封禁触角 {idx}"
        def cmd_save(exp, idx):
            if exp is None or idx<0 or idx>=len(exp.tentacles): return "❌ 无效ID"
            t = exp.tentacles[int(idx)]; astr = v8.vector_to_artist_string(t.vector); v8.save_favorite(astr, t.vector); return f"⭐ 已收藏"
        def cmd_protect(exp, idx):
            if exp is None or idx<0 or idx>=len(exp.tentacles): return "❌ 无效ID"
            t = exp.tentacles[int(idx)]; pzones = v8.load_protect_zones(); new_id = len(pzones)
            zone = v8.ProtectZone(new_id, t.vector.copy(), 0.15, v8.MIXED_ARTISTS_COUNT); pzones[new_id] = zone
            v8.save_protect_zones(pzones); return f"🛡️ 已创建保护区_{new_id}"
        def cmd_promote(exp, idx):
            if exp is None or idx<0 or idx>=len(exp.tentacles): return "❌ 无效ID"
            if not exp.tentacles[int(idx)].is_weak: return "⚠️ 不是弱触角"
            exp.promote_tentacle(int(idx)); return f"✅ 触角 {idx} 已转正"
        def cmd_scout(exp, idx, cap):
            if exp is None or idx<0 or idx>=len(exp.tentacles): return "❌ 无效ID"
            t = exp.tentacles[int(idx)]; new_id = max((sp['id'] for sp in exp.scout_points), default=-1)+1
            exp.scout_points.append({"id":new_id,"vector":t.vector.tolist(),"capacity":int(cap),"assigned":[],"active":True})
            v8.save_scout_points(exp.scout_points); return f"🔭 已创建探测点 {new_id}"
        def cmd_spread(exp, n, strength):
            if exp is None: return "❌ 未初始化"
            exp.spread_weakest_tentacles(int(n), strength); return f"🌱 已扩散 {int(n)} 个触角"

        def on_ban(exp, idx):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化", exp
            msg = cmd_ban(exp, int(idx)); gr.Info(msg); return msg, exp
        def on_save(exp, idx):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化"; msg = cmd_save(exp, int(idx)); gr.Info(msg); return msg
        def on_protect(exp, idx):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化", exp
            msg = cmd_protect(exp, int(idx)); gr.Info(msg); return msg, exp
        def on_promote(exp, idx):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化", exp
            msg = cmd_promote(exp, int(idx)); gr.Info(msg); return msg, exp
        def on_scout(exp, idx, cap):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化", exp
            msg = cmd_scout(exp, int(idx), int(cap)); gr.Info(msg); return msg, exp
        def on_spread(exp, n, strength):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化", exp
            msg = cmd_spread(exp, int(n), strength); gr.Info(msg); return msg, exp

        ban_btn.click(fn=on_ban, inputs=[explorer_state, cmd_idx], outputs=[cmd_log, explorer_state])
        save_btn.click(fn=on_save, inputs=[explorer_state, cmd_idx], outputs=[cmd_log])
        protect_btn.click(fn=on_protect, inputs=[explorer_state, cmd_idx], outputs=[cmd_log, explorer_state])
        promote_btn.click(fn=on_promote, inputs=[explorer_state, cmd_idx], outputs=[cmd_log, explorer_state])
        scout_btn.click(fn=on_scout, inputs=[explorer_state, cmd_idx, scout_cap], outputs=[cmd_log, explorer_state])
        spread_btn.click(fn=on_spread, inputs=[explorer_state, spread_n, spread_strength], outputs=[cmd_log, explorer_state])
        ban_text_btn.click(fn=lambda d: f"🚫 已封禁: {d[:50]}", inputs=[ban_desc], outputs=[cmd_log])

        # -- Ban 区 --
        refresh_ban_btn.click(fn=get_ban_md, outputs=[ban_md])
        def on_unban(bid):
            bl = v8.load_ban_list()
            for b in bl:
                if b['id'] == int(bid): b['active'] = False; v8.save_ban_list(bl); return f"🔓 已解封 ID {bid}", get_ban_md()
            return f"❌ 未找到 ID {bid}", get_ban_md()
        unban_btn.click(fn=on_unban, inputs=[unban_id], outputs=[cmd_log, ban_md])
        def on_clear_bans():
            bl = v8.load_ban_list(); [b.update(active=False) for b in bl]; v8.save_ban_list(bl); return get_ban_md()
        clear_ban_btn.click(fn=on_clear_bans, outputs=[ban_md])

        # -- 保护区 --
        refresh_pz_btn.click(fn=get_pz_md, outputs=[pz_md])
        def on_clear_pz():
            fp = _PROJ / "data/protect_zones_anima/protect_zones.json"
            if fp.exists(): fp.unlink(); return get_pz_md()
            return get_pz_md()
        clear_pz_btn.click(fn=on_clear_pz, outputs=[pz_md])
        def on_release(exp, zid, remove_after=False):
            if exp is None: gr.Warning("请先初始化"); return "请先初始化", exp
            pzones = v8.load_protect_zones(); zone = pzones.get(int(zid))
            if zone is None: return "❌ 未找到", exp
            if zone.best_vector is None: return "⚠️ 无最佳向量", exp
            exp.replace_weakest_with_vector(zone.best_vector)
            if remove_after: del pzones[int(zid)]; v8.save_protect_zones(pzones); return "✅ 已释放并删除", exp
            return "✅ 已释放", exp
        release_btn.click(fn=on_release, inputs=[explorer_state, release_zid], outputs=[cmd_log, explorer_state])
        def on_release_del(exp, zid): return on_release(exp, zid, remove_after=True)
        release_del_btn.click(fn=on_release_del, inputs=[explorer_state, release_zid], outputs=[cmd_log, explorer_state])
        def on_pz_delete(zid):
            pzones = v8.load_protect_zones()
            if int(zid) not in pzones: return f"❌ 未找到", get_pz_md()
            del pzones[int(zid)]; v8.save_protect_zones(pzones); return f"🗑️ 已删除", get_pz_md()
        pz_delete_btn.click(fn=on_pz_delete, inputs=[release_zid], outputs=[cmd_log, pz_md])

        # -- 培育 --
        def refresh_cultivate_grid(pop, page, psize):
            ret = []; css_parts = []; start = page*psize; end = min(start+psize, len(pop))
            for czi in range(16):
                if start <= czi < end:
                    r = pop[czi]; pm = " ⭐" if r.get('is_parent') else ""
                    img = f'<img src="{r["url"]}" style="width:100%;border-radius:6px;display:block">' if r.get('url') else f'<div style="width:100%;height:200px;border:1px dashed gray;border-radius:6px;display:flex;align-items:center;justify-content:center;color:gray;">❌</div>'
                    ret.extend([img, f"图{czi+1}{pm}: {r['artist_str'][:60]}...", 0.0])
                    css_parts.append(f".cza-{czi}{{display:block!important}}")
                else: ret.extend(["","",0.0]); css_parts.append(f".cza-{czi}{{display:none!important}}")
            ret.append(f"<style>{' '.join(css_parts)}</style>"); return tuple(ret)
        _cz_out = []
        for _i in range(16): _cz_out.extend([cultivate_cell_images[_i], cultivate_cell_infos[_i], cultivate_scores[_i]])
        _cz_out.append(cultivate_grid_css)
        cultivate_batch_state.change(fn=refresh_cultivate_grid, inputs=[cultivate_batch_state, cultivate_page_state, cultivate_page_size], outputs=_cz_out)
        cultivate_page_state.change(fn=refresh_cultivate_grid, inputs=[cultivate_batch_state, cultivate_page_state, cultivate_page_size], outputs=_cz_out)
        cultivate_page_size.change(fn=refresh_cultivate_grid, inputs=[cultivate_batch_state, cultivate_page_state, cultivate_page_size], outputs=_cz_out)
        def on_cultivate_prev(page, batch, psize): return max(0, page-1) if batch else 0
        def on_cultivate_next(page, batch, psize): return min(((len(batch)-1)//max(1,psize)), page+1) if batch else 0
        def on_cultivate_page_info(page, batch, psize):
            if not batch: return "**0/0** 共0图"
            return f"**{page+1}/{max(1,(len(batch)+psize-1)//psize)}** 共{len(batch)}图"
        cultivate_prev_btn.click(fn=on_cultivate_prev, inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size], outputs=[cultivate_page_state])
        cultivate_next_btn.click(fn=on_cultivate_next, inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size], outputs=[cultivate_page_state])
        cultivate_page_state.change(fn=on_cultivate_page_info, inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size], outputs=[cultivate_page_info])
        cultivate_batch_state.change(fn=on_cultivate_page_info, inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size], outputs=[cultivate_page_info])

        def on_cultivate_activate(zone_id, pop_size, n_parents):
            pzones = v8.load_protect_zones(); zone = pzones.get(int(zone_id))
            if zone is None: gr.Warning(f"未找到保护区 {zone_id}"); return None, "### ❌ 未找到", "", []
            info = f"### ✅ 保护区_{int(zone_id)}\n候选: {len(zone.candidate_artists)} | 最佳: {zone.best_score:.2f} | 已培育 {zone.generation} 轮 | {int(pop_size)}图/{int(n_parents)}父代"
            batch = zone._evo_population if hasattr(zone, '_evo_population') and zone._evo_population else []
            return zone, info, "", batch
        cultivate_activate_btn.click(fn=on_cultivate_activate, inputs=[cultivate_zone_input, cultivate_pop_size, cultivate_n_parents], outputs=[cultivate_zone_state, cultivate_artist_info, cultivate_status, cultivate_batch_state])

        def on_cultivate_round_gen(zone, pop_size):
            empty_g = "<p style='color:gray;text-align:center;'>等待培育</p>"
            if zone is None: yield ("### ❌ 请先加载", empty_g, ""); return
            if not hasattr(zone, '_evo_population') or not zone._evo_population: zone.evo_init(int(pop_size))
            population = zone._evo_population
            if zone.no_improve_rounds >= v8.PROTECT_CONVERGE_ROUNDS: yield ("### ✅ 已收敛", empty_g, "已收敛"); return
            server = get_setting('anima_comfy', v8.COMFYUI_SERVER); gen = zone.generation; total = len(population)
            yield (f"### 🧬 第{gen}轮培育中... {total}张图", empty_g, f"⏳ 排队...")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            pending = {}
            with ThreadPoolExecutor(max_workers=total) as exc:
                futures = {}
                for c in population:
                    prompt = f"{v8.BASE_POSITIVE_PROMPT}, {c['artist_str']}"
                    fname = f"protect_{zone.zone_id}_gen{gen}_c{c['id']}"; seed = gen*1000+zone.zone_id+c['id']
                    wf = v8.build_workflow(prompt, v8.BASE_NEGATIVE_PROMPT, seed, fname)
                    fut = exc.submit(lambda w=wf: requests.post(f"http://{server}/prompt", json={"prompt": w}, timeout=10).json()['prompt_id'])
                    futures[fut] = c
                for fut in as_completed(futures):
                    c = futures[fut]
                    try: pid = fut.result(); pending[pid] = c
                    except: c['url'] = None; c['error'] = 'submit_fail'
            taco_times = []; batch_t0 = time.time(); completed = 0
            while pending:
                elapsed = time.time()-batch_t0; max_t = max(taco_times) if taco_times else 0
                if taco_times and max_t > elapsed: eta_str = f" | 预计剩余 ~{max_t-elapsed:.0f}s"
                elif taco_times: eta_str = " | 预计即将完成"
                else: eta_str = " | 剩余时间...（正在计算）"
                yield (f"### ⏳ {completed}/{total} ({elapsed:.0f}s){eta_str}", empty_g, "")
                pids = list(pending.keys())
                try: h = requests.get(f"http://{server}/history", timeout=10).json()
                except Exception: time.sleep(1); continue
                for pid in pids:
                    if pid in h:
                        c = pending.pop(pid); c['url'] = None
                        err = _check_comfyui_error(h[pid])
                        if err:
                            if 'lora' in err.lower() and v8.MODE == 'turbo':
                                v8.MODE = 'base'; v8.WORKFLOW_TEMPLATE_PATH = Path("workflows/t2i_anima_base.json")
                                v8.STEPS = 30; v8.CFG = 5; gr.Warning("已切换为base，请重新培育")
                                yield (f"### ❌ LoRA缺失，已切换为base模式", "", ""); return
                            yield (f"### ❌ 错误: {err}", "", ""); return
                        for out in h[pid].get('outputs',{}).values():
                            if 'images' in out: c['url'] = img_info_to_url(out['images'][0], server); break
                        taco_times.append(time.time()-batch_t0-sum(taco_times)); completed += 1
                if pending and time.time()-batch_t0 > 180:
                    for pid, c in list(pending.items()): c['url'] = None; c['error'] = '超时'; pending.pop(pid); completed += 1
                if pending: time.sleep(1)
            for c in population:
                if 'url' not in c: c['url'] = None
            yield ("### ✅ 全部生成完毕，等待评分", empty_g, population)
        cultivate_round_btn.click(fn=on_cultivate_round_gen, inputs=[cultivate_zone_state, cultivate_pop_size], outputs=[cultivate_artist_info, cultivate_status, cultivate_batch_state])
        # Oops: the gallery output is mangled. Let me fix below.

        def on_cultivate_submit(zone, n_parents, *scores):
            if zone is None: gr.Warning("请先加载"); return zone, ""
            if not hasattr(zone, '_evo_population') or not zone._evo_population: gr.Warning("请先培育"); return zone, ""
            pop = zone._evo_population; sd = {}
            for i in range(len(pop)): v = scores[i] if i < len(scores) else 0; sd[i] = max(-10., min(10., float(v or 0)))
            result = zone.evo_submit_scores(sd, int(n_parents))
            pzones = v8.load_protect_zones(); pzones[zone.zone_id] = zone; v8.save_protect_zones(pzones)
            parents = result.get('parents',[]); ps = ", ".join(f"图{p+1}⭐" for p in parents)
            status = f"**📊 第{result['generation']-1}轮**\n父代: {ps}\n最佳: {result['best_score']:.2f}\n无提升: {result['no_improve_rounds']}轮"
            if result['converged']: status += "\n\n✅ 已收敛！"
            else: status += "\n\n💡 继续培育"
            return zone, status
        cultivate_submit_btn.click(fn=on_cultivate_submit, inputs=[cultivate_zone_state, cultivate_n_parents] + cultivate_scores, outputs=[cultivate_zone_state, cultivate_status])

        # -- 探测点 --
        refresh_scout_btn.click(fn=get_scout_md, inputs=[explorer_state], outputs=[scout_md])
        def on_rm_scout(exp, sid):
            if exp is None: return "请先初始化", exp
            for sp in exp.scout_points:
                if sp['id'] == int(sid): sp['active'] = False; v8.save_scout_points(exp.scout_points); return f"🗑️ 已删除", exp
            return f"❌ 未找到", exp
        rm_scout_btn.click(fn=on_rm_scout, inputs=[explorer_state, rm_scout_id], outputs=[cmd_log, explorer_state])

        # -- 收藏 --
        refresh_fav_btn.click(fn=get_fav_md, outputs=[fav_md])
        def on_fav_remove(idx):
            fp = _PROJ / "data/favorites_anima.json"
            if not fp.exists(): return "📋 暂无收藏", get_fav_md()
            with open(fp, 'r', encoding='utf-8') as f: favs = json.load(f)
            if int(idx)<1 or int(idx)>len(favs): return f"❌ 无效编号", get_fav_md()
            favs.pop(int(idx)-1);
            with open(fp, 'w', encoding='utf-8') as f: json.dump(favs, f, indent=2, ensure_ascii=False)
            return "🗑️ 已移除", get_fav_md()
        fav_rm_btn.click(fn=on_fav_remove, inputs=[fav_rm_idx], outputs=[cmd_log, fav_md])

        # -- 连接测试 --
        def test_conn(addr):
            try: r = requests.get(f"http://{addr}/history", timeout=10); return f"✅ {addr}" if r.status_code==200 else f"⚠️ {r.status_code}"
            except Exception as e: return f"❌ {e}"
        test_btn.click(fn=test_conn, inputs=[comfy_test_addr], outputs=[conn_result])

        # -- 保存设置 --
        def on_save(mode, srv, wf, w, h, st, cf, samp, sch, unet, pos, neg, nt, ev, mg, sth, mc, ss, temp,
                    bs, bm, br, bp, bd, pp, pm, pc, wn, pr, pl, mw, debug, plot, trace):
            s = {k:v for k,v in zip(
                ['anima_mode','anima_comfy','anima_wf','anima_w','anima_h','anima_st','anima_cf','anima_samp','anima_sch','anima_unet','anima_pos','anima_neg',
                 'anima_nt','anima_ev','anima_mg','anima_sth','anima_mc','anima_ss','anima_temp',
                 'anima_bs','anima_bm','anima_br','anima_bp','anima_bd','anima_pp','anima_pm','anima_pc',
                 'anima_wn','anima_pr','anima_pl','anima_mw','anima_debug','anima_plot','anima_trace'],
                [mode,srv,wf,int(w),int(h),int(st),float(cf),samp,sch,unet,pos,neg,
                 int(nt),int(ev),int(mg),float(sth),int(mc),float(ss),float(temp),
                 float(bs),float(bm),float(br),float(bp),float(bd),int(pp),int(pm),int(pc),
                 int(wn),int(pr),float(pl),int(mw),bool(debug),plot,int(trace)])}
            save_settings(s)
            v8.MODE = mode; v8.UNET_NAME = unet; v8.COMFYUI_SERVER = srv
            v8.WIDTH = int(w); v8.HEIGHT = int(h); v8.STEPS = int(st); v8.CFG = float(cf)
            v8.SAMPLER_NAME = samp; v8.SCHEDULER = sch
            v8.BASE_POSITIVE_PROMPT = pos; v8.BASE_NEGATIVE_PROMPT = neg
            v8.N_TENTACLES = int(nt); v8.EVAL_BUDGET = int(ev); v8.MAX_GENERATIONS = int(mg)
            v8.SIMILARITY_THRESHOLD = float(sth); v8.MIXED_ARTISTS_COUNT = int(mc); v8.TENTACLE_STEP_SIZE = float(ss); v8.TEMPERATURE = float(temp)
            v8.BAN_SELECT_THRESH = float(bs); v8.BAN_MUTATE_THRESH = float(bm); v8.BAN_REBIRTH_THRESH = float(br)
            v8.BAN_PENALTY_COEFFICIENT = float(bp); v8.BAN_DECAY_RATE = float(bd)
            v8.PROTECT_POP_SIZE = int(pp); v8.PROTECT_MAX_CANDIDATES = int(pm); v8.PROTECT_CONVERGE_ROUNDS = int(pc)
            v8.WEAK_CONSECUTIVE_NEGATIVE = int(wn); v8.PROMOTE_SUGGEST_ROUNDS = int(pr)
            v8.PULL_STRENGTH = float(pl); v8.MAX_WEAK_TENTACLES = int(mw)
            v8.DEBUG_MODE = bool(debug); v8.PLOT_STYLE = plot; v8.TRACE_HISTORY_LEN = int(trace)
            if mode == "turbo": v8.WORKFLOW_TEMPLATE_PATH = Path("workflows/t2i_anima_turbo.json")
            else: v8.WORKFLOW_TEMPLATE_PATH = Path("workflows/t2i_anima_base.json")
            return "✅ Anima 设置已保存（部分参数需下次重启程序或初始化新会话）"
        save_btn.click(fn=on_save,
            inputs=[s_mode,s_srv,s_wf,s_w,s_h,s_st,s_cf,s_samp,s_sch,s_unet,s_pos,s_neg,s_nt,s_ev,s_mg,s_sth,s_mc,s_ss, s_tmp,
                    s_bs,s_bm,s_br,s_bp,s_bd,s_pp,s_pm,s_pc,s_wn,s_pr,s_pl,s_mw,s_debug,s_plot,s_trace],
            outputs=[save_log])

        # -- 保存/载入 Anima 探索状态 --
        EXPLORER_SAVE_ANIMA = _PROJ / "data/explorer_save_anima.json"
        def save_explorer_state(exp):
            if exp is None: return
            data = {"version":1,"n_tentacles":exp.n_tentacles,"eval_budget":exp.eval_budget,"generation":exp.generation,
                    "global_best":None,"tentacles":[],"scout_points":exp.scout_points}
            if exp.global_best is not None:
                data["global_best"] = {"indices":exp.global_best[0].tolist(),"weights":exp.global_best[1].tolist(),"score":exp.global_best[2]}
            for t in exp.tentacles:
                data["tentacles"].append({"indices":t.indices.tolist(),"weights":t.weights.tolist(),"birth_gen":t.birth_gen,
                    "last_eval_gen":t.last_eval_gen,"score_history":[[g,s] for g,s in t.score_history],
                    "vector":t.vector.tolist(),"step_size":t.step_size,"active":t.active,"is_weak":t.is_weak,"scout_id":t.scout_id})
            with open(EXPLORER_SAVE_ANIMA,'w',encoding='utf-8') as f: json.dump(data,f,indent=2,ensure_ascii=False)
        def load_explorer_state():
            if not EXPLORER_SAVE_ANIMA.exists(): return None
            try:
                with open(EXPLORER_SAVE_ANIMA,'r',encoding='utf-8') as f: data = json.load(f)
                exp = v8.SlimeMoldExplorerAnima.__new__(v8.SlimeMoldExplorerAnima)
                exp.n_tentacles=data["n_tentacles"];exp.eval_budget=data["eval_budget"];exp.generation=data["generation"]
                exp.recent_evaluated_vectors=[]
                if data.get("global_best"):
                    exp.global_best=(np.array(data["global_best"]["indices"],dtype=np.int64),
                                    np.array(data["global_best"]["weights"],dtype=np.float32),data["global_best"]["score"])
                else: exp.global_best=None
                exp.tentacles=[]
                for td in data["tentacles"]:
                    t=v8.Tentacle.__new__(v8.Tentacle)
                    t.indices=np.array(td["indices"],dtype=np.int64);t.weights=np.array(td["weights"],dtype=np.float32)
                    t.birth_gen=td.get("birth_gen",0);t.last_eval_gen=td.get("last_eval_gen",0)
                    t.score_history=[(g,s) for g,s in td.get("score_history",[])]
                    t.vector=np.array(td["vector"],dtype=np.float32);t.step_size=td.get("step_size",0.085)
                    t.active=td.get("active",True);t.is_weak=td.get("is_weak",False)
                    t.scout_id=td.get("scout_id");t.trace=deque(maxlen=v8.TRACE_HISTORY_LEN);exp.tentacles.append(t)
                exp.scout_points=data.get("scout_points",[])
                return exp
            except Exception:
                try:EXPLORER_SAVE_ANIMA.unlink()
                except:pass
                return None

        # -- 启动构建助手 --
        def on_launch_build():
            import subprocess; cwd=Path(__file__).parent
            try:
                subprocess.Popen([sys.executable,"build_webui_anima.py"],cwd=str(cwd))
                gr.Info("Anima构建助手已启动->http://127.0.0.1:17327"); return "OK->http://127.0.0.1:17327"
            except Exception as e: gr.Error(str(e)); return f"FAIL:{e}"
        build_launch_btn.click(fn=on_launch_build,outputs=[build_launch_msg])

        # -- 导入导出 --
        def on_export():
            src=EXPLORER_SAVE_ANIMA
            if not src.exists():gr.Warning("暂无进度");return None
            import shutil;exp_dir=_PROJ / "data/exports";exp_dir.mkdir(parents=True,exist_ok=True)
            ts=time.strftime("%Y%m%d_%H%M%S")
            try:
                with open(src,'r',encoding='utf-8') as f:gen=json.load(f).get("generation","?")
            except:gen="?"
            dst=exp_dir/f"explorer_anima_gen{gen}_{ts}.json";shutil.copy2(src,dst);return str(dst)
        export_btn.click(fn=on_export,outputs=[export_file])

        def on_import(f):
            if f is None:gr.Warning("请先选择文件");return "### 未选择"
            import shutil
            if isinstance(f,dict):src=Path(f.get('name',''))
            elif isinstance(f,str):src=Path(f)
            else:src=Path(str(f))
            if not src.exists() or src.suffix!='.json':return "### 文件无效"
            try:
                with open(src,'r',encoding='utf-8') as fh:data=json.load(fh)
                if not isinstance(data,dict)or"tentacles"not in data or"generation"not in data:return "### 格式无效"
            except:return "### 无法解析"
            shutil.copy2(src,EXPLORER_SAVE_ANIMA)
            gr.Info(f"已导入第{data.get('generation','?')}代，请刷新");return f"### 已导入\n\n**请刷新页面**"
        import_btn.click(fn=on_import,inputs=[import_file],outputs=[import_msg])

        # -- 应用设置到 Anima V8 --
        def apply_v8_settings(s):
            m={'anima_mode':'MODE','anima_comfy':'COMFYUI_SERVER','anima_wf':'WORKFLOW_TEMPLATE_PATH',
               'anima_w':'WIDTH','anima_h':'HEIGHT','anima_st':'STEPS','anima_cf':'CFG',
               'anima_samp':'SAMPLER_NAME','anima_sch':'SCHEDULER','anima_unet':'UNET_NAME',
               'anima_pos':'BASE_POSITIVE_PROMPT','anima_neg':'BASE_NEGATIVE_PROMPT',
               'anima_nt':'N_TENTACLES','anima_ev':'EVAL_BUDGET','anima_mg':'MAX_GENERATIONS',
               'anima_sth':'SIMILARITY_THRESHOLD','anima_mc':'MIXED_ARTISTS_COUNT','anima_ss':'TENTACLE_STEP_SIZE',
               'anima_temp':'TEMPERATURE',
               'anima_bs':'BAN_SELECT_THRESH','anima_bm':'BAN_MUTATE_THRESH','anima_br':'BAN_REBIRTH_THRESH',
               'anima_bp':'BAN_PENALTY_COEFFICIENT','anima_bd':'BAN_DECAY_RATE',
               'anima_pp':'PROTECT_POP_SIZE','anima_pm':'PROTECT_MAX_CANDIDATES','anima_pc':'PROTECT_CONVERGE_ROUNDS',
               'anima_wn':'WEAK_CONSECUTIVE_NEGATIVE','anima_pr':'PROMOTE_SUGGEST_ROUNDS',
               'anima_pl':'PULL_STRENGTH','anima_mw':'MAX_WEAK_TENTACLES',
               'anima_debug':'DEBUG_MODE','anima_plot':'PLOT_STYLE','anima_trace':'TRACE_HISTORY_LEN'}
            for k,vn in m.items():
                if k in s: setattr(v8,vn,s[k])

        # -- 自动加载 --
        def on_auto_load():
            try:
                s=load_settings(); apply_v8_settings(s)
                spread=s.get('spread_anima',0.5)
                buffer_clear();buffer_flush_to_file()
                exp=load_explorer_state()
                if exp is not None: gr.Info(f"已恢复Anima探索进度(第{exp.generation}代)")
                else: exp=create_explorer(spread);save_explorer_state(exp)
                batch=getattr(exp,"last_batch",[])
                pz=v8.load_protect_zones()
                return (exp,f"### {get_stats_text(exp,pz)}",batch,0,get_ban_md(),get_pz_md(),get_scout_md(exp),"")
            except Exception as e:
                import traceback;traceback.print_exc()
                exp=create_explorer(0.5);pz=v8.load_protect_zones();save_explorer_state(exp)
                return (exp,f"### {get_stats_text(exp,pz)}",[],0,get_ban_md(),get_pz_md(),get_scout_md(exp),"")
        app.load(fn=on_auto_load, outputs=[explorer_state,stats_display,batch_state,page_state,ban_md,pz_md,scout_md,progress_status])

    return app

if __name__ == "__main__":
    if "--debug" in sys.argv:
        v8.DEBUG_MODE = True
        print("[DEBUG] Anima 调试模式已开启")
    print("🧫 Anima 启动...")
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=17326, share=False, css=CSS, theme=gr.themes.Soft())
