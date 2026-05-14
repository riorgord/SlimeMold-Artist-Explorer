#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
画师串小助手 - Gradio WebUI
黏菌风格探索器图形界面
"""

import gradio as gr
import numpy as np
import requests
import time
import json
import os
from collections import deque
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Windows GBK 编码修复（V8的emoji打印需要）
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'gb2312'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJ = Path(__file__).parent.parent

# ===============================================
# 向量库完整性检查
# ===============================================
_VECTOR_DIR = _PROJ / "outputs_2048"
_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
_VECTOR_REQUIRED = [
    _VECTOR_DIR / "artist_vectors_2048.npy",
    _VECTOR_DIR / "artist_ids_2048.json",
    _VECTOR_DIR / "artist_index_2048.faiss",
    _VECTOR_DIR / "artist_metadata_2048.csv",
]
_VECTOR_MISSING = [f for f in _VECTOR_REQUIRED if not f.exists()]
if _VECTOR_MISSING:
    print("=" * 60)
    print("❌ 向量库文件缺失！")
    print("   目录 outputs_2048/ 已创建")
    print("   缺失文件：")
    for _m in _VECTOR_MISSING:
        print(f"   - {_m.name}")
    print("=" * 60)
    print("是否现在启动向量库构建助手？(y/n): ", end="")
    try:
        ans = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans in ("y", "yes"):
        import subprocess
        print("🚀 正在启动 build_webui.py ...")
        subprocess.Popen([sys.executable, str(_PROJ / "webui/build_webui.py")])
    sys.exit(1)

# ===============================================
# 导入 V8 核心引擎
# ===============================================
sys.path.insert(0, str(_PROJ))
from engines.interactive_style_explorer8 import (
    SlimeMoldExplorerV6, ProtectZone, Tentacle,
    vectors_all, artist_ids_all, index, metadata_df, id_to_name, name_to_id,
    vector_to_artist_string, blend_from_indices,
    build_workflow, queue_prompt, wait_for_prompt,
    encode_text_to_vector,
    load_ban_list, save_ban_list,
    load_protect_zones, save_protect_zones,
    load_scout_points, save_scout_points, save_favorite,
    add_ban, is_near_ban, decay_ban_penalty,
    plot_tentacles, pca_model,
    # 所有配置参数（供设置面板读取默认值）
    COMFYUI_SERVER, COMFYUI_OUTPUT_DIR, WORKFLOW_TEMPLATE_PATH, VECTOR_DIR,
    BASE_POSITIVE_PROMPT, BASE_NEGATIVE_PROMPT, RUN_OUTPUT_DIR,
    N_TENTACLES, EVAL_BUDGET, MAX_GENERATIONS, SIMILARITY_THRESHOLD, MIXED_ARTISTS_COUNT,
    BAN_SELECT_THRESH, BAN_MUTATE_THRESH, BAN_REBIRTH_THRESH,
    BAN_PENALTY_COEFFICIENT, BAN_DECAY_RATE, STEP_SIZE, TEMPERATURE,
    PROTECT_POP_SIZE, PROTECT_MAX_CANDIDATES, PROTECT_CONVERGE_ROUNDS,
    WEAK_CONSECUTIVE_NEGATIVE, PROMOTE_SUGGEST_ROUNDS, PULL_STRENGTH, MAX_WEAK_TENTACLES,
    DEBUG_MODE, PLOT_STYLE, TRACE_HISTORY_LEN,
    WIDTH, HEIGHT, STEPS, CFG, SAMPLER_NAME, SCHEDULER, CHECKPOINT_NAME,
)

WIDTH = 1024
HEIGHT = 1024
STEPS = 20
CFG = 5
SAMPLER_NAME = "euler_ancestral"
SCHEDULER = "simple"

def _check_comfyui_error(res) -> Optional[str]:
    """检查 ComfyUI 返回的错误"""
    status = res.get('status', {})
    if status.get('status_str') == 'error':
        msgs = status.get('messages', [])
        for m in msgs:
            msg_text = str(m) if isinstance(m, str) else str(m[1]) if isinstance(m, (list,tuple)) and len(m)>1 else str(m)
            msg_lower = msg_text.lower()
            if 'lora' in msg_lower:
                return f"⚠️ LoRA 缺失！{msg_text[:200]}"
            if 'model' in msg_lower or 'checkpoint' in msg_lower or 'unet' in msg_lower or 'load' in msg_lower:
                return f"⚠️ 模型加载失败！请确认设置里填的底模文件名是否正确、是否在 ComfyUI models/checkpoints/ 目录下\n{msg_text[:200]}"
            return f"⚠️ ComfyUI 执行错误：{msg_text[:200]}"
    return None

os.makedirs(_PROJ / "data", exist_ok=True)  # 保证运行时数据目录存在
os.makedirs(_PROJ / "data/webui_output", exist_ok=True)


# ===============================================
# 设置持久化
# ===============================================
SETTINGS_FILE = _PROJ / "data/webui_settings.json"


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_settings(settings: dict):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_setting(key: str, default):
    s = load_settings()
    return s.get(key, default)


# ===============================================
# 评分缓存（翻页时暂存分数）
# ===============================================
SCORES_CACHE_DIR = _PROJ / "data/cache"
SCORES_CACHE_PATH = SCORES_CACHE_DIR / "scores_cache.json"


def load_scores_cache() -> dict:
    if SCORES_CACHE_PATH.exists():
        try:
            with open(SCORES_CACHE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if DEBUG_MODE:
                print(f"[DEBUG] load_scores_cache: {len(data)} entries loaded")
            return data
        except Exception:
            if DEBUG_MODE:
                print(f"[DEBUG] load_scores_cache: corrupt, returning empty")
            return {}
    if DEBUG_MODE:
        print("[DEBUG] load_scores_cache: no cache file")
    return {}


def save_scores_cache(scores: dict):
    SCORES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False)
    if DEBUG_MODE:
        print(f"[DEBUG] save_scores_cache: {len(scores)} entries written")


def clear_scores_cache():
    if SCORES_CACHE_PATH.exists():
        SCORES_CACHE_PATH.unlink()
        if DEBUG_MODE:
            print("[DEBUG] clear_scores_cache: deleted")
    else:
        if DEBUG_MODE:
            print("[DEBUG] clear_scores_cache: no cache to clear")


# ===============================================
# 评分内存缓冲（去抖动：不每次change都写文件）
# ===============================================
_score_buffer: Dict[str, float] = {}


def buffer_save_score(idx: int, value: float):
    _score_buffer[str(idx)] = value


def buffer_get_scores() -> dict:
    return _score_buffer


def buffer_clear():
    _score_buffer.clear()


def buffer_flush_to_file():
    """翻页/提交时持久化到cache文件"""
    SCORES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(_score_buffer, f, ensure_ascii=False)
    if DEBUG_MODE:
        print(f"[DEBUG] buffer_flush: {len(_score_buffer)} entries written to cache file")


def apply_settings_to_v8(settings: dict):
    """将设置写入 V8 模块变量（全局生效，新探索器实例受影响）"""
    import engines.interactive_style_explorer8 as v8
    mappings = {
        'comfyui_server': 'COMFYUI_SERVER',
        'base_positive': 'BASE_POSITIVE_PROMPT',
        'base_negative': 'BASE_NEGATIVE_PROMPT',
        'width': 'WIDTH',
        'height': 'HEIGHT',
        'steps': 'STEPS',
        'cfg': 'CFG',
        'sampler': 'SAMPLER_NAME',
        'scheduler': 'SCHEDULER',
        'checkpoint_name': 'CHECKPOINT_NAME',
        'n_tentacles': 'N_TENTACLES',
        'eval_budget': 'EVAL_BUDGET',
        'max_generations': 'MAX_GENERATIONS',
        'similarity_thresh': 'SIMILARITY_THRESHOLD',
        'mixed_count': 'MIXED_ARTISTS_COUNT',
        'step_size': 'STEP_SIZE',
        'temperature': 'TEMPERATURE',
        'ban_select_thresh': 'BAN_SELECT_THRESH',
        'ban_mutate_thresh': 'BAN_MUTATE_THRESH',
        'ban_rebirth_thresh': 'BAN_REBIRTH_THRESH',
        'ban_penalty_coef': 'BAN_PENALTY_COEFFICIENT',
        'ban_decay': 'BAN_DECAY_RATE',
        'protect_pop': 'PROTECT_POP_SIZE',
        'protect_max_cand': 'PROTECT_MAX_CANDIDATES',
        'protect_converge': 'PROTECT_CONVERGE_ROUNDS',
        'weak_negative': 'WEAK_CONSECUTIVE_NEGATIVE',
        'promote_rounds': 'PROMOTE_SUGGEST_ROUNDS',
        'pull_strength': 'PULL_STRENGTH',
        'max_weak': 'MAX_WEAK_TENTACLES',
        'debug_mode': 'DEBUG_MODE',
        'plot_style': 'PLOT_STYLE',
        'trace_len': 'TRACE_HISTORY_LEN',
    }
    for key, varname in mappings.items():
        if key in settings:
            setattr(v8, varname, settings[key])
    # 同步到当前模块的全局变量
    for key, varname in mappings.items():
        if key in settings:
            globals()[varname] = settings[key]


# ===============================================
# 图片地址工具：用 ComfyUI /view API，不依赖本地路径
# ===============================================
def img_info_to_url(img_info: dict, server: str = None) -> str:
    srv = server or COMFYUI_SERVER
    filename = img_info['filename']
    subfolder = img_info.get('subfolder', '')
    img_type = img_info.get('type', 'output')
    return f"http://{srv}/view?filename={filename}&subfolder={subfolder}&type={img_type}"


# ===============================================
# 探索器包装函数
# ===============================================
def create_explorer(spread: float) -> SlimeMoldExplorerV6:
    if DEBUG_MODE:
        print(f"[DEBUG] create_explorer: n_tentacles={N_TENTACLES}, eval_budget={EVAL_BUDGET}, spread={spread:.2f}")
    return SlimeMoldExplorerV6(
        n_tentacles=N_TENTACLES,
        eval_budget=EVAL_BUDGET,
        spread=spread
    )


# ===============================================
# 探索器状态持久化（保存/载入）
# ===============================================
EXPLORER_SAVE_PATH = _PROJ / "data/explorer_save.json"


def save_explorer_state(explorer: SlimeMoldExplorerV6):
    """保存探索器状态到 JSON（numpy→list，下次启动可恢复）"""
    data = {
        "version": 1,
        "n_tentacles": explorer.n_tentacles,
        "eval_budget": explorer.eval_budget,
        "generation": explorer.generation,
        "global_best": None,
        "tentacles": [],
        "scout_points": explorer.scout_points,
        "last_batch": getattr(explorer, "last_batch", []),
    }
    if explorer.global_best is not None:
        data["global_best"] = {
            "indices": explorer.global_best[0].tolist(),
            "weights": explorer.global_best[1].tolist(),
            "score": explorer.global_best[2],
        }
    for t in explorer.tentacles:
        data["tentacles"].append({
            "indices": t.indices.tolist(),
            "weights": t.weights.tolist(),
            "birth_gen": t.birth_gen,
            "last_eval_gen": t.last_eval_gen,
            "score_history": [[g, s] for g, s in t.score_history],
            "vector": t.vector.tolist(),
            "step_size": t.step_size,
            "active": t.active,
            "is_weak": t.is_weak,
            "scout_id": t.scout_id,
        })
    with open(EXPLORER_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if DEBUG_MODE:
        print(f"[DEBUG] save_explorer_state: gen={explorer.generation}, {len(explorer.tentacles)} tentacles saved")


def load_explorer_state() -> Optional[SlimeMoldExplorerV6]:
    """从 JSON 恢复探索器状态，失败时返回 None"""
    if not EXPLORER_SAVE_PATH.exists():
        return None
    try:
        with open(EXPLORER_SAVE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] load_explorer_state: 文件损坏 - {e}")
        return None
    # 反序列化可能因存档结构不匹配而失败，整体保护
    try:
        explorer = SlimeMoldExplorerV6.__new__(SlimeMoldExplorerV6)
        explorer.n_tentacles = data["n_tentacles"]
        explorer.eval_budget = data["eval_budget"]
        explorer.generation = data["generation"]
        explorer.recent_evaluated_vectors = []
        if data.get("global_best"):
            explorer.global_best = (
                np.array(data["global_best"]["indices"], dtype=np.int64),
                np.array(data["global_best"]["weights"], dtype=np.float32),
                data["global_best"]["score"],
            )
        else:
            explorer.global_best = None
        explorer.tentacles = []
        for t_data in data["tentacles"]:
            t = Tentacle.__new__(Tentacle)
            t.indices = np.array(t_data["indices"], dtype=np.int64)
            t.weights = np.array(t_data["weights"], dtype=np.float32)
            t.birth_gen = t_data.get("birth_gen", 0)
            t.last_eval_gen = t_data.get("last_eval_gen", 0)
            t.score_history = [(g, s) for g, s in t_data.get("score_history", [])]
            t.vector = np.array(t_data["vector"], dtype=np.float32)
            t.step_size = t_data.get("step_size", 0.12)
            t.active = t_data.get("active", True)
            t.is_weak = t_data.get("is_weak", False)
            t.scout_id = t_data.get("scout_id")
            t.trace = deque(maxlen=TRACE_HISTORY_LEN)
            explorer.tentacles.append(t)
        explorer.scout_points = data.get("scout_points", load_scout_points())
        explorer.last_batch = data.get("last_batch", [])
        if DEBUG_MODE:
            print(f"[DEBUG] load_explorer_state: gen={explorer.generation}, {len(explorer.tentacles)} tentacles loaded, "
                  f"batch={len(explorer.last_batch)} images")
        return explorer
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] load_explorer_state: 反序列化失败 - {e}")
            import traceback
            traceback.print_exc()
        # 损坏的存档文件移除，避免下次继续报错
        try:
            EXPLORER_SAVE_PATH.unlink()
        except Exception:
            pass
        return None


def get_stats_text(explorer: SlimeMoldExplorerV6, pzones: dict) -> str:
    if explorer is None:
        return "尚未初始化"
    act, art, clus = explorer.stats()
    weak = sum(1 for t in explorer.tentacles if t.is_weak)
    if DEBUG_MODE:
        print(f"[DEBUG] stats: gen={explorer.generation}, 活跃={act}/{explorer.n_tentacles}, "
              f"弱={weak}, 画师={art}, 簇={clus}, 探测点={len(explorer.scout_points)}, 保护区={len(pzones)}")
    return (
        f"🟢 活跃 {act}/{explorer.n_tentacles}　"
        f"🎨 覆盖画师 {art}　"
        f"📊 覆盖簇 {clus}　"
        f"🔗 弱触角 {weak}　"
        f"🔭 探测点 {len(explorer.scout_points)}　"
        f"🛡️ 保护区 {len(pzones)}　"
        f"🧬 第 {explorer.generation} 代"
    )


def get_tentacle_marker(t) -> str:
    """返回触角状态标记（空串表示无标记）"""
    marks = []
    if t.is_weak:
        marks.append("🔗弱")
    if t.scout_id is not None:
        marks.append(f"🔭S{t.scout_id}")
    return " ".join(marks)


def run_explore_round(explorer: SlimeMoldExplorerV6, server: str):
    """执行一轮全局探索，返回（批次数据, 统计文字, PCA图）"""
    if explorer is None:
        return [], "请先初始化探索器", None

    if DEBUG_MODE:
        act_cnt, art_cnt, clus_cnt = explorer.stats()
        weak_cnt = sum(1 for t in explorer.tentacles if t.is_weak)
        print(f"[DEBUG] === 第 {explorer.generation+1} 代 全局探索 ===")
        print(f"[DEBUG] 活跃={act_cnt}/{explorer.n_tentacles}, 弱触角={weak_cnt}, 覆盖画师={art_cnt}, 覆盖簇={clus_cnt}")

    explorer.generation += 1
    selected = explorer.select_tentacles_to_evaluate()
    if not selected:
        return [], "⚠️ 无可用触角", None

    if DEBUG_MODE:
        print(f"[DEBUG] run_explore_round: 选中 {len(selected)} 个触角: {selected}")
        print(f"[DEBUG] 参数: steps={STEPS}, cfg={CFG}, sampler={SAMPLER_NAME}, scheduler={SCHEDULER}")
        print(f"[DEBUG] 服务器: {server}")

    batch_items = []

    for idx in selected:
        t = explorer.tentacles[idx]
        artist_str = vector_to_artist_string(t.vector)
        prompt = f"{BASE_POSITIVE_PROMPT}, {artist_str}"
        fname = f"gen{explorer.generation:02d}_t{idx:03d}"
        seed = explorer.generation * 100 + idx
        wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
        pid = queue_prompt(wf)
        t0 = time.time()
        res = wait_for_prompt(pid)
        elapsed = time.time() - t0
        img_url = None
        if res:
            for out in res.get('outputs', {}).values():
                if 'images' in out:
                    img_url = img_info_to_url(out['images'][0], server)
                    break
        if DEBUG_MODE:
            marker_parts = []
            if t.is_weak:
                marker_parts.append("弱")
            if t.scout_id is not None:
                marker_parts.append(f"探测S{t.scout_id}")
            marker_str = "[" + ",".join(marker_parts) + "]" if marker_parts else ""
            print(f"[DEBUG]   t#{idx}: seed={seed}, file={fname}, elapsed={elapsed:.1f}s, img={'OK' if img_url else 'FAIL'}{marker_str}")
        if img_url:
            marker = get_tentacle_marker(t)
            batch_items.append({
                "idx": idx,
                "url": img_url,
                "artist_str": artist_str,
                "marker": marker,
                "seed": seed,
            })
        time.sleep(0.3)

    if not batch_items:
        return [], "❌ 所有图片生成失败", None

    fig = _make_plot(explorer, load_protect_zones())
    stats = get_stats_text(explorer, load_protect_zones())

    if DEBUG_MODE:
        print(f"[DEBUG] === 第 {explorer.generation} 代 完成: 成功={len(batch_items)}/{len(selected)} ===")

    return batch_items, stats, fig


def run_explore_round_gen(explorer: SlimeMoldExplorerV6, server: str):
    """Generator 版探索，每次 yield (进度文字, 批次数据, 统计文字, PCA图)"""
    if explorer is None:
        yield "探索器未初始化", [], "请先初始化探索器", None
        return

    if DEBUG_MODE:
        act_cnt, art_cnt, clus_cnt = explorer.stats()
        weak_cnt = sum(1 for t in explorer.tentacles if t.is_weak)
        print(f"[DEBUG] === 第 {explorer.generation+1} 代 全局探索 ===")
        print(f"[DEBUG] 活跃={act_cnt}/{explorer.n_tentacles}, 弱触角={weak_cnt}, 覆盖画师={art_cnt}, 覆盖簇={clus_cnt}")

    explorer.generation += 1
    selected = explorer.select_tentacles_to_evaluate()
    if not selected:
        yield "⚠️ 无可用触角", [], "⚠️ 无可用触角", None
        return

    if DEBUG_MODE:
        print(f"[DEBUG] run_explore_round: 选中 {len(selected)} 个触角: {selected}")
        print(f"[DEBUG] 参数: steps={STEPS}, cfg={CFG}, sampler={SAMPLER_NAME}, scheduler={SCHEDULER}")
        print(f"[DEBUG] 服务器: {server}")

    total = len(selected)
    batch_items = []
    taco_times = []  # 已完成触角的耗时，用于计算倒计时
    yield f"⏳ 第 {explorer.generation} 代: 选中 {total} 个触角，开始执行...", batch_items, "", None

    for i, idx in enumerate(selected):
        t = explorer.tentacles[idx]
        artist_str = vector_to_artist_string(t.vector)
        prompt = f"{BASE_POSITIVE_PROMPT}, {artist_str}"
        fname = f"gen{explorer.generation:02d}_t{idx:03d}"
        seed = explorer.generation * 100 + idx
        wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
        pid = queue_prompt(wf)

        # 逐秒轮询，每次 yield 更新进度
        poll_start = time.time()
        res = None
        while True:
            elapsed = time.time() - poll_start
            avg_t = sum(taco_times) / len(taco_times) if taco_times else 0
            eta = avg_t * (total - i - 1) + max(0, avg_t - elapsed) if taco_times else 0
            eta_str = f" | 预计剩余 ~{eta:.0f}s" if taco_times else " | 剩余时间...（正在计算）"
            yield f"⏳ 触角 {i+1}/{total} (ID {idx}) 生图中... {elapsed:.0f}s{eta_str}", [], "", None
            try:
                resp = requests.get(f"http://{COMFYUI_SERVER}/history/{pid}", timeout=10)
                hist = resp.json()
            except Exception as e:
                yield f"⏳ 触角 {i+1}/{total} (ID {idx}) 请求失败 ({e})，重试中...", [], "", None
                time.sleep(2)
                continue
            if pid in hist:
                res = hist[pid]
                err = _check_comfyui_error(res)
                if err:
                    gr.Error(err)
                    yield f"❌ 触角 {i+1}/{total} (ID {idx}) 错误", [], "", None; return
                elapsed_done = time.time() - poll_start
                taco_times.append(elapsed_done)
                yield f"✅ 触角 {i+1}/{total} (ID {idx}) 完成 ({elapsed_done:.0f}s)", [], "", None
                break
            if elapsed > 120:
                yield f"⏳ 触角 {i+1}/{total} (ID {idx}) 超时", [], "", None
                break
            time.sleep(1)

        # 处理生成结果
        img_url = None
        if res:
            for out in res.get('outputs', {}).values():
                if 'images' in out:
                    img_url = img_info_to_url(out['images'][0], server)
                    break

        if DEBUG_MODE:
            marker_parts = []
            if t.is_weak:
                marker_parts.append("弱")
            if t.scout_id is not None:
                marker_parts.append(f"探测S{t.scout_id}")
            marker_str = "[" + ",".join(marker_parts) + "]" if marker_parts else ""
            print(f"[DEBUG]   t#{idx}: seed={seed}, file={fname}, elapsed={time.time()-poll_start:.1f}s, img={'OK' if img_url else 'FAIL'}{marker_str}")

        if img_url:
            marker = get_tentacle_marker(t)
            batch_items.append({
                "idx": idx,
                "url": img_url,
                "artist_str": artist_str,
                "marker": marker,
                "seed": seed,
            })
        time.sleep(0.3)

    # 最终统计
    fig = _make_plot(explorer, load_protect_zones()) if batch_items else None
    stats = get_stats_text(explorer, load_protect_zones())
    explorer.last_batch = batch_items  # 暂存供保存
    save_explorer_state(explorer)  # 探索完成后自动保存状态
    yield f"✅ 第 {explorer.generation} 代完成！{len(batch_items)}/{total} 张成功", batch_items, stats, fig


def submit_scores(explorer: SlimeMoldExplorerV6, score_data):
    """提交评分，更新触角（支持 dict/scores_state, DataFrame, 或 list）"""
    if explorer is None:
        return get_stats_text(explorer, load_protect_zones()), None
    if score_data is None:
        return get_stats_text(explorer, load_protect_zones()), None

    # Dict 模式（从 gr.Render scores_state 传入）
    if isinstance(score_data, dict):
        if not score_data:
            return get_stats_text(explorer, load_protect_zones()), None
        selected = []
        scores = []
        for idx_str, s in score_data.items():
            if s is None:
                continue
            idx = int(idx_str)
            s = max(-10.0, min(10.0, float(s)))
            selected.append(idx)
            scores.append(s)
        if not selected:
            return get_stats_text(explorer, load_protect_zones()), None
        if DEBUG_MODE:
            score_pairs = [f"#{idx}={s:.1f}" for idx, s in zip(selected, scores)]
            print(f"[DEBUG] submit_scores (dict): {len(selected)} scores: [{', '.join(score_pairs)}]")
        explorer.update_tentacles(selected, scores)
        if explorer.generation % 3 == 0:
            explorer.exchange_info()
        fig = _make_plot(explorer, load_protect_zones())
        return get_stats_text(explorer, load_protect_zones()), fig

    # DataFrame / list 模式（向后兼容）
    if hasattr(score_data, 'empty') and score_data.empty:
        return get_stats_text(explorer, load_protect_zones()), None

    if hasattr(score_data, 'to_dict'):
        rows = score_data.values.tolist()
    else:
        rows = list(score_data) if score_data else []

    selected = []
    scores = []
    for row in rows:
        try:
            idx = int(row[0])
            s = float(row[2]) if len(row) > 2 and row[2] not in (None, "", "None") else 0.0
        except (ValueError, TypeError, IndexError):
            continue
        s = max(-10.0, min(10.0, s))
        selected.append(idx)
        scores.append(s)

    if not selected:
        return get_stats_text(explorer, load_protect_zones()), None
    if DEBUG_MODE:
        score_pairs = [f"#{idx}={s:.1f}" for idx, s in zip(selected, scores)]
        print(f"[DEBUG] submit_scores (list): {len(selected)} scores: [{', '.join(score_pairs)}]")

    explorer.update_tentacles(selected, scores)
    if explorer.generation % 3 == 0:
        explorer.exchange_info()

    fig = _make_plot(explorer, load_protect_zones())
    return get_stats_text(explorer, load_protect_zones()), fig


# ===============================================
# PCA 绘图（返回 matplotlib figure 供 gr.Plot 用）
# ===============================================
def _make_plot(explorer: SlimeMoldExplorerV6, protect_zones: dict = None):
    if pca_model is None:
        return None
    active = [t for t in explorer.tentacles if t.active]
    if not active:
        return None
    if DEBUG_MODE:
        print(f"[DEBUG] _make_plot: 第{explorer.generation}代, 活跃={len(active)}触角")

    fig, ax = plt.subplots(figsize=(10, 8))

    if PLOT_STYLE == "full":
        from interactive_style_explorer8 import pca_vectors_2d
        if pca_vectors_2d is not None:
            ax.scatter(pca_vectors_2d[:, 0], pca_vectors_2d[:, 1],
                       c='lightgray', alpha=0.3, s=5)

    # 轨迹
    for t in active:
        if len(t.trace) >= 2:
            pts = [pca_model.transform(vec.reshape(1, -1))[0]
                   for _, vec in t.trace]
            xs, ys = zip(*pts)
            alphas = np.linspace(0.2, 1.0, len(xs))
            for i in range(len(xs) - 1):
                ax.plot(xs[i:i+2], ys[i:i+2], color='gray',
                        alpha=alphas[i], linewidth=1)

    # 触角点
    scores = [t.recent_avg_score() for t in active]
    norm_scores = (np.array(scores) + 10) / 20
    colors = plt.cm.RdYlGn(norm_scores)[:, :3]
    t_vecs = np.array([t.vector for t in active])
    t_2d = pca_model.transform(t_vecs)

    weak_mask = [t.is_weak for t in active]
    if any(weak_mask):
        wi = np.where(weak_mask)[0]
        ax.scatter(t_2d[wi, 0], t_2d[wi, 1], c='yellow', s=120,
                   marker='D', edgecolors='black', linewidth=1.5, label='弱触角')
    normal_mask = [not t.is_weak for t in active]
    if any(normal_mask):
        ni = np.where(normal_mask)[0]
        ax.scatter(t_2d[ni, 0], t_2d[ni, 1], c=colors[ni], s=100,
                   marker='X', edgecolors='white', linewidth=1.5)

    # Ban区
    ban_list = load_ban_list()
    active_bans = [b for b in ban_list if b.get('active', True)]
    if active_bans:
        try:
            ban_vecs = np.array([b['vector'] for b in active_bans], dtype=np.float32)
            ban_2d = pca_model.transform(ban_vecs)
            ax.scatter(ban_2d[:, 0], ban_2d[:, 1], c='red', s=150,
                       marker='o', alpha=0.6, label='Ban区')
        except Exception:
            pass

    # 保护区
    if protect_zones:
        for zid, zone in protect_zones.items():
            z2d = pca_model.transform(zone.center_vector.reshape(1, -1))[0]
            ax.scatter(z2d[0], z2d[1], c='cyan', s=200, marker='*',
                       alpha=0.8, label=f'保护区_{zid}')

    # 探测点
    for sp in explorer.scout_points:
        sv = np.array(sp['vector'], dtype=np.float32)
        s2d = pca_model.transform(sv.reshape(1, -1))[0]
        ax.scatter(s2d[0], s2d[1], c='green', s=200, marker='o',
                   alpha=0.5, label='探测点')

    ax.legend(fontsize=9)
    ax.set_title(f"触角分布 (第{explorer.generation}代)", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig


# ===============================================
# 快捷指令函数
# ===============================================
def cmd_ban_tentacle(explorer: SlimeMoldExplorerV6, idx: int) -> str:
    if explorer is None or idx < 0 or idx >= len(explorer.tentacles):
        return "❌ 无效触角ID"
    if DEBUG_MODE:
        print(f"[DEBUG] cmd_ban: 封禁触角 {idx}")
    explorer.ban_current_tentacle(idx)
    return f"🚫 已封禁触角 {idx}"


def cmd_save_tentacle(explorer: SlimeMoldExplorerV6, idx: int) -> str:
    if explorer is None or idx < 0 or idx >= len(explorer.tentacles):
        return "❌ 无效触角ID"
    t = explorer.tentacles[idx]
    astr = vector_to_artist_string(t.vector)
    save_favorite(astr, t.vector)
    if DEBUG_MODE:
        print(f"[DEBUG] cmd_save: 收藏触角 {idx}")
    return f"⭐ 已收藏触角 {idx}: {astr[:50]}..."


def cmd_create_protect(explorer: SlimeMoldExplorerV6, idx: int) -> str:
    if explorer is None or idx < 0 or idx >= len(explorer.tentacles):
        return "❌ 无效触角ID"
    t = explorer.tentacles[idx]
    pzones = load_protect_zones()
    new_id = len(pzones)
    zone = ProtectZone(new_id, t.vector.copy(), 0.15, MIXED_ARTISTS_COUNT)
    pzones[new_id] = zone
    save_protect_zones(pzones)
    if DEBUG_MODE:
        print(f"[DEBUG] cmd_protect: 创建保护区_{new_id}, 候选={len(zone.candidate_artists)}")
    return f"🛡️ 已创建保护区_{new_id}，候选画师 {len(zone.candidate_artists)} 个"


def cmd_create_scout(explorer: SlimeMoldExplorerV6, idx: int, capacity: int) -> str:
    if explorer is None or idx < 0 or idx >= len(explorer.tentacles):
        return "❌ 无效触角ID"
    t = explorer.tentacles[idx]
    new_id = max((sp['id'] for sp in explorer.scout_points), default=-1) + 1
    explorer.scout_points.append({
        "id": new_id,
        "vector": t.vector.tolist(),
        "capacity": capacity,
        "assigned": [],
        "active": True
    })
    save_scout_points(explorer.scout_points)
    if DEBUG_MODE:
        print(f"[DEBUG] cmd_scout: 创建探测点 {new_id}, 容量={capacity}, 来源触角={idx}")
    return f"🔭 已创建探测点 {new_id}，容量 {capacity}"


def cmd_promote(explorer: SlimeMoldExplorerV6, idx: int) -> str:
    if explorer is None or idx < 0 or idx >= len(explorer.tentacles):
        return "❌ 无效触角ID"
    if not explorer.tentacles[idx].is_weak:
        return "⚠️ 该触角不是弱触角"
    explorer.promote_tentacle(idx)
    return f"✅ 触角 {idx} 已转正"


def cmd_ban_by_text(desc: str) -> str:
    if not desc.strip():
        return "❌ 请输入描述"
    try:
        if DEBUG_MODE:
            print(f"[DEBUG] cmd_ban_by_text: 开始编码描述 '{desc[:40]}'...")
        vec = encode_text_to_vector(desc)
        add_ban(desc, vec, "manual_text", 0)
        if DEBUG_MODE:
            print(f"[DEBUG] cmd_ban_by_text: 封禁完成")
        return f"🚫 已封禁: {desc[:50]}"
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] cmd_ban_by_text: 失败: {e}")
        return f"❌ 封禁失败: {e}"


def cmd_spread_tentacles(explorer: SlimeMoldExplorerV6, n: int, strength: float) -> str:
    if explorer is None:
        return "❌ 未初始化"
    if DEBUG_MODE:
        print(f"[DEBUG] cmd_spread: n={n}, strength={strength:.2f}")
    explorer.spread_weakest_tentacles(n, strength)
    return f"🌱 已扩散 {n} 个触角，强度 {strength}"


# ===============================================
# Ban区管理函数
# ===============================================
def get_ban_markdown() -> str:
    ban_list = load_ban_list()
    active = [b for b in ban_list if b.get('active', True)]
    if not active:
        return "*暂无Ban区*"
    lines = ["| ID | 描述 | 惩罚强度 | 创建代数 |",
             "|---|---|---|---|"]
    for b in active:
        lines.append(f"| {b['id']} | {b['description'][:50]} | {b.get('penalty',1.0):.2f} | {b.get('created_gen',0)} |")
    return "\n".join(lines)


def unban_by_id(ban_id: int) -> str:
    ban_list = load_ban_list()
    for b in ban_list:
        if b['id'] == ban_id:
            b['active'] = False
            save_ban_list(ban_list)
            return f"🔓 已解封 ID {ban_id}"
    return f"❌ 未找到 ID {ban_id}"


def clear_all_bans() -> str:
    ban_list = load_ban_list()
    if not ban_list:
        return "📋 暂无Ban区"
    count = len(ban_list)
    for b in ban_list:
        b['active'] = False
    save_ban_list(ban_list)
    return f"🗑️ 已清空 {count} 个Ban区"


# ===============================================
# 保护区管理函数
# ===============================================
def get_protect_zone_markdown() -> str:
    """返回保护区的 Markdown 表格文本"""
    pzones = load_protect_zones()
    if not pzones:
        return "*暂无保护区*"
    lines = ["| ID | 中心画师 | 候选数 | 最佳分 | 无提升轮数 |",
             "|---|---|---|---|---|"]
    for zid, zone in pzones.items():
        best_str = zone.best_artist_str or vector_to_artist_string(zone.center_vector, top_k=zone.mixed_count)
        lines.append(f"| {zid} | {best_str} | {len(zone.candidate_artists)} | {zone.best_score:.2f} | {zone.no_improve_rounds} |")
    return "\n".join(lines)


def release_protect_to_global(explorer: SlimeMoldExplorerV6, zone_id: int, remove_after: bool = False) -> str:
    pzones = load_protect_zones()
    zone = pzones.get(int(zone_id))
    if zone is None:
        return f"❌ 未找到保护区 {zone_id}"
    if zone.best_vector is None:
        return "⚠️ 该保护区尚无最佳向量"
    explorer.replace_weakest_with_vector(zone.best_vector)
    if remove_after:
        del pzones[int(zone_id)]
        save_protect_zones(pzones)
        return f"✅ 保护区_{zone_id} 已释放最优菌丝并删除"
    return f"✅ 保护区_{zone_id} 最优菌丝已释放到全局"


def remove_protect_zone(zone_id: int) -> str:
    pzones = load_protect_zones()
    if int(zone_id) not in pzones:
        return f"❌ 未找到保护区 {zone_id}"
    del pzones[int(zone_id)]
    save_protect_zones(pzones)
    return f"🗑️ 已删除保护区_{zone_id}"


def clear_all_protect_zones() -> str:
    filepath = _PROJ / "data/protect_zones/protect_zones.json"
    if filepath.exists():
        filepath.unlink()
        return "🗑️ 已清空全部保护区"
    return "📋 暂无保护区"


# ===============================================
# 探测点管理
# ===============================================
def get_scout_markdown(explorer: SlimeMoldExplorerV6) -> str:
    active_sp = [sp for sp in explorer.scout_points if sp.get('active', True)]
    if not active_sp:
        return "*暂无探测点*"
    lines = ["| ID | 容量 | 已分配 | 位置 |",
             "|---|---|---|---|"]
    for sp in active_sp:
        pos = vector_to_artist_string(np.array(sp['vector']), top_k=1)[:40]
        lines.append(f"| {sp['id']} | {sp['capacity']} | {len(sp.get('assigned', []))} | {pos} |")
    return "\n".join(lines)


def remove_scout(explorer: SlimeMoldExplorerV6, scout_id: int) -> str:
    for sp in explorer.scout_points:
        if sp['id'] == scout_id:
            sp['active'] = False
            save_scout_points(explorer.scout_points)
            return f"🗑️ 已删除探测点 {scout_id}"
    return f"❌ 未找到探测点 {scout_id}"


# ===============================================
# 收藏管理
# ===============================================
def get_favorites_markdown() -> str:
    fav_file = _PROJ / "data/favorites.json"
    if not fav_file.exists():
        return "*暂无收藏*"
    with open(fav_file, 'r', encoding='utf-8') as f:
        favs = json.load(f)
    if not favs:
        return "*暂无收藏*"
    lines = ["| # | 画师串 | 收藏时间 |",
             "|---|---|---|"]
    for i, f_entry in enumerate(favs):
        ts = f_entry.get('timestamp', '')[:16].replace('T', ' ')
        lines.append(f"| {i+1} | {f_entry.get('artist_string', '')[:60]} | {ts} |")
    return "\n".join(lines)


def remove_favorite_by_index(idx: int) -> str:
    fav_file = _PROJ / "data/favorites.json"
    if not fav_file.exists():
        return "📋 暂无收藏"
    with open(fav_file, 'r', encoding='utf-8') as f:
        favs = json.load(f)
    if idx < 1 or idx > len(favs):
        return f"❌ 无效编号 (1~{len(favs)})"
    removed = favs.pop(idx - 1)
    with open(fav_file, 'w', encoding='utf-8') as f:
        json.dump(favs, f, indent=2, ensure_ascii=False)
    return f"🗑️ 已移除: {removed.get('artist_string', '')[:40]}..."


# ===============================================
# 工具函数：测试 ComfyUI 连接
# ===============================================
def test_comfyui_connection(server: str) -> str:
    if DEBUG_MODE:
        print(f"[DEBUG] test_comfyui_connection: 测试 {server}...")
    try:
        import requests
        r = requests.get(f"http://{server}/history", timeout=10)
        if r.status_code == 200:
            if DEBUG_MODE:
                print(f"[DEBUG] test_comfyui_connection: 成功 ({server})")
            return f"✅ ComfyUI 连接成功 ({server})"
        else:
            if DEBUG_MODE:
                print(f"[DEBUG] test_comfyui_connection: 状态码 {r.status_code}")
            return f"⚠️ 状态码 {r.status_code}"
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] test_comfyui_connection: 失败 - {e}")
        return f"❌ 连接失败: {e}"


# ===============================================
# Render handler 工具函数（模块级，避免 @gr.render 闭包 KeyError）
# ===============================================
def _render_handle_score(value, idx):
    val = max(-10.0, min(10.0, float(value))) if value is not None else 0.0
    buffer_save_score(int(idx), val)


def _render_handle_ban(exp, idx):
    if exp is None:
        gr.Warning("请先初始化探索器")
        return exp
    msg = cmd_ban_tentacle(exp, int(idx))
    gr.Info(msg)
    return exp


def _render_handle_save(exp, idx):
    if exp is None:
        gr.Warning("请先初始化探索器")
        return
    msg = cmd_save_tentacle(exp, int(idx))
    gr.Info(msg)


def _render_handle_protect(exp, idx):
    if exp is None:
        gr.Warning("请先初始化探索器")
        return exp
    msg = cmd_create_protect(exp, int(idx))
    gr.Info(msg)
    return exp


def _render_handle_scout(exp, idx):
    if exp is None:
        gr.Warning("请先初始化探索器")
        return exp
    msg = cmd_create_scout(exp, int(idx), 3)
    gr.Info(msg)
    return exp


def _render_handle_save_image(url: str, idx: int):
    """下载并保存单张图片（含 ComfyUI 元数据）到 output/"""
    if not url or idx < 0:
        gr.Warning("无可保存图片")
        return
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        out_dir = _PROJ / "data/output"
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"tentacle_{idx}.png"
        with open(fname, 'wb') as f:
            f.write(resp.content)
        gr.Info(f"已保存: {fname}")
    except Exception as e:
        gr.Error(f"保存失败: {e}")


# ===============================================
# Gradio 界面
# ===============================================
CSS = """
.gr-box {border-radius: 8px;}
.gallery {min-height: 300px;}
footer {display: none !important;}
"""


def build_app():
    with gr.Blocks(title="画师串小助手 - 黏菌风格探索器") as app:
        gr.Markdown("# 🧫 画师串小助手 - 黏菌风格探索器")

        # ---- 全局状态 ----
        explorer_state = gr.State(None)

        # ============================================
        # Tab 1: 风格探索（主界面）
        # ============================================
        with gr.Tab("🧫 风格探索 SDXL"):
            # ---- 顶部状态栏 ----
            stats_display = gr.Markdown("### 尚未初始化")
            progress_status = gr.Markdown("### ")

            # ---- 控制区（合并按钮）----
            with gr.Row():
                spread_slider = gr.Slider(0.0, 1.0, value=get_setting('spread', 0.5), step=0.05,
                                          label="初始扩散度")
                init_btn = gr.Button("🆕 初始化新会话", variant="secondary", scale=1)
                main_btn = gr.Button("🚀 探索 & 提交", variant="primary", scale=2)

            # ---- 状态组件（供动态渲染使用）----
            batch_state = gr.State([])
            page_state = gr.State(0)
            page_size_state = gr.State(4)
            cultivate_zone_state = gr.State(None)
            COLS = 4
            MAX_GRID_ITEMS = 16

            # ---- 每页数量控制（在渲染外面，不会被重绘打断）----
            with gr.Row():
                psize_input = gr.Number(value=get_setting('page_size', 4), label="每页显示",
                                         precision=0, minimum=1, maximum=16, scale=1)

            psize_input.change(
                fn=lambda v: save_settings({**load_settings(), 'page_size': int(v)}) or (int(v), 0),
                inputs=[psize_input],
                outputs=[page_size_state, page_state]
            )

            # ---- 单元格可见性 CSS（预创建网格用）----
            grid_css = gr.HTML("<style>" + " ".join([f".gc-{i}{{display:none!important}}" for i in range(16)]) + "</style>")

            # ---- 翻页控件（在渲染外面，不会被重绘打断）----
            with gr.Row():
                prev_btn = gr.Button("◀ 上一页", size="sm", scale=1)
                page_info = gr.Markdown("**0/0** 共0条")
                next_btn = gr.Button("下一页 ▶", size="sm", scale=1)

            def on_prev_page(page, batch, psize):
                if not batch:
                    return 0
                return max(0, page - 1)

            def on_next_page(page, batch, psize):
                if not batch:
                    return 0
                total = max(1, (len(batch) + psize - 1) // psize)
                return min(total - 1, page + 1)

            def on_update_page_info(page, batch, psize):
                if not batch:
                    return "**0/0** 共0条"
                total = max(1, (len(batch) + psize - 1) // psize)
                return f"**{page+1}/{total}** 共{len(batch)}条"

            prev_btn.click(
                fn=on_prev_page,
                inputs=[page_state, batch_state, page_size_state],
                outputs=[page_state]
            )
            next_btn.click(
                fn=on_next_page,
                inputs=[page_state, batch_state, page_size_state],
                outputs=[page_state]
            )
            page_state.change(
                fn=on_update_page_info,
                inputs=[page_state, batch_state, page_size_state],
                outputs=[page_info]
            )
            batch_state.change(
                fn=on_update_page_info,
                inputs=[page_state, batch_state, page_size_state],
                outputs=[page_info]
            )

            # ---- 预创建评分网格（16 个固定槽位，无 @gr.render，避免 KeyError）----
            gr.Markdown("🚫=封禁触角  ⭐=收藏  🛡️=创建保护区  🔭=创建探测点  💾=保存图片  |  评分 -10~10")

            cell_states: list[gr.State] = []
            cell_images: list[gr.HTML] = []
            cell_scores: list[gr.Number] = []
            cell_urls: list[gr.State] = []
            cell_artists: list[gr.Markdown] = []

            for grid_i in range(MAX_GRID_ITEMS):
                if grid_i % COLS == 0:
                    row_context = gr.Row().__enter__()
                with gr.Column(elem_classes=[f"gc-{grid_i}"]):
                    idx_state = gr.State(-1)
                    img_html = gr.HTML("")
                    url_state = gr.State("")
                    with gr.Row():
                        score_n = gr.Number(value=0, minimum=-10, maximum=10,
                                             step=1, show_label=False, scale=2)
                        score_n.input(fn=_render_handle_score,
                                       inputs=[score_n, idx_state], outputs=[])
                        gr.Button("🚫", scale=1).click(
                            fn=_render_handle_ban,
                            inputs=[explorer_state, idx_state],
                            outputs=[explorer_state])
                        gr.Button("⭐", scale=1).click(
                            fn=_render_handle_save,
                            inputs=[explorer_state, idx_state],
                            outputs=[])
                        gr.Button("🛡️", scale=1).click(
                            fn=_render_handle_protect,
                            inputs=[explorer_state, idx_state],
                            outputs=[explorer_state])
                        gr.Button("🔭", scale=1).click(
                            fn=_render_handle_scout,
                            inputs=[explorer_state, idx_state],
                            outputs=[explorer_state])
                        gr.Button("💾", scale=1).click(
                            fn=_render_handle_save_image,
                            inputs=[url_state, idx_state],
                            outputs=[])
                    artist_md = gr.Markdown("")
                if grid_i % COLS == COLS - 1:
                    row_context.__exit__(None, None, None)

                cell_states.append(idx_state)
                cell_images.append(img_html)
                cell_scores.append(score_n)
                cell_urls.append(url_state)
                cell_artists.append(artist_md)

            # ---- 网格刷新函数 ----
            def refresh_grid(batch, page, psize):
                if not batch:
                    ret = []
                    for _ in range(MAX_GRID_ITEMS):
                        ret.extend([-1, "", 0.0, "", ""])
                    css_parts = [f".gc-{i}{{display:none!important}}" for i in range(MAX_GRID_ITEMS)]
                    ret.append(f"<style>{' '.join(css_parts)}</style>")
                    return ret

                start = page * psize
                end = min(start + psize, len(batch))
                page_items = batch[start:end]
                scores = buffer_get_scores()

                ret = []
                css_parts = []
                for i in range(MAX_GRID_ITEMS):
                    if i < len(page_items):
                        item = page_items[i]
                        idx = item["idx"]
                        marker_str = f" {item['marker']}" if item.get("marker") else ""
                        score_val = scores.get(str(idx), 0)
                        img = f'<img src="{item["url"]}" style="width:100%;border-radius:6px;display:block">'
                        artist = f"`#{idx}{marker_str}` {item['artist_str']}"
                        ret.extend([idx, img, float(score_val), item["url"], artist])
                        css_parts.append(f".gc-{i}{{display:block!important}}")
                    else:
                        ret.extend([-1, "", 0.0, "", ""])
                        css_parts.append(f".gc-{i}{{display:none!important}}")

                ret.append(f"<style>{' '.join(css_parts)}</style>")
                return ret

            # 刷新输出：每个 cell 5 个组件 + 1 CSS = 81
            _grid_out = []
            for i in range(MAX_GRID_ITEMS):
                _grid_out.extend([cell_states[i], cell_images[i],
                                  cell_scores[i], cell_urls[i], cell_artists[i]])
            _grid_out.append(grid_css)

            batch_state.change(fn=refresh_grid,
                               inputs=[batch_state, page_state, page_size_state],
                               outputs=_grid_out)
            page_state.change(fn=refresh_grid,
                              inputs=[batch_state, page_state, page_size_state],
                              outputs=_grid_out)
            page_size_state.change(fn=refresh_grid,
                                   inputs=[batch_state, page_state, page_size_state],
                                   outputs=_grid_out)

            # ---- PCA 可视化 ----
            with gr.Accordion("📊 PCA 触角分布图", open=False):
                pca_plot = gr.Plot(label="触角分布")

            # ---- 快捷指令区 ----
            with gr.Accordion("⚡ 快捷指令", open=False):
                with gr.Row():
                    cmd_idx = gr.Number(value=0, label="触角ID", precision=0)
                with gr.Row():
                    ban_btn = gr.Button("🚫 Ban此触角", size="sm")
                    save_btn = gr.Button("⭐ 收藏此触角", size="sm")
                    protect_btn = gr.Button("🛡️ 创建保护区", size="sm")
                    promote_btn = gr.Button("🔗 转正弱触角", size="sm")

                with gr.Row():
                    scout_capacity = gr.Number(value=3, label="探测点容量", precision=0)
                    scout_btn = gr.Button("🔭 创建探测点", size="sm")

                with gr.Row():
                    spread_n = gr.Number(value=3, label="扩散触角数", precision=0)
                    spread_strength = gr.Slider(0.0, 1.0, value=0.3, label="扩散强度")
                    spread_btn = gr.Button("🌱 扩散弱触角", size="sm")

                with gr.Row():
                    ban_desc = gr.Textbox(label="文本封禁描述", placeholder="输入描述如：丑陋风格, bad hands")
                    ban_text_btn = gr.Button("📝 文本封禁", size="sm")

                cmd_log = gr.Textbox(label="指令反馈", interactive=False)

            # ---- Ban区管理 ----
            with gr.Accordion("🚫 Ban区管理", open=False):
                ban_markdown = gr.Markdown("*暂无Ban区*")
                refresh_ban_btn = gr.Button("🔄 刷新Ban区", size="sm")
                clear_ban_btn = gr.Button("🗑️ 清空所有Ban区", size="sm", variant="secondary")
                with gr.Row():
                    unban_id = gr.Number(value=0, label="解封ID", precision=0)
                    unban_btn = gr.Button("🔓 解封", size="sm")

            # ---- 保护区管理 ----
            with gr.Accordion("🛡️ 保护区管理", open=False):
                pz_markdown = gr.Markdown("*暂无保护区*")
                refresh_pz_btn = gr.Button("🔄 刷新保护区", size="sm")
                clear_pz_btn = gr.Button("🗑️ 清空所有保护区", size="sm", variant="secondary")
                with gr.Row():
                    release_zid = gr.Number(value=0, label="保护区ID", precision=0)
                    release_btn = gr.Button("📤 释放到全局", size="sm")
                    release_del_btn = gr.Button("📤 释放到全局并删除", size="sm")
                    pz_delete_btn = gr.Button("🗑️ 删除保护区", size="sm")

            # ---- 保护区培育 ----
            with gr.Accordion("🧬 保护区培育（进化模式）", open=False):
                gr.Markdown(
                    "**进化规则**: 每轮生成 N 张图 → 评分 → 高分者成为父代⭐ "
                    "→ 突变+杂交产生下一代 → 循环至收敛"
                )
                with gr.Row():
                    cultivate_zone_input = gr.Number(value=0, label="保护区ID", precision=0, minimum=0)
                    cultivate_activate_btn = gr.Button("🎯 加载保护区", variant="secondary", size="sm")
                    cultivate_pop_size = gr.Number(value=get_setting('cultivate_pop_size', 4), visible=False)
                    cultivate_n_parents = gr.Number(value=get_setting('cultivate_n_parents', 2), visible=False)
                with gr.Row():
                    cultivate_round_btn = gr.Button("🌱 培育一轮", variant="primary", scale=2)
                cultivate_artist_info = gr.Markdown("### 尚未开始培育")
                cultivate_status = gr.Markdown("")
                # 预建 16 格培育栅格
                cultivate_batch_state = gr.State([])
                cultivate_page_state = gr.State(0)
                cultivate_page_size = gr.State(4)
                with gr.Row():
                    cultivate_psize = gr.Number(value=4, label="每页显示", precision=0, minimum=1, maximum=16)
                cultivate_psize.change(fn=lambda v: (int(v), 0), inputs=[cultivate_psize],
                                       outputs=[cultivate_page_size, cultivate_page_state])
                cultivate_grid_css = gr.HTML("<style>" + " ".join([f".cz-{i}{{display:none!important}}" for i in range(16)]) + "</style>")
                with gr.Row():
                    cultivate_prev_btn = gr.Button("◀ 上一页", size="sm", scale=1)
                    cultivate_page_info = gr.Markdown("**0/0** 共0图")
                    cultivate_next_btn = gr.Button("下一页 ▶", size="sm", scale=1)
                cultivate_cell_images = []
                cultivate_cell_infos = []
                cultivate_scores = []
                CZ_COLS = 4
                CZ_MAX = 16
                for cz_i in range(CZ_MAX):
                    if cz_i % CZ_COLS == 0:
                        _cz_row = gr.Row().__enter__()
                    with gr.Column(elem_classes=[f"cz-{cz_i}"]):
                        cultivate_cell_images.append(gr.HTML(""))
                        cultivate_cell_infos.append(gr.Markdown(""))
                        cultivate_scores.append(gr.Number(value=0, minimum=-10, maximum=10, step=1,
                                                         precision=0, show_label=False, scale=2))
                    if cz_i % CZ_COLS == CZ_COLS - 1 or cz_i == CZ_MAX - 1:
                        _cz_row.__exit__(None, None, None)
                with gr.Row():
                    cultivate_submit_btn = gr.Button("📊 提交全部评分", variant="primary")

            # ---- 探测点管理 ----
            with gr.Accordion("🔭 探测点管理", open=False):
                scout_markdown = gr.Markdown("*暂无探测点*")
                refresh_scout_btn = gr.Button("🔄 刷新探测点", size="sm")
                with gr.Row():
                    rm_scout_id = gr.Number(value=0, label="删除探测点ID", precision=0)
                    rm_scout_btn = gr.Button("🗑️ 删除", size="sm")

            # ---- 收藏 ----
            with gr.Accordion("⭐ 收藏", open=False):
                fav_markdown = gr.Markdown("*暂无收藏*")
                refresh_fav_btn = gr.Button("🔄 刷新收藏", size="sm")
                with gr.Row():
                    fav_remove_idx = gr.Number(value=1, label="移除编号", precision=0, minimum=1)
                    fav_remove_btn = gr.Button("🗑️ 移除", size="sm")

        # ============================================
        # Tab 2: 设置
        # ============================================
        with gr.Tab("⚙️ 设置"):
            gr.Markdown("## 设置 — 配置 · 连接 · 诊断")

            # ---- 连接测试（不嵌套，直接暴露）----
            # ---- 连接测试（不嵌套，直接暴露）----
            with gr.Row():
                comfy_addr = gr.Textbox(value=COMFYUI_SERVER, label="ComfyUI 地址",
                                        info="格式: ip:port")
                test_conn_btn = gr.Button("🔌 测试连接", variant="secondary", scale=1)
                conn_result = gr.Textbox(label="连接状态", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 设置（修改后点底部保存）")

            # ---- 设置（仅一层 Accordion，不嵌套）----
            # 1. ComfyUI 连接
            with gr.Accordion("🔌 ComfyUI 连接", open=False):
                s_comfy_server = gr.Textbox(value=get_setting('comfyui_server', COMFYUI_SERVER),
                                            label="ComfyUI 服务地址", info="格式: ip:port")
                s_workflow_path = gr.Textbox(value=get_setting('workflow_template', str(WORKFLOW_TEMPLATE_PATH)),
                                             label="工作流模板路径")

            # 2. 生图参数
            with gr.Accordion("🎨 生图参数", open=False):
                with gr.Row():
                    s_width = gr.Number(value=get_setting('width', WIDTH), label="图像宽", precision=0)
                    s_height = gr.Number(value=get_setting('height', HEIGHT), label="图像高", precision=0)
                    s_steps = gr.Number(value=get_setting('steps', STEPS), label="采样步数", precision=0)
                    s_cfg = gr.Number(value=get_setting('cfg', CFG), label="CFG", precision=1)
                with gr.Row():
                    s_sampler = gr.Dropdown(choices=["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2s_ancestral", "ddim", "uni_pc"],
                                            value=get_setting('sampler', SAMPLER_NAME), label="采样器")
                    s_scheduler = gr.Dropdown(choices=["simple", "karras", "exponential", "sgm_uniform"],
                                              value=get_setting('scheduler', SCHEDULER), label="调度器")
                s_checkpoint = gr.Textbox(value=get_setting('checkpoint_name', CHECKPOINT_NAME),
                                          label="底模文件名", info="ComfyUI models/checkpoints 目录下的 .safetensors 文件名")
                s_base_pos = gr.Textbox(value=get_setting('base_positive', BASE_POSITIVE_PROMPT),
                                        label="基础正面提示词", lines=2)
                s_base_neg = gr.Textbox(value=get_setting('base_negative', BASE_NEGATIVE_PROMPT),
                                        label="基础负面提示词", lines=3)

            # 3. 黏菌探索器
            with gr.Accordion("🧬 黏菌探索器", open=False):
                with gr.Row():
                    s_n_tentacles = gr.Number(value=get_setting('n_tentacles', N_TENTACLES),
                                              label="触角总数", precision=0)
                    s_eval_budget = gr.Number(value=get_setting('eval_budget', EVAL_BUDGET),
                                              label="每轮生图数", precision=0)
                    s_max_gen = gr.Number(value=get_setting('max_generations', MAX_GENERATIONS),
                                          label="最大轮数", precision=0)
                with gr.Row():
                    s_sim_thresh = gr.Slider(0.5, 1.0, value=get_setting('similarity_thresh', SIMILARITY_THRESHOLD),
                                             step=0.01, label="触角相似度阈值")
                    s_mixed_count = gr.Number(value=get_setting('mixed_count', MIXED_ARTISTS_COUNT),
                                              label="画师串长度", precision=0)
                    s_step_size = gr.Slider(0.01, 0.5, value=get_setting('step_size', STEP_SIZE),
                                            step=0.01, label="变异步长")
                    s_temperature = gr.Slider(0.1, 5.0, value=get_setting('temperature', TEMPERATURE),
                                              step=0.1, label="权重温度",
                                              info=">1 主画师更突出，<1 更均匀")

            # 4. Ban区
            with gr.Accordion("🚫 Ban区参数", open=False):
                with gr.Row():
                    s_ban_select = gr.Slider(0.5, 1.0, value=get_setting('ban_select_thresh', BAN_SELECT_THRESH),
                                             step=0.01, label="选择惩罚阈值")
                    s_ban_mutate = gr.Slider(0.5, 1.0, value=get_setting('ban_mutate_thresh', BAN_MUTATE_THRESH),
                                             step=0.01, label="变异回退阈值")
                    s_ban_rebirth = gr.Slider(0.5, 1.0, value=get_setting('ban_rebirth_thresh', BAN_REBIRTH_THRESH),
                                              step=0.01, label="重生禁止阈值")
                with gr.Row():
                    s_ban_penalty = gr.Slider(0.5, 5.0, value=get_setting('ban_penalty_coef', BAN_PENALTY_COEFFICIENT),
                                              step=0.1, label="软斥力系数")
                    s_ban_decay = gr.Slider(0.1, 1.0, value=get_setting('ban_decay', BAN_DECAY_RATE),
                                            step=0.05, label="衰减速率")

            # 5. 保护区
            with gr.Accordion("🛡️ 保护区参数", open=False):
                with gr.Row():
                    s_protect_pop = gr.Number(value=get_setting('protect_pop', PROTECT_POP_SIZE),
                                              label="每轮菌丝数", precision=0)
                    s_protect_max = gr.Number(value=get_setting('protect_max_cand', PROTECT_MAX_CANDIDATES),
                                              label="候选画师上限", precision=0)
                    s_protect_conv = gr.Number(value=get_setting('protect_converge', PROTECT_CONVERGE_ROUNDS),
                                               label="收敛轮数", precision=0)
                with gr.Row():
                    s_cultivate_pop = gr.Number(value=get_setting('cultivate_pop_size', 4), label="培育每轮生图数", precision=0, minimum=2, maximum=12)
                    s_cultivate_parents = gr.Number(value=get_setting('cultivate_n_parents', 2), label="培育父代数", precision=0, minimum=1, maximum=4)

            # 6. 弱触角 & 探测点
            with gr.Accordion("🔗 弱触角 & 探测点", open=False):
                with gr.Row():
                    s_weak_neg = gr.Number(value=get_setting('weak_negative', WEAK_CONSECUTIVE_NEGATIVE),
                                           label="触发弱触角（连续负分）", precision=0)
                    s_promote_rounds = gr.Number(value=get_setting('promote_rounds', PROMOTE_SUGGEST_ROUNDS),
                                                 label="建议转正（连续正分）", precision=0)
                    s_pull = gr.Slider(0.0, 1.0, value=get_setting('pull_strength', PULL_STRENGTH),
                                       step=0.05, label="绳子拉力系数")
                    s_max_weak = gr.Number(value=get_setting('max_weak', MAX_WEAK_TENTACLES),
                                           label="弱触角上限", precision=0)

            # 7. 调试 & 显示
            with gr.Accordion("📊 调试 & 显示", open=False):
                s_debug = gr.Checkbox(value=get_setting('debug_mode', DEBUG_MODE), label="调试模式")
                with gr.Row():
                    s_plot_style = gr.Dropdown(choices=["full", "lite"],
                                               value=get_setting('plot_style', PLOT_STYLE), label="绘图风格")
                    s_trace = gr.Number(value=get_setting('trace_len', TRACE_HISTORY_LEN),
                                        label="轨迹显示步数", precision=0)

            # 保存按钮（在所有设置Accordion外面）
            settings_log = gr.Textbox(label="设置反馈", interactive=False)
            save_settings_btn = gr.Button("💾 保存全部设置", variant="primary")

            # ---- 向量库说明 ----
            with gr.Accordion("📖 关于向量库", open=False):
                gr.Markdown(
                    """
                    ### 向量库基于以下配置构建：

                    - **底模适配**: SDXL 系列（CLIPTextEncode, 2048 维 pooled_output）
                    - **编码器**: ComfyUI `CLIPTextEncode` + `SaveCondition` 节点
                    - **向量提取**: `torch.load` 读取 `.ckpt`，取 `last_hidden_state` mean-pool
                    - **索引方式**: FAISS 余弦相似度（IndexFlatIP, 向量 L2 归一化）

                    ### 换底模后重建

                    运行 `python build_webui.py` 打开向量库构建助手，按步骤操作：

                    1. 准备画师标签 Markdown 表（`| id | name |`）
                    2. 准备好有 `SaveCondition` 这个节点的插件
                    3. 步骤 ① 编码 → 从 ComfyUI 拷贝 `.ckpt` 到 `data/conditions/`
                    4. 步骤 ② 提取向量 → 构建 FAISS 索引
                    5. 步骤 ③（可选）→ 生成 PCA 风格地图

                    > 向量空间由编码器决定。换了底模或编码提示词后必须重建。
                    """
                )
                build_launch_btn = gr.Button("🚀 启动向量库构建助手", variant="secondary")
                build_launch_msg = gr.Markdown("")

            # ---- 导入导出 ----
            with gr.Accordion("📦 导入导出进度", open=False):
                gr.Markdown("导出当前探索进度到本地备份，或从备份文件恢复进度。")
                with gr.Row():
                    export_btn = gr.Button("📥 导出进度", variant="secondary")
                    export_file = gr.File(label="下载", interactive=False, file_count="single",
                                          file_types=[".json"], visible=True)
                gr.Markdown("---")
                with gr.Row():
                    import_file = gr.File(label="选择备份文件", file_count="single", file_types=[".json"])
                    import_btn = gr.Button("📤 导入进度", variant="secondary")
                import_msg = gr.Markdown("")


        # ============================================
        # 事件绑定
        # ============================================

        # -- 初始化 --
        def on_init(spread):
            buffer_clear()
            buffer_flush_to_file()
            save_settings({**load_settings(), 'spread': spread})
            if DEBUG_MODE:
                print(f"[DEBUG] on_init: spread={spread:.2f}")
            exp = create_explorer(spread)
            save_explorer_state(exp)  # 覆盖保存为新会话
            pz = load_protect_zones()
            if DEBUG_MODE:
                act, art, clus = exp.stats()
                print(f"[DEBUG] on_init: 活跃={act}/{exp.n_tentacles}, 画师={art}, 簇={clus}, 保护区={len(pz)}")
            gr.Info("🆕 新会话已初始化")
            return exp, f"### {get_stats_text(exp, pz)}", [], 0, get_ban_markdown(), get_protect_zone_markdown(), get_scout_markdown(exp)

        init_btn.click(
            fn=on_init,
            inputs=[spread_slider],
            outputs=[explorer_state, stats_display, batch_state, page_state, ban_markdown, pz_markdown, scout_markdown]
        )

        # -- 主操作：提交评分(如有) + 探索下一轮 --
        def on_main_action(exp):
            import traceback
            try:
                if exp is None:
                    gr.Warning("请先初始化探索器")
                    yield [], "### ❌ 请先初始化", None, exp, 0, ""
                    return
                gr.Info("开始探索...")
                server = get_setting('comfyui_server', COMFYUI_SERVER)

                if DEBUG_MODE:
                    print(f"[DEBUG] === on_main_action: 第 {exp.generation+1} 代 ===")

                # 1. 提交评分
                scores_dict = buffer_get_scores()
                if scores_dict:
                    yield [], "### 探索中...", None, exp, 0, "⏳ 提交评分中..."
                    if DEBUG_MODE:
                        print(f"[DEBUG] 提交前一轮评分: {len(scores_dict)} 个触角")
                    submit_scores(exp, scores_dict)
                    buffer_clear()
                    buffer_flush_to_file()

                # 2. 探索下一轮（generator 逐步 yield 进度）
                for progress_text, batch_items, stats, fig in run_explore_round_gen(exp, server):
                    stats_md = f"### {stats}" if stats else ""
                    plot_val = fig if fig is not None else gr.update()
                    yield batch_items, stats_md, plot_val, exp, 0, progress_text

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[探索] 错误: {e}\n{tb}")
                gr.Error(str(e))
                yield [], f"### ❌ {e}", None, exp, 0, f"❌ {e}"

        main_btn.click(
            fn=on_main_action,
            inputs=[explorer_state],
            outputs=[batch_state, stats_display, pca_plot, explorer_state, page_state, progress_status]
        )

        # -- 快捷指令 --
        def on_ban(exp, idx):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化", exp
            msg = cmd_ban_tentacle(exp, int(idx))
            gr.Info(msg)
            return msg, exp

        ban_btn.click(
            fn=on_ban, inputs=[explorer_state, cmd_idx],
            outputs=[cmd_log, explorer_state]
        )

        def on_save(exp, idx):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化"
            msg = cmd_save_tentacle(exp, int(idx))
            gr.Info(msg)
            return msg

        save_btn.click(
            fn=on_save, inputs=[explorer_state, cmd_idx],
            outputs=[cmd_log]
        )

        def on_protect(exp, idx):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化", exp
            msg = cmd_create_protect(exp, int(idx))
            gr.Info(msg)
            return msg, exp

        protect_btn.click(
            fn=on_protect, inputs=[explorer_state, cmd_idx],
            outputs=[cmd_log, explorer_state]
        )

        def on_promote(exp, idx):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化", exp
            msg = cmd_promote(exp, int(idx))
            gr.Info(msg)
            return msg, exp

        promote_btn.click(
            fn=on_promote, inputs=[explorer_state, cmd_idx],
            outputs=[cmd_log, explorer_state]
        )

        def on_scout(exp, idx, cap):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化", exp
            msg = cmd_create_scout(exp, int(idx), int(cap))
            gr.Info(msg)
            return msg, exp

        scout_btn.click(
            fn=on_scout, inputs=[explorer_state, cmd_idx, scout_capacity],
            outputs=[cmd_log, explorer_state]
        )

        def on_spread(exp, n, strength):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化", exp
            msg = cmd_spread_tentacles(exp, int(n), strength)
            gr.Info(msg)
            return msg, exp

        spread_btn.click(
            fn=on_spread, inputs=[explorer_state, spread_n, spread_strength],
            outputs=[cmd_log, explorer_state]
        )

        def on_ban_text(desc):
            msg = cmd_ban_by_text(desc)
            gr.Info(msg)
            return msg

        ban_text_btn.click(
            fn=on_ban_text, inputs=[ban_desc],
            outputs=[cmd_log]
        )

        # -- 刷新Ban区 --
        refresh_ban_btn.click(
            fn=get_ban_markdown,
            outputs=[ban_markdown]
        )

        # -- 解封 --
        def on_unban(bid):
            msg = unban_by_id(int(bid))
            gr.Info(msg)
            return msg, get_ban_markdown()

        unban_btn.click(
            fn=on_unban, inputs=[unban_id],
            outputs=[cmd_log, ban_markdown]
        )

        # -- 清空Ban区 --
        def on_clear_bans():
            msg = clear_all_bans()
            gr.Info(msg)
            return msg, get_ban_markdown()

        clear_ban_btn.click(
            fn=on_clear_bans,
            outputs=[cmd_log, ban_markdown]
        )

        # -- 刷新保护区 --
        refresh_pz_btn.click(
            fn=get_protect_zone_markdown,
            outputs=[pz_markdown]
        )

        # -- 清空保护区 --
        def on_clear_pz():
            msg = clear_all_protect_zones()
            gr.Info(msg)
            return msg, get_protect_zone_markdown()

        clear_pz_btn.click(
            fn=on_clear_pz,
            outputs=[cmd_log, pz_markdown]
        )

        # -- 释放保护区 --
        def on_release(exp, zid, remove_after=False):
            if exp is None:
                gr.Warning("请先初始化")
                return "请先初始化", exp
            msg = release_protect_to_global(exp, int(zid), remove_after=remove_after)
            gr.Info(msg)
            return msg, exp

        release_btn.click(
            fn=on_release, inputs=[explorer_state, release_zid],
            outputs=[cmd_log, explorer_state]
        )

        def on_release_del(exp, zid):
            return on_release(exp, zid, remove_after=True)

        release_del_btn.click(
            fn=on_release_del, inputs=[explorer_state, release_zid],
            outputs=[cmd_log, explorer_state]
        )

        def on_pz_delete(zid):
            msg = remove_protect_zone(int(zid))
            gr.Info(msg)
            return msg, get_protect_zone_markdown()

        pz_delete_btn.click(
            fn=on_pz_delete,
            inputs=[release_zid],
            outputs=[cmd_log, pz_markdown]
        )

        # -- 培育网格刷新（和主网格一样套路）--
        def refresh_cultivate_grid(pop, page, psize):
            ret = []
            css_parts = []
            total = len(pop)
            start = page * psize
            end = min(start + psize, total)
            for cz_i in range(CZ_MAX):
                if start <= cz_i < end:
                    r = pop[cz_i]
                    pm = " ⭐" if r.get('is_parent') else ""
                    img = f'<img src="{r["url"]}" style="width:100%;border-radius:6px;display:block">' if r.get('url') else '<div style="width:100%;height:200px;border:1px dashed gray;border-radius:6px;display:flex;align-items:center;justify-content:center;color:gray;">❌</div>'
                    ret.extend([img, f"图{cz_i+1}{pm}: {r['artist_str'][:60]}...", 0.0])
                    css_parts.append(f".cz-{cz_i}{{display:block!important}}")
                else:
                    ret.extend(["", "", 0.0])
                    css_parts.append(f".cz-{cz_i}{{display:none!important}}")
            ret.append(f"<style>{' '.join(css_parts)}</style>")
            return tuple(ret)

        # -- 培育翻页 --
        def on_cultivate_prev(page, batch, psize):
            return max(0, page - 1) if batch else 0

        def on_cultivate_next(page, batch, psize):
            if not batch:
                return 0
            return min(((len(batch) - 1) // max(1, psize)), page + 1)

        def on_cultivate_page_info(page, batch, psize):
            if not batch:
                return "**0/0** 共0图"
            total_p = max(1, (len(batch) + psize - 1) // psize)
            return f"**{page+1}/{total_p}** 共{len(batch)}图"

        cultivate_prev_btn.click(fn=on_cultivate_prev,
                                  inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size],
                                  outputs=[cultivate_page_state])
        cultivate_next_btn.click(fn=on_cultivate_next,
                                  inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size],
                                  outputs=[cultivate_page_state])

        _cz_grid_out = []
        for _cz_i in range(CZ_MAX):
            _cz_grid_out.extend([cultivate_cell_images[_cz_i], cultivate_cell_infos[_cz_i], cultivate_scores[_cz_i]])
        _cz_grid_out.append(cultivate_grid_css)

        cultivate_batch_state.change(fn=refresh_cultivate_grid,
                                      inputs=[cultivate_batch_state, cultivate_page_state, cultivate_page_size],
                                      outputs=_cz_grid_out)
        cultivate_page_state.change(fn=refresh_cultivate_grid,
                                     inputs=[cultivate_batch_state, cultivate_page_state, cultivate_page_size],
                                     outputs=_cz_grid_out)
        cultivate_page_size.change(fn=refresh_cultivate_grid,
                                    inputs=[cultivate_batch_state, cultivate_page_state, cultivate_page_size],
                                    outputs=_cz_grid_out)
        cultivate_page_state.change(fn=on_cultivate_page_info,
                                     inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size],
                                     outputs=[cultivate_page_info])
        cultivate_batch_state.change(fn=on_cultivate_page_info,
                                      inputs=[cultivate_page_state, cultivate_batch_state, cultivate_page_size],
                                      outputs=[cultivate_page_info])

        # -- 保护区培育（进化模式）--
        def on_cultivate_activate(zone_id, pop_size, n_parents):
            pzones = load_protect_zones()
            zone = pzones.get(int(zone_id))
            if zone is None:
                gr.Warning(f"未找到保护区 {zone_id}")
                return (None, "### ❌ 未找到保护区", "", [])
            has_pop = hasattr(zone, '_evo_population') and zone._evo_population
            ps = int(pop_size)
            info = (
                f"### ✅ 已加载保护区_{int(zone_id)}\n"
                f"- 候选画师: {len(zone.candidate_artists)} 个\n"
                f"- 当前最佳: {zone.best_score:.2f}\n"
                f"- 已培育 {zone.generation} 轮\n"
                f"- 每轮 {ps} 图, 父代 {int(n_parents)} 个"
            )
            batch = zone._evo_population if has_pop else []
            return (zone, info, "", batch)

        cultivate_activate_btn.click(
            fn=on_cultivate_activate,
            inputs=[cultivate_zone_input, cultivate_pop_size, cultivate_n_parents],
            outputs=[cultivate_zone_state, cultivate_artist_info, cultivate_status, cultivate_batch_state]
        )

        def on_cultivate_round_gen(zone, pop_size):
            if zone is None:
                yield ("### ❌ 请先加载保护区", "", [])
                return

            if not hasattr(zone, '_evo_population') or not zone._evo_population:
                zone.evo_init(int(pop_size))
            population = zone._evo_population

            if zone.no_improve_rounds >= PROTECT_CONVERGE_ROUNDS:
                yield ("### ✅ 已收敛，可释放最优到全局", "已收敛，可释放最优到全局", population)
                return

            server = get_setting('comfyui_server', COMFYUI_SERVER)
            gen = zone.generation
            total = len(population)

            yield (f"### 🧬 第 {gen} 轮培育中... {total} 张图", f"⏳ 排队中...", [])

            from concurrent.futures import ThreadPoolExecutor, as_completed
            pending = {}
            with ThreadPoolExecutor(max_workers=total) as exc:
                futures = {}
                for c in population:
                    prompt = f"{BASE_POSITIVE_PROMPT}, {c['artist_str']}"
                    fname = f"protect_{zone.zone_id}_gen{gen}_c{c['id']}"
                    seed = gen * 1000 + zone.zone_id + c['id']
                    wf = build_workflow(prompt, BASE_NEGATIVE_PROMPT, seed, fname)
                    fut = exc.submit(lambda w=wf: (
                        requests.post(f"http://{server}/prompt", json={"prompt": w}, timeout=10).json()['prompt_id']
                    ))
                    futures[fut] = c
                for fut in as_completed(futures):
                    c = futures[fut]
                    try:
                        pid = fut.result()
                        pending[pid] = c
                    except Exception as e:
                        c['url'] = None
                        c['error'] = str(e)

            taco_times = []
            batch_t0 = time.time()
            completed = 0
            while pending:
                elapsed = time.time() - batch_t0
                max_t = max(taco_times) if taco_times else 0
                if taco_times and max_t > elapsed:
                    eta_str = f" | 预计剩余 ~{max_t - elapsed:.0f}s"
                elif taco_times:
                    eta_str = " | 预计即将完成"
                else:
                    eta_str = " | 剩余时间...（正在计算）"
                yield (f"### ⏳ 第 {gen} 轮: {completed}/{total} 完成 ({elapsed:.0f}s){eta_str}", "", [])
                pids = list(pending.keys())
                try:
                    h = requests.get(f"http://{server}/history", timeout=10).json()
                except Exception:
                    time.sleep(1)
                    continue
                for pid in pids:
                    if pid in h:
                        c = pending.pop(pid)
                        c['url'] = None
                        err = _check_comfyui_error(h[pid])
                        if err: gr.Error(err); yield f"### ❌ {err}", "", ""; return
                        for out in h[pid].get('outputs', {}).values():
                            if 'images' in out:
                                c['url'] = img_info_to_url(out['images'][0], server)
                                break
                        taco_times.append(time.time() - batch_t0 - sum(taco_times))
                        completed += 1
                if pending and time.time() - batch_t0 > 180:
                    for pid, c in list(pending.items()):
                        c['url'] = None
                        c['error'] = '超时'
                        pending.pop(pid)
                        completed += 1
                if pending:
                    time.sleep(1)

            for c in population:
                if 'url' not in c:
                    c['url'] = None
            yield ("### ✅ 全部生成完毕，等待评分", "", population)

        cultivate_round_btn.click(
            fn=on_cultivate_round_gen,
            inputs=[cultivate_zone_state, cultivate_pop_size],
            outputs=[cultivate_artist_info, cultivate_status, cultivate_batch_state]
        )

        def on_cultivate_submit(zone, n_parents, *scores):
            if zone is None:
                gr.Warning("请先加载保护区")
                return zone, ""
            if not hasattr(zone, '_evo_population') or not zone._evo_population:
                gr.Warning("请先点击「培育一轮」")
                return zone, ""
            pop = zone._evo_population
            scores_dict = {}
            for i in range(len(pop)):
                v = scores[i] if i < len(scores) else 0
                if v is None:
                    v = 0.0
                scores_dict[i] = max(-10.0, min(10.0, float(v)))
            result = zone.evo_submit_scores(scores_dict, int(n_parents))
            # 持久化
            pzones = load_protect_zones()
            pzones[zone.zone_id] = zone
            save_protect_zones(pzones)
            parents = result.get('parents', [])
            parent_str = ", ".join(f"图{p+1}⭐" for p in parents)
            status = (
                f"**📊 第 {result['generation']-1} 轮评分结果**\n"
                f"- 父代: {parent_str}\n"
                f"- 历史最佳: {result['best_score']:.2f}\n"
                f"- 无提升轮数: {result['no_improve_rounds']}"
            )
            if result['converged']:
                status += "\n\n✅ **已收敛！** 可点击「释放最优到全局」"
                if result.get('best_artist_str'):
                    status += f"\n🏆 最优画师串: {result['best_artist_str'][:80]}..."
            else:
                status += "\n\n💡 父代已进入下一轮，点击「培育一轮」查看新种群"
            return zone, status

        cultivate_submit_btn.click(
            fn=on_cultivate_submit,
            inputs=[cultivate_zone_state, cultivate_n_parents] + cultivate_scores,
            outputs=[cultivate_zone_state, cultivate_status]
        )

        # -- 刷新探测点 --
        def on_refresh_scouts(exp):
            if exp is None:
                return []
            return get_scout_markdown(exp)

        refresh_scout_btn.click(
            fn=on_refresh_scouts, inputs=[explorer_state],
            outputs=[scout_markdown]
        )

        # -- 删除探测点 --
        def on_rm_scout(exp, sid):
            if exp is None:
                return "请先初始化", exp
            msg = remove_scout(exp, int(sid))
            return msg, exp

        rm_scout_btn.click(
            fn=on_rm_scout, inputs=[explorer_state, rm_scout_id],
            outputs=[cmd_log, explorer_state]
        )

        # -- 刷新收藏 --
        refresh_fav_btn.click(
            fn=get_favorites_markdown,
            outputs=[fav_markdown]
        )

        def on_fav_remove(idx):
            msg = remove_favorite_by_index(int(idx))
            gr.Info(msg)
            return msg, get_favorites_markdown()

        fav_remove_btn.click(
            fn=on_fav_remove,
            inputs=[fav_remove_idx],
            outputs=[cmd_log, fav_markdown]
        )

        # -- 连接测试 --
        test_conn_btn.click(
            fn=test_comfyui_connection,
            inputs=[comfy_addr],
            outputs=[conn_result]
        )

        # -- 启动向量库构建助手 --
        def on_launch_build():
            import subprocess, sys, os
            cwd = Path(__file__).parent
            try:
                subprocess.Popen(
                    [sys.executable, "build_webui.py"],
                    cwd=str(cwd),
                )
                gr.Info("🚀 向量库构建助手已启动 → http://127.0.0.1:17325")
                return "✅ 已启动 → http://127.0.0.1:17325"
            except Exception as e:
                gr.Error(f"启动失败: {e}")
                return f"❌ 启动失败: {e}"

        build_launch_btn.click(
            fn=on_launch_build,
            outputs=[build_launch_msg]
        )

        # -- 保存设置 --
        def on_save_settings(comfy_srv, wf_path, w, h, steps, cfg, sampler, sched,
                             ckpt, pos, neg, n_tent, budget, max_gen, sim_thresh, mixed,
                             ban_sel, ban_mut, ban_reb, ban_pen, ban_dec,
                             prot_pop, prot_max, prot_conv,
                             weak_neg, prom_rounds, pull, max_wk, step_size, temp,
                             debug, plot_style, trace):
            s = {
                'comfyui_server': comfy_srv,
                'workflow_template': wf_path,
                'width': int(w), 'height': int(h), 'steps': int(steps), 'cfg': float(cfg),
                'sampler': sampler, 'scheduler': sched,
                'checkpoint_name': ckpt or CHECKPOINT_NAME,
                'base_positive': pos, 'base_negative': neg,
                'n_tentacles': int(n_tent), 'eval_budget': int(budget),
                'max_generations': int(max_gen), 'similarity_thresh': float(sim_thresh),
                'mixed_count': int(mixed),
                'ban_select_thresh': float(ban_sel), 'ban_mutate_thresh': float(ban_mut),
                'ban_rebirth_thresh': float(ban_reb), 'ban_penalty_coef': float(ban_pen),
                'ban_decay': float(ban_dec),
                'protect_pop': int(prot_pop), 'protect_max_cand': int(prot_max),
                'protect_converge': int(prot_conv),
                'weak_negative': int(weak_neg), 'promote_rounds': int(prom_rounds),
                'pull_strength': float(pull), 'max_weak': int(max_wk),
                'step_size': float(step_size),
                'debug_mode': bool(debug), 'plot_style': plot_style,
                'trace_len': int(trace),
            }
            save_settings(s)
            apply_settings_to_v8(s)
            if DEBUG_MODE:
                debug_val = s.get('debug_mode', False)
                print(f"[DEBUG] 设置已保存（部分参数需下次重启程序或初始化新会话）: debug_mode={debug_val}, eval_budget={s.get('eval_budget')}, "
                      f"n_tentacles={s.get('n_tentacles')}, steps={s.get('steps')}")
            return "✅ 设置已保存并生效（部分参数需下次重启程序或初始化新会话）"

        save_settings_btn.click(
            fn=on_save_settings,
            inputs=[s_comfy_server, s_workflow_path,
                    s_width, s_height, s_steps, s_cfg, s_sampler, s_scheduler,
                    s_checkpoint, s_base_pos, s_base_neg,
                    s_n_tentacles, s_eval_budget, s_max_gen, s_sim_thresh, s_mixed_count,
                    s_ban_select, s_ban_mutate, s_ban_rebirth, s_ban_penalty, s_ban_decay,
                    s_protect_pop, s_protect_max, s_protect_conv,
                    s_weak_neg, s_promote_rounds, s_pull, s_max_weak,
                    s_step_size, s_temperature, s_debug, s_plot_style, s_trace],
            outputs=[settings_log]
        )

        # -- 导入导出进度 --
        def on_export_progress():
            src = _PROJ / "data/explorer_save.json"
            if not src.exists():
                gr.Warning("暂无探索进度可导出")
                return None, "### ❌ 暂无进度"
            import shutil
            exp_dir = _PROJ / "data/exports"
            exp_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            # 读取当前代数
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                gen = data.get("generation", "?")
            except Exception:
                gen = "?"
            dst = exp_dir / f"explorer_gen{gen}_{ts}.json"
            shutil.copy2(src, dst)
            return str(dst), f"### ✅ 已导出: explorer_gen{gen}_{ts}.json"

        export_btn.click(
            fn=on_export_progress,
            outputs=[export_file, import_msg]
        )

        def on_import_progress(f):
            if f is None:
                gr.Warning("请先选择备份文件")
                return "### ❌ 未选择文件"
            import shutil
            # 兼容不同 Gradio 版本的 File 返回值
            if isinstance(f, dict):
                src = Path(f.get('name', ''))
            elif isinstance(f, str):
                src = Path(f)
            else:
                src = Path(str(f))
            if not src.exists() or src.suffix != '.json':
                return "### ❌ 文件无效"
            try:
                with open(src, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if not isinstance(data, dict) or "tentacles" not in data or "generation" not in data:
                    return "### ❌ 文件格式无效（缺少 tentacles 或 generation）"
            except json.JSONDecodeError:
                return "### ❌ 无法解析 JSON"
            dst = _PROJ / "data/explorer_save.json"
            shutil.copy2(src, dst)
            gen = data.get("generation", "?")
            n = len(data.get("tentacles", []))
            gr.Info(f"✅ 已导入第 {gen} 代进度（{n} 个触角），请刷新页面生效")
            return f"### ✅ 已导入第 {gen} 代（{n} 触角）\n\n⚠️ **请刷新页面** 使进度生效"

        import_btn.click(
            fn=on_import_progress,
            inputs=[import_file],
            outputs=[import_msg]
        )

        # -- 页面加载时自动加载设置 + 初始化 --
        def on_auto_load():
            try:
                s = load_settings()
                apply_settings_to_v8(s)
                spread = s.get('spread', 0.5)
                if DEBUG_MODE:
                    print(f"[DEBUG] on_auto_load: spread={spread:.2f}, settings_count={len(s)}")
                buffer_clear()
                buffer_flush_to_file()
                exp = load_explorer_state()
                if exp is not None:
                    if DEBUG_MODE:
                        act, art, clus = exp.stats()
                        print(f"[DEBUG] on_auto_load: 已恢复探索进度 gen={exp.generation}, 活跃={act}/{exp.n_tentacles}")
                else:
                    exp = create_explorer(spread)
                    save_explorer_state(exp)
                pz = load_protect_zones()
                batch = getattr(exp, "last_batch", [])
                stats = f"### {get_stats_text(exp, pz)}"
                return exp, stats, batch, 0, get_ban_markdown(), get_protect_zone_markdown(), get_scout_markdown(exp), ""
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] on_auto_load 失败: {e}，回退到新会话")
                exp = create_explorer(0.5)
                save_explorer_state(exp)
                pz = load_protect_zones()
                return exp, f"### {get_stats_text(exp, pz)}", [], 0, get_ban_markdown(), get_protect_zone_markdown(), get_scout_markdown(exp), ""
                apply_settings_to_v8(s)
                spread = s.get('spread', 0.5)
                if DEBUG_MODE:
                    print(f"[DEBUG] on_auto_load: spread={spread:.2f}, settings_count={len(s)}")
                buffer_clear()
                buffer_flush_to_file()
                exp = load_explorer_state()
                if exp is not None:
                    if DEBUG_MODE:
                        act, art, clus = exp.stats()
                        print(f"[DEBUG] on_auto_load: 已恢复探索进度 gen={exp.generation}, 活跃={act}/{exp.n_tentacles}")
                else:
                    exp = create_explorer(spread)
                    save_explorer_state(exp)
                pz = load_protect_zones()
                batch = getattr(exp, "last_batch", [])
                stats = f"### {get_stats_text(exp, pz)}"
                return exp, stats, batch, 0, get_ban_markdown(), get_protect_zone_markdown(), get_scout_markdown(exp), ""
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] on_auto_load 失败: {e}，回退到新会话")
                exp = create_explorer(0.5)
                save_explorer_state(exp)
                pz = load_protect_zones()
                return exp, f"### {get_stats_text(exp, pz)}", [], 0, get_ban_markdown(), get_protect_zone_markdown(), get_scout_markdown(exp), ""

        app.load(
            fn=on_auto_load,
            outputs=[explorer_state, stats_display, batch_state, page_state, ban_markdown, pz_markdown, scout_markdown, progress_status]
        )

    return app


# ===============================================
# 入口
# ===============================================
if __name__ == "__main__":
    print("🧫 启动画师串小助手 WebUI...")
    if DEBUG_MODE:
        print(f"[DEBUG] 配置: server={COMFYUI_SERVER}, n_tentacles={N_TENTACLES}, eval_budget={EVAL_BUDGET}")
        print(f"[DEBUG] 生图参数: {WIDTH}x{HEIGHT}, steps={STEPS}, cfg={CFG}, sampler={SAMPLER_NAME}")
        print(f"[DEBUG] 输出目录: {RUN_OUTPUT_DIR}")
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=17324,
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )
