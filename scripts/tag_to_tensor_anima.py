import requests
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===============================================
# 1. 解析 Markdown 表格（使用修复后的版本）
# ===============================================
def parse_artist_md_table(md_file_path: str) -> List[Dict]:
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
                except:
                    artist['id'] = None
                try:
                    artist['post_count'] = int(artist['post_count'])
                except:
                    artist['post_count'] = 0
                try:
                    artist['uniqueness_score'] = float(artist['uniqueness_score'])
                except:
                    artist['uniqueness_score'] = 0.0
                artists.append(artist)
        else:
            if in_table and passed_header:
                break

    print(f"✅ 成功解析 {len(artists)} 位画师信息")
    return artists

# ===============================================
# 2. ComfyUI API 交互函数
# ===============================================
def queue_prompt(server: str, workflow: Dict) -> str:
    r = requests.post(f"http://{server}/prompt", json={"prompt": workflow})
    r.raise_for_status()
    return r.json()['prompt_id']

def get_history(server: str, prompt_id: str) -> Optional[Dict]:
    try:
        r = requests.get(f"http://{server}/history/{prompt_id}")
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                return data[prompt_id]
    except:
        pass
    return None

def is_queue_empty(server: str) -> bool:
    """检查 ComfyUI 队列是否为空（通过 /queue 接口）"""
    try:
        r = requests.get(f"http://{server}/queue")
        if r.status_code == 200:
            data = r.json()
            # 如果 running 和 pending 都为空，则队列空闲
            return len(data.get('queue_running', [])) == 0 and len(data.get('queue_pending', [])) == 0
    except:
        pass
    return True  # 出错时默认 true，避免死等

# ===============================================
# 3. 单个画师处理任务
# ===============================================
def process_artist(
    artist: Dict,
    server: str,
    wf_template: Dict,
    conditions_dir: Path,
    prompt_prefix: str
) -> Tuple[int, bool, str]:
    artist_id = artist['id']
    artist_name = artist['name']
    prompt = f"{prompt_prefix} @{artist_name}"
    filename = f"anima_artist_{artist_id}"

    # 构建工作流
    wf = json.loads(json.dumps(wf_template))
    for node in wf.values():
        if node.get('class_type') == 'JjkText':
            node['inputs']['text'] = prompt
        if node.get('class_type') == 'SaveCondition':
            node['inputs']['filename'] = filename

    # 提交任务
    try:
        prompt_id = queue_prompt(server, wf)
    except Exception as e:
        return artist_id, False, f"提交失败: {e}"

    # 轮询等待完成（最多等 60 秒，实际需要 <1 秒）
    start = time.time()
    while time.time() - start < 60:
        history = get_history(server, prompt_id)
        if history is not None:
            break
        time.sleep(0.2)

    # 检查输出文件
    expected_path = conditions_dir / f"{filename}.ckpt"
    waited = 0
    while not expected_path.exists() and waited < 3:
        time.sleep(0.3)
        waited += 0.3

    if expected_path.exists():
        return artist_id, True, str(expected_path)
    else:
        return artist_id, False, "文件未生成"

# ===============================================
# 4. 主函数：并发调度
# ===============================================
def main():
    # ---- 配置 ----
    COMFYUI_SERVER = "127.0.0.1:8188"
    MD_FILE_PATH = "./artists/artists40k.md"
    WORKFLOW_TEMPLATE = "workflows/tag_to_tensor_anima.json"
    # ⚠️ 运行前请改成你自己的 ComfyUI models/conditions/ 目录！
    CONDITIONS_DIR = Path("<你的ComfyUI目录>/models/conditions")
    OUTPUT_CSV = "./data/anima_vectors_metadata.csv"
    PROMPT_PREFIX = "1girl, newest, score_9, score_8, masterpiece, best quality"  # @ 自动追加

    MAX_CONCURRENT_SUBMIT = 100  # 同时最多保持多少个待处理任务（避免挤爆队列）
    TOTAL_WORKERS = 15          # 线程池大小

    # ---- 加载模板 ----
    with open(WORKFLOW_TEMPLATE, 'r', encoding='utf-8') as f:
        wf_template = json.load(f)

    # ---- 解析画师 ----
    artists = parse_artist_md_table(MD_FILE_PATH)
    total = len(artists)

    print(f"\n🚀 开始并发提取 Anima 画师向量")
    print(f"   画师总数: {total}")
    print(f"   预计耗时: {total * 0.2 / 60:.1f} ~ {total * 0.25 / 60:.1f} 分钟（理论下限）")
    print(f"   并发策略: 线程池{max}_workers，保持队列≤{MAX_CONCURRENT_SUBMIT}\n")

    # ---- 分批提交 ----
    metadata_records = []
    failed = []
    start_time = time.time()

    # 我们把所有画师分成小批，每批提交MAX_CONCURRENT_SUBMIT个，然后等待队列清空
    for batch_start in range(0, total, MAX_CONCURRENT_SUBMIT):
        batch_end = min(batch_start + MAX_CONCURRENT_SUBMIT, total)
        batch = artists[batch_start:batch_end]
        batch_size = len(batch)

        print(f"📦 批次 [{batch_start+1}-{batch_end}]/{total} 提交中...", end=" ")

        # 使用线程池提交这一批
        futures = []
        with ThreadPoolExecutor(max_workers=TOTAL_WORKERS) as executor:
            for artist in batch:
                future = executor.submit(
                    process_artist,
                    artist,
                    COMFYUI_SERVER,
                    wf_template,
                    CONDITIONS_DIR,
                    PROMPT_PREFIX
                )
                futures.append(future)

        print(f"已提交 {batch_size} 个任务，等待队列清空...")

        # 等待队列完全空闲（表示所有提交的任务都已处理完）
        # 同时也可以检查 futures 是否完成，但用队列状态更准确
        while not is_queue_empty(COMFYUI_SERVER):
            time.sleep(1)

        # 收集结果
        for future in futures:
            try:
                artist_id, success, msg = future.result()
                if success:
                    metadata_records.append({
                        'artist_id': artist_id,
                        'artist_name': next(a['name'] for a in artists if a['id'] == artist_id),
                        'ckpt_filename': f"anima_artist_{artist_id}.ckpt",
                        'ckpt_path': msg,
                    })
                else:
                    failed.append((artist_id, msg))
                    print(f"  ❌ 画师 {artist_id}: {msg}")
            except Exception as e:
                print(f"  ❌ 线程异常: {e}")

        elapsed = time.time() - start_time
        processed = batch_end
        speed = processed / elapsed
        eta = (total - processed) / speed if speed > 0 else 0
        print(f"   已完成 {processed}/{total}, 速度 {speed:.1f} 个/秒, 预计剩余 {eta/60:.1f} 分钟\n")

    # ---- 保存元数据 ----
    if metadata_records:
        df = pd.DataFrame(metadata_records)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\n🎉 成功 {len(metadata_records)}/{total} 个画师")
        print(f"📋 元数据: {OUTPUT_CSV}")
    else:
        print("\n⚠️ 未成功任何画师")

    if failed:
        print(f"⚠️ 失败 {len(failed)} 个，前10个: {failed[:10]}")

    total_time = time.time() - start_time
    print(f"⏱️ 总耗时: {total_time/60:.1f} 分钟")

if __name__ == "__main__":
    main()