import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re
from tqdm import tqdm  # 进度条库

# ===============================================
# 1. 解析Markdown文件中的画师表格
# ===============================================
def parse_artist_md_table(md_file_path: str) -> List[Dict]:
    """从Markdown文件中解析画师表格，返回包含画师信息的字典列表。"""
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
                artist_data = dict(zip(headers, parts))
                try:
                    artist_data['id'] = int(artist_data['id'])
                except (ValueError, TypeError, KeyError):
                    artist_data['id'] = None
                try:
                    artist_data['post_count'] = int(artist_data['post_count'])
                except (ValueError, TypeError, KeyError):
                    artist_data['post_count'] = 0
                try:
                    artist_data['uniqueness_score'] = float(artist_data['uniqueness_score'])
                except (ValueError, TypeError, KeyError):
                    artist_data['uniqueness_score'] = 0.0
                artists.append(artist_data)
        else:
            if in_table and passed_header:
                break

    print(f"✅ 成功解析 {len(artists)} 位画师信息")
    return artists

# ===============================================
# 2. ComfyUI API 交互函数
# ===============================================
def queue_prompt(server_address: str, workflow: Dict) -> str:
    """提交工作流到ComfyUI队列，返回prompt_id"""
    p = {"prompt": workflow}
    response = requests.post(f"http://{server_address}/prompt", json=p)
    response.raise_for_status()
    return response.json()['prompt_id']

def wait_for_prompt(server_address: str, prompt_id: str, timeout: int = 60) -> Optional[Dict]:
    """等待ComfyUI完成一个prompt的执行"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f"http://{server_address}/history/{prompt_id}")
        history = response.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(1)
    print(f"⚠️ 超时: 等待 prompt_id {prompt_id} 完成")
    return None

def update_workflow_prompt(workflow: Dict, artist_name: str, prompt_prefix: str = "1girl, ") -> Dict:
    """根据画师名修改工作流中的提示词"""
    full_prompt = f"{prompt_prefix}by {artist_name}"
    for node_id, node in workflow.items():
        if node.get('class_type') == 'CLIPTextEncode':
            node['inputs']['text'] = full_prompt
            return workflow
    raise ValueError("在工作流中未找到 CLIPTextEncode 节点")

def update_workflow_save_condition_filename(workflow: Dict, artist_id: int) -> Dict:
    """根据画师ID修改 SaveCondition 节点的保存文件名，使用 artist_{id} 格式"""
    filename = f"artist_{artist_id}"
    for node_id, node in workflow.items():
        if node.get('class_type') == 'SaveCondition':
            node['inputs']['filename'] = filename
            return workflow
    print("⚠️ 未找到 SaveCondition 节点，请确认工作流已正确导出")
    return workflow

# ===============================================
# 画师标签 → 条件向量编码器（步骤 1 / 3）
#
# 前置条件：
#   1. ComfyUI 已安装 SaveCondition 节点（编码结果写入 models/conditions/）
#   2. 准备好 Markdown 画师表（至少 | id | name | 两列）
#   3. 准备好 ComfyUI 工作流模板（含 CLIPTextEncode + SaveCondition）
#
# 产出：
#   ComfyUI/models/conditions/artist_*.ckpt  需手动拷贝到本项目 data/conditions/
#   artist_vectors_metadata.csv              手动放到 data/ 后供 build_index 使用
# ===============================================
def main():
    # ---- 配置参数 ----
    COMFYUI_SERVER = "127.0.0.1:8188"                             # ComfyUI服务地址
    MD_FILE_PATH = "./artists/artists40k.md"                        # Markdown画师表路径
    WORKFLOW_TEMPLATE = "workflows/artist_workflow_api.json"       # 编码工作流模板
    OUTPUT_CSV = "./data/artist_vectors_metadata.csv"              # 元数据CSV输出路径

    # ComfyUI 条件保存目录（编码结果写入这里，完成后手动拷贝到 data/conditions/）
    # ⚠️ 运行前请改成你自己的 ComfyUI models/conditions/ 目录！
    CONDITIONS_DIR = Path("<你的ComfyUI目录>/models/conditions")
    
    # ---- 加载工作流模板 ----
    try:
        with open(WORKFLOW_TEMPLATE, 'r', encoding='utf-8') as f:
            base_workflow = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到工作流模板文件: {WORKFLOW_TEMPLATE}")
        return
    
    # ---- 解析画师列表 ----
    artists = parse_artist_md_table(MD_FILE_PATH)
    if not artists:
        print("❌ 未解析到任何画师，请检查Markdown文件格式。")
        return

    # ---- 准备记录元数据 ----
    metadata_records = []
    total_start_time = time.time()  # 总计时起点

    # ---- 使用 tqdm 显示总进度条 ----
    print("\n🚀 开始批量生成画师条件向量...")
    for artist in tqdm(artists, desc="🎨 总进度", unit="位画师", ncols=100):
        artist_id = artist['id']
        artist_name = artist['name']
        single_start_time = time.time()  # 单个画师计时起点
        
        # 复制并修改工作流
        workflow = json.loads(json.dumps(base_workflow))  # 深拷贝
        workflow = update_workflow_prompt(workflow, artist_name)
        workflow = update_workflow_save_condition_filename(workflow, artist_id)
        
        try:
            # 提交任务并等待结果
            prompt_id = queue_prompt(COMFYUI_SERVER, workflow)
            
            result = wait_for_prompt(COMFYUI_SERVER, prompt_id)
            if result is None:
                tqdm.write(f"  ⚠️ {artist_name} 等待超时，跳过")
                continue
                
            # 检查文件是否生成
            expected_filename = f"artist_{artist_id}.ckpt"
            expected_path = CONDITIONS_DIR / expected_filename
            
            # 等待文件写入完成（最多等待5秒）
            waited = 0
            while not expected_path.exists() and waited < 5:
                time.sleep(0.5)
                waited += 0.5
            
            if expected_path.exists():
                file_size = expected_path.stat().st_size
                elapsed = time.time() - single_start_time
                tqdm.write(
                    f"  ✅ {artist_name} (ID: {artist_id}) 完成，"
                    f"文件大小 {file_size:,} 字节，耗时 {elapsed:.2f} 秒"
                )
                
                # 记录元数据（包含耗时）
                metadata_records.append({
                    'artist_id': artist_id,
                    'artist_name': artist_name,
                    'post_count': artist.get('post_count', 0),
                    'uniqueness_score': artist.get('uniqueness_score', 0.0),
                    'ckpt_filename': expected_filename,
                    'ckpt_path': str(expected_path),
                    'processing_time_sec': round(elapsed, 2)  # 新增耗时字段
                })
            else:
                tqdm.write(f"  ⚠️ 未找到生成的文件: {expected_filename}")
                
        except Exception as e:
            tqdm.write(f"  ❌ {artist_name} 处理出错: {e}")
        
        # 短暂停顿，避免请求过快
        time.sleep(0.3)
    
    # ---- 总体统计报告 ----
    total_elapsed = time.time() - total_start_time
    print("\n" + "="*50)
    print(f"⏱️  总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
    print(f"✅ 成功处理: {len(metadata_records)} / {len(artists)} 位画师")
    
    if metadata_records:
        # 计算平均耗时
        avg_time = sum(r['processing_time_sec'] for r in metadata_records) / len(metadata_records)
        print(f"📊 平均每个画师耗时: {avg_time:.2f} 秒")
    
    # ---- 保存元数据CSV ----
    if metadata_records:
        df = pd.DataFrame(metadata_records)
        # 确保输出目录存在
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"📋 元数据已保存至: {OUTPUT_CSV}")
    else:
        print("\n⚠️ 没有成功处理任何画师，请检查ComfyUI服务是否正常运行")

if __name__ == "__main__":
    main()