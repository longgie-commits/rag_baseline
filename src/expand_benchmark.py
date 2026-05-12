import json
import csv
import random
import re
import sys
import os
sys.path.append('src')
import retrieve
from retrieve import call_remote_model

# 配置
CHUNKS_PATH = "data/rag_chunks.json" # 假设你有这个切片备份，如果没有我会从检索结果里挖
BENCHMARK_PATH = "data/benchmark.csv"
OUTPUT_PATH = "data/benchmark_v2.csv"
TARGET_COUNT = 100

def generate_stem_qa(chunk_text):
    """利用 Qwen-8B 基于公式块生成高难度 QA"""
    prompt = f"""你是一位 STEM 领域的专家。请根据以下教材片段，生成一个高质量的问答对。
要求：
1. 问题必须涉及片段中的【公式推导】或【计算逻辑】。
2. 问题应具有挑战性，能够测试出 AI 是否会产生“符号幻觉”。
3. 提供准确的参考答案（Reference Answer）。
4. 明确标注出证据来源（Evidence Span）。

教材片段：
{chunk_text}

请按以下 JSON 格式输出：
{{
  "question": "...",
  "reference_answer": "...",
  "evidence_span": "...",
  "type": "formula_reasoning"
}}
"""
    try:
        response = call_remote_model(prompt)
        # 提取 JSON
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"生成失败: {e}")
    return None

def main():
    # 1. 加载现有题目
    existing_data = []
    with open(BENCHMARK_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        existing_data = list(reader)
    
    current_ids = [int(item['id']) for item in existing_data]
    next_id = max(current_ids) + 1 if current_ids else 1
    
    print(f"当前题目数: {len(existing_data)}，目标总数: {TARGET_COUNT}")
    
    # 2. 挖掘公式块 (利用检索系统的 search 功能随机抓取)
    new_items = []
    needed = TARGET_COUNT - len(existing_data)
    
    # 模拟一些关键词来搜公式块
    keywords = ["公式", "推导", "卡尔曼", "贝叶斯", "状态转移", "协方差", "似然", "证据", "模糊", "神经网络"]
    
    while len(new_items) < needed:
        kw = random.choice(keywords)
        # 随机抓一些块
        results = retrieve.search(kw, top_k=10, chunk_mode="st_mtfc")
        
        # 筛选出带 [FORMULA] 标签的
        formula_results = [r for r in results if retrieve.identify_structure_tag(r['text']) == "FORMULA"]
        
        for r in formula_results:
            if len(new_items) >= needed: break
            
            print(f"正在基于 ID {r['chunk_id']} 生成第 {len(new_items)+1}/{needed} 道新题...")
            qa = generate_stem_qa(r['text'])
            
            if qa:
                qa['id'] = str(next_id)
                # 补全 CSV 字段
                qa['chapter'] = "Unknown" # 自动生成的暂时归类
                qa['section'] = "Unknown"
                qa['difficulty'] = "hard"
                qa['answerability'] = "yes"
                new_items.append(qa)
                next_id += 1
                
    # 3. 合并并保存
    all_data = existing_data + new_items
    fieldnames = existing_data[0].keys() if existing_data else ['id', 'chapter', 'section', 'question', 'type', 'answerability', 'reference_answer', 'evidence_span', 'difficulty', 'notes']
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in all_data:
            # 确保字段对齐
            row = {k: item.get(k, "") for k in fieldnames}
            writer.writerow(row)
            
    print(f"成功！题库已扩充至 {len(all_data)} 题，保存在 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
