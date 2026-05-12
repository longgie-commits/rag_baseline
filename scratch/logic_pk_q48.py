import sys
import io
import json
import csv
from pathlib import Path

# 设置环境
sys.path.append('src')
import retrieve
import evaluate

def run_pk():
    target_id = '48'
    benchmark_path = 'data/benchmark_v2.csv'

    with open(benchmark_path, 'r', encoding='utf-8-sig') as f:
        q = [row for row in csv.DictReader(f) if row['id'] == target_id][0]

    retrieve.init_models()
    query = q['question']
    gold_ans = q['reference_answer']
    gold_span = q['evidence_span']

    print(f'\n--- Performing Logic PK for Q{target_id} ---')
    print('| Method | Score | HR | CA | GA | Rationale |')
    print('| :--- | :--- | :--- | :--- | :--- | :--- |')

    # 1. Long-Context
    res_long = retrieve.search(query, chunk_mode='standard', top_k=20)
    ans_long = retrieve.generate_answer(query, res_long, advanced_mode=False)
    e_long = evaluate.evaluate_multi_metrics('', ans_long, gold_ans, gold_span)
    print(f'| Long-Context (8B) | {e_long["score"]} | {1-int(e_long["faithfulness"])} | {e_long.get("overlap_ca", "-")} | {e_long.get("overlap_ga", "-")} | Information Overload |')

    # 2. ST-MTFC v2
    res_v2 = retrieve.search(query, chunk_mode='st_mtfc', top_k=3)
    ans_v2 = retrieve.generate_answer(query, res_v2, advanced_mode=True, chunk_mode='st_mtfc')
    e_v2 = evaluate.evaluate_multi_metrics('', ans_v2, gold_ans, gold_span)
    print(f'| ST-MTFC v2 | {e_v2["score"]} | {1-int(e_v2["faithfulness"])} | {e_v2.get("overlap_ca", "-")} | {e_v2.get("overlap_ga", "-")} | Logic Welded |')

    # 3. ST-MTFC v3 (Logic Chain)
    # 模拟逻辑链：在 v2 检索基础上，主动追加变量定义检索
    extra_defs = retrieve.search(query + ' 变量定义', chunk_mode='st_mtfc', top_k=2)
    res_v3 = res_v2 + extra_defs
    ans_v3 = retrieve.generate_answer(query, res_v3, advanced_mode=True, chunk_mode='st_mtfc')
    e_v3 = evaluate.evaluate_multi_metrics('', ans_v3, gold_ans, gold_span)
    print(f'| **ST-MTFC v3 (Chain)** | **{e_v3["score"]}** | **{1-int(e_v3["faithfulness"])}** | {e_v3.get("overlap_ca", "-")} | {e_v3.get("overlap_ga", "-")} | **Logical Chain Anchored** |')

if __name__ == '__main__':
    run_pk()
