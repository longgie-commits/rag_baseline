import csv
import sys
import os
sys.path.append('src')
import evaluate
import retrieve
from evaluate import evaluate_multi_metrics
from retrieve import search, generate_answer, consolidate_context

# 配置
CSV_PATH = "data/case_study_pilot2_qwen8b.csv"
BENCHMARK_PATH = "data/benchmark_v2.csv"
REPORT_PATH = "data/eval_report_qwen8b_full100.md"

def repair():
    print("--- 正在启动结果修复程序 ---")
    
    # 1. 加载题目库
    with open(BENCHMARK_PATH, 'r', encoding='utf-8-sig') as f:
        benchmark = {it['id']: it for it in csv.DictReader(f)}
    
    # 2. 读取现有结果
    results = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        results = list(csv.DictReader(f))
    
    repaired_count = 0
    for i, row in enumerate(results):
        # 寻找因为报错导致的 0 分案例
        if row['answer_score'] == '0' and ('error' in row['failure_type'].lower() or 'timeout' in row['failure_type'].lower() or '超时' in row['failure_type']):
            qid = row['question_id']
            mode = row['method']
            print(f"正在修复: Q{qid} ({mode})...")
            
            item = benchmark[qid]
            try:
                # 重新运行 RAG
                top_k_results = search(item['question'], top_k=3, chunk_mode=mode)
                context = consolidate_context(top_k_results, chunk_mode=mode)
                answer = generate_answer(item['question'], top_k_results, chunk_mode=mode)
                
                # 重新评测
                metrics = evaluate.evaluate_multi_metrics(context, answer, item['reference_answer'], item['evidence_span'])
                
                # 更新结果行
                results[i]['answer_score'] = str(metrics['score'])
                results[i]['overlap_gc'] = f"{metrics['overlap_gc']:.2f}"
                results[i]['overlap_ca'] = str(metrics['overlap_ca'])
                results[i]['overlap_ga'] = str(metrics['overlap_ga'])
                results[i]['formula_match'] = "True" if metrics['formula_match'] else "False"
                results[i]['failure_type'] = metrics['failure_type']
                
                repaired_count += 1
                print(f"修复成功: New Score = {metrics['score']}")
            except Exception as e:
                print(f"再次修复失败: {e}")

    if repaired_count > 0:
        # 3. 保存回 CSV
        fieldnames = results[0].keys()
        with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        # 4. 重新生成报告
        print("正在更新 Markdown 报告...")
        evaluate.generate_report(results)
        print(f"全部完成！共修复 {repaired_count} 处网络错误。")
    else:
        print("未发现需要修复的网络错误案例。")

if __name__ == "__main__":
    repair()
