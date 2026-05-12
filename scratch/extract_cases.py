import csv
import json

targets = ['22', '31', '35']
methods = ['mtfc_v2', 'st_mtfc']
results_path = 'data/case_study_pilot2_qwen8b.csv'

comparison = []
with open(results_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['question_id'] in targets and row['method'] in methods:
            comparison.append({
                'ID': row['question_id'],
                'Method': row['method'],
                'Score': row['answer_score'],
                'CA': row['overlap_ca'],
                'GA': row['overlap_ga'],
                'Failure': row['failure_type'],
                'Answer': row['answer'][:300] + '...' # 只截取前300字
            })

print(json.dumps(comparison, ensure_ascii=False, indent=2))
