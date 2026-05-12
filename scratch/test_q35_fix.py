import sys
sys.path.append('src')
import csv
import evaluate
import retrieve
from retrieve import search, generate_answer

# 配置
BENCHMARK_PATH = "data/benchmark.csv"
TEST_ID = "35"

def test_q35_fix():
    print(f"--- 正在测试 Q{TEST_ID} 的修复效果 (ST-MTFC 模式) ---")
    
    # 1. 加载数据
    with open(BENCHMARK_PATH, 'r', encoding='utf-8-sig') as f:
        data = list(csv.DictReader(f))
        item = next(it for it in data if it['id'] == TEST_ID)

    # 2. 检索与生成 (st_mtfc 模式)
    top_k_results = search(item['question'], top_k=3, chunk_mode="st_mtfc")
    context = retrieve.consolidate_context(top_k_results, chunk_mode="st_mtfc")
    
    print("\n[Generated Answer (New Prompt)]:")
    answer = generate_answer(item['question'], top_k_results, chunk_mode="st_mtfc")
    print(answer)
    
    # 3. 评测
    print("\n[Judge Evaluation]:")
    metrics = evaluate.evaluate_multi_metrics(context, answer, item['reference_answer'], item.get('evidence_span', ""))
    print(f"Score: {metrics['score']}")
    print(f"Failure Type: {metrics['failure_type']}")
    print(f"CA Overlap: {metrics['overlap_ca']}")

if __name__ == "__main__":
    test_q35_fix()
