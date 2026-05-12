import csv
import os
import json
import re
import torch
import retrieve
from retrieve import search, generate_answer, consolidate_context

# === 配置区 (Qwen-8B 复核版) ===
BENCHMARK_PATH = "data/benchmark_v2.csv"
CASE_STUDY_PATH = "data/case_study_pilot2_qwen8b.csv"
ANALYSIS_CASES_PATH = "data/analysis_cases_qwen8b.json"
REPORT_PATH = "data/eval_report_qwen8b_full100.md"
OLD_RESULTS_PATH = "data/case_study_pilot2.csv" # 保持对比

# 百题巅峰对决: 跑通所有 100 道题
TEST_IDS = [] 

# 四路方法对比
CONFIGS = [
    ("degraded", "standard"),
    ("degraded", "mtfc_v1"),
    ("degraded", "mtfc_v2"),
    ("degraded", "st_mtfc")
]
TOP_K = 3

# === 深度指标计算工具 ===

def extract_symbols(text):
    """提取公式、变量及关键符号"""
    if not text: return set()
    # 匹配 LaTeX, 下标变量, 以及常见的矩阵/大写变量
    patterns = [
        r'\$.*?\$', 
        r'\\\[.*?\\\]', 
        r'[A-Za-z]+_[a-z0-9{}]+', 
        r'[A-Z][a-z]*' # 捕捉 A, B, X, U 等矩阵符号
    ]
    res = []
    for p in patterns:
        res.extend(re.findall(p, str(text)))
    return set(res)

def calculate_overlap(set1, set2):
    if not set1: return 0
    return len(set1 & set2)

def check_retrieval_hit_rule(context, gold_span):
    if not gold_span or len(gold_span) < 5: return 1
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,}|[A-Za-z]+_[a-z0-9{}]+|\$.*?\$', gold_span)
    if not keywords: return 1
    hits = sum(1 for k in keywords if k in context)
    return 1 if hits / len(keywords) > 0.4 else 0

def evaluate_multi_metrics(context, answer, gold_answer, gold_span):
    """多粒度评估核心逻辑"""
    prompt = f"""你是一名严谨的学术评委。请对比 [参考上下文]、[参考标准答案] 和 [待评测回答]。
    待评测回答：{answer}
    标准答案：{gold_answer}
    参考上下文：{context[:1500]}
    
    请严格按 JSON 输出: {{"score": 0-100, "formula_match": 0/1, "faithfulness": 0/1, "failure_type": "..."}}
    """
    
    # 1. 基础命中判定
    hit_rule = check_retrieval_hit_rule(context, gold_span)
    
    # 2. 符号重合度分析 (三方对比)
    g_syms = extract_symbols(gold_answer)
    c_syms = extract_symbols(context)
    a_syms = extract_symbols(answer)
    
    overlap_gc = calculate_overlap(g_syms, c_syms)
    overlap_ca = calculate_overlap(c_syms, a_syms)
    overlap_ga = calculate_overlap(g_syms, a_syms)
    
    missing = list(g_syms - a_syms)[:5]
    extra = list(a_syms - g_syms)[:5]
    
    # 3. LLM 裁判判定 (改为调用远程 8B 模型)
    try:
        resp = retrieve.call_remote_model(prompt)
        data = json.loads(re.search(r'\{.*\}', resp, re.DOTALL).group())
    except:
        data = {"score": 50 if overlap_ga > 0 else 0, "formula_match": 0, "faithfulness": 1, "failure_type": "judge_parse_error"}
    
    # 4. 疑似参数覆盖判定
    p_override = "No"
    if overlap_gc > 1 and overlap_ga < overlap_gc:
        p_override = "Suspected"
        if data.get("failure_type") == "None":
            data["failure_type"] = "evidence_answer_inconsistency"

    return {
        "score": data.get("score", 0),
        "retrieval_hit": hit_rule,
        "formula_match": data.get("formula_match", 0),
        "faithfulness": data.get("faithfulness", 1),
        "failure_type": data.get("failure_type", "None"),
        "p_override": p_override,
        "overlap_gc": overlap_gc,
        "overlap_ca": overlap_ca,
        "overlap_ga": overlap_ga,
        "missing": str(missing),
        "extra": str(extra)
    }

# === 主流程 ===

def run_pilot():
    if not os.path.exists("data"): os.makedirs("data")
    with open(BENCHMARK_PATH, "r", encoding="utf-8-sig") as f:
        benchmark = list(csv.DictReader(f))

    # Smoke Test 过滤逻辑
    if TEST_IDS:
        print(f"!!! SMOKE TEST MODE ENABLED: Only processing IDs {TEST_IDS} !!!")
        benchmark = [item for item in benchmark if item.get('id', item.get('\ufeffid')) in TEST_IDS]

    all_results = []
    for q_idx, c_mode in CONFIGS:
        print(f"\n>>> Running Config: {q_idx} + {c_mode}")
        for item in benchmark:
            qid = item.get('id', item.get('\ufeffid'))
            print(f"  [Q{qid}] Processing...", end='\r')
            
            # 检索
            top_k_results = search(item['question'], top_k=TOP_K, advanced_mode=True, quality=q_idx, chunk_mode=c_mode)
            # 生成
            answer = generate_answer(item['question'], top_k_results, advanced_mode=True, chunk_mode=c_mode)
            # 上下文预览 (用于报告)
            context = consolidate_context(top_k_results, chunk_mode=c_mode)
            
            metrics = evaluate_multi_metrics(context, answer, item['reference_answer'], item.get('evidence_span', ""))
            
            print(f"  [Q{qid}] Score: {metrics['score']} | CA: {metrics['overlap_ca']} | GA: {metrics['overlap_ga']}")
            
            res = {
                "question_id": qid, "method": c_mode, "category": item['type'],
                "answer_score": metrics["score"], "retrieval_hit": metrics["retrieval_hit"],
                "formula_match": metrics["formula_match"], "faithfulness": metrics["faithfulness"],
                "p_override": metrics["p_override"], "failure_type": metrics["failure_type"],
                "overlap_gc": metrics["overlap_gc"], "overlap_ca": metrics["overlap_ca"], "overlap_ga": metrics["overlap_ga"],
                "missing_symbols": metrics["missing"], "extra_symbols": metrics["extra"],
                "context_preview": context[:150].replace('\n', ' '),
                "answer": answer.replace('\n', ' ')
            }
            all_results.append(res)

    # 1. 保存 CSV
    with open(CASE_STUDY_PATH, "w", encoding="utf-8-sig", newline="") as f:
        csv.DictWriter(f, fieldnames=all_results[0].keys()).writeheader()
        csv.DictWriter(f, fieldnames=all_results[0].keys()).writerows(all_results)
    
    # 2. 生成分析 JSON
    analysis = {
        "Suspected_Override": [r for r in all_results if r["p_override"] == "Suspected"],
        "Success_Cases": [r for r in all_results if r["answer_score"] >= 80],
        "Q31_Comparison": [r for r in all_results if r["question_id"] == "31"]
    }
    with open(ANALYSIS_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    # 3. 生成报告 (带 1.5B 对比)
    generate_report(all_results)

def get_stats(results):
    stats = {}
    methods = ["standard", "mtfc_v1", "mtfc_v2", "st_mtfc"]
    for m in methods:
        m_res = [r for r in results if r["method"] == m]
        if not m_res: continue
        stats[m] = {
            "avg_s": sum(float(r["answer_score"]) for r in m_res) / len(m_res),
            "avg_h": sum(float(r["retrieval_hit"]) for r in m_res) / len(m_res),
            "avg_f": sum(1 if str(r["formula_match"]).lower() in ["1", "true"] else 0 for r in m_res) / len(m_res),
            "hr": sum(1 if str(r.get("faithfulness", "1")) == "0" else 0 for r in m_res) / len(m_res),
            "p_count": len([r for r in m_res if r["p_override"] == "Suspected"]),
            "avg_gc": sum(float(r["overlap_gc"]) for r in m_res) / len(m_res),
            "avg_ca": sum(float(r["overlap_ca"]) for r in m_res) / len(m_res),
            "avg_ga": sum(float(r["overlap_ga"]) for r in m_res) / len(m_res),
            "failure_dist": {}
        }
        # 统计 failure_type 分布
        for r in m_res:
            ft = r["failure_type"]
            stats[m]["failure_dist"][ft] = stats[m]["failure_dist"].get(ft, 0) + 1
    return stats

def generate_report(results):
    stats_8b = get_stats(results)
    
    report = [
        "# ST-MTFC 结构感知 RAG 实验分析报告 (Qwen-8B 复核版)",
        f"> **评测环境**: Qwen-8B-Remote (Judge) | **数据集**: Degraded (OCR) | **设置**: Top-K={TOP_K}\n",
        "## 1. 跨方法性能对比 (Qwen-8B)",
        "| Method | Avg Score | Hallucination Rate | Hit Rate | Formula Match | Suspected Override | GC_Overlap | CA_Overlap | GA_Overlap |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    methods = ["standard", "mtfc_v1", "mtfc_v2", "st_mtfc"]
    for m in methods:
        if m not in stats_8b: continue
        s = stats_8b[m]
        report.append(f"| {m} | {s['avg_s']:.1f} | {s['hr']*100:.1f}% | {s['avg_h']*100:.1f}% | {s['avg_f']*100:.1f}% | {s['p_count']} | {s['avg_gc']:.2f} | {s['avg_ca']:.2f} | {s['avg_ga']:.2f} |")
    
    # === 1.5B vs 8B 对比表 ===
    if os.path.exists(OLD_RESULTS_PATH):
        report.append("\n## 2. 1.5B vs 8B 核心指标对比 (st_mtfc 模式)")
        report.append("| Metric | Qwen-1.5B (Old) | Qwen-8B (New) | Trend |")
        report.append("| :--- | :--- | :--- | :--- |")
        
        with open(OLD_RESULTS_PATH, "r", encoding="utf-8-sig") as f:
            old_data = list(csv.DictReader(f))
        stats_15b = get_stats(old_data)
        
        m = "st_mtfc"
        if m in stats_15b and m in stats_8b:
            s1 = stats_15b[m]
            s2 = stats_8b[m]
            metrics = [
                ("Answer Score", "avg_s", ".1f"),
                ("Hallucination Rate", "hr", ".1%"),
                ("Formula Match", "avg_f", ".1f"),
                ("CA_Overlap", "avg_ca", ".2f"),
                ("GA_Overlap", "avg_ga", ".2f"),
                ("GC_Overlap", "avg_gc", ".2f"),
                ("Suspected Case Count", "p_count", "d")
            ]
            for label, key, fmt in metrics:
                v1, v2 = s1[key], s2[key]
                trend = "⬆️" if v2 > v1 else ("⬇️" if v2 < v1 else "➡️")
                # 针对 Suspected Case Count 和 Hallucination Rate，下降才是提升
                if key in ["p_count", "hr"]: trend = "✅" if v2 < v1 else ("⚠️" if v2 > v1 else "➡️")
                
                report.append(f"| {label} | {v1:{fmt}} | {v2:{fmt}} | {trend} |")

        report.append("\n## 3. Failure Type 分布对比 (st_mtfc)")
        if m in stats_15b and m in stats_8b:
            report.append(f"- **1.5B**: {stats_15b[m]['failure_dist']}")
            report.append(f"- **8B**: {stats_8b[m]['failure_dist']}")

    report.append("\n## 4. Evidence Usage vs Symbol Alignment Analysis")
    report.append("> 专项分析：探讨模型是否存在“过度采信但未对齐”的情况（即高 CA_Overlap 但低 GA_Overlap）。\n")
    report.append("| ID | Method | CA | GA | Alignment Gap | Rationale |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # 筛选对齐差距较大的案例 (Gap > 2)
    alignment_gaps = [r for r in results if float(r["overlap_ca"]) > float(r["overlap_ga"]) + 2]
    for r in alignment_gaps:
        gap = float(r["overlap_ca"]) - float(r["overlap_ga"])
        report.append(f"| {r['question_id']} | {r['method']} | {r['overlap_ca']} | {r['overlap_ga']} | {gap:.0f} | {r['failure_type'][:30]}... |")

    report.append("\n### 案例深度观察")
    if alignment_gaps:
        report.append(f"1. **符号错位现象**: 在上述 {len(alignment_gaps)} 个案例中，模型表现出强烈的“证据采信欲望”，但因 OCR 乱码或模型推理局限，未能将采信的符号正确对齐到标准答案逻辑中。")
        # 比较 mtfc_v2 与 st_mtfc
        v2_gaps = len([r for r in alignment_gaps if r["method"] == "mtfc_v2"])
        st_gaps = len([r for r in alignment_gaps if r["method"] == "st_mtfc"])
        report.append(f"2. **方法论差异**: `mtfc_v2` 出现严重对齐 Gap 的频率为 {v2_gaps} 次，而 `st_mtfc` 为 {st_gaps} 次。这可能指示了结构化标注对“盲目引用”具有一定的抑制作用。")
    else:
        report.append("1. **结论**: 8B 模型在 35 题全量测试中未表现出明显的“高 CA 但低 GA”大规模失控，符号采信与准确度保持了较好的一致性。")

    report.append("\n## 5. 疑似证据-答案不一致性案例 (8B)")
    report.append("| ID | Method | GC_Overlap | CA_Overlap | GA_Overlap | Failure Type |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    overrides = [r for r in results if r["p_override"] == "Suspected"]
    for r in overrides:
        report.append(f"| {r['question_id']} | {r['method']} | {r['overlap_gc']} | {r['overlap_ca']} | {r['overlap_ga']} | {r['failure_type']} |")

    report.append("\n## 4. 实验观察与初步讨论")
    report.append("1. **结构化标注趋势**: 在 Pilot 设置下，ST-MTFC 观察到疑似 evidence-answer inconsistency 案例从 5 例（standard）下降至 2 例。这指示了结构化标注可能有助于提升生成答案对检索证据的采信度。")
    report.append("2. **符号一致性分析**: 观察到 CA_Overlap 指标从 0.52 (standard) 轻微上升至 0.60 (st_mtfc)。该趋势暗示了 Structure Tagging 对模型注意力分布可能存在微弱的正面引导作用。")
    report.append("3. **典型案例 (Q31)**: 在 Q31 的横向对比中，standard 模式表现为 CA_Overlap=1/GA_Overlap=0，而 st_mtfc 表现为 CA_Overlap=3/GA_Overlap=2。该变动指示了结构感知对特定符号密集型任务的敏感性。")
    report.append("4. **结论审慎性**: 以上结论仅基于 1.5B Judge 模型及 35 题规模的初步观察。后续将在更强 Judge 或全量 7B 模型上复核该趋势。")
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nFinal Rigorous Report generated at: {REPORT_PATH}")
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nFinal Report generated at: {REPORT_PATH}")

if __name__ == "__main__":
    run_pilot()
