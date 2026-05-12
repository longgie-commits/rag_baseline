import csv
import re
import os

RESULTS_PATH = "data/case_study_pilot2.csv"
BENCHMARK_PATH = "data/benchmark.csv"

def get_syms(text):
    if not text: return set()
    # 匹配大写字母(矩阵/变量)、下标变量、LaTeX
    return set(re.findall(r'[A-Za-z]+_[a-z0-9{}]+|\$.*?\$|[A-Z][a-z]*', str(text)))

def analyze():
    with open(BENCHMARK_PATH, "r", encoding="utf-8-sig") as f:
        benchmark = {r.get('id', r.get('\ufeffid')): r for r in csv.DictReader(f)}
    with open(RESULTS_PATH, "r", encoding="utf-8-sig") as f:
        results = list(csv.DictReader(f))

    print("\n### Systemic Evidence-Answer Inconsistency Discovery")
    print("| ID | Method | Gold_Ctx_Overlap | Gold_Ans_Overlap | Suspected Reason | Answer Snippet |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    found_count = 0
    for r in results:
        qid = r['question_id']
        gold_text = benchmark[qid]['reference_answer']
        
        g_syms = get_syms(gold_text)
        c_syms = get_syms(r['context_preview'])
        a_syms = get_syms(r['answer'])
        
        g_c = len(g_syms & c_syms)
        g_a = len(g_syms & a_syms)
        
        # 判定标准：证据里有(>1个符号)，但回答里没用全(重合度下降)
        if g_c > 1 and g_a < g_c:
            reason = "suspected_parametric_override"
            print(f"| {qid} | {r['method']} | {g_c} | {g_a} | {reason} | {r['answer'][:40]}... |")
            found_count += 1
            
    if found_count == 0:
        print("No cases found with current heuristics.")

if __name__ == "__main__":
    analyze()
