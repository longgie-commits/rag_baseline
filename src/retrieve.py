import json
import pickle
import re
import sys
from pathlib import Path

import jieba
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"
BM25_PATH = INDEX_DIR / "bm25.index"

MODEL_NAME = "shibing624/text2vec-base-chinese"
RERANKER_NAME = "BAAI/bge-reranker-base"

# 服务器配置
REMOTE_API_URL = "http://ps2.tail07443e.ts.net:2640/chat"

import psutil

def get_mem_info():
    mem = psutil.virtual_memory()
    return f"RAM Usage: {mem.percent}% (Used: {mem.used/1024**3:.1f}G / Total: {mem.total/1024**3:.1f}G)"

# 全局变量占位
gen_model = None
gen_tokenizer = None
embedding_model = None
reranker_model = None

def call_remote_model(prompt: str, history: list = None):
    """通用调用师兄服务器上的 Qwen-8B 接口"""
    payload = {
        "query": prompt,
        "use_kb": False,  # 强制使用我们自己的 RAG 逻辑，关闭服务器端的
        "history": history if history else [],
        "return_thinking": False
    }
    try:
        response = requests.post(REMOTE_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("answer", "Error: No answer in response")
    except Exception as e:
        return f"Remote Call Error: {str(e)}"

def init_models():
    global embedding_model, reranker_model
    if embedding_model is not None:
        return
        
    print(f"--- 启动模型加载序列 (云端生成模式) ---")
    print(get_mem_info())
    
    print("1. 正在加载 Embedding 模型...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    print("2. 正在加载 Reranker 模型...")
    reranker_model = CrossEncoder(RERANKER_NAME, max_length=512, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("3. 生成模型: 已配置为远程调用 (Qwen-8B @ ps2.tail07443e.ts.net)")
    print("--- 基础模型加载成功！ ---")
    print(get_mem_info())

def load_indexes_and_meta(quality="degraded", chunk_mode="mtfc"):
    # 确保模型已加载
    init_models()
    
    # ST-MTFC 复用 mtfc_v2 的物理索引，但增加逻辑增强
    target_mode = "mtfc_v2" if chunk_mode == "st_mtfc" else chunk_mode
    target_dir = PROJECT_ROOT / "index" / f"{quality}_{target_mode}"
    
    index_path = target_dir / "faiss.index"
    meta_path = target_dir / "meta.json"
    bm25_path = target_dir / "bm25.index"

    if not index_path.exists() or not meta_path.exists() or not bm25_path.exists():
        raise FileNotFoundError(f"Missing index files in {target_dir}. Please run build_index.py --ablation_all first.")

    faiss_index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    return faiss_index, bm25_index, meta

def rewrite_query(query: str) -> list:
    """ 前置子引擎：使用 LLM 动态改写/拆解 Query """
    prompt = (
        f"你是一个搜索关键词提取专家。目标是拆解用户的查询，以提高文档检索召回率。\n"
        f"如果原问题长且隐晦，或存在多个对比对象（如'XX和YY的区别'），请提取出2-3个最适合扔给搜索引擎的简短子查询。\n"
        f"如果问题很简单，就只输出它自己。\n"
        f"必须仅输出纯法的JSON数组格式，如 [\"查询A\", \"查询B\"]。不要带```json标记，不要解释。\n"
        f"【原问题】：{query}"
    )
    
    # 调用远程模型进行查询改写
    resp = call_remote_model(prompt)
    
    # 解析拆解的查询
    try:
        clean = resp.replace("```json", "").replace("```", "").strip()
        start = clean.find("[")
        end = clean.rfind("]")
        if start != -1 and end != -1:
            sub_queries = json.loads(clean[start:end+1])
            if isinstance(sub_queries, list) and sub_queries:
                return [str(q) for q in sub_queries]
    except:
        pass
    return [query]

def hybrid_search_pool(sub_queries: list, faiss_index, bm25_index, meta, embedding_model, pool_size=20):
    """ 多路召回：用 Faiss 和 BM25 分别搜刮尽可能多的候选人组合成超集 """
    candidate_dict = {}
    
    # 针对每一个拆解出来的子查询都进行独立多路召回
    for q in sub_queries:
        # ---- 1. Dense (Faiss) 召回 ----
        q_vec = embedding_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = faiss_index.search(q_vec, pool_size)
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx not in candidate_dict:
                candidate_dict[idx] = meta[idx]
        
        # ---- 2. Sparse (BM25) 召回 ----
        tokenized_q = jieba.lcut(q)
        bm25_scores = bm25_index.get_scores(tokenized_q)
        top_n_idx = np.argsort(bm25_scores)[::-1][:pool_size]
        for idx in top_n_idx:
            if bm25_scores[idx] > 0 and idx not in candidate_dict: # 只录入有得分的
                candidate_dict[idx] = meta[idx]
                
    return list(candidate_dict.values())

def identify_structure_tag(text):
    """升级版启发式识别：支持更复杂的符号识别和语义功能分类"""
    # 1. 变量定义/符号解释识别 (VARIABLE)
    var_keywords = ["其中", "式中", "表示", "定义为", "为……参数", "is defined as", "where", "denotes"]
    if any(k in text for k in var_keywords) and (len(text) < 300):
        return "VARIABLE"
    
    # 2. 公式识别 (FORMULA) - 增强正则匹配
    formula_patterns = [
        r'\$.*?\$', r'\\\[.*?\\\]', r'[A-Za-z]+_[a-z0-9{}]+', 
        r'=', r'≈', r'Σ', r'∑', r'∫', r'±', r'→', r'\\infty',
        r'exp\(.*?\)', r'log\(.*?\)', r'\\partial'
    ]
    if any(re.search(p, text) for p in formula_patterns):
        return "FORMULA"
    
    # 3. 说明性文本 (EXPLANATION)
    return "EXPLANATION"

def consolidate_context(reranked_results: list, chunk_mode: str = "standard") -> str:
    """ 
    组织证据：支持语义上下文链接 (Semantic Context Linking)
    ST-MTFC v2: 自动耦合相邻的公式与变量定义。
    """
    reranked_results.sort(key=lambda x: x["chunk_id"])
    
    processed_items = []
    for item in reranked_results:
        content = item["text"]
        tag = identify_structure_tag(content) if chunk_mode == "st_mtfc" else None
        processed_items.append({"chunk_id": item["chunk_id"], "text": content, "tag": tag})

    stitched_blocks = []
    current_unit = []
    last_chunk_id = -999
    
    i = 0
    while i < len(processed_items):
        item = processed_items[i]
        content = item["text"]
        tag = item["tag"]

        # ST-MTFC v2 核心逻辑：检测逻辑三元组 (EXPLANATION + FORMULA + VARIABLE)
        if chunk_mode == "st_mtfc" and i + 1 < len(processed_items):
            next_item = processed_items[i+1]
            if next_item["chunk_id"] == item["chunk_id"] + 1:
                # 场景 1：公式 + 变量定义 (Coupling Formula and its Variable anchors)
                # 场景 2：解释说明 + 公式 (Coupling Physical Context and its Formula)
                if (tag == "FORMULA" and next_item["tag"] == "VARIABLE") or \
                   (tag == "EXPLANATION" and next_item["tag"] == "FORMULA") or \
                   (tag == "VARIABLE" and next_item["tag"] == "FORMULA"):
                    coupled_text = f"<STEM_LOGIC_UNIT>\n[{tag}]\n{content}\n[{next_item['tag']}]\n{next_item['text']}\n</STEM_LOGIC_UNIT>"
                    stitched_blocks.append(coupled_text)
                    i += 2
                    last_chunk_id = next_item["chunk_id"]
                    continue

        # 普通缝合逻辑
        if chunk_mode == "st_mtfc":
            content = f"[{tag}]\n{content}"

        if item["chunk_id"] == last_chunk_id + 1:
            if stitched_blocks:
                stitched_blocks[-1] += f"\n{content}"
            else:
                stitched_blocks.append(content)
        else:
            stitched_blocks.append(content)
        
        last_chunk_id = item["chunk_id"]
        i += 1
        
    return "\n\n... (文档跳转) ...\n\n".join(stitched_blocks)


def search(query, top_k=5, advanced_mode=True, quality="degraded", chunk_mode="mtfc_v2"):
    """ 主管道出口方法 """
    faiss_index, bm25_index, meta = load_indexes_and_meta(quality, chunk_mode)
    
    if not advanced_mode:
        q_vec = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = faiss_index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item = meta[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    # === 全新系统架构 (Pilot 2 增强版) ===
    sub_queries = rewrite_query(query)
    
    # 提取 Query 中的章节线索 (如 "第2章", "卡尔曼" -> 2章)
    chapter_hint = None
    if "卡尔曼" in query or "第2章" in query: chapter_hint = "第2"
    if "证据理论" in query or "DS" in query or "第4章" in query: chapter_hint = "第4"
    if "贝叶斯" in query or "第3章" in query: chapter_hint = "第3"

    candidates = hybrid_search_pool(sub_queries, faiss_index, bm25_index, meta, embedding_model, pool_size=40)
    
    if not candidates: return []
        
    pairs = [[query, doc["text"]] for doc in candidates]
    rerank_scores = reranker_model.predict(pairs)
    
    def get_chap_num(s):
        m = re.search(r'(\d+)', s)
        return m.group(1) if m else s
    
    hint_num = get_chap_num(chapter_hint) if chapter_hint else None
    
    for i, score in enumerate(rerank_scores):
        final_score = float(score)
        
        # 章节约束硬惩罚: 如果启用了 v2/v3 且章节不匹配，得分大幅削减
        if chunk_mode in ["mtfc_v2", "st_mtfc"] and hint_num:
            doc_chap = candidates[i].get("chapter", "")
            chunk_chap_num = get_chap_num(doc_chap)
            
            if hint_num != chunk_chap_num:
                final_score -= 10.0 # 极大幅度惩罚跨章内容
                
        candidates[i]["rerank_score"] = final_score
        
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


def generate_answer(query, top_k_texts, advanced_mode=True, chunk_mode="standard"):
    # 4. 缝合证据链 (Context Stitching) 去除时空错乱感
    if advanced_mode:
        context = consolidate_context(top_k_texts, chunk_mode=chunk_mode)
    else:
        context = "\n".join([t["text"] for t in top_k_texts])
    
    prompt = f"请根据以下背景信息回答问题。如果背景中没有相关信息，请回答不知道。回答应尽量严谨、专业。注意：若背景中公式存在OCR识别错误，请优先保证逻辑准确，不要盲目套用错误符号。\n\n背景信息：\n{context}\n\n问题：\n{query}\n\n回答："
    
    # 调用远程模型生成最终答案
    answer = call_remote_model(prompt)
    return answer


def main():
    print("\n\n=========================================")
    print(" RAG 高级图谱检索装载完成 (支持BM25混合, Reranker 和 自动改写)")
    print("=========================================")
    
    while True:
        query = input("\n👉 Enter your query: ").strip()
        if not query: continue
        if query.lower() in ['quit', 'exit', 'q']: break

        print("正在进行智能检索链，请稍候...")
        results = search(query, top_k=5, advanced_mode=True)
        answer = generate_answer(query, results, advanced_mode=True)

        print("\n=== AI 综合答卷 ===")
        print(answer)

if __name__ == "__main__":
    main()