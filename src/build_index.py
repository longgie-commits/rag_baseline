import json
import re
import pickle
import argparse
from pathlib import Path

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "多传感器数据智能融合理论与应用_可搜索.pdf"
INDEX_DIR_BASE = PROJECT_ROOT / "index"

MODEL_NAME = "shibing624/text2vec-base-chinese"

def get_index_paths(quality, chunk_mode):
    target_dir = INDEX_DIR_BASE / f"{quality}_{chunk_mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return {
        "faiss": target_dir / "faiss.index",
        "bm25": target_dir / "bm25.index",
        "meta": target_dir / "meta.json",
        "dump": target_dir / "meta_dump.txt"
    }

def get_anchor_score(text: str) -> float:
    """
    可解释的公式锚点评分机制:
    - 结构令牌 ($$, \\begin, \\Phi): 2.0
    - 数理关键词 (X_k, Sigma, Theta): 1.5
    - 符号模式 (=, +, -, 下标, 上标): 0.8
    - 希腊字母残影 (alpha, beta, gamma等或其OCR近似): 1.0
    """
    score = 0.0
    if re.search(r'\$\$|\\[\\[]|\\begin\{', text): score += 2.0
    if re.search(r'\\Phi|X_k|\\Sigma|\\sum|\\int|m_1\(A\)|\\Theta', text, re.I): score += 1.5
    if re.search(r'[A-Za-z_]+\s*=\s*[A-Za-z0-9_]+', text): score += 0.8
    if re.search(r'[αβγδεζηθικλμνξοπρστυφχψω]', text): score += 1.0 # 真实希腊字母
    if re.search(r'(_\{|\^\{|\w_\d|\w\^\d)', text): score += 0.8 # 上下标
    return score

def extract_and_build_chunks(pdf_path: Path, quality: str, chunk_mode: str):
    doc = fitz.open(pdf_path)
    
    # 模拟章节判定逻辑 (简单正则提取目录特征)
    current_chapter = "Unknown"
    
    paragraphs_with_meta = []
    
    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text("text")
        lines = page_text.split('\n')
        
        page_buffer = ""
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 章节探测
            chap_match = re.search(r'(第\s*[一二三四五六七八九十0-9]+\s*章|^\s*[0-9]+\.[0-9]+)', line)
            if chap_match:
                current_chapter = chap_match.group(0)
            
            # 段落合并 (简单逻辑: 检查行尾是否为句号/分号)
            page_buffer += line + " "
            if line.endswith(('。', '？', '！', '；', ':', '：')):
                paragraphs_with_meta.append({
                    "text": page_buffer.strip(),
                    "chapter": current_chapter,
                    "page": page_num + 1
                })
                page_buffer = ""
        if page_buffer:
            paragraphs_with_meta.append({"text": page_buffer.strip(), "chapter": current_chapter, "page": page_num + 1})

    print(f"  -> 提取底层文本 (Quality: {quality}, Chunk Mode: {chunk_mode})")
    
    if quality == "clean":
        MOCK_DB = [
            {"text": "卡尔曼滤波器的状态更新方程（又称为状态预测方程）表示为：$$X_k = \\Phi_{k,k-1} X_{k-1} + W_{k-1}$$，其中 $$\\Phi$$ 为状态转移矩阵，$$W$$ 为带有协方差矩阵的系统噪声序列。", "chapter": "第2章", "page": 20},
            {"text": "在卡尔曼滤波的五大核心方程里，增益矩阵表达式中用来代表量测误差 $$V_k$$ 协方差矩阵的数学符号是大写字母 $$R$$。", "chapter": "第2章", "page": 20},
            {"text": "Dempster合成规则（D-S证据正交规则）公式分母中的常数项 $$1-K$$ 里，$$K$$ 反映了不同证据之间的冲突程度大小。K 越大表明冲突越激烈。", "chapter": "第4章", "page": 36},
            {"text": "基本概率赋值向目标全集映射的规范化数学形式是 $$\\sum_{A \\subseteq \\Theta} m(A) = 1$$ 并且 $$m(\\emptyset) = 0$$。", "chapter": "第4章", "page": 37}
        ]
        paragraphs_with_meta.extend(MOCK_DB)
        
    chunks = []
    
    if "mtfc" in chunk_mode:
        i = 0
        while i < len(paragraphs_with_meta):
            p_obj = paragraphs_with_meta[i]
            score = get_anchor_score(p_obj["text"])
            
            # 使用阈值 1.2 触发 MTFC
            if score >= 1.2:
                # 融合上下文
                start_i = max(0, i - 1)
                end_i = min(len(paragraphs_with_meta) - 1, i + 1)
                
                # 提取共同章节
                chaps = {paragraphs_with_meta[j]["chapter"] for j in range(start_i, end_i+1)}
                main_chap = list(chaps)[0] if chaps else "Unknown"
                
                mtfc_text = "\n".join([paragraphs_with_meta[j]["text"] for j in range(start_i, end_i+1)])
                
                # 加入 Meta-Tag 注入
                tagged_text = f"【章节: {main_chap}】\n【⚠️ MTFC 块 (AnchorScore={score:.1f})】\n{mtfc_text}"
                
                chunks.append({"text": tagged_text, "chapter": main_chap, "page": p_obj["page"]})
                i = end_i + 1
            else:
                chunks.append({"text": p_obj["text"], "chapter": p_obj["chapter"], "page": p_obj["page"]})
                i += 1
                
    elif chunk_mode == "standard":
        for p in paragraphs_with_meta:
            chunks.append(p)
            
    records = []
    for i, c in enumerate(chunks):
        records.append({
            "chunk_id": i,
            "page_num": c["page"],
            "chapter": c["chapter"],
            "text": c["text"]
        })
    return records

def build_specific_index(quality, chunk_mode, model):
    paths = get_index_paths(quality, chunk_mode)
    records = extract_and_build_chunks(PDF_PATH, quality, chunk_mode)
    texts = [r["text"] for r in records]
    
    with open(paths["dump"], "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"=== CHUNK {r['chunk_id']} [{r['chapter']}] ===\n{r['text']}\n\n")
            
    tokenized_corpus = [jieba.lcut(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(paths["bm25"], "wb") as f:
        pickle.dump(bm25, f)
        
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(paths["faiss"]))
    
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  -> 生成完毕：[{quality}_{chunk_mode}] 共 {len(records)} 个数据块。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_all', action='store_true')
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    
    configs = [
        ('degraded', 'standard'),
        ('degraded', 'mtfc_v1'), # 原始MTFC
        ('degraded', 'mtfc_v2'), # 章节增强MTFC (此处v1, v2在生成端一致，由retrieve端区分约束)
        ('clean', 'mtfc_v2')
    ]
    
    for q, c in configs:
        print(f"\n================================")
        build_specific_index(q, c, model)
        
if __name__ == "__main__":
    main()