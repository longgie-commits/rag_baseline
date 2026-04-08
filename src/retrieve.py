import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

MODEL_NAME = "shibing624/text2vec-base-chinese"
GEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 专精中文的轻量级大模型

# 加载生成模型
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME)
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)


def load_index_and_meta():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, meta


def search(query, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    index, meta = load_index_and_meta()

    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = meta[idx]
        results.append({
            "score": float(score),
            "page_num": item["page_num"],
            "chunk_in_page": item["chunk_in_page"],
            "text": item["text"]
        })

    return results


def generate_answer(query, top_k_texts):
    # 构建标准的问答 Prompt (已汉化)
    context = "\n".join([text["text"] for text in top_k_texts])
    prompt = f"请基于以下给出的【背景信息】，简明扼要地回答【问题】。如果背景信息中没有直接答案，请提炼相关内容回答。\n\n【背景信息】:\n{context}\n\n【问题】: {query}"
    
    messages = [
        {"role": "system", "content": "你是一个智能的问答助手。你必须根据提供的背景信息用中文回答用户的问题。"},
        {"role": "user", "content": prompt}
    ]

    # 将 messages 转换为 Qwen 专属的 ChatML 模板格式
    text = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize prompt
    inputs = gen_tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
    input_length = inputs.input_ids.shape[1]

    # Generate response
    outputs = gen_model.generate(
        **inputs, 
        max_new_tokens=150, 
        num_return_sequences=1,
        pad_token_id=gen_tokenizer.eos_token_id
    )

    # Decode only the newly generated tokens
    gen_tokens = outputs[0][input_length:]
    answer = gen_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return answer


def main():
    query = input("Enter your query: ").strip()
    if not query:
        print("Empty query.")
        return

    # 获取检索结果
    top_k_results = search(query, top_k=5)

    # 根据检索结果生成回答
    answer = generate_answer(query, top_k_results)

    # 输出生成的答案
    print("\n=== AI 生成的回答 ===")
    print(answer)


if __name__ == "__main__":
    main()