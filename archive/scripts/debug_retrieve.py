import sys
sys.path.append('src')
from retrieve import search, generate_answer

def run_debug(query):
    print(f"QUERY: {query}")
    res = search(query, top_k=3)
    text = "\n".join([r['text'] for r in res])
    try:
        print("--- RETRIEVED TEXT ---")
        print(text.encode('utf-8', errors='ignore').decode('utf-8'))
    except Exception as e:
        print("Error printing:", e)
    
    ans = generate_answer(query, res)
    print("--- LLM ANSWER ---")
    print(ans)

run_debug("多传感器数据智能融合是什么？")
