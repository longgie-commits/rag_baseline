import json
from pathlib import Path

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "数据挖掘原理.pdf"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

MODEL_NAME = "shibing624/text2vec-base-chinese"


def read_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        text = text.replace("\u00a0", " ")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

        if text:
            pages.append({
                "page_num": page_num + 1,
                "text": text
            })

    return pages


def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        start = end - overlap

    return chunks


def build_chunks(pages):
    records = []

    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"]

        chunks = chunk_text(page_text, chunk_size=400, overlap=50)

        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id": len(records),
                "page_num": page_num,
                "chunk_in_page": i,
                "text": chunk
            })

    return records


def embed_texts(texts, model, batch_size=32):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading PDF: {PDF_PATH}")
    pages = read_pdf(PDF_PATH)
    print(f"Loaded {len(pages)} pages.")

    print("[2/4] Chunking text...")
    records = build_chunks(pages)
    print(f"Built {len(records)} chunks.")

    texts = [r["text"] for r in records]

    print(f"[3/4] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[4/4] Encoding and building FAISS index...")
    embeddings = embed_texts(texts, model)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved metadata to: {META_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()