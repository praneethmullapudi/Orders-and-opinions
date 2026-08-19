"""RAG pipeline over customer reviews: embed reviews at setup, retrieve + synthesize at query time."""
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.llm import chat

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_CSV = ROOT / "data" / "processed" / "reviews_for_embedding.csv"
INDEX_PATH = ROOT / "data" / "processed" / "reviews.faiss"
META_PATH = ROOT / "data" / "processed" / "reviews_meta.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder: SentenceTransformer | None = None

RAG_SYSTEM_PROMPT = """You are a customer insights analyst. Given a question and a set of \
customer review excerpts, write a short, grounded answer (2-4 sentences) that summarizes \
what the reviews say. Only use information present in the excerpts. Reference review numbers \
like [1], [2] when citing a specific point."""


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def build_index() -> None:
    df = pd.read_csv(REVIEWS_CSV)
    embedder = _get_embedder()
    embeddings = embedder.encode(df["review_text"].tolist(), show_progress_bar=True, batch_size=64)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)
    print(f"Indexed {len(df)} reviews to {INDEX_PATH}")


def _load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("FAISS index not built yet. Run: python src/rag_agent.py --build")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    index, meta = _load_index()
    embedder = _get_embedder()
    query_vec = embedder.encode([question]).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, ids = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        row = dict(meta[idx])
        row["score"] = float(score)
        results.append(row)
    return results


def answer_question(question: str, top_k: int = 5) -> dict:
    reviews = retrieve(question, top_k=top_k)
    if not reviews:
        return {"answer": "No relevant reviews found.", "reviews": []}

    excerpts = "\n".join(
        f"[{i + 1}] (score {r['review_score']}/5, category={r['product_category']}) {r['review_text']}"
        for i, r in enumerate(reviews)
    )
    answer = chat(RAG_SYSTEM_PROMPT, f"Question: {question}\n\nReview excerpts:\n{excerpts}")
    return {"answer": answer, "reviews": reviews}


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console defaults to cp1252

    if "--build" in sys.argv:
        build_index()
    else:
        out = answer_question("What do customers complain about most in the electronics category?")
        print(out["answer"])
