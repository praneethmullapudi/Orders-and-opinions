"""LLM-based router: decide whether a question needs the SQL pipeline or the RAG pipeline."""
from src.llm import chat

ROUTER_SYSTEM_PROMPT = """You classify user questions about an e-commerce dataset into one route.

- "sql": questions about numbers, aggregates, counts, revenue, trends, dates, categories, \
  delivery times, payments -- anything answerable by querying structured order/product/customer tables.
- "rag": questions about opinions, complaints, sentiment, or what customers said in free-text reviews.

Reply with exactly one word: sql or rag."""


def route(question: str) -> str:
    reply = chat(ROUTER_SYSTEM_PROMPT, question).strip().lower()
    return "rag" if "rag" in reply else "sql"


if __name__ == "__main__":
    for q in [
        "What were our top 5 product categories by revenue?",
        "What do customers complain about most in the electronics category?",
    ]:
        print(q, "->", route(q))
