# AI Data Analyst Assistant

A Streamlit chat app that lets you ask natural-language questions over the [Olist Brazilian E-Commerce dataset](https://kaggle.com/datasets/olistbr/brazilian-ecommerce) — orders, products, payments, and customer reviews — and get grounded, explainable answers.

Each question is routed automatically to one of two pipelines:

- **Text-to-SQL** for structured questions ("top 5 categories by revenue", "average payment by type")
- **RAG (retrieval-augmented generation)** for questions about review sentiment ("what do customers complain about in electronics?")

Every answer comes with a "how I got this" panel showing the exact SQL query run, or the review excerpts retrieved.

## Architecture

```
User question
     │
     ▼
 ┌─────────┐
 │  Router  │  (LLM classifies: SQL question or RAG question?)
 └────┬────┘
      │
 ┌────┴─────────────────┐
 ▼                       ▼
SQL PATH               RAG PATH
question → LLM        question → embed → FAISS
generates SQL          retrieve top-k reviews
→ run on SQLite         → LLM synthesizes answer
→ LLM formats            with citations to reviews
  plain-English answer
      │                       │
      └───────────┬───────────┘
                   ▼
        Streamlit Chat UI
```

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Groq API (Llama 3.3 70B) |
| Structured data | SQLite |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, local) |
| Vector store | FAISS |
| UI | Streamlit |

## Setup

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Get a free [Groq API key](https://console.groq.com/keys) and add it:

   ```bash
   cp .env.example .env
   # edit .env and set GROQ_API_KEY
   ```

3. Download the [Olist dataset CSVs](https://kaggle.com/datasets/olistbr/brazilian-ecommerce) into `data/raw/`.

4. Build the SQLite DB and the review-embedding export:

   ```bash
   python src/data_prep.py
   ```

5. Build the FAISS index over review text:

   ```bash
   python src/rag_agent.py --build
   ```

6. Run the app:

   ```bash
   streamlit run app.py
   ```

## Project structure

```
data/
  raw/            # Olist CSVs (not committed)
  processed/      # olist.db (SQLite) + FAISS index (not committed)
notebooks/
  analysis.ipynb  # EDA on the dataset
src/
  data_prep.py    # CSVs -> SQLite, prep review text
  sql_agent.py    # text-to-SQL pipeline
  rag_agent.py    # RAG pipeline over reviews
  router.py       # SQL vs RAG classifier
  llm.py          # shared Groq client
outputs/
  figures/        # architecture diagram, EDA charts
  reports/        # key insights summary
app.py            # Streamlit entry point
```

## Notes

- SQL generation is restricted to `SELECT`-only queries — any other statement is rejected before execution.
- Embeddings run locally (no API cost); only the router, SQL generation, and answer synthesis call the Groq API.
