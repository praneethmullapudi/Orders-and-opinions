# Project Plan: AI Data Analyst Assistant

## 1. Overview

**Project title:** AI Data Analyst Assistant
**One-line description:** A Streamlit chat app that lets users ask natural-language questions over e-commerce order data and customer reviews, automatically routing each question to either a text-to-SQL pipeline or a RAG (retrieval-augmented generation) pipeline.

**Problem statement:** Business stakeholders can't write SQL or dig through thousands of reviews manually. This app lets them ask plain-English questions like *"What were our top 5 product categories by revenue?"* or *"What do customers complain about most in the electronics category?"* and get grounded, explainable answers.

**Why this project:** Demonstrates LLM application development (not just LLM usage), tool-use/function-calling patterns, and explainable AI design — all high-signal skills for AI/Data Science roles in 2026.

---

## 2. Dataset

- **Source:** [Olist Brazilian E-Commerce dataset (Kaggle)](https://kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Size:** ~100K orders across 9 relational CSV files
- **Key tables:** orders, order_items, products, customers, payments, order_reviews (with free-text review comments)
- **Structured layer:** loaded into SQLite for text-to-SQL querying
- **Unstructured layer:** review text embedded and stored in FAISS for RAG querying

---

## 3. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq API (Llama 3.3) | Free tier, fast inference, no local server needed (deployable to Streamlit Cloud) |
| Structured data | SQLite | Lightweight, zero-setup relational store |
| Embeddings | sentence-transformers (local) | Free, runs locally, no API cost |
| Vector store | FAISS | Free, fast, simple local vector search |
| UI | Streamlit | Matches existing tech stack, fast to build, portfolio-friendly |
| Env/secrets | python-dotenv | Keep API keys out of code |

---

## 4. Architecture

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
   (answer + "how I got this"
    expander showing SQL query
    or retrieved review snippets)
```

---

## 5. Methodology / Build Steps

1. **Data prep** (`src/data_prep.py`)
   - Load raw Olist CSVs from `data/raw/`
   - Clean, join, and load into a SQLite DB at `data/processed/olist.db`
   - Clean and export review text for embedding

2. **SQL pipeline** (`src/sql_agent.py`)
   - Pass DB schema + user question to Groq LLM
   - LLM generates SQL query
   - Execute query on SQLite, handle errors gracefully
   - Pass result back to LLM to format a plain-English answer

3. **RAG pipeline** (`src/rag_agent.py`)
   - Embed all review text with sentence-transformers at setup time, store in FAISS index
   - At query time: embed question → retrieve top-k similar reviews → pass to LLM as context → generate grounded answer with review citations

4. **Router** (`src/router.py`)
   - LLM-based classifier: given the question, decide SQL vs RAG (or route to both if ambiguous)

5. **Streamlit app** (`app.py`)
   - Chat interface
   - Displays answer + expandable "how I got this" section (SQL query used, or reviews retrieved)
   - Sidebar with example questions to guide users

6. **Documentation & packaging**
   - README (project overview, setup instructions, architecture diagram)
   - requirements.txt, .env.example, .gitignore
   - LinkedIn post
   - Key insights summary

---

## 6. Expected Outputs / Deliverables

- [ ] `src/data_prep.py` — data loading & SQLite build script
- [ ] `src/sql_agent.py` — text-to-SQL pipeline
- [ ] `src/rag_agent.py` — RAG pipeline over reviews
- [ ] `src/router.py` — question router
- [ ] `app.py` — Streamlit chat app
- [ ] `notebooks/analysis.ipynb` — exploratory data analysis on Olist dataset
- [ ] `README.md` — full project documentation
- [ ] `requirements.txt`, `.env.example`, `.gitignore`
- [ ] Architecture diagram (image in `outputs/figures/`)
- [ ] Key insights summary (`outputs/reports/`)
- [ ] LinkedIn post (ready to publish)

---

## 7. Folder Structure

```
ai-analyst-assistant/
│
├── data/
│   ├── raw/              # Original Olist CSVs (user-downloaded from Kaggle)
│   └── processed/        # olist.db (SQLite) + FAISS index
│
├── notebooks/
│   └── analysis.ipynb    # EDA on the dataset
│
├── src/
│   ├── data_prep.py      # Load CSVs → SQLite, prep review text
│   ├── sql_agent.py      # Text-to-SQL pipeline
│   ├── rag_agent.py      # RAG pipeline over reviews
│   └── router.py         # Question router (SQL vs RAG)
│
├── outputs/
│   ├── figures/           # Architecture diagram, EDA charts
│   └── reports/           # Key insights summary
│
├── app.py                 # Streamlit chat app (entry point)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 8. Status / Progress Tracker

| Step | Status |
|---|---|
| Project scaffolding | ✅ Done |
| Dataset download (user, from Kaggle) | ✅ Done |
| `data_prep.py` | ✅ Done |
| `sql_agent.py` | ✅ Done |
| `rag_agent.py` | ✅ Done |
| `router.py` | ✅ Done |
| `app.py` (Streamlit UI) | ✅ Done |
| EDA notebook | ✅ Done (executed, charts in `outputs/figures/`) |
| README | ✅ Done |
| Architecture diagram | ✅ Done (`outputs/figures/architecture_diagram.png`) |
| Key insights summary | ✅ Done (`outputs/reports/key_insights.md`) |
| LinkedIn post | ✅ Done (`outputs/reports/linkedin_post.md`) |

---

## 9. Notes for Claude (context for future sessions)

- User is an MSc student building a CV/LinkedIn portfolio (see reusable system prompt for full standards: README, code quality, storytelling, LinkedIn post format).
- Groq chosen over OpenAI/Claude API specifically to keep this project free — do not swap providers without asking.
- Deliverables must follow the user's standard folder structure and quality checklist (all charts titled/labeled, insights business-relevant, code clean and commented, no hardcoded paths).
- Next action once this file is placed in the project folder: build `src/data_prep.py` once the Olist CSVs are available in `data/raw/`.
