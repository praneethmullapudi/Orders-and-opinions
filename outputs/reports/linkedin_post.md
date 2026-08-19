I built an AI analyst that answers plain-English questions over 100K e-commerce orders — and shows its work.

The problem: business stakeholders can't write SQL, and nobody has time to read thousands of customer reviews by hand. So I built a Streamlit app that routes each question to the right pipeline automatically.

Ask "What were our top 5 categories by revenue?" → it writes and runs the SQL, then explains the numbers.
Ask "What do customers complain about in electronics?" → it retrieves the most relevant reviews via semantic search and summarizes what they actually say, with citations.

An LLM router decides which path to take. No manual toggling between "SQL mode" and "search mode" — you just ask.

Stack: Groq (Llama) for generation, SQLite for structured queries, sentence-transformers + FAISS for retrieval, Streamlit for the UI. Every answer comes with a "how I got this" panel — the exact SQL run, or the review excerpts retrieved — so nothing is a black box.

A few things I learned building this:
- Routing structured vs. unstructured questions to different pipelines beats forcing one approach to do both jobs.
- Grounding answers in retrieved evidence (and showing that evidence) matters more for trust than a slicker UI.
- Restricting generated SQL to SELECT-only, with a keyword denylist, is a small guardrail that removes a whole class of risk.

Dataset: Olist Brazilian E-Commerce (Kaggle), ~100K orders across 9 tables plus free-text reviews.

#AI #LLM #DataScience #RAG #Python
