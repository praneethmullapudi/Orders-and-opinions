# Key Insights — Olist E-Commerce Analysis

Derived from `notebooks/analysis.ipynb` and live runs of the SQL/RAG pipelines.

## Revenue
- Top category by revenue: **beleza_saude** (health & beauty), R$1,258,681, followed by **relogios_presentes** (watches & gifts) and **cama_mesa_banho** (bed/bath/table).
- See `outputs/figures/revenue_by_category.png` for the top-10 breakdown.

## Delivery performance
- Average delivery time: **12.1 days** from purchase to customer receipt.
- Only **8.1%** of delivered orders arrived later than the estimated delivery date — most orders beat their estimate.
- See `outputs/figures/delivery_time_distribution.png` and `order_status_distribution.png`.

## Customer sentiment
- **42,687** reviews have usable free-text comments, feeding the RAG pipeline.
- Most-reviewed categories by text volume: bed/bath/table, health & beauty, sports & leisure, computer accessories.
- Review scores skew positive; see `outputs/figures/review_score_distribution.png` for the full 1–5 star breakdown.
- Electronics complaints cluster around two failure modes: items that don't work/connect as advertised, and wrong-item fulfillment (e.g. a different hard-drive model shipped than ordered).

## Pipeline validation
- Router correctly classifies numeric/aggregate questions as `sql` and opinion/sentiment questions as `rag`.
- SQL pipeline generates read-only SQL, executes it, and grounds its answer in the returned rows.
- RAG pipeline retrieves top-k relevant reviews via FAISS cosine similarity and cites them by number in its answer.
