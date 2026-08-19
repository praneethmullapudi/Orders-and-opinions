"""AI Data Analyst Assistant -- Streamlit chat app over Olist e-commerce data."""
import random
import sqlite3

import pandas as pd
import streamlit as st

from src import rag_agent, sql_agent
from src.llm import MODEL as GROQ_MODEL
from src.router import route

st.set_page_config(page_title="Orders & Opinions", page_icon="📊", layout="wide")

USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "📊"

SQL_EXAMPLES = [
    "What were our top 5 product categories by revenue?",
    "How many orders were delivered late vs on time?",
    "What's the average payment value by payment type?",
]
RAG_EXAMPLES = [
    "What do customers complain about most in the electronics category?",
    "What do customers say about delivery speed in their reviews?",
]

DATA_DICTIONARY = {
    "orders": "One row per order, with status and purchase/delivery timestamps.",
    "order_items": "One row per product within an order -- an order can have several.",
    "order_payments": "Payment method and value for each order (credit card, boleto, etc).",
    "order_reviews": "Customer star rating (1-5) and free-text comment after delivery.",
    "customers": "Customer location (city/state) linked to each order.",
    "products": "Product catalog: category, weight, and dimensions.",
    "sellers": "Seller location for each product listing.",
    "geolocation": "Zip-code-level latitude/longitude for customers and sellers.",
    "product_category_translation": "Maps Portuguese category names to English.",
}

SQL_GLOSSARY = [
    ("SELECT", "Chooses which columns to return."),
    ("FROM", "The table the query reads from."),
    ("JOIN ... ON", "Combines rows from two tables that share a matching key (e.g. product_id)."),
    ("WHERE", "Filters individual rows before any grouping happens."),
    ("GROUP BY", "Collapses rows into groups (e.g. one per category) so they can be aggregated."),
    ("SUM / COUNT / AVG", "Aggregate functions that compute one number per group."),
    ("ORDER BY", "Sorts the result rows, often paired with DESC for highest-first."),
    ("LIMIT", "Caps how many rows come back."),
]


@st.cache_data
def get_table_info() -> list[dict]:
    conn = sqlite3.connect(sql_agent.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]
        info = []
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [row[1] for row in cur.fetchall()]
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            info.append({"table": t, "columns": cols, "rows": count})
        return info
    finally:
        conn.close()


@st.cache_data
def get_table_sample(table: str) -> pd.DataFrame:
    conn = sqlite3.connect(sql_agent.DB_PATH)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10", conn)
    finally:
        conn.close()


@st.cache_data
def get_dataset_stats() -> dict:
    conn = sqlite3.connect(sql_agent.DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM orders")
        orders = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM order_reviews "
            "WHERE review_comment_message IS NOT NULL AND TRIM(review_comment_message) != ''"
        )
        reviews = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT product_category_name) FROM products")
        categories = cur.fetchone()[0]

        cur.execute(
            "SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, COUNT(*) "
            "FROM orders WHERE order_purchase_timestamp IS NOT NULL "
            "GROUP BY month ORDER BY month"
        )
        orders_by_month = cur.fetchall()[-7:]

        cur.execute(
            "SELECT strftime('%Y-%m', review_creation_date) AS month, COUNT(*) "
            "FROM order_reviews "
            "WHERE review_comment_message IS NOT NULL AND TRIM(review_comment_message) != '' "
            "GROUP BY month ORDER BY month"
        )
        reviews_by_month = cur.fetchall()[-7:]

        return {
            "orders": orders,
            "reviews": reviews,
            "categories": categories,
            "orders_by_month": orders_by_month,
            "reviews_by_month": reviews_by_month,
        }
    finally:
        conn.close()


def _sparkline_svg(values: list[int], spark_id: str, width: int = 220, height: int = 60) -> str:
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    step = width / (len(values) - 1)
    points = [
        (i * step, height - ((v - lo) / span) * (height - 8) - 4) for i, v in enumerate(values)
    ]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"0,{height} {path} {width},{height}"
    last_x, last_y = points[-1]
    return f"""<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" class="stat-spark">
        <polygon points="{area}" fill="url(#sparkFill-{spark_id})" />
        <polyline points="{path}" fill="none" stroke="#5eead4" stroke-width="2.5"
            stroke-linejoin="round" stroke-linecap="round" />
        <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="#5eead4" />
        <defs>
            <linearGradient id="sparkFill-{spark_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.45" />
                <stop offset="100%" stop-color="#14b8a6" stop-opacity="0" />
            </linearGradient>
        </defs>
    </svg>"""


def _delta_label(series: list[int]) -> tuple[str, bool] | None:
    if len(series) < 2 or series[-2] == 0:
        return None
    diff = series[-1] - series[-2]
    pct = diff / series[-2] * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:,} ({sign}{pct:.1f}%) vs prev. month", diff < 0


def stat_card(
    label: str,
    value: str,
    delta: tuple[str, bool] | None,
    series: list[int],
    spark_id: str,
) -> str:
    if delta:
        delta_text, is_negative = delta
        delta_class = "stat-delta stat-delta-negative" if is_negative else "stat-delta"
        delta_html = f'<div class="{delta_class}">{delta_text}</div>'
    else:
        delta_html = ""
    spark_html = _sparkline_svg(series, spark_id, height=40) if len(series) >= 2 else ""
    return (
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>{delta_html}{spark_html}</div>'
    )


# ponytail: Streamlit's theme (config.toml) is fixed at server start with no Python
# setter, so a real in-app light/dark switch has to override colors via injected CSS.
LIGHT_MODE_CSS = """
<style>
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
[data-testid="stAppScrollToBottomContainer"], .hero-bar {
    background-color: #ffffff !important;
    color: #111111 !important;
}
[data-testid="stSidebar"] {
    background-color: #f5f5f7 !important;
    color: #111111 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"],
h1, h2, h3, p, span, label, li, .stMarkdown, .stCaption {
    color: #111111 !important;
}
[data-testid="stExpander"], [data-testid="stChatInput"], [data-testid="stChatInput"] div,
[data-testid="stChatMessage"] {
    background-color: #f5f5f7 !important;
    border-color: #d1d5db !important;
}
[data-testid="stChatInputTextArea"] {
    color: #111111 !important;
}
[data-testid="stChatInputTextArea"]::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}
[data-testid="stBaseButton-secondary"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    border-color: #d1d5db !important;
}
/* ponytail: Vega sets the chart's own background as an inline style (config.toml's
   dark theme), which normal CSS can't beat -- only !important overrides an inline
   style, so it has to be targeted here specifically. Axis/label text comes from the
   same dark theme (light gray, for the black background above), so it needs the
   same override or it goes near-invisible on the new white background. */
[data-testid="stVegaLiteChart"] svg {
    background-color: #ffffff !important;
}
[data-testid="stVegaLiteChart"] text {
    fill: #111111 !important;
}
</style>
"""

# ponytail: sidebar has no fixed "pages" to route between (just example-question
# buttons + session controls), so the reference nav look is mapped onto those --
# green gradient pill panel, rounded logo badge, white pill for the standout action.
SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a7a5e 0%, #0d4436 45%, #05130f 85%, #000000 100%);
}
.sidebar-brand {
    width: 46px; height: 46px; border-radius: 50%;
    background: radial-gradient(circle at 35% 32%, #99f6e4, #14b8a6 55%, #0f766e 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; margin: 0 auto 1.1rem;
    box-shadow: 0 0 22px 4px rgba(20,184,166,0.4);
}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #eafff8 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { opacity: 0.65; }
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 20px !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.08) !important;
    border: none !important;
    border-radius: 999px !important;
    color: #f2fffb !important;
    justify-content: flex-start !important;
    padding: 0.55rem 1rem !important;
    transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.94) !important;
    color: #0d4436 !important;
    transform: translateY(-1px);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #ffffff !important;
    color: #0d4436 !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600;
    justify-content: flex-start !important;
    padding: 0.55rem 1rem !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.14) !important; }
</style>
"""

# ponytail: reference card has a floating tooltip bubble over one data point --
# skipped, the end-point dot + delta line carries the same info without per-point
# bubble placement math for every card.
STAT_CARD_CSS = """
<style>
.stat-card {
    background: linear-gradient(160deg, #142a26 0%, #0a1512 60%, #000000 100%);
    border: 1px solid rgba(94,234,212,0.18);
    border-radius: 22px;
    padding: 0.85rem 1.2rem 0.7rem;
    box-shadow: 0 0 30px rgba(20,184,166,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
    height: 100%;
    margin-bottom: 0.85rem;
}
.stat-label {
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #5eead4; opacity: 0.85; font-weight: 600; margin-bottom: 0.35rem;
}
.stat-value { font-size: 1.9rem; font-weight: 800; color: #f2fffb; line-height: 1.1; }
.stat-delta { font-size: 0.8rem; color: #5eead4; opacity: 0.8; margin-top: 0.3rem; }
.stat-delta-negative { color: #fb7185; }
.stat-spark { width: 100%; height: 40px; margin-top: 0.5rem; display: block; }
</style>
"""

# ponytail: glossy-hero look (glow orb, gradient title, hover-glow buttons) via CSS
# only -- no new deps, works with the existing teal theme accent (#14b8a6).
# ponytail: tried pinning this header via position:fixed/sticky so it stays visible
# while scrolling, but st.chat_input's container auto-scrolls to its bottom on every
# rerun -- even on the landing page, whenever content is taller than the viewport
# (near-guaranteed with 3 stat cards + example buttons). That forced scroll doesn't
# stop for a fixed header, it just slides the content up underneath it, permanently
# hiding the top of the page behind an opaque bar. No padding value fixes it since
# the scroll offset is a moving target. Plain in-flow header: no overlap, ever.
HERO_CSS = """
<style>
.hero-bar {
    display: flex; align-items: center; justify-content: flex-start;
    gap: 0.85rem; background-color: #000000;
    padding: 0.6rem 0 0.9rem 0;
}
.hero-orb {
    width: 50px; height: 50px; border-radius: 50%; flex-shrink: 0;
    background: radial-gradient(circle at 35% 32%, #99f6e4, #14b8a6 45%, #0f766e 80%);
    box-shadow: 0 0 24px 5px rgba(20,184,166,0.45), 0 0 50px 12px rgba(20,184,166,0.22);
    animation: hero-pulse 3.2s ease-in-out infinite;
}
@keyframes hero-pulse {
    0%, 100% { box-shadow: 0 0 24px 5px rgba(20,184,166,0.45), 0 0 50px 12px rgba(20,184,166,0.22); }
    50% { box-shadow: 0 0 32px 8px rgba(20,184,166,0.6), 0 0 62px 16px rgba(20,184,166,0.32); }
}
.hero-title {
    text-align: left; font-size: 1.7rem; font-weight: 800; margin: 0; line-height: 1.2;
    background: linear-gradient(90deg, #99f6e4, #14b8a6 55%, #0f766e);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-subtitle { text-align: left; opacity: 0.7; margin: 0.1rem 0 0; font-size: 0.85rem; }
[data-testid="stBaseButton-secondary"] {
    border-radius: 14px !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: #14b8a6 !important;
    box-shadow: 0 0 18px rgba(20,184,166,0.35);
    transform: translateY(-1px);
}
</style>
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.markdown('<div class="sidebar-brand">📊</div>', unsafe_allow_html=True)
    light_mode = st.toggle("☀️ Light mode", value=False)
    st.divider()

    st.subheader("💡 Try asking")
    with st.container(border=True):
        st.markdown("**📊 Numbers & trends**")
        st.caption("Routed to the SQL pipeline")
        for q in SQL_EXAMPLES:
            if st.button(q, use_container_width=True, key=f"sql_{q}", icon="📈"):
                st.session_state["pending_question"] = q

    with st.container(border=True):
        st.markdown("**💬 Customer sentiment**")
        st.caption("Routed to the RAG pipeline")
        for q in RAG_EXAMPLES:
            if st.button(q, use_container_width=True, key=f"rag_{q}", icon="💬"):
                st.session_state["pending_question"] = q

    st.divider()
    session_controls = st.container()

st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
st.markdown(HERO_CSS, unsafe_allow_html=True)
if light_mode:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

is_landing = len(st.session_state["messages"]) == 0

st.markdown(
    '<div class="hero-bar"><div class="hero-orb"></div><div>'
    '<div class="hero-title">Orders & Opinions</div>'
    '<div class="hero-subtitle">One chat box, two pipelines -- SQL crunches the numbers, '
    "RAG reads what customers actually said.</div>"
    "</div></div>",
    unsafe_allow_html=True,
)

stats = get_dataset_stats()
orders_series = [c for _, c in stats["orders_by_month"]]
reviews_series = [c for _, c in stats["reviews_by_month"]]

st.markdown(STAT_CARD_CSS, unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(
    stat_card("Orders", f"{stats['orders']:,}", _delta_label(orders_series), orders_series, "orders"),
    unsafe_allow_html=True,
)
c2.markdown(
    stat_card(
        "Reviews with text", f"{stats['reviews']:,}", _delta_label(reviews_series), reviews_series, "reviews"
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    stat_card("Product categories", str(stats["categories"]), None, [], "categories"),
    unsafe_allow_html=True,
)

# ponytail: st.chat_input keeps the whole main area pinned to the bottom on every
# rerun (even inside a fragment), so any expander/checkbox click there jumps the
# scroll. st.dialog renders as an overlay outside that scrolling container instead.
@st.dialog("🗂️ Explore the dataset & learn SQL", width="large")
def show_dataset_explorer() -> None:
    st.caption(
        "Every table behind the SQL pipeline, what it contains, and a sample of its rows -- "
        "plus a quick glossary for reading the SQL shown in \"How I got this\"."
    )
    for t in get_table_info():
        st.markdown(f"**{t['table']}** — {t['rows']:,} rows")
        st.caption(DATA_DICTIONARY.get(t["table"], ""))
        st.code(", ".join(t["columns"]), language=None)
        if st.checkbox("Preview 10 sample rows", key=f"preview_{t['table']}"):
            st.dataframe(get_table_sample(t["table"]), use_container_width=True)
        st.divider()

    st.markdown("**📘 New to SQL? Quick glossary**")
    for kw, desc in SQL_GLOSSARY:
        st.markdown(f"- **`{kw}`** — {desc}")


# ponytail: folded "Explore the dataset" into this row (instead of its own row below)
# so the landing page is short enough that Streamlit's forced scroll-to-bottom on
# st.chat_input doesn't push the hero off the top of the viewport -- see HERO_CSS.
if is_landing:
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        if st.button("Revenue & Trends", icon="📊", use_container_width=True):
            st.session_state["pending_question"] = SQL_EXAMPLES[0]
    with fc2:
        if st.button("Customer Sentiment", icon="💬", use_container_width=True):
            st.session_state["pending_question"] = RAG_EXAMPLES[0]
    with fc3:
        if st.button("Surprise Me", icon="🎲", use_container_width=True):
            st.session_state["pending_question"] = random.choice(SQL_EXAMPLES + RAG_EXAMPLES)
    with fc4:
        if st.button("🗂️ Explore the dataset", use_container_width=True):
            show_dataset_explorer()
else:
    if st.button("🗂️ Explore the dataset & learn SQL"):
        show_dataset_explorer()

st.divider()

def render_result_chart(df: pd.DataFrame | None) -> None:
    """Bar chart for 2-column category/number results. ponytail: 2-col heuristic, extend if wider result shapes need charts."""
    if df is None or df.empty or len(df.columns) != 2:
        return
    cat_col, val_col = df.columns
    if pd.api.types.is_numeric_dtype(df[val_col]) and not pd.api.types.is_numeric_dtype(df[cat_col]):
        st.bar_chart(df.set_index(cat_col)[val_col], color="#14b8a6")


for msg in st.session_state["messages"]:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        render_result_chart(msg.get("df"))
        if msg.get("detail"):
            with st.expander("How I got this"):
                st.write(msg["detail"])

question = st.chat_input("Ask a question about orders or reviews...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking..."):
            # ponytail: system boundary (external Groq API) -- catch broadly so a
            # transient failure shows an error instead of crashing mid-turn and
            # leaving the user's question stuck in history with no reply.
            try:
                path = route(question)
                df = None
                if path == "sql":
                    result = sql_agent.answer_question(question)
                    st.write(result["answer"])
                    df = result["result"]
                    render_result_chart(df)
                    answer, detail = result["answer"], f"**SQL used:**\n```sql\n{result['sql']}\n```"
                    with st.expander("How I got this"):
                        st.markdown(detail)
                        if df is not None:
                            st.dataframe(df)
                else:
                    result = rag_agent.answer_question(question)
                    st.write(result["answer"])
                    detail_lines = [
                        f"- (score {r['review_score']}/5, {r['product_category']}) {r['review_text']}"
                        for r in result["reviews"]
                    ]
                    answer = result["answer"]
                    detail = "**Reviews retrieved:**\n" + "\n".join(detail_lines)
                    with st.expander("How I got this"):
                        st.markdown(detail)
            except Exception as exc:
                answer = f"Something went wrong answering that: {exc}"
                detail = None
                df = None
                st.error(answer)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "detail": detail, "df": df}
    )

n_asked = len(st.session_state["messages"]) // 2
with session_controls:
    st.caption(f"🗨️ {n_asked} question{'s' if n_asked != 1 else ''} asked this session")
    if st.button(
        "🗑️ Clear conversation", use_container_width=True, disabled=n_asked == 0, type="primary"
    ):
        st.session_state["messages"] = []
        st.rerun()
    st.divider()
    st.caption(f"Powered by Groq ({GROQ_MODEL}) · SQLite + FAISS")
