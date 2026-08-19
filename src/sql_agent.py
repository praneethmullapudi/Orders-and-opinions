"""Text-to-SQL pipeline: question -> SQL -> execute on SQLite -> plain-English answer."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.llm import chat

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "olist.db"

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA)\b", re.IGNORECASE
)

SQL_SYSTEM_PROMPT = """You are a SQLite expert. Given a database schema and a question, \
write a single read-only SQL query (SELECT only) that answers the question.

Rules:
- Output ONLY the SQL query, no explanation, no markdown code fences.
- Use only tables/columns from the schema below.
- Always add a LIMIT (max 100) unless the question asks for an aggregate/count.
- Timestamps are stored as SQLite TEXT in "YYYY-MM-DD HH:MM:SS" format.

Schema:
{schema}
"""

ANSWER_SYSTEM_PROMPT = """You are a data analyst. Given a user's question, the SQL query \
that was run, and the resulting rows, write a short, plain-English answer (2-4 sentences). \
Cite concrete numbers from the results. If the results are empty, say so plainly."""


def get_schema() -> str:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        lines = []
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cur.fetchall()]
            lines.append(f"{table}({', '.join(cols)})")
        return "\n".join(lines)
    finally:
        conn.close()


def generate_sql(question: str, schema: str) -> str:
    raw = chat(SQL_SYSTEM_PROMPT.format(schema=schema), question)
    # strip markdown fences if the model adds them anyway
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return sql


def run_sql(sql: str) -> pd.DataFrame:
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN_KEYWORDS.search(sql):
        raise ValueError("Query contains a disallowed keyword.")
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def answer_question(question: str) -> dict:
    schema = get_schema()
    sql = generate_sql(question, schema)
    try:
        df = run_sql(sql)
    except Exception as exc:
        return {"answer": f"I couldn't run that query: {exc}", "sql": sql, "result": None}

    result_preview = df.head(20).to_string(index=False)
    answer = chat(
        ANSWER_SYSTEM_PROMPT,
        f"Question: {question}\nSQL: {sql}\nResults:\n{result_preview}",
    )
    return {"answer": answer, "sql": sql, "result": df}


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # ponytail: Windows console defaults to cp1252

    out = answer_question("What were the top 5 product categories by revenue?")
    print(out["sql"])
    print(out["answer"])
