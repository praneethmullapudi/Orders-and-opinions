"""Load the raw Olist CSVs into a SQLite DB and export review text for embedding.

Run: python src/data_prep.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "olist.db"
REVIEWS_TXT_PATH = ROOT / "data" / "processed" / "reviews_for_embedding.csv"

TABLES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "product_category_translation": "product_category_name_translation.csv",
}

TIMESTAMP_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def load_csvs() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, filename in TABLES.items():
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        df = pd.read_csv(path)
        for col in TIMESTAMP_COLS.get(name, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        frames[name] = df
    return frames


def clean(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # products: fill missing category with "unknown", translate to English
    products = frames["products"].copy()
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    translation = frames["product_category_translation"]
    products = products.merge(translation, on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )
    frames["products"] = products

    # reviews: drop rows with no review_id, keep blank comment text as ""
    reviews = frames["order_reviews"].copy()
    reviews = reviews.dropna(subset=["review_id", "order_id"])
    reviews["review_comment_title"] = reviews["review_comment_title"].fillna("")
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("")
    frames["order_reviews"] = reviews

    return frames


def write_sqlite(frames: dict[str, pd.DataFrame]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        for name, df in frames.items():
            df.to_sql(name, conn, index=False)
        conn.commit()
    finally:
        conn.close()


def export_reviews_for_embedding(frames: dict[str, pd.DataFrame]) -> None:
    reviews = frames["order_reviews"]
    orders = frames["orders"][["order_id", "customer_id"]]
    items = frames["order_items"][["order_id", "product_id"]].drop_duplicates("order_id")
    products = frames["products"][["product_id", "product_category_name_english"]]

    merged = (
        reviews.merge(orders, on="order_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
    )
    merged["review_text"] = (merged["review_comment_title"] + " " + merged["review_comment_message"]).str.strip()
    merged = merged[merged["review_text"] != ""]

    out = merged[
        [
            "review_id",
            "order_id",
            "review_score",
            "review_text",
            "product_category_name_english",
        ]
    ].rename(columns={"product_category_name_english": "product_category"})

    REVIEWS_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEWS_TXT_PATH, index=False)
    print(f"Exported {len(out)} review texts to {REVIEWS_TXT_PATH}")


def main() -> None:
    frames = load_csvs()
    frames = clean(frames)
    write_sqlite(frames)
    print(f"Wrote {len(frames)} tables to {DB_PATH}")
    export_reviews_for_embedding(frames)


if __name__ == "__main__":
    main()
