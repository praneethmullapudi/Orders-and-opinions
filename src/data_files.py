"""Ensures processed data files (too large for git) exist locally, fetching them from a
GitHub Release if missing -- used on Streamlit Cloud where the repo alone isn't enough."""
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/praneethmullapudi/Orders-and-opinions/releases/download/data-v1"
REQUIRED_FILES = ["olist.db", "reviews.faiss", "reviews_meta.pkl"]


def missing_files(data_dir: Path) -> list[str]:
    return [f for f in REQUIRED_FILES if not (data_dir / f).exists()]


def download_missing(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in missing_files(data_dir):
        urllib.request.urlretrieve(f"{RELEASE_URL}/{filename}", data_dir / filename)
