"""Self-check for src.data_files.missing_files -- run directly: python test_data_files.py"""
import tempfile
from pathlib import Path

from src.data_files import REQUIRED_FILES, missing_files


def demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        assert missing_files(data_dir) == REQUIRED_FILES

        for f in REQUIRED_FILES[:-1]:
            (data_dir / f).touch()
        assert missing_files(data_dir) == [REQUIRED_FILES[-1]]

        (data_dir / REQUIRED_FILES[-1]).touch()
        assert missing_files(data_dir) == []
    print("ok")


if __name__ == "__main__":
    demo()
