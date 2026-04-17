from __future__ import annotations

from pathlib import Path


def load_paths(txt_path: str | Path) -> list[str]:
    txt_path = Path(txt_path)
    with txt_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
