"""Convert stock CSV snapshots to compressed Parquet and drop the CSVs.

Run after fetch_data.py:  python scripts/compact_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def compact(folder: Path):
    total_before = total_after = 0
    for csv in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv)
        for c in df.columns:
            if c not in ("date", "code"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        out = csv.with_suffix(".parquet")
        df.to_parquet(out, index=False, compression="zstd")
        total_before += csv.stat().st_size
        total_after += out.stat().st_size
        csv.unlink()
    print(f"{folder.relative_to(ROOT)}: {total_before/1e6:.1f} MB -> {total_after/1e6:.1f} MB")


if __name__ == "__main__":
    compact(DATA / "stocks" / "price")
    compact(DATA / "stocks" / "value")
