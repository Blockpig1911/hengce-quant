"""Fetch quarterly financial indicators needed for RMW / CMA factors."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT = DATA / "stocks" / "fin_hist"
OUT.mkdir(parents=True, exist_ok=True)

COLS = {
    "日期": "date",
    "净资产收益率(%)": "roe",
    "总资产增长率(%)": "asset_growth",
    "销售毛利率(%)": "gross_margin",
    "资产负债率(%)": "debt_ratio",
    "净利润增长率(%)": "profit_growth",
}


def fetch(code: str):
    import akshare as ak
    path = OUT / f"{code}.csv"
    if path.exists():
        return code, 0
    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2017")
    if df.empty:
        return code, 0
    df = df.rename(columns=COLS)
    keep = ["date"] + [c for c in COLS.values() if c != "date" and c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return code, len(df)


def main():
    codes = pd.read_csv(DATA / "meta" / "universe.csv", dtype={"code": str})["code"].tolist()
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                code, n = fut.result()
            except Exception as exc:  # noqa: BLE001
                code, n = futures[fut], f"ERR {str(exc)[:40]}"
            if done % 50 == 0:
                print(f"{done}/{len(codes)} {time.time()-t0:.0f}s {code} {n}", flush=True)
    print("done", len(codes), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
