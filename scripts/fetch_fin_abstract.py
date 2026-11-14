"""Fetch quarterly ROE / growth / leverage via the faster financial abstract API."""
from __future__ import annotations

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

TARGET = {
    "净资产收益率(ROE)": "roe",
    "毛利率": "gross_margin",
    "资产负债率": "debt_ratio",
    "营业总收入增长率": "rev_growth",
    "归属母公司净利润增长率": "profit_growth",
}


def fetch(code: str):
    import akshare as ak

    path = OUT / f"{code}.csv"
    if path.exists() and path.stat().st_size > 100:
        return code, 0
    d = ak.stock_financial_abstract(symbol=code)
    if d.empty:
        return code, 0
    date_cols = [c for c in d.columns if str(c).isdigit() and len(str(c)) == 8]
    rows = []
    for label, name in TARGET.items():
        sub = d[d["指标"].astype(str) == label]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for c in date_cols:
            val = pd.to_numeric(row.get(c), errors="coerce")
            if pd.notna(val):
                rows.append({"date": c, "field": name, "value": val})
    if not rows:
        return code, 0
    long = pd.DataFrame(rows)
    wide = long.pivot(index="date", columns="field", values="value").reset_index()
    wide["date"] = pd.to_datetime(wide["date"], format="%Y%m%d")
    wide = wide.sort_values("date")
    wide.to_csv(path, index=False, encoding="utf-8-sig")
    return code, len(wide)


def main():
    codes = pd.read_csv(DATA / "meta" / "all_stocks.csv", dtype={"code": str})["code"].tolist()
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(fetch, c): c for c in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                code, n = fut.result()
            except Exception as exc:  # noqa: BLE001
                code, n = futures[fut], f"ERR {str(exc)[:30]}"
            if done % 100 == 0:
                print(f"{done}/{len(codes)} {time.time()-t0:.0f}s {code} {n}", flush=True)
    print("done", len(codes), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
