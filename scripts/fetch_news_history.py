"""Build a daily news-sentiment history from CCTV's evening news.

Run once (about 3 minutes): python scripts/fetch_news_history.py
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.news import score_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "news"
OUT.mkdir(parents=True, exist_ok=True)


def main(days: int = 35):
    import akshare as ak

    rows = []
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        try:
            df = ak.news_cctv(date=d.strftime("%Y%m%d"))
            if df.empty:
                continue
            text = " ".join((df["title"].fillna("") + " " + df["content"].fillna("")).tolist())
            raw, pos, neg = score_text(text)
            rows.append({"date": d, "新闻数": len(df), "情绪": raw,
                         "情绪分": 50 + 50 * raw, "正面词": pos, "负面词": neg})
            print(d, len(df), round(50 + 50 * raw, 1))
        except Exception as exc:  # noqa: BLE001
            print(d, "skip", str(exc)[:60])
        time.sleep(0.4)
    out = pd.DataFrame(rows).sort_values("date")
    out.to_csv(OUT / "cctv_daily.csv", index=False, encoding="utf-8-sig")
    print("saved", out.shape)


if __name__ == "__main__":
    main()
