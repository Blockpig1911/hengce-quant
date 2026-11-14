"""Fetch one year of daily quotes for every A-share via iFinD batch API."""
from __future__ import annotations

import sys
import time
import tomllib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "all_a"
OUT.mkdir(parents=True, exist_ok=True)
BATCH = OUT / "batches"
BATCH.mkdir(exist_ok=True)

FIELDS = "open;high;low;close;volume;amount;turnoverRatio;pe_ttm;pb"


def to_ths(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return f"{code}.BJ"


def main():
    import iFinDPy as ths

    sec = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
    code = ths.THS_iFinDLogin(sec["THS_ACCOUNT"], sec["THS_PASSWORD"])
    if code not in (0, "0"):
        raise SystemExit(f"login failed: {code}")
    allsh = pd.read_csv(DATA / "meta" / "all_stocks.csv", dtype={"code": str})
    allsh["code"] = allsh["code"].str.zfill(6)
    codes = allsh["code"].tolist()
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    t0 = time.time()
    ok = 0
    for i in range(0, len(codes), 20):
        path = BATCH / f"batch_{i//20:03d}.parquet"
        if path.exists():
            ok += 1
            continue
        batch = codes[i:i + 20]
        try:
            r = ths.THS_HistoryQuotes(",".join(to_ths(c) for c in batch), FIELDS,
                                      "", start, end)
            if r.get("errorcode") != 0:
                raise RuntimeError(r.get("errmsg"))
            frames = []
            for tb in r.get("tables", []):
                table = tb.get("table") or {}
                dates = tb.get("time") or []
                if not dates or not table:
                    continue
                n = len(dates)
                d = {k: v[:n] for k, v in table.items() if isinstance(v, list)}
                d["date"] = dates
                d["code"] = tb["thscode"].split(".")[0]
                frames.append(pd.DataFrame(d))
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(path, index=False, compression="zstd")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            # split a failed batch into 10-code halves
            for j in (0, 10):
                sub = batch[j:j + 10]
                p2 = BATCH / f"batch_{i//20:03d}_{j}.parquet"
                if p2.exists():
                    continue
                try:
                    r = ths.THS_HistoryQuotes(",".join(to_ths(c) for c in sub), FIELDS,
                                              "", start, end)
                    frames = []
                    for tb in r.get("tables", []):
                        table = tb.get("table") or {}
                        dates = tb.get("time") or []
                        if not dates or not table:
                            continue
                        n = len(dates)
                        d = {k: v[:n] for k, v in table.items() if isinstance(v, list)}
                        d["date"] = dates
                        d["code"] = tb["thscode"].split(".")[0]
                        frames.append(pd.DataFrame(d))
                    if frames:
                        pd.concat(frames, ignore_index=True).to_parquet(p2, index=False, compression="zstd")
                except Exception:
                    pass
            print("batch fail", i, str(exc)[:60], flush=True)
        if (i // 20) % 20 == 0:
            print(f"{i}/{len(codes)} {time.time()-t0:.0f}s", flush=True)
    ths.THS_iFinDLogout()
    parts = [pd.read_parquet(p) for p in BATCH.glob("batch_*.parquet")]
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"])
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c not in ("code", "date"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df.to_parquet(OUT / "history.parquet", index=False, compression="zstd")
    print("saved", df.shape, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
