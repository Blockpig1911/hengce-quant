"""Fetch a realtime snapshot for every A-share (price, PE, PB)."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "all_a" / "latest.csv"


def to_ths(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return f"{code}.BJ"


def main():
    import iFinDPy as ths

    sec = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8"))
    if ths.THS_iFinDLogin(sec["THS_ACCOUNT"], sec["THS_PASSWORD"]) not in (0, "0"):
        raise SystemExit("login failed")
    allsh = pd.read_csv(DATA / "meta" / "all_stocks.csv", dtype={"code": str})
    allsh["code"] = allsh["code"].str.zfill(6)
    rows = []
    codes = allsh["code"].tolist()
    for i in range(0, len(codes), 500):
        batch = codes[i:i + 500]
        r = ths.THS_RealtimeQuotes(",".join(to_ths(c) for c in batch),
                                   "latest;preClose;changeRatio;pe_ttm;pb", "")
        for tb in r.get("tables", []):
            t = tb.get("table") or {}
            rows.append({
                "code": tb["thscode"].split(".")[0],
                "price": (t.get("latest") or [None])[0],
                "pre_close": (t.get("preClose") or [None])[0],
                "pct_chg": ((t.get("changeRatio") or [None])[0] or 0) / 100,
                "pe": (t.get("pe_ttm") or [None])[0],
                "pb": (t.get("pb") or [None])[0],
                "time": (tb.get("time") or [""])[0],
            })
        print(i, len(rows), flush=True)
    ths.THS_iFinDLogout()
    df = pd.DataFrame(rows).drop_duplicates("code")
    df = df.merge(allsh, on="code", how="left")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved", df.shape, OUT)


if __name__ == "__main__":
    main()
