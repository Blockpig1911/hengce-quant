"""Data access layer. All loaders are read-only and memoised."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import DATA, FUTURES_META, INDEX_META, SECTOR_META


def _read(path: Path, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    return df


def _stock_path(kind: str, code: str) -> Path | None:
    for ext in (".parquet", ".csv"):
        p = DATA / "stocks" / kind / f"{code}{ext}"
        if p.exists():
            return p
    return None


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@lru_cache(maxsize=512)
def load_index(symbol: str) -> pd.DataFrame:
    df = _read(DATA / "indices" / f"{symbol}.csv")
    return _numeric(df, ["open", "high", "low", "close", "volume", "amount"])


@lru_cache(maxsize=64)
def load_sector(code: str) -> pd.DataFrame:
    df = _read(DATA / "sectors" / f"{code}.csv")
    return _numeric(df, ["open", "high", "low", "close", "volume"])


@lru_cache(maxsize=128)
def load_futures(symbol: str) -> pd.DataFrame:
    df = _read(DATA / "futures" / f"{symbol}.csv")
    return _numeric(df, ["open", "high", "low", "close", "volume", "open_interest"])


@lru_cache(maxsize=512)
def load_stock_price(code: str) -> pd.DataFrame:
    path = _stock_path("price", code)
    if path is None:
        return pd.DataFrame()
    df = _read(path)
    return _numeric(df, ["open", "high", "low", "close", "volume", "amount", "turnover"])


@lru_cache(maxsize=512)
def load_stock_value(code: str) -> pd.DataFrame:
    path = _stock_path("value", code)
    if path is None:
        return pd.DataFrame()
    df = _read(path)
    return _numeric(df, ["close", "pct_chg", "mcap", "float_mcap", "pe_ttm", "pb", "ps"])


@lru_cache(maxsize=1)
def load_all_stocks() -> pd.DataFrame:
    path = DATA / "meta" / "all_stocks.csv"
    if not path.exists():
        return pd.DataFrame(columns=["code", "name"])
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df


@lru_cache(maxsize=800)
def load_stock_fin_hist(code: str) -> pd.DataFrame:
    path = DATA / "stocks" / "fin_hist" / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).set_index("date").sort_index()


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    path = DATA / "meta" / "universe.csv"
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df


@lru_cache(maxsize=512)
def load_stock_fin(code: str) -> dict:
    path = DATA / "stocks" / "fin" / f"{code}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def load_macro(name: str) -> pd.DataFrame:
    path = DATA / "macro" / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_candidates = ["日期", "信用交易日期", "月份", "date"]
    for c in date_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str).str.replace("年", "-").str.replace("月份", "")
                                   .str.replace("月", ""), errors="coerce")
            df = df.dropna(subset=[c]).set_index(c).sort_index()
            break
    return df


@lru_cache(maxsize=128)
def load_inventory(symbol: str) -> pd.DataFrame:
    path = DATA / "inventory" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read(path)
    return _numeric(df, ["inventory", "change"])


@lru_cache(maxsize=32)
def load_index_valuation(name: str, kind: str = "pe") -> pd.DataFrame:
    path = DATA / "indices" / f"{kind}_{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).set_index("日期").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------- panels
def close_panel(mapping: dict[str, str] | None = None, kind: str = "index") -> pd.DataFrame:
    """Return a wide close-price frame (columns = display names)."""
    mapping = mapping or {k: v[0] for k, v in INDEX_META.items()}
    out = {}
    for key, name in mapping.items():
        df = load_index(key) if kind == "index" else load_sector(key)
        if df.empty:
            continue
        out[name] = df["close"]
    panel = pd.DataFrame(out).sort_index()
    return panel


def sector_panel() -> pd.DataFrame:
    out = {}
    for code, (name, _) in SECTOR_META.items():
        df = load_sector(code)
        if not df.empty:
            out[name] = df["close"]
    return pd.DataFrame(out).sort_index()


def futures_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or list(FUTURES_META)
    out = {}
    for sym in symbols:
        df = load_futures(sym)
        if not df.empty:
            out[FUTURES_META[sym][0]] = df["close"]
    return pd.DataFrame(out).sort_index()


def stock_close_panel(codes: list[str]) -> pd.DataFrame:
    out = {}
    for code in codes:
        df = load_stock_price(code)
        if not df.empty and len(df) > 250:
            out[code] = df["close"]
    return pd.DataFrame(out).sort_index()


def stock_monthly() -> pd.DataFrame:
    """Month-end adjusted close for every stock, columns = codes."""
    codes = load_universe()["code"].tolist()
    panel = stock_close_panel(codes)
    return panel.resample("ME").last()


def macro_series(name: str, column: str) -> pd.Series:
    df = load_macro(name)
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def bond_10y() -> pd.Series:
    s = macro_series("bond_yield", "中国国债收益率10年")
    return s.dropna()


def bond_2y() -> pd.Series:
    return macro_series("bond_yield", "中国国债收益率2年").dropna()


def _macro_value(name: str, label: str) -> pd.Series:
    df = load_macro(name)
    if df.empty or "今值" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df["今值"], errors="coerce").dropna()
    s.name = label
    return s.sort_index()


def cpi_yoy() -> pd.Series:
    df = load_macro("cpi_yoy")
    if df.empty or "全国-同比增长" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df["全国-同比增长"], errors="coerce").dropna()
    s.name = "cpi"
    return s.sort_index()


def ppi_yoy() -> pd.Series:
    return _macro_value("ppi_yoy", "ppi")


def pmi() -> pd.Series:
    df = load_macro("pmi")
    if df.empty or "制造业-指数" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df["制造业-指数"], errors="coerce").dropna()
    s.name = "pmi"
    return s.sort_index()


def m2_yoy() -> pd.Series:
    return macro_series("m2", "货币和准货币(M2)-同比增长")


def margin_balance() -> pd.Series:
    df = load_macro("margin_sse")
    if df.empty:
        return pd.Series(dtype=float)
    col = "融资融券余额" if "融资融券余额" in df.columns else "融资余额"
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.sort_index()


def usdcny() -> pd.Series:
    df = load_macro("usdcny")
    if df.empty:
        return pd.Series(dtype=float)
    col = "央行中间价" if "央行中间价" in df.columns else df.columns[-1]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.sort_index()


def data_freshness() -> dict:
    meta_path = DATA / "meta" / "fetch_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
