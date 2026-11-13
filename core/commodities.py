"""Commodity seasonality, inventory cycle and positioning scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .config import FUTURES_META
from .indicators import momentum, price_percentile, returns

MONTH_LABELS = ["1月", "2月", "3月", "4月", "5月", "6月",
                "7月", "8月", "9月", "10月", "11月", "12月"]


def seasonality(symbol: str, years: int = 10) -> pd.Series:
    df = data.load_futures(symbol)
    if df.empty:
        return pd.Series(dtype=float)
    px = df["close"].dropna()
    px = px[px.index >= px.index.max() - pd.Timedelta(days=365 * years)]
    m = px.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.Series(dtype=float)
    return m.groupby(m.index.month).mean()


def seasonality_matrix(years: int = 10) -> pd.DataFrame:
    cols = {}
    for sym, (name, cat, _) in FUTURES_META.items():
        s = seasonality(sym, years)
        if len(s) == 12:
            cols[name] = s
    df = pd.DataFrame(cols).T
    df.columns = MONTH_LABELS
    return df


def seasonality_hit_rate(symbol: str, month: int, years: int = 10) -> float:
    df = data.load_futures(symbol)
    if df.empty:
        return np.nan
    px = df["close"].dropna()
    px = px[px.index >= px.index.max() - pd.Timedelta(days=365 * years)]
    m = px.resample("ME").last().pct_change().dropna()
    vals = m[m.index.month == month]
    return float((vals > 0).mean()) if len(vals) else np.nan


def inventory_metrics(symbol: str) -> dict:
    inv = data.load_inventory(symbol)
    if inv.empty:
        return {}
    s = inv["inventory"].dropna()
    if len(s) < 10:
        return {}
    latest = float(s.iloc[-1])
    chg20 = latest - float(s.iloc[-21]) if len(s) > 21 else latest - float(s.iloc[0])
    pct = float((s.iloc[-1] >= s).mean())
    return {"latest": latest, "chg20": chg20, "pct": pct,
            "pct_chg20": chg20 / float(s.iloc[-21]) if len(s) > 21 and s.iloc[-21] else np.nan,
            "days": len(s)}


def commodity_profile(symbol: str) -> dict:
    df = data.load_futures(symbol)
    if df.empty:
        return {}
    name, cat, inv_name = FUTURES_META.get(symbol, (symbol, "", None))
    px = df["close"]
    out = {
        "symbol": symbol, "name": name, "category": cat,
        "last": float(px.iloc[-1]),
        "ret_1m": float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) > 21 else np.nan,
        "ret_3m": float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) > 63 else np.nan,
        "ret_12m": float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) > 252 else np.nan,
        "mom_12_1": momentum(px),
        "vol_60d": float(returns(px).iloc[-60:].std() * np.sqrt(252)),
        "price_pct_5y": price_percentile(px, 5),
        "drawdown": float(px.iloc[-1] / px.cummax().iloc[-1] - 1),
    }
    if inv_name:
        out.update(inventory_metrics(symbol))
    return out


def commodity_table() -> pd.DataFrame:
    rows = [commodity_profile(s) for s in FUTURES_META]
    return pd.DataFrame([r for r in rows if r])
