"""Composite scoring, buy-zone estimation and probabilistic return projections."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .config import FUTURES_META, INDEX_META, SECTOR_META
from .indicators import (atr, drawdown, macro_regime, momentum, price_percentile,
                         returns, sma, trend_score, vol_score)
from .commodities import commodity_profile, inventory_metrics

SCORE_BANDS = [
    (75, "积极配置", "#1f6f4a"),
    (60, "逢低布局", "#2e8b6f"),
    (45, "中性观望", "#8a8f98"),
    (30, "谨慎减持", "#c07a2c"),
    (0, "回避", "#a33b4a"),
]


def action_for(score: float) -> tuple[str, str]:
    if pd.isna(score):
        return "数据不足", "#8a8f98"
    for cut, label, color in SCORE_BANDS:
        if score >= cut:
            return label, color
    return "回避", "#a33b4a"


ACTION_COLORS = {"积极配置": "#1f6f4a", "逢低布局": "#2e8b6f", "左侧布局": "#2f7fa8",
                 "趋势持有": "#1f6f4a", "中性观望": "#8a8f98", "谨慎减持": "#c07a2c",
                 "回避": "#a33b4a", "数据不足": "#8a8f98"}


def quadrant_action(odds: float, timing: float) -> tuple[str, str]:
    """赔率（便宜程度）× 胜率（趋势强度）二维决策。"""
    if pd.isna(odds) or pd.isna(timing):
        return "数据不足", ACTION_COLORS["数据不足"]
    if odds >= 60 and timing >= 55:
        return "积极配置", ACTION_COLORS["积极配置"]
    if odds >= 60 and timing < 55:
        return "左侧布局", ACTION_COLORS["左侧布局"]
    if odds < 60 and timing >= 60:
        return "趋势持有", ACTION_COLORS["趋势持有"]
    if odds < 40 and timing < 40:
        return "回避", ACTION_COLORS["回避"]
    if odds < 40:
        return "谨慎减持", ACTION_COLORS["谨慎减持"]
    return "中性观望", ACTION_COLORS["中性观望"]


def _roll_pct(s: pd.Series, window: int, min_periods: int = 500) -> pd.Series:
    s = s.dropna()
    if len(s) < min_periods:
        return pd.Series(np.nan, index=s.index)
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: (x <= x[-1]).mean(), raw=True)


def momentum_percentile(price: pd.Series, years: int = 8) -> float:
    s = price.dropna()
    if len(s) < 400:
        return np.nan
    m = s.resample("ME").last()
    mom = m / m.shift(12) - 1
    mom = mom.dropna()
    mom = mom.iloc[-years * 12:]
    if len(mom) < 24:
        return np.nan
    return float((mom <= mom.iloc[-1]).mean())


def valuation_snapshot(pe: pd.Series, years: int = 8) -> dict:
    s = pe.dropna()
    if len(s) < 24:
        return {}
    try:
        gap = s.index.to_series().diff().dt.days.median()
    except Exception:
        gap = None
    n = min(len(s), int(years * 252) if gap is not None and gap < 10 else int(years * 12))
    w = s.iloc[-n:]
    cur = float(w.iloc[-1])
    return {
        "pe": cur,
        "pct": float((w <= cur).mean()),
        "median": float(w.median()),
        "q30": float(w.quantile(0.30)),
        "q70": float(w.quantile(0.70)),
        "min": float(w.min()),
        "max": float(w.max()),
    }


def _wavg(pairs: list[tuple[float, float]]) -> float:
    num = sum(v * w for v, w in pairs if not np.isnan(v))
    den = sum(w for v, w in pairs if not np.isnan(v))
    return num / den if den else 50.0


def equity_diagnosis(symbol: str, pe_name: str | None = None) -> dict:
    df = data.load_index(symbol)
    name = INDEX_META.get(symbol, (symbol, ""))[0]
    px = df["close"]
    val = {}
    if pe_name:
        pe_df = data.load_index_valuation(pe_name, "pe")
        if not pe_df.empty:
            pe_col = "滚动市盈率" if "滚动市盈率" in pe_df.columns else pe_df.columns[-1]
            val = valuation_snapshot(pe_df[pe_col])
    price_pct = price_percentile(px, 5)
    valuation_score = 100 * (1 - val["pct"]) if val else np.nan
    position_score = 100 * (1 - price_pct) if not np.isnan(price_pct) else np.nan
    ts = trend_score(px)
    mp = momentum_percentile(px)
    vs = vol_score(px)
    dd = float(drawdown(px).iloc[-1])
    dd_pct = float((drawdown(px).iloc[-756:] < dd).mean() * 100)

    if val:
        odds = valuation_score
        timing = _wavg([(ts, 0.55), (mp, 0.30), (vs, 0.15)])
        score = _wavg([(odds, 0.40), (timing, 0.60)])
        basis = "PE估值分位"
    else:
        odds = position_score
        timing = _wavg([(ts, 0.55), (mp, 0.30), (vs, 0.15)])
        score = _wavg([(odds, 0.35), (timing, 0.65)])
        basis = "价格分位（缺估值历史）"
    action, color = quadrant_action(odds, timing)
    return {
        "symbol": symbol, "name": name, "type": "宽基指数",
        "price": float(px.iloc[-1]), "score": float(score),
        "odds": float(odds), "timing": float(timing), "action": action, "color": color,
        "valuation": val, "price_pct": price_pct, "valuation_score": valuation_score,
        "position_score": position_score, "trend": ts, "momentum_pct": mp,
        "vol_score": vs, "drawdown": dd, "basis": basis,
        "ret_1m": float(px.iloc[-1] / px.iloc[-21] - 1),
        "ret_12m": float(px.iloc[-1] / px.iloc[-252] - 1),
    }


def sector_diagnosis(code: str) -> dict:
    df = data.load_sector(code)
    name = SECTOR_META.get(code, (code, ""))[0]
    px = df["close"]
    pct = price_percentile(px, 5)
    ts = trend_score(px)
    mp = momentum_percentile(px)
    vs = vol_score(px)
    odds = 100 * (1 - pct)
    timing = _wavg([(ts, 0.50), (mp, 0.35), (vs, 0.15)])
    score = _wavg([(odds, 0.35), (timing, 0.65)])
    action, color = quadrant_action(odds, timing)
    return {
        "symbol": code, "name": name, "type": "行业指数",
        "price": float(px.iloc[-1]), "score": float(score),
        "odds": float(odds), "timing": float(timing), "action": action, "color": color,
        "price_pct": pct, "trend": ts, "momentum_pct": mp, "vol_score": vs,
        "drawdown": float(drawdown(px).iloc[-1]),
        "ret_1m": float(px.iloc[-1] / px.iloc[-21] - 1),
        "ret_3m": float(px.iloc[-1] / px.iloc[-63] - 1),
        "ret_12m": float(px.iloc[-1] / px.iloc[-252] - 1),
        "basis": "价格分位（缺估值历史）",
    }


def commodity_diagnosis(symbol: str) -> dict:
    p = commodity_profile(symbol)
    if not p:
        return {}
    inv = inventory_metrics(symbol) if FUTURES_META[symbol][2] else {}
    price_score = 100 * (1 - p["price_pct_5y"])
    ts = trend_score(data.load_futures(symbol)["close"])
    mp = momentum_percentile(data.load_futures(symbol)["close"])
    vs = vol_score(data.load_futures(symbol)["close"])
    if inv:
        inv_score = 100 * (1 - inv["pct"])
        odds = _wavg([(price_score, 0.60), (inv_score, 0.40)])
    else:
        inv_score = np.nan
        odds = price_score
    timing = _wavg([(ts, 0.55), (mp, 0.35), (vs, 0.10)])
    score = _wavg([(odds, 0.45), (timing, 0.55)])
    action, color = quadrant_action(odds, timing)
    p.update({"score": float(score), "type": "商品", "price": p.get("last"),
              "price_pct": p.get("price_pct_5y"), "odds": float(odds),
              "timing": float(timing), "action": action, "color": color,
              "inventory_score": inv_score, "trend": ts, "momentum_pct": mp,
              "vol_score": vs, "basis": "价格分位+库存+趋势" if inv else "价格分位+趋势"})
    return p


def stock_diagnosis(code: str) -> dict:
    px_df = data.load_stock_price(code)
    if px_df.empty:
        latest = data.load_all_a_latest()
        if latest.empty or not (latest["code"] == code).any():
            return {}
        r = latest[latest["code"] == code].iloc[0]
        action, color = quadrant_action(50, 50)
        return {"symbol": code, "name": r.get("name", code), "type": "个股快照",
                "price": float(r["price"]), "score": 50.0, "odds": 50.0, "timing": 50.0,
                "action": action, "color": color, "valuation": {}, "valuation_score": np.nan,
                "quality": np.nan, "roe": np.nan, "gross_margin": np.nan, "debt_ratio": np.nan,
                "trend": np.nan, "momentum_pct": np.nan, "vol_score": np.nan,
                "drawdown": np.nan, "ret_1m": np.nan, "ret_12m": np.nan,
                "basis": "仅实时快照", "pe": r.get("pe"), "pb": r.get("pb")}
    px = px_df["close"]
    val_df = data.load_stock_value(code)
    val = {}
    if not val_df.empty:
        pe = val_df["pe_ttm"].dropna()
        val = valuation_snapshot(pe)
    fin = data.load_stock_fin(code)
    uni = data.load_universe()
    name = uni.loc[uni["code"] == code, "name"].iloc[0] if (uni["code"] == code).any() else code

    val_score = 100 * (1 - val["pct"]) if val else np.nan
    roe = fin.get("roe")
    gm = fin.get("gross_margin")
    debt = fin.get("debt_ratio")
    quality_parts = []
    if roe is not None:
        quality_parts.append(float(np.clip(roe / 25 * 100, 0, 100)))
    if gm is not None:
        quality_parts.append(float(np.clip(gm / 50 * 100, 0, 100)))
    quality = np.mean(quality_parts) if quality_parts else np.nan
    ts = trend_score(px)
    mp = momentum_percentile(px)
    vs = vol_score(px)
    odds = val_score
    timing = _wavg([(ts, 0.50), (quality, 0.20), (mp, 0.20), (vs, 0.10)])
    score = _wavg([(odds, 0.45), (timing, 0.55)])
    action, color = quadrant_action(odds, timing)
    return {
        "symbol": code, "name": name, "type": "个股", "price": float(px.iloc[-1]),
        "score": float(score), "odds": float(odds), "timing": float(timing),
        "action": action, "color": color,
        "valuation": val, "valuation_score": val_score,
        "quality": quality, "roe": roe, "gross_margin": gm, "debt_ratio": debt,
        "trend": ts, "momentum_pct": mp, "vol_score": vs,
        "drawdown": float(drawdown(px).iloc[-1]),
        "ret_1m": float(px.iloc[-1] / px.iloc[-21] - 1),
        "ret_12m": float(px.iloc[-1] / px.iloc[-252] - 1),
        "basis": "PE分位+质量+趋势",
    }


def buy_zone_from_pe(price: pd.Series, pe: pd.Series, low_q: float = 0.30,
                     high_q: float = 0.55) -> dict:
    """Map valuation quantiles to price levels via implied EPS."""
    px = price.dropna()
    s = pe.dropna()
    if len(s) < 24 or px.empty:
        return {}
    cur_pe = float(s.iloc[-1])
    cur_px = float(px.iloc[-1])
    if cur_pe <= 0:
        return {}
    eps = cur_px / cur_pe
    pe_lo = float(s.quantile(low_q))
    pe_hi = float(s.quantile(high_q))
    lo, hi = sorted([eps * pe_lo, eps * pe_hi])
    return {
        "eps": eps, "pe_lo": pe_lo, "pe_hi": pe_hi,
        "buy_low": lo, "buy_high": hi,
        "discount_to_high": hi / cur_px - 1,
        "cur_pe": cur_pe,
    }


@lru_cache(maxsize=1)
def erp_series() -> pd.DataFrame:
    """Equity risk premium: CSI300 earnings yield minus 10Y CGB yield."""
    pe_df = data.load_index_valuation("沪深300", "pe")
    if pe_df.empty:
        return pd.DataFrame()
    col = "滚动市盈率" if "滚动市盈率" in pe_df.columns else pe_df.columns[-1]
    pe = pe_df[col].dropna()
    y = data.bond_10y()
    if pe.empty or y.empty:
        return pd.DataFrame()
    y_m = y.reindex(pe.index, method="ffill") / 100.0
    df = pd.DataFrame({"ep": 1 / pe, "yield": y_m})
    df["erp"] = df["ep"] - df["yield"]
    df = df.dropna()
    df.index.name = "date"
    return df


def erp_snapshot(years: int = 8) -> dict:
    df = erp_series()
    if df.empty:
        return {}
    s = df["erp"].iloc[-years * 12:]
    cur = float(s.iloc[-1])
    return {"erp": cur, "pct": float((s <= cur).mean()), "median": float(s.median()),
            "q30": float(s.quantile(0.3)), "q70": float(s.quantile(0.7))}


@lru_cache(maxsize=1)
def current_regime() -> dict:
    reg = macro_regime(data.pmi(), data.cpi_yoy())
    if reg.empty:
        return {}
    last = reg.iloc[-1]
    return {"regime": str(last["regime"]), "pmi": float(last["pmi"]),
            "cpi": float(last["cpi"]), "date": str(reg.index[-1].date())}


def buy_zone_from_price(price: pd.Series, years: int = 5, low_q: float = 0.30,
                        high_q: float = 0.55) -> dict:
    px = price.dropna()
    if len(px) < 250:
        return {}
    n = min(len(px), int(years * 252))
    w = px.iloc[-n:]
    lo = float(w.quantile(low_q))
    hi = float(w.quantile(high_q))
    cur = float(px.iloc[-1])
    return {"buy_low": min(lo, hi), "buy_high": max(lo, hi),
            "discount_to_high": max(lo, hi) / cur - 1,
            "cur_pe": None, "eps": None, "pe_lo": None, "pe_hi": None}


def stop_level(px_df: pd.DataFrame, mult: float = 2.0) -> dict:
    a = atr(px_df, 14).iloc[-1]
    last = float(px_df["close"].iloc[-1])
    ma200 = sma(px_df["close"], 200).iloc[-1]
    stop = last - mult * a
    return {"atr": float(a), "stop": float(stop), "stop_pct": float(stop / last - 1),
            "ma200": float(ma200) if not pd.isna(ma200) else np.nan}


def monte_carlo(price: pd.Series, horizon: int = 252, n: int = 2000,
                block: int = 5, seed: int = 42) -> dict:
    r = returns(price).dropna().iloc[-756:]
    if len(r) < 120:
        return {}
    rng = np.random.default_rng(seed)
    vals = r.values
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(vals) - block, size=(n, nb))
    samples = np.empty((n, nb * block))
    for i in range(nb):
        idx = (starts[:, i, None] + np.arange(block)[None, :])
        samples[:, i * block:(i + 1) * block] = vals[idx]
    sims = np.cumprod(1 + samples[:, :horizon], axis=1)
    terminal = sims[:, -1] - 1
    running_max = np.maximum.accumulate(sims, axis=1)
    mdd = (sims / running_max - 1).min(axis=1)
    last = float(price.iloc[-1])
    return {
        "last": last,
        "p05": float(np.quantile(terminal, 0.05)),
        "p25": float(np.quantile(terminal, 0.25)),
        "p50": float(np.quantile(terminal, 0.50)),
        "p75": float(np.quantile(terminal, 0.75)),
        "p95": float(np.quantile(terminal, 0.95)),
        "prob_up": float((terminal > 0).mean()),
        "exp": float(terminal.mean()),
        "mdd_median": float(np.median(mdd)),
        "mdd_p95": float(np.quantile(mdd, 0.05)),
        "terminal": terminal.tolist(),
        "prices": {k: last * (1 + v) for k, v in {
            "p05": np.quantile(terminal, 0.05), "p25": np.quantile(terminal, 0.25),
            "p50": np.quantile(terminal, 0.50), "p75": np.quantile(terminal, 0.75),
            "p95": np.quantile(terminal, 0.95)}.items()},
    }


def conditional_forward(pe: pd.Series, price: pd.Series, horizon: int = 12,
                        bandwidth: float = 0.10) -> dict:
    """Forward return distribution conditional on being near today's valuation percentile."""
    s = pe.dropna()
    px = price.reindex(s.index).dropna()
    s = s.reindex(px.index)
    if len(s) < 60:
        return {}
    pct = s.rolling(96, min_periods=24).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    fwd = px.shift(-horizon) / px - 1
    df = pd.DataFrame({"pct": pct, "fwd": fwd}).dropna()
    if len(df) < 40:
        return {}
    cur = float(df["pct"].iloc[-1])
    sub = df[(df["pct"] - cur).abs() <= bandwidth]["fwd"]
    if len(sub) < 15:
        sub = df["fwd"]
    return {"current_pct": cur, "n": len(sub), "mean": float(sub.mean()),
            "median": float(sub.median()), "win": float((sub > 0).mean()),
            "p25": float(sub.quantile(0.25)), "p75": float(sub.quantile(0.75)),
            "horizon": horizon}


@lru_cache(maxsize=1)
def breadth() -> dict:
    codes = data.load_universe()["code"].tolist()
    above200, above60, total = 0, 0, 0
    rets_20 = []
    for code in codes:
        df = data.load_stock_price(code)
        if df.empty or "close" not in df.columns:
            continue
        px = df["close"].dropna()
        if len(px) < 220:
            continue
        total += 1
        if px.iloc[-1] > px.rolling(200).mean().iloc[-1]:
            above200 += 1
        if px.iloc[-1] > px.rolling(60).mean().iloc[-1]:
            above60 += 1
        rets_20.append(px.iloc[-1] / px.iloc[-21] - 1)
    return {"n": total, "above200": above200 / total if total else np.nan,
            "above60": above60 / total if total else np.nan,
            "median_ret_20": float(np.median(rets_20)) if rets_20 else np.nan}


@lru_cache(maxsize=1)
def market_temperature() -> dict:
    pe = data.load_index_valuation("沪深300", "pe")
    val_pct = np.nan
    if not pe.empty:
        col = "滚动市盈率" if "滚动市盈率" in pe.columns else pe.columns[-1]
        val_pct = valuation_snapshot(pe[col])["pct"]
    px = data.load_index("sh000300")["close"]
    mom = momentum_percentile(px)
    vs = vol_score(px)
    b = breadth()
    margin = data.margin_balance()
    margin_chg = margin.iloc[-1] / margin.iloc[-61] - 1 if len(margin) > 61 else np.nan
    margin_pct = float((margin.pct_change(60).dropna().iloc[-756:] <= margin_chg).mean()) if len(margin) > 120 else np.nan
    parts = {
        "估值（反向）": (1 - val_pct) * 100 if not np.isnan(val_pct) else np.nan,
        "趋势动量": mom * 100 if not np.isnan(mom) else np.nan,
        "市场宽度": b["above200"] * 100 if not np.isnan(b["above200"]) else np.nan,
        "低波动": vs,
        "杠杆情绪": margin_pct * 100 if not np.isnan(margin_pct) else np.nan,
    }
    weights = {"估值（反向）": 0.35, "趋势动量": 0.25, "市场宽度": 0.20,
               "低波动": 0.10, "杠杆情绪": 0.10}
    num = sum(parts[k] * weights[k] for k in parts if not np.isnan(parts[k]))
    den = sum(weights[k] for k in parts if not np.isnan(parts[k]))
    temp = num / den if den else np.nan
    return {"temperature": temp, "parts": parts, "valuation_pct": val_pct,
            "breadth": b, "margin_chg": margin_chg}


@lru_cache(maxsize=1)
def sector_rotation_backtest() -> dict:
    """Momentum rotation with a market trend filter.

    Rule: if CSI300 closes above its 200-day MA at month-end, hold the top-5
    industries by 12-1 momentum; otherwise stay in cash. Monthly rebalance,
    10bp one-way cost. The composite score is still returned for the quadrant.
    """
    from .config import SECTOR_META
    from .indicators import sma

    closes = {name: data.load_sector(code)["close"]
              for code, (name, _) in SECTOR_META.items()}
    px = pd.DataFrame(closes).dropna(how="all")
    m = px.resample("ME").last()
    rets = m.pct_change()
    fwd = rets.shift(-1)

    price_pct = pd.DataFrame({c: _roll_pct(m[c], 60, 24) for c in m.columns})
    trend = m / m.rolling(12).mean() - 1
    mom = m / m.shift(12) - 1
    vol = rets.rolling(6).std()

    parts = [-(price_pct - 0.5), trend, mom, -vol]
    zs = []
    for f in parts:
        mu = f.mean(axis=1)
        sd = f.std(axis=1).replace(0, np.nan)
        zs.append(f.sub(mu, axis=0).div(sd, axis=0))
    valid = sum(z.notna().astype(int) for z in zs) == len(zs)
    score = sum(zs) / len(zs)
    score = score.where(valid)

    hs = data.load_index("sh000300")["close"]
    risk_on = (hs > sma(hs, 200)).resample("ME").last()

    rows, w_prev = [], None
    dates = score.index[score.index >= "2018-01-01"]
    for dt in dates[:-1]:
        s_mom = mom.loc[dt].dropna()
        if len(s_mom) < 10:
            continue
        if not bool(risk_on.get(dt, False)):
            w = pd.Series(0.0, index=m.columns)
        else:
            picks = s_mom.nlargest(5).index
            w = pd.Series(0.0, index=m.columns)
            w[picks] = 1 / 5
        turnover = 1.0 if w_prev is None else (w - w_prev).abs().sum()
        gross = float((w * fwd.loc[dt]).sum())
        rows.append({"date": dt, "net": gross - 0.001 * turnover,
                     "turnover": turnover, "gross": gross})
        w_prev = w

    res = pd.DataFrame(rows).set_index("date")
    bench = fwd.mean(axis=1).reindex(res.index)
    res["bench"] = bench
    res["nav"] = (1 + res["net"]).cumprod()
    res["nav_bench"] = (1 + bench.fillna(0)).cumprod()
    return {"returns": res, "score": score, "momentum": mom, "risk_on": risk_on}


def all_signals() -> pd.DataFrame:
    rows = []
    for sym, (name, group) in INDEX_META.items():
        pe_name = {"sh000300": "沪深300", "sh000905": "中证500", "sh000016": "上证50"}.get(sym)
        d = equity_diagnosis(sym, pe_name)
        rows.append(d)
    for code, (name, _) in SECTOR_META.items():
        rows.append(sector_diagnosis(code))
    for sym, (name, cat, _) in FUTURES_META.items():
        d = commodity_diagnosis(sym)
        if d:
            rows.append(d)
    out = pd.DataFrame(rows)
    return out
