"""Performance, valuation and technical indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns(price: pd.Series, log: bool = False) -> pd.Series:
    price = price.dropna()
    r = np.log(price / price.shift(1)) if log else price.pct_change()
    return r.dropna()


def drawdown(price: pd.Series) -> pd.Series:
    price = price.dropna()
    peak = price.cummax()
    return price / peak - 1.0


def max_drawdown(price: pd.Series) -> float:
    dd = drawdown(price)
    return float(dd.min()) if len(dd) else np.nan


def annualized_return(price: pd.Series, periods: int = TRADING_DAYS) -> float:
    price = price.dropna()
    if len(price) < 2:
        return np.nan
    total = price.iloc[-1] / price.iloc[0]
    years = len(price) / periods
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annualized_vol(price: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns(price)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(price: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns(price)
    if len(r) < 2:
        return np.nan
    excess = r - rf / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino(price: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns(price)
    if len(r) < 2:
        return np.nan
    excess = r - rf / periods
    downside = excess[excess < 0].std(ddof=1)
    if not downside:
        return np.nan
    return float(excess.mean() / downside * np.sqrt(periods))


def calmar(price: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(price)
    if not mdd or np.isnan(mdd):
        return np.nan
    return annualized_return(price, periods) / abs(mdd)


def var_cvar(r: pd.Series, q: float = 0.05) -> tuple[float, float]:
    r = r.dropna()
    if len(r) < 20:
        return np.nan, np.nan
    var = float(np.quantile(r, q))
    cvar = float(r[r <= var].mean())
    return var, cvar


def beta_alpha(asset: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> tuple[float, float]:
    df = pd.concat([returns(asset), returns(bench)], axis=1, join="inner").dropna()
    if len(df) < 30:
        return np.nan, np.nan
    x, y = df.iloc[:, 1], df.iloc[:, 0]
    var = x.var(ddof=1)
    beta = float(y.cov(x) / var) if var else np.nan
    alpha = float((y.mean() - beta * x.mean()) * periods)
    return beta, alpha


def perf_stats(price: pd.Series, periods: int = TRADING_DAYS, rf: float = 0.0) -> dict:
    price = price.dropna()
    if len(price) < 3:
        return {}
    r = returns(price)
    mdd = max_drawdown(price)
    var, cvar = var_cvar(r)
    ann = annualized_return(price, periods)
    vol = annualized_vol(price, periods)
    return {
        "累计收益": price.iloc[-1] / price.iloc[0] - 1,
        "年化收益": ann,
        "年化波动": vol,
        "夏普比率": sharpe(price, rf, periods),
        "索提诺比率": sortino(price, rf, periods),
        "最大回撤": mdd,
        "卡玛比率": calmar(price, periods),
        "日胜率": float((r > 0).mean()),
        "VaR95": var,
        "CVaR95": cvar,
        "样本天数": len(price),
    }


def metrics_from_returns(r: pd.Series, periods: int = 12, rf: float = 0.0) -> dict:
    """Metrics for a periodic (monthly) return series."""
    r = r.dropna()
    if len(r) < 3:
        return {}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (periods / len(nav)) - 1
    vol = r.std(ddof=1) * np.sqrt(periods)
    dd = nav / nav.cummax() - 1
    return {
        "累计收益": nav.iloc[-1] - 1,
        "年化收益": ann,
        "年化波动": vol,
        "夏普比率": ann / vol if vol else np.nan,
        "最大回撤": float(dd.min()),
        "卡玛比率": ann / abs(dd.min()) if dd.min() < 0 else np.nan,
        "月胜率": float((r > 0).mean()),
        "样本月数": len(r),
    }


# ---------------------------------------------------------------- valuation
def pct_rank(series: pd.Series, window: int | None = None, min_periods: int = 60) -> pd.Series:
    """Rolling percentile rank (0-1) of each observation within its trailing window."""
    s = series.dropna()
    if window is None:
        return s.expanding(min_periods=min_periods).apply(
            lambda x: (x <= x[-1]).mean(), raw=True)
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: (x <= x[-1]).mean(), raw=True)


def latest_percentile(series: pd.Series, window: int | None = None) -> float:
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    sub = s.iloc[-window:] if window else s
    if len(sub) < 8:
        return np.nan
    return float((sub <= sub.iloc[-1]).mean())


def valuation_percentile(pe: pd.Series, years: int = 8) -> pd.Series:
    """Percentile of valuation within a trailing window measured in years."""
    s = pe.dropna()
    n = max(int(years * 12), 24)
    return pct_rank(s, window=n, min_periods=24)


def forward_return(price: pd.Series, periods: int = 12) -> pd.Series:
    price = price.dropna()
    return price.shift(-periods) / price - 1


def valuation_forward_table(pe: pd.Series, price: pd.Series, horizon: int = 12,
                            years: int = 8, n_buckets: int = 5) -> pd.DataFrame:
    """Bucket current valuation percentile and realised forward returns."""
    pe = pe.dropna()
    px = price.reindex(pe.index).dropna()
    pe = pe.reindex(px.index)
    pct = valuation_percentile(pe, years=years)
    fwd = forward_return(px, horizon)
    df = pd.DataFrame({"pe_pct": pct, "fwd": fwd}).dropna()
    if len(df) < 30:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["pe_pct"], n_buckets, labels=False, duplicates="drop")
    out = df.groupby("bucket").agg(
        估值分位下限=("pe_pct", "min"),
        估值分位上限=("pe_pct", "max"),
        未来收益均值=("fwd", "mean"),
        未来收益中位数=("fwd", "median"),
        正收益概率=("fwd", lambda x: (x > 0).mean()),
        样本数=("fwd", "size"),
    ).reset_index()
    return out


def price_percentile(price: pd.Series, years: int = 5) -> float:
    s = price.dropna()
    n = int(years * 252)
    return latest_percentile(s, window=n if len(s) > n else None)


# ---------------------------------------------------------------- technical
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(2, n // 2)).mean()


def rsi(price: pd.Series, n: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def trend_score(price: pd.Series) -> float:
    """0-100 composite trend score from MA alignment and slope."""
    p = price.dropna()
    if len(p) < 220:
        return np.nan
    last = p.iloc[-1]
    ma20, ma60, ma200 = sma(p, 20).iloc[-1], sma(p, 60).iloc[-1], sma(p, 200).iloc[-1]
    score = 50.0
    score += 12 if last > ma20 else -12
    score += 12 if ma20 > ma60 else -12
    score += 14 if ma60 > ma200 else -14
    slope = p.iloc[-1] / p.iloc[-21] - 1
    score += float(np.clip(slope * 300, -15, 15))
    return float(np.clip(score, 0, 100))


def momentum(price: pd.Series, months: int = 12, skip: int = 1) -> float:
    p = price.dropna()
    if len(p) < 252 + 22:
        return np.nan
    n = int(months * 21)
    end = p.iloc[-1 - skip * 21] if skip else p.iloc[-1]
    start = p.iloc[-1 - n]
    return float(end / start - 1)


def vol_score(price: pd.Series, window: int = 60) -> float:
    """Lower recent volatility -> higher score (0-100) versus own 3y history."""
    r = returns(price)
    if len(r) < 260:
        return np.nan
    v = r.rolling(window).std() * np.sqrt(252)
    hist = v.dropna().iloc[-756:]
    cur = v.iloc[-1]
    if len(hist) < 60 or np.isnan(cur):
        return np.nan
    return float((hist > cur).mean() * 100)


def max_single_day(price: pd.Series, window: int = 20) -> float:
    r = returns(price).iloc[-window:]
    return float(r.max()) if len(r) else np.nan


def uncertainty_score(price: pd.Series) -> float:
    """0-100 where lower recent volatility/drawdown = higher certainty."""
    p = price.dropna()
    if len(p) < 260:
        return np.nan
    vol_pct = vol_score(p)
    dd = drawdown(p).iloc[-1]
    dd_hist = drawdown(p).iloc[-756:]
    dd_pct = float((dd_hist < dd).mean() * 100)
    return float(np.nanmean([vol_pct, dd_pct]))


# ---------------------------------------------------------------- macro regime
def macro_regime(pmi_series: pd.Series, cpi_series: pd.Series) -> pd.DataFrame:
    """Monthly growth × inflation regime.

    Growth: PMI >= 50 and 3-month change >= 0 -> expansion.
    Inflation: CPI YoY >= 2.5% or 3-month change > 0.3pp -> inflation.
    """
    if pmi_series.empty or cpi_series.empty:
        return pd.DataFrame()
    g = pmi_series.resample("ME").last().dropna()
    c = cpi_series.resample("ME").last().dropna()
    df = pd.DataFrame({"pmi": g, "cpi": c}).dropna()
    if df.empty:
        return pd.DataFrame()
    growth_up = (df["pmi"] >= 50) & (df["pmi"].diff(3) >= 0)
    infl_up = (df["cpi"] >= 2.5) | (df["cpi"].diff(3) > 0.3)
    mapping = {(True, False): "复苏", (True, True): "过热",
               (False, True): "滞胀", (False, False): "衰退"}
    df["growth"] = growth_up
    df["inflation"] = infl_up
    df["regime"] = [mapping[(a, b)] for a, b in zip(growth_up, infl_up)]
    return df


def regime_name(growth_up: bool, infl_up: bool) -> str:
    return {(True, False): "复苏", (True, True): "过热",
            (False, True): "滞胀", (False, False): "衰退"}[(growth_up, infl_up)]
