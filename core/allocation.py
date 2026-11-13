"""Cross-asset return construction, risk parity and tactical allocation backtests."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .config import FUTURES_META

TRADING = 252


@lru_cache(maxsize=1)
def bond_total_return() -> pd.Series:
    """Synthetic 10Y government bond total-return index from yield changes.

    Monthly approximation: carry y/12 minus duration times yield change.
    """
    y = data.bond_10y().dropna()
    if y.empty:
        return pd.Series(dtype=float)
    m = y.resample("ME").last().dropna()
    dy = m.diff() / 100.0
    carry = m.shift(1) / 100.0 / 12.0
    ret = carry - 7.5 * dy
    return ret.dropna()


@lru_cache(maxsize=1)
def monthly_asset_returns() -> pd.DataFrame:
    eq = data.load_index("sh000300")["close"].resample("ME").last().pct_change()
    gold = data.load_futures("AU0")["close"].resample("ME").last().pct_change()

    comm_rets = []
    for sym in FUTURES_META:
        df = data.load_futures(sym)
        if len(df) > 300:
            comm_rets.append(df["close"].resample("ME").last().pct_change())
    comm = pd.concat(comm_rets, axis=1).mean(axis=1)

    bond = bond_total_return()
    df = pd.DataFrame({"股票": eq, "债券": bond, "黄金": gold, "商品": comm}).dropna()
    return df[df.index >= "2010-01-01"]


PORTFOLIO_ASSETS = ["沪深300", "中证500", "债券", "黄金", "商品"]


@lru_cache(maxsize=1)
def portfolio_monthly_returns() -> pd.DataFrame:
    base = monthly_asset_returns().rename(columns={"股票": "沪深300"})
    zz500 = data.load_index("sh000905")["close"].resample("ME").last().pct_change()
    df = pd.DataFrame({"沪深300": base["沪深300"], "中证500": zz500,
                       "债券": base["债券"], "黄金": base["黄金"], "商品": base["商品"]})
    df = df.dropna()
    return df[df.index >= "2010-01-01"]


@lru_cache(maxsize=1)
def bond_daily_return() -> pd.Series:
    y = data.bond_10y().dropna()
    if y.empty:
        return pd.Series(dtype=float)
    dy = y.diff() / 100.0
    carry = y.shift(1) / 100.0 / TRADING
    return (carry - 7.5 * dy).dropna()


@lru_cache(maxsize=1)
def portfolio_daily_returns() -> pd.DataFrame:
    eq300 = data.load_index("sh000300")["close"].pct_change()
    eq500 = data.load_index("sh000905")["close"].pct_change()
    gold = data.load_futures("AU0")["close"].pct_change()
    comms = []
    for sym in FUTURES_META:
        df = data.load_futures(sym)
        if len(df) > 300:
            comms.append(df["close"].pct_change())
    comm = pd.concat(comms, axis=1).mean(axis=1)
    df = pd.DataFrame({"沪深300": eq300, "中证500": eq500, "债券": bond_daily_return(),
                       "黄金": gold, "商品": comm})
    df = df.dropna(how="all")
    return df[df.index >= "2010-01-01"]


def portfolio_backtest(weights: dict[str, float], cost: float = 0.001) -> dict:
    rets = portfolio_monthly_returns()
    w = pd.Series(weights).reindex(rets.columns).fillna(0).astype(float)
    if w.sum() <= 0:
        w = pd.Series(1 / len(rets.columns), index=rets.columns)
    w = w / w.sum()
    wdf = pd.DataFrame([w.values] * len(rets), index=rets.index, columns=rets.columns)
    bt = _rebalance_backtest(wdf, rets, cost=cost)
    return {"returns": bt, "weights": w}


def risk_contribution(weights: dict[str, float]) -> pd.Series:
    rets = portfolio_monthly_returns()
    w = pd.Series(weights).reindex(rets.columns).fillna(0).astype(float)
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    cov = rets.cov().values * 12
    port_var = float(w.values @ cov @ w.values)
    if port_var <= 0:
        return pd.Series(0.0, index=w.index)
    rc = w.values * (cov @ w.values) / port_var
    return pd.Series(rc, index=w.index)


STRESS_WINDOWS = [
    ("2015 股灾", "2015-06-12", "2016-02-29"),
    ("2018 贸易摩擦", "2018-01-24", "2018-12-28"),
    ("2020 新冠冲击", "2020-01-23", "2020-03-23"),
    ("2022–24 成长回撤", "2021-12-13", "2024-02-05"),
    ("2026 调整", "2026-01-01", "2026-12-31"),
]


def stress_test(weights: dict[str, float]) -> pd.DataFrame:
    rets = portfolio_daily_returns()
    w = pd.Series(weights).reindex(rets.columns).fillna(0).astype(float)
    if w.sum() <= 0:
        w = pd.Series(1 / len(rets.columns), index=rets.columns)
    w = w / w.sum()
    rows = []
    for label, start, end in STRESS_WINDOWS:
        sub = rets.loc[start:end]
        if sub.empty:
            continue
        cum = (1 + sub).cumprod().iloc[-1] - 1
        rows.append({"区间": label, "组合": float((w * cum).sum()),
                     "60/40": float(0.6 * cum.get("沪深300", 0) + 0.4 * cum.get("债券", 0)),
                     "沪深300": float(cum.get("沪深300", np.nan))})
    return pd.DataFrame(rows)


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    n = len(cov)
    x0 = np.ones(n) / n
    target = 1.0 / n

    def obj(w):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 1e9
        rc = w * (cov @ w) / port_var
        return float(((rc - target) ** 2).sum() * 1e4)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(res.x, 0, 1)
    return w / w.sum() if w.sum() > 0 else x0


def _rebalance_backtest(weights: pd.DataFrame, returns: pd.DataFrame,
                        cost: float = 0.001) -> pd.DataFrame:
    idx = returns.index.intersection(weights.index)
    w = weights.reindex(idx)
    r = returns.reindex(idx)
    prev = None
    rows = []
    for dt in idx:
        wt = w.loc[dt].fillna(0)
        if prev is None:
            to = wt.abs().sum()
        else:
            to = (wt - prev).abs().sum()
        gross = float((wt * r.loc[dt]).sum())
        net = gross - cost * to
        rows.append({"date": dt, "gross": gross, "net": net, "turnover": to})
        prev = wt
    out = pd.DataFrame(rows).set_index("date")
    out["nav"] = (1 + out["net"]).cumprod()
    return out


@lru_cache(maxsize=1)
def strategy_backtests() -> dict[str, pd.DataFrame]:
    rets = monthly_asset_returns()
    assets = list(rets.columns)
    index = rets.index

    # 1) 60/40
    w6040 = pd.DataFrame(0.0, index=index, columns=assets)
    w6040["股票"] = 0.6
    w6040["债券"] = 0.4
    bt_6040 = _rebalance_backtest(w6040, rets, cost=0.001)

    # 2) ERC risk parity, trailing 36m covariance, min 24m
    rows = {}
    for i, dt in enumerate(index):
        hist = rets.iloc[max(0, i - 36):i]
        if len(hist) < 24:
            rows[dt] = np.ones(len(assets)) / len(assets)
            continue
        cov = hist.cov().values * 12
        rows[dt] = risk_parity_weights(cov)
    w_erc = pd.DataFrame.from_dict(rows, orient="index", columns=assets)
    bt_erc = _rebalance_backtest(w_erc, rets, cost=0.001)

    # 3) dual momentum: top 2 by 12-1 momentum, only if positive; else bonds
    w_mom = pd.DataFrame(0.0, index=index, columns=assets)
    for i, dt in enumerate(index):
        if i < 12:
            w_mom.loc[dt, "债券"] = 1.0
            continue
        mom = (1 + rets.iloc[i - 12:i - 1]).prod() - 1
        picks = mom.sort_values(ascending=False).head(2)
        picks = picks[picks > 0]
        if len(picks) == 0:
            w_mom.loc[dt, "债券"] = 1.0
        else:
            for a in picks.index:
                w_mom.loc[dt, a] = 1.0 / len(picks)
    bt_mom = _rebalance_backtest(w_mom, rets, cost=0.001)

    # 4) HENGCE: ERC base + momentum tilt + 10% vol target
    w_h = pd.DataFrame(0.0, index=index, columns=assets)
    for i, dt in enumerate(index):
        if i < 36:
            w_h.loc[dt] = w_erc.loc[dt] if dt in w_erc.index else np.ones(len(assets)) / len(assets)
            continue
        base = w_erc.loc[dt].values.astype(float)
        mom = ((1 + rets.iloc[i - 12:i - 1]).prod() - 1).values
        rank = pd.Series(mom).rank(pct=True).values
        tilt = np.where(mom > 0, 1 + 0.6 * (rank - 0.5) * 2, 0.0)
        w = base * tilt
        if w.sum() <= 0:
            w = base
        w = w / w.sum()
        w = np.minimum(w, 0.45)
        w = w / w.sum()
        vol = rets.iloc[i - 12:i].std().values * np.sqrt(12)
        port_vol = float(np.sqrt(w @ np.diag(vol ** 2) @ w))
        scale = min(1.0, 0.10 / port_vol) if port_vol > 0 else 1.0
        w_h.loc[dt] = w * scale
    bt_h = _rebalance_backtest(w_h, rets, cost=0.001)

    return {"60/40": bt_6040, "风险平价": bt_erc, "双动量": bt_mom,
            "衡策组合": bt_h, "_weights": w_h, "_erc_weights": w_erc}


def strategy_summary(backs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    backs = backs or strategy_backtests()
    from .indicators import metrics_from_returns
    rows = []
    for name, bt in backs.items():
        if name.startswith("_"):
            continue
        m = metrics_from_returns(bt["net"], periods=12)
        m["策略"] = name
        rows.append(m)
    cols = ["策略", "累计收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "卡玛比率", "月胜率", "样本月数"]
    return pd.DataFrame(rows)[cols]


def correlation_matrix() -> pd.DataFrame:
    return monthly_asset_returns().corr()


def regime_asset_performance() -> pd.DataFrame:
    from .indicators import macro_regime

    rets = monthly_asset_returns()
    reg = macro_regime(data.pmi(), data.cpi_yoy())
    if reg.empty:
        return pd.DataFrame()
    reg = reg[["regime"]]
    df = rets.join(reg, how="inner").dropna()
    if df.empty:
        return pd.DataFrame()
    # annualise average monthly return and hit rate per regime
    out = df.groupby("regime").agg({c: "mean" for c in rets.columns})
    out = (1 + out) ** 12 - 1
    counts = df.groupby("regime").size().rename("月数")
    return out.join(counts)
