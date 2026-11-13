"""Cross-sectional factor construction, IC analysis and quantile backtests."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data

FACTORS = {
    "MOM": "动量（12-1月）",
    "REV": "短期反转（1月）",
    "LOWVOL": "低波动（60日）",
    "SIZE": "小市值（流通市值）",
    "VALUE": "价值（盈利收益率）",
    "LIQUID": "低换手（20日）",
    "TREND": "趋势（价格/MA200）",
    "LOTTERY": "低彩票偏好（20日最大单日涨幅）",
    "HML": "价值（账面市值比）",
    "RMW": "盈利（ROE）",
    "CMA": "低股本扩张（投资代理）",
    "PMO": "低异常换手",
    "HIGH52": "接近52周高点",
}
# sign = +1 means high raw value is good, -1 means low raw value is good
FACTOR_SIGN = {"MOM": 1, "REV": 1, "LOWVOL": 1, "SIZE": 1,
               "VALUE": 1, "LIQUID": 1, "TREND": 1, "LOTTERY": 1}


def _month_end(s: pd.Series) -> pd.Series:
    return s.resample("ME").last()


@lru_cache(maxsize=1)
def build_panel() -> pd.DataFrame:
    """Long panel: one row per (month-end, stock) with point-in-time factors."""
    uni = data.load_universe()
    rows = []
    for code in uni["code"]:
        px = data.load_stock_price(code)
        if px.empty or len(px) < 400:
            continue
        val = data.load_stock_value(code)
        close = px["close"].astype(float)
        r = close.pct_change()
        feat = pd.DataFrame({
            "close": close,
            "vol60": r.rolling(60).std() * np.sqrt(252),
            "turn20": px["turnover"].rolling(20).mean() if "turnover" in px else np.nan,
            "trend": close / close.rolling(200).mean() - 1,
            "maxret20": r.rolling(20).max(),
            "turn252": px["turnover"].rolling(252).mean() if "turnover" in px else np.nan,
            "high52": close / close.rolling(252).max(),
        })
        m = feat.resample("ME").last()
        m["ret_1m"] = m["close"].pct_change()
        m["ret_next"] = m["close"].shift(-1) / m["close"] - 1
        m["mom_12_1"] = m["close"].shift(1) / m["close"].shift(13) - 1
        fh = data.load_stock_fin_hist(code)
        if not fh.empty:
            av = fh.copy()
            if "asset_growth" not in av.columns:
                av["asset_growth"] = np.nan
            av = av[["roe", "asset_growth"]]
            av.index = av.index + pd.Timedelta(days=120)
            av = av.sort_index().reindex(m.index, method="ffill")
            m = m.join(av)
        else:
            m["roe"] = np.nan
            m["asset_growth"] = np.nan
        if not val.empty:
            vm = val[["pe_ttm", "pb", "float_mcap", "mcap"]].resample("ME").last()
            m = m.join(vm, how="left")
            m["shares"] = m["mcap"] / m["close"]
            m["share_growth"] = m["shares"].pct_change(12)
        else:
            m["pe_ttm"] = np.nan
            m["pb"] = np.nan
            m["float_mcap"] = np.nan
        m["code"] = code
        m.index.name = "date"
        rows.append(m.reset_index())
    panel = pd.concat(rows, ignore_index=True)
    panel["ep"] = np.where(panel["pe_ttm"] > 0, 1 / panel["pe_ttm"], np.nan)
    panel["size"] = np.log(panel["float_mcap"].where(panel["float_mcap"] > 0))
    panel["lowvol"] = -panel["vol60"]
    panel["rev"] = -panel["ret_1m"]
    panel["liquid"] = -panel["turn20"]
    panel["lottery"] = -panel["maxret20"]
    panel["value"] = panel["ep"]
    panel["size_z"] = -panel["size"]
    panel["bm"] = np.where(panel["pb"] > 0, 1 / panel["pb"], np.nan)
    panel["cma"] = -panel["asset_growth"].fillna(panel["share_growth"])
    panel["pmo"] = -(panel["turn20"] / panel["turn252"] - 1)
    return panel


def _winsorize(s: pd.Series, q: float = 0.02) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


RAW_COLS = {
    "MOM": "mom_12_1", "REV": "rev", "LOWVOL": "lowvol", "SIZE": "size_z",
    "VALUE": "value", "LIQUID": "liquid", "TREND": "trend", "LOTTERY": "lottery",
    "HML": "bm", "RMW": "roe", "CMA": "cma", "PMO": "pmo", "HIGH52": "high52",
}


def _score_panel(p: pd.DataFrame) -> pd.DataFrame:
    p = p.dropna(subset=["ret_next"])
    p = p[p["date"] >= "2018-12-01"].copy()
    zcols = []
    for name, raw in RAW_COLS.items():
        col = f"z_{name}"
        p[col] = p.groupby("date")[raw].transform(
            lambda x: ((_winsorize(x) - _winsorize(x).mean()) / _winsorize(x).std(ddof=0))
            if x.notna().sum() > 20 and _winsorize(x).std(ddof=0) > 0 else np.nan)
        zcols.append(col)
    p["score"] = p[zcols].mean(axis=1, skipna=True)
    p["n_factors"] = p[zcols].notna().sum(axis=1)
    p.loc[p["n_factors"] < 5, "score"] = np.nan
    return p


@lru_cache(maxsize=1)
def scored_panel() -> pd.DataFrame:
    cache = data.DATA / "derived" / "factor_panel.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return _score_panel(build_panel())


def ic_series(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly Spearman IC of each factor and the composite score."""
    from scipy import stats

    p = panel if panel is not None else scored_panel()
    rows = []
    zcols = [f"z_{k}" for k in FACTORS] + ["score"]
    for dt, g in p.groupby("date"):
        g = g.dropna(subset=["ret_next"])
        if len(g) < 30:
            continue
        row = {"date": dt}
        for c in zcols:
            sub = g[[c, "ret_next"]].dropna()
            row[c] = stats.spearmanr(sub[c], sub["ret_next"]).statistic if len(sub) > 20 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ic.columns:
        s = ic[c].dropna()
        if len(s) < 12:
            continue
        std = s.std(ddof=1)
        rows.append({
            "因子": c.replace("z_", "").replace("score", "综合"),
            "IC均值": s.mean(),
            "IC标准差": std,
            "ICIR": s.mean() / std if std else np.nan,
            "t值": s.mean() / std * np.sqrt(len(s)) if std else np.nan,
            "正IC占比": (s > 0).mean(),
            "样本月数": len(s),
        })
    return pd.DataFrame(rows).sort_values("IC均值", ascending=False)


def ic_by_year(ic: pd.DataFrame) -> pd.DataFrame:
    return ic.groupby(ic.index.year).mean()


def factor_correlation(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Average cross-sectional Spearman correlation between factor scores."""
    p = panel if panel is not None else scored_panel()
    zcols = [f"z_{k}" for k in FACTORS]
    mats = []
    for _, g in p.groupby("date"):
        if len(g) < 30:
            continue
        mats.append(g[zcols].corr(method="spearman"))
    if not mats:
        return pd.DataFrame()
    out = sum(mats) / len(mats)
    out.index = [c.replace("z_", "") for c in out.index]
    out.columns = [c.replace("z_", "") for c in out.columns]
    out.index.name = "因子"
    return out


def quantile_returns(panel: pd.DataFrame | None = None, n: int = 5,
                     col: str = "score", stat: str = "mean") -> pd.DataFrame:
    p = panel if panel is not None else scored_panel()
    p = p.dropna(subset=[col, "ret_next"])
    out = []
    for dt, g in p.groupby("date"):
        if len(g) < n * 10:
            continue
        g = g.copy()
        try:
            g["q"] = pd.qcut(g[col], n, labels=False, duplicates="drop")
        except ValueError:
            continue
        means = g.groupby("q")["ret_next"].agg(stat)
        means["date"] = dt
        out.append(means)
    df = pd.DataFrame(out).set_index("date").sort_index()
    df.columns = [f"Q{int(c)+1}" for c in df.columns]
    return df.dropna(how="all")


def backtest_factor(panel: pd.DataFrame | None = None, n_long: int = 30,
                    n_short: int = 30, cost: float = 0.002,
                    benchmark: str = "sh000300", long_only: bool = False) -> dict:
    """Monthly long-short (or long-only) backtest with transaction costs.

    cost is one-way cost per unit of turnover (0.002 = 20bp).
    """
    p = panel if panel is not None else scored_panel()
    p = p.dropna(subset=["score", "ret_next"])
    dates = sorted(p["date"].unique())
    prev_w = None
    rows = []
    for dt in dates:
        g = p[p["date"] == dt].sort_values("score", ascending=False)
        if len(g) < n_long + n_short:
            continue
        longs = g.head(n_long)["code"].tolist()
        shorts = g.tail(n_short)["code"].tolist() if not long_only else []
        rets = g.set_index("code")["ret_next"]
        w = pd.Series(0.0, index=rets.index)
        w[longs] = 1.0 / n_long
        if not long_only:
            w[shorts] = -1.0 / n_short
        if prev_w is None:
            turnover = w.abs().sum()
        else:
            common = w.index.union(prev_w.index)
            turnover = (w.reindex(common, fill_value=0) -
                        prev_w.reindex(common, fill_value=0)).abs().sum()
        gross = float((w * rets).sum())
        net = gross - cost * turnover
        rows.append({"date": dt, "gross": gross, "net": net, "turnover": turnover,
                     "n": len(g)})
        prev_w = w
    res = pd.DataFrame(rows).set_index("date")
    bench = data.load_index(benchmark)["close"].resample("ME").last().pct_change()
    bench = bench.reindex(res.index)
    ew = p.groupby("date")["ret_next"].mean().reindex(res.index)
    res["bench"] = bench
    res["ew"] = ew
    res["excess"] = res["net"] - bench.fillna(0)
    return {"returns": res, "n_long": n_long, "n_short": n_short, "cost": cost}


def composite_weights(ic: pd.DataFrame) -> pd.Series:
    """Average IC used as a transparent factor weight (full-sample, for reporting)."""
    s = ic.mean()
    s = s.drop(labels=["score"], errors="ignore")
    s = s / s.abs().sum()
    return s


MODEL_FACTORS = {
    "SMB": ("size", -1),
    "HML": ("bm", 1),
    "RMW": ("roe", 1),
    "CMA": ("cma", 1),
    "MOM": ("mom_12_1", 1),
    "PMO": ("pmo", 1),
}
MODEL_SETS = {
    "FF3": ["MKT", "SMB", "HML"],
    "FF5": ["MKT", "SMB", "HML", "RMW", "CMA"],
    "CH4": ["MKT", "SMB", "HML", "PMO"],
}
MODEL_LABELS = {
    "MKT": "市场", "SMB": "小市值", "HML": "价值(BM)", "RMW": "盈利(ROE)",
    "CMA": "低股本扩张", "MOM": "动量", "PMO": "低异常换手",
}


def model_factor_returns(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly long-short returns for the classic factor models."""
    p = panel if panel is not None else scored_panel()
    p = p.dropna(subset=["ret_next"])
    rows = []
    for dt, g in p.groupby("date"):
        if len(g) < 50:
            continue
        row = {"date": dt, "MKT": g["ret_next"].mean() - 0.02 / 12}
        for name, (col, sign) in MODEL_FACTORS.items():
            s = g[col]
            valid = s.notna()
            if valid.sum() < 40:
                row[name] = np.nan
                continue
            lo, hi = s[valid].quantile([0.3, 0.7])
            if sign > 0:
                long_m, short_m = s >= hi, s <= lo
            else:
                long_m, short_m = s <= lo, s >= hi
            row[name] = g.loc[long_m, "ret_next"].mean() - g.loc[short_m, "ret_next"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index()


def factor_return_summary(fr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in fr.columns:
        s = fr[c].dropna()
        if len(s) < 12:
            continue
        vol = s.std(ddof=1) * np.sqrt(12)
        ann = s.mean() * 12
        rows.append({"因子": c, "年化收益": ann, "年化波动": vol,
                     "夏普": ann / vol if vol else np.nan,
                     "t值": s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if s.std(ddof=1) else np.nan,
                     "月胜率": (s > 0).mean(), "样本月数": len(s)})
    return pd.DataFrame(rows)


def custom_backtest(weights: dict[str, float], panel: pd.DataFrame | None = None,
                    cost: float = 0.002) -> dict:
    p = panel if panel is not None else scored_panel()
    w = {k: float(v) for k, v in weights.items() if f"z_{k}" in p.columns and v}
    if not w:
        w = {k: 1.0 for k in FACTORS}
    total = sum(abs(v) for v in w.values())
    score = sum(p[f"z_{k}"] * (v / total) for k, v in w.items())
    q = p.copy()
    q["score"] = score
    q["n_factors"] = sum(p[f"z_{k}"].notna() for k in w)
    q.loc[q["n_factors"] < 5, "score"] = np.nan
    return backtest_factor(q, long_only=True, cost=cost)
