"""Pre-compute heavy analytics into data/derived so the app starts fast."""
from __future__ import annotations

import json
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import allocation, commodities, data, factors, signals
from core.config import DATA
from core.indicators import metrics_from_returns, valuation_forward_table

OUT = DATA / "derived"
OUT.mkdir(exist_ok=True)


def save(df: pd.DataFrame, name: str, index: bool = False):
    df.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig", index=index)
    print(f"  saved {name}.csv {df.shape}")


def main():
    pd.set_option("display.width", 120)

    print("[1/7] factor panel")
    panel = factors._score_panel(factors.build_panel())
    panel.to_parquet(OUT / "factor_panel.parquet", index=False, compression="zstd")
    print("  panel", panel.shape, panel["date"].min(), panel["date"].max())

    print("[2/7] IC")
    ic = factors.ic_series(panel)
    save(ic, "factor_ic", index=True)
    summary = factors.ic_summary(ic)
    save(summary.reset_index(drop=True), "factor_summary")
    mfr = factors.model_factor_returns(panel)
    save(mfr, "factor_model_returns", index=True)
    save(factors.factor_return_summary(mfr).reset_index(drop=True), "factor_model_summary")
    save(factors.ic_by_year(ic).rename_axis("year"), "factor_ic_yearly", index=True)
    save(factors.factor_correlation(panel), "factor_correlation", index=True)

    # IC decay for composite at 1/3/6/12 months
    decay = []
    base = panel.dropna(subset=["score"]).copy()
    for h in [1, 2, 3, 6, 12]:
        fwd = base.groupby("code")["close"].shift(-h) / base["close"] - 1
        tmp = base[["date", "score"]].copy()
        tmp["fwd"] = fwd
        tmp = tmp.dropna()
        vals = []
        for _, g in tmp.groupby("date"):
            if len(g) > 30:
                vals.append(g["score"].corr(g["fwd"], method="spearman"))
        decay.append({"horizon": h, "ic": float(np.nanmean(vals)) if vals else np.nan})
    save(pd.DataFrame(decay).reset_index(drop=True), "factor_decay")

    print("[3/7] factor backtests")
    ls = factors.backtest_factor(panel, long_only=False)
    lo = factors.backtest_factor(panel, long_only=True)
    lsr = ls["returns"]
    save(lsr, "factor_ls_returns", index=True)
    save(lo["returns"], "factor_lo_returns", index=True)
    q = factors.quantile_returns(panel)
    save(q, "factor_quantiles", index=True)
    save(factors.quantile_returns(panel, stat="median"), "factor_quantiles_median", index=True)

    rows = []
    for label, r in [("多空组合", lsr["net"]), ("多头组合", lo["returns"]["net"])]:
        m = metrics_from_returns(r, periods=12)
        m["组合"] = label
        rows.append(m)
    m = metrics_from_returns(lsr["net"] - lsr["bench"].fillna(0), periods=12)
    m["组合"] = "多空超额"
    rows.append(m)
    if "ew" in lsr.columns:
        m = metrics_from_returns(lsr["ew"], periods=12)
        m["组合"] = "股票池等权"
        rows.append(m)
    save(pd.DataFrame(rows).reset_index(drop=True), "factor_metrics")

    print("[4/7] allocation")
    backs = allocation.strategy_backtests()
    nav = pd.DataFrame({k: v["net"] for k, v in backs.items() if not k.startswith("_")})
    nav = (1 + nav).cumprod()
    save(nav, "strategy_nav", index=True)
    save(allocation.strategy_summary(backs).reset_index(drop=True), "strategy_summary")
    w = backs["_weights"].copy()
    w.index.name = "date"
    w = w.reset_index().melt(id_vars="date", var_name="asset", value_name="weight")
    save(w, "strategy_weights")
    corr = allocation.correlation_matrix()
    corr.index.name = "资产"
    save(corr, "asset_correlation", index=True)
    reg = allocation.regime_asset_performance()
    save(reg.reset_index(drop=True), "regime_performance")

    print("[5/7] commodity")
    ct = commodities.commodity_table()
    save(ct.reset_index(drop=True), "commodity_table")
    seas = commodities.seasonality_matrix()
    seas.index.name = "品种"
    save(seas, "seasonality", index=True)

    print("[6/7] financial snapshot")
    fin_rows = []
    uni = data.load_all_stocks()
    yj = pd.read_csv(DATA / "meta" / "yjbb_latest.csv", dtype={"股票代码": str})
    yj["股票代码"] = yj["股票代码"].str.zfill(6)
    yj = yj.set_index("股票代码")
    snap = data.load_all_a_latest().set_index("code") if not data.load_all_a_latest().empty else pd.DataFrame()
    for _, u in uni.iterrows():
        code = u["code"]
        f = data.load_stock_fin(code)
        h = data.load_stock_fin_hist(code)
        if not f and not h.empty:
            f = h.iloc[-1].to_dict()
        f = f or {}
        v = data.load_stock_value(code)
        last = v.iloc[-1] if not v.empty else {}
        pe = last.get("pe_ttm") if hasattr(last, "get") else np.nan
        pb = last.get("pb") if hasattr(last, "get") else np.nan
        if (pd.isna(pe) or pd.isna(pb)) and not snap.empty and code in snap.index:
            pe = snap.loc[code, "pe"] if pd.isna(pe) else pe
            pb = snap.loc[code, "pb"] if pd.isna(pb) else pb
        ag = float(h["asset_growth"].dropna().iloc[-1]) if not h.empty and "asset_growth" in h else np.nan
        y = yj.loc[code] if code in yj.index else None
        roe = f.get("roe") if pd.notna(f.get("roe")) else (y["净资产收益率"] if y is not None else np.nan)
        gm = f.get("gross_margin") if pd.notna(f.get("gross_margin")) else (y["销售毛利率"] if y is not None else np.nan)
        rg = f.get("rev_growth") if pd.notna(f.get("rev_growth")) else (y["营业总收入-同比增长"] if y is not None else np.nan)
        pg = f.get("profit_growth") if pd.notna(f.get("profit_growth")) else (y["净利润-同比增长"] if y is not None else np.nan)
        fin_rows.append({
            "code": code, "name": u["name"], "roe": roe,
            "gross_margin": gm, "net_margin": f.get("net_margin"),
            "debt_ratio": f.get("debt_ratio"), "rev_growth": rg,
            "profit_growth": pg, "asset_growth": ag,
            "industry": (y["所处行业"] if y is not None else ""),
            "pe": pe, "pb": pb,
            "mcap": last.get("mcap") if hasattr(last, "get") else np.nan,
        })
    save(pd.DataFrame(fin_rows), "fin_snapshot")

    print("[6/7] signals")
    sig = signals.all_signals()
    sig["score"] = sig["score"].round(1)
    save(sig.reset_index(drop=True), "signals")

    ctm = pd.read_csv(OUT / "commodity_table.csv")
    add = sig[sig["type"] == "商品"][["symbol", "score", "action", "odds", "timing"]]
    ctm = ctm.merge(add, on="symbol", how="left")
    save(ctm.reset_index(drop=True), "commodity_table")
    erp = signals.erp_series()
    if not erp.empty:
        save(erp, "erp", index=True)
    (OUT / "regime_current.json").write_text(
        json.dumps(signals.current_regime(), ensure_ascii=False, indent=2), encoding="utf-8")
    temp = signals.market_temperature()
    (OUT / "market_temp.json").write_text(
        json.dumps({k: (v if not isinstance(v, dict) else v) for k, v in temp.items()},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (OUT / "breadth.json").write_text(
        json.dumps(temp.get("breadth", {}), ensure_ascii=False, default=float), encoding="utf-8")

    print("[7/7] sector rotation + valuation")
    rot = signals.sector_rotation_backtest()
    save(rot["returns"], "sector_rotation_returns", index=True)
    save(rot["score"].reset_index(), "sector_scores")

    pe = data.load_index_valuation("沪深300", "pe")
    if not pe.empty:
        col = "滚动市盈率" if "滚动市盈率" in pe.columns else pe.columns[-1]
        px = data.load_index("sh000300")["close"].resample("ME").last()
        vf = valuation_forward_table(pe[col], px, horizon=12)
        save(vf.reset_index(drop=True), "valuation_forward")

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "factor_panel_rows": int(len(panel)),
        "stocks": int(panel["code"].nunique()),
        "start": str(panel["date"].min().date()),
        "end": str(panel["date"].max().date()),
        "market_temperature": float(temp.get("temperature", np.nan)),
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("done", meta)


if __name__ == "__main__":
    main()
