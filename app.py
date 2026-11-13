"""衡策 HENGCE — 跨资产量化决策平台 (Streamlit entry point)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import allocation, commodities, data, factors, live, news, signals
from core.config import (ASSET_COLORS, DATA, DISCLAIMER, FUTURES_META,
                         INDEX_META, PALETTE, PLOTLY_FONT, SECTOR_META)
from core.indicators import drawdown, metrics_from_returns, returns, sma

st.set_page_config(page_title="衡策 HENGCE · 跨资产量化平台", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root { --navy:#1f3a5f; --teal:#2f6f5e; --gold:#a8842c; --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; }
html, body, [class*="css"] { font-family: "Inter","Microsoft YaHei","PingFang SC",sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
section[data-testid="stSidebar"] { background:#f7f8fa; border-right:1px solid #e5e7eb; }
section[data-testid="stSidebar"] * { color:#374151; }
section[data-testid="stSidebar"] .stRadio label { padding:5px 2px; font-size:14px; }
div[data-testid="stMetric"] { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px;
  padding:14px 16px; box-shadow:none; overflow:visible; }
div[data-testid="stMetricLabel"] { color:#6b7280; font-size:12px; }
div[data-testid="stMetricValue"] { color:#111827; font-size:22px; font-weight:600; }
div[data-testid="stMetricDelta"] { font-size:12px; line-height:1.4; }
.hero { padding:0 0 18px; margin-bottom:20px; border-bottom:1px solid #e5e7eb; }
.hero h1 { margin:0; font-size:28px; letter-spacing:0; font-weight:600; color:#111827; }
.hero p { margin:8px 0 0; color:#6b7280; font-size:14px; }
.hero .tag { display:none; }
.card { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:16px 18px;
  box-shadow:none; height:100%; }
.card h4 { margin:0 0 6px; font-size:14px; color:#111827; font-weight:600; }
.card p { margin:0; color:#4b5563; font-size:13px; line-height:1.6; }
.pill { display:inline-block; padding:2px 8px; border-radius:4px; color:#fff; font-size:11px; }
.small { color:#6b7280; font-size:12px; }
.insight { background:#f9fafb; border-left:2px solid #1f3a5f; padding:12px 16px;
  border-radius:2px; font-size:13px; line-height:1.7; color:#374151; }
.insight b { color:#111827; }
div.stButton>button { border-radius:6px; border:1px solid #d1d5db; }
h3 { font-size:18px !important; font-weight:600 !important; margin-top:26px !important; }
h4 { font-size:15px !important; font-weight:600 !important; margin-top:20px !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ACTION_COLORS = signals.ACTION_COLORS
ASSET_ORDER = ["A股宽基", "行业指数", "商品", "个股"]


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def _load_derived(name: str, mtime: float) -> pd.DataFrame:
    p = DATA / "derived" / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ["date", "月份"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def load_derived(name: str) -> pd.DataFrame:
    p = DATA / "derived" / f"{name}.csv"
    return _load_derived(name, p.stat().st_mtime if p.exists() else 0)


@st.cache_data(show_spinner=False)
def _load_json(name: str, mtime: float) -> dict:
    p = DATA / "derived" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_json(name: str) -> dict:
    p = DATA / "derived" / name
    return _load_json(name, p.stat().st_mtime if p.exists() else 0)


@st.cache_data(show_spinner=False)
def get_signals() -> pd.DataFrame:
    df = load_derived("signals")
    if df.empty:
        df = signals.all_signals()
    return df


@st.cache_data(show_spinner=False)
def get_market_temp() -> dict:
    t = load_json("market_temp.json")
    return t if t else signals.market_temperature()


def fmt_pct(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:+.{digits}f}%"


def fmt_num(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{digits}f}"


def fig_base(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(family=PLOTLY_FONT, size=12.5, color="#243447"),
        title=dict(text=title, font=dict(size=15, color="#12304f")) if title else None,
        margin=dict(l=10, r=16, t=44 if title else 16, b=58),
        hoverlabel=dict(font_family=PLOTLY_FONT, font_size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#dfe6ed")
    fig.update_yaxes(gridcolor="#eef2f6", zeroline=False)
    return fig


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'


def hero(title: str, subtitle: str, tags: list[str] | None = None):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def card(title: str, body: str) -> str:
    return f'<div class="card"><h4>{title}</h4><p>{body}</p></div>'


def gauge(value: float, title: str, height: int = 260) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": " / 100", "font": {"size": 30, "color": "#12304f"}},
        title={"text": title, "font": {"size": 14, "color": "#6b7a8c"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#c9d3dc"},
               "bar": {"color": "#1f4e79", "thickness": 0.28},
               "bgcolor": "#ffffff", "borderwidth": 0,
               "steps": [{"range": [0, 30], "color": "#f2d7da"},
                         {"range": [30, 45], "color": "#f7e4d0"},
                         {"range": [45, 60], "color": "#eceff2"},
                         {"range": [60, 75], "color": "#d9ece4"},
                         {"range": [75, 100], "color": "#c5e2d5"}]}))
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=40, b=10),
                      font=dict(family=PLOTLY_FONT))
    return fig


def price_figure(df: pd.DataFrame, buy_zone: dict | None = None, stop: float | None = None,
                 title: str = "", height: int = 430, zone_label: str = "估值中枢区间") -> go.Figure:
    px = df["close"].dropna() if "close" in df else df.iloc[:, 0]
    plot = px[px.index >= px.index.max() - pd.Timedelta(days=365 * 3)]
    fig = go.Figure()
    if buy_zone and buy_zone.get("buy_low"):
        fig.add_hrect(y0=buy_zone["buy_low"], y1=buy_zone["buy_high"],
                      fillcolor="rgba(46,139,111,.13)", line_width=0,
                      annotation_text=zone_label, annotation_position="top left",
                      annotation_font=dict(size=11, color="#2e8b6f"))
    fig.add_trace(go.Scatter(x=plot.index, y=plot, name="收盘价",
                             line=dict(color="#12304f", width=1.8)))
    for n, color in [(20, "#c9a227"), (60, "#2f7fa8"), (200, "#b35c2e")]:
        ma = sma(px, n).reindex(plot.index)
        fig.add_trace(go.Scatter(x=plot.index, y=ma, name=f"MA{n}",
                                 line=dict(color=color, width=1)))
    if stop:
        fig.add_hline(y=stop, line_dash="dot", line_color="#a33b4a",
                      annotation_text=f"止损 {stop:,.2f}", annotation_font_color="#a33b4a")
    fig.update_layout(hovermode="x unified")
    return fig_base(fig, height=height, title=title)


def drawdown_fig(nav: pd.Series, name: str, height: int = 220) -> go.Figure:
    dd = nav / nav.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name=name,
                               line=dict(color="#a33b4a", width=1)))
    fig.update_yaxes(tickformat=".0%")
    return fig_base(fig, height=height)


def table_style(df: pd.DataFrame, pct_cols: list[str] | None = None,
                num_cols: list[str] | None = None):
    pct_cols = pct_cols or []
    num_cols = num_cols or []
    fmt = {c: "{:+.1%}" for c in pct_cols if c in df.columns}
    fmt.update({c: "{:,.2f}" for c in num_cols if c in df.columns})
    return df.style.format(fmt, na_rep="—")


def explain(text: str):
    with st.expander("这页在做什么？", expanded=False):
        st.write(text)


def action_badge(score: float, label: str | None = None, color: str | None = None) -> str:
    if label is None:
        label, _ = signals.action_for(score)
    return pill(label, color or ACTION_COLORS.get(label, "#8a8f98"))


# ------------------------------------------------------------------ page 0
def page_overview():
    temp = get_market_temp()
    sig = get_signals()
    b = temp.get("breadth", {})
    pe_pct = temp.get("valuation_pct", np.nan)
    temp_v = temp.get("temperature", np.nan)
    hs300 = data.load_index("sh000300")["close"]
    ret20 = hs300.iloc[-1] / hs300.iloc[-21] - 1
    bond = data.bond_10y()
    erp = (1 / 13.0) - bond.iloc[-1] / 100 if not bond.empty else np.nan

    hero("A股总览 · 衡策", "先看市场贵不贵、趋势强不强、新闻偏暖还是偏冷，再决定要不要出手。",
         ["A股 · 行业 · 因子", "沪深800深度覆盖 + 全A搜索",
          "实时盯盘 + 模拟盘", "公开数据 · 可复现"])
    explain("这一页是“出门前看天气”：沪深300的市盈率处在历史什么位置，指数在均线之上还是之下，"
            "多少股票还在200日线上，新闻整体偏正面还是偏负面。它不告诉你买哪只，只告诉你现在适合重仓还是谨慎。")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("市场温度", f"{temp_v:.0f} / 100" if not np.isnan(temp_v) else "—",
              "偏冷" if temp_v < 45 else "偏热")
    c2.metric("沪深300 PE分位", f"{pe_pct*100:.0f}%" if not np.isnan(pe_pct) else "—",
              "近8年历史分位")
    c3.metric("沪深300 20日", fmt_pct(ret20), "短期动量")
    c4.metric("站上200日线", f"{b.get('above200', np.nan)*100:.0f}%",
              f"{b.get('n', 0)} 只成分股")
    c5.metric("10年期国债", f"{bond.iloc[-1]:.2f}%" if not bond.empty else "—", "无风险利率锚")

    col1, col2 = st.columns([2.15, 1])
    with col1:
        names = ["上证指数", "沪深300", "中证500", "创业板指", "科创50", "中证1000"]
        series = []
        for n in names:
            sym = {"上证指数": "sh000001", "沪深300": "sh000300", "中证500": "sh000905",
                   "创业板指": "sz399006", "科创50": "sh000688", "中证1000": "sh000852"}[n]
            series.append(data.load_index(sym)["close"].rename(n))
        panel = pd.concat(series, axis=1).sort_index()
        panel = panel[panel.index >= panel.index.max() - pd.Timedelta(days=365)]
        norm = panel / panel.iloc[0] * 100
        fig = go.Figure()
        for i, n in enumerate(norm.columns):
            fig.add_trace(go.Scatter(x=norm.index, y=norm[n], name=n,
                                     line=dict(width=1.9, color=PALETTE[i])))
        fig.update_yaxes(title="净值（起点=100）")
        fig = fig_base(fig, height=360, title="近一年 A 股主要指数")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.plotly_chart(gauge(temp_v, "市场温度计", 300), width="stretch")

    st.markdown("#### 温度计拆解")
    parts = temp.get("parts", {})
    if parts:
        pv = pd.DataFrame({"维度": list(parts), "得分": list(parts.values())}).dropna()
        fig = px.bar(pv, x="得分", y="维度", orientation="h", range_x=[0, 100],
                     color="得分", color_continuous_scale=["#a33b4a", "#e9edf1", "#2e8b6f"])
        fig.update_layout(coloraxis_showscale=False, height=260)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_base(fig, 260), width="stretch")

    st.markdown("#### 股债性价比与宏观状态")
    left, right = st.columns([2.1, 1])
    with left:
        erp = load_derived("erp")
        if not erp.empty:
            erp = erp.set_index("date")
            erp = erp[erp.index >= erp.index.max() - pd.Timedelta(days=365 * 10)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=erp.index, y=erp["erp"] * 100, name="ERP",
                                     line=dict(color="#1f4e79", width=2)))
            mu, sd = erp["erp"].mean() * 100, erp["erp"].std() * 100
            fig.add_hline(y=mu, line_dash="dash", line_color="#8a8f98",
                          annotation_text=f"均值 {mu:.1f}%", annotation_font_color="#8a8f98")
            fig.add_hrect(y0=mu - sd, y1=mu + sd, fillcolor="rgba(31,78,121,.06)", line_width=0)
            fig.update_yaxes(title="盈利收益率 − 10年国债（%）")
            st.plotly_chart(fig_base(fig, 320, "股债性价比：越高表示股票相对债券越便宜"), width="stretch")
    with right:
        reg = load_json("regime_current.json")
        snap = signals.erp_snapshot()
        desc = {"复苏": "增长扩张、通胀温和，股票与商品占优",
                "过热": "增长强、通胀升，商品与黄金占优，债券承压",
                "滞胀": "增长弱、通胀高，黄金与现金占优",
                "衰退": "增长与通胀双弱，债券占优，股票等待盈利底"}
        label = reg.get("regime", "—")
        try:
            _ns = news.market_snapshot(load_news())
            news_line = (f"<p>新闻情绪 <b>{_ns.get('score', float('nan')):.0f} / 100</b>，"
                         f"正面 {_ns.get('pos', 0)} 条 / 负面 {_ns.get('neg', 0)} 条</p>")
        except Exception:
            news_line = ""
        st.markdown(f"""
        <div class="card">
          <h4>当前宏观状态</h4>
          <p style="font-size:22px;color:#12304f;margin:4px 0"><b>{label}</b></p>
          <p>{desc.get(label, '')}</p>
          <hr style="border:none;border-top:1px solid #eef2f6;margin:10px 0">
          <p>制造业 PMI <b>{reg.get('pmi', float('nan')):.1f}</b> · CPI 同比 <b>{reg.get('cpi', float('nan')):.1f}%</b></p>
          <p>股债性价比 ERP <b>{snap.get('erp', float('nan'))*100:.1f}%</b>，
          处于近8年 <b>{snap.get('pct', float('nan'))*100:.0f}%</b> 分位</p>
          {news_line}
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 今日机会与风险")
    sig_a = sig[sig["type"].isin(["宽基指数", "行业指数"])] if not sig.empty else sig
    if not sig_a.empty:
        top = sig_a.sort_values("score", ascending=False).head(3)
        bottom = sig_a.sort_values("score").head(3)
        cols = st.columns(6)
        for i, (_, r) in enumerate(top.iterrows()):
            with cols[i]:
                st.markdown(card(f"{r['name']}", f"{action_badge(r['score'], r.get('action'), r.get('color'))}<br><span class='small'>综合评分 {r['score']:.0f}</span><br><span class='small'>{r['type']}</span>"),
                            unsafe_allow_html=True)
        for j, (_, r) in enumerate(bottom.iterrows()):
            with cols[3 + j]:
                st.markdown(card(f"{r['name']}", f"{action_badge(r['score'], r.get('action'), r.get('color'))}<br><span class='small'>综合评分 {r['score']:.0f}</span><br><span class='small'>{r['type']}</span>"),
                            unsafe_allow_html=True)
        st.markdown(
            f"<div class='insight'>综合评分最高的资产是 <b>{top.iloc[0]['name']}</b>（{top.iloc[0]['score']:.0f}分，"
            f"{top.iloc[0]['action']}），最低的是 <b>{bottom.iloc[0]['name']}</b>（{bottom.iloc[0]['score']:.0f}分，"
            f"{bottom.iloc[0]['action']}）。评分不是收益预测，而是“赔率+胜率+趋势”的相对排序；"
            f"建议结合下方资产诊断中的买入区间与止损位使用。</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 宽基指数估值一览")
    rows = []
    for sym, pe_name in [("sh000300", "沪深300"), ("sh000905", "中证500"), ("sh000016", "上证50")]:
        pe = data.load_index_valuation(pe_name, "pe")
        if pe.empty:
            continue
        col = "滚动市盈率" if "滚动市盈率" in pe.columns else pe.columns[-1]
        s = pe[col].dropna()
        cur = s.iloc[-1]
        rows.append({"指数": INDEX_META[sym][0], "PE(TTM)": cur,
                     "8年分位": (s.iloc[-96:] <= cur).mean(),
                     "中位数": s.iloc[-96:].median(),
                     "最低": s.iloc[-96:].min(), "最高": s.iloc[-96:].max()})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(table_style(df, pct_cols=["8年分位"], num_cols=["PE(TTM)", "中位数", "最低", "最高"]),
                     width="stretch", hide_index=True)


# ------------------------------------------------------------------ page 0b
INDEX_MAP = {
    "000300.SH": ("sh000300", "沪深300"),
    "000905.SH": ("sh000905", "中证500"),
    "399006.SZ": ("sz399006", None),
    "000001.SH": ("sh000001", None),
    "000688.SH": ("sh000688", None),
    "399001.SZ": ("sz399001", None),
}


@st.cache_data(ttl=3600, show_spinner=False)
def daily_score(code: str) -> tuple[float, float]:
    if code in INDEX_MAP:
        sym, pe_name = INDEX_MAP[code]
        d = signals.equity_diagnosis(sym, pe_name)
        return float(d.get("score", 50)), float(d.get("trend", 50))
    d = signals.stock_diagnosis(code.split(".")[0])
    if not d:
        return float("nan"), float("nan")
    return float(d.get("score", 50)), float(d.get("trend", 50))


def _paper_new() -> dict:
    return {"cash": 1_000_000.0, "positions": {}, "trades": [], "equity": [],
            "ticks": 0, "bench0": None, "start": datetime.now().strftime("%m-%d %H:%M")}


def _paper_tick(selected: list[str], allow_trade: bool, news_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    from datetime import datetime as _dt

    q = live.fetch_quotes(selected)
    if q.empty:
        return q, st.session_state.paper
    acc = st.session_state.paper
    today = _dt.now().date()
    rows = []
    equity_before = acc["cash"] + sum(
        p["shares"] * float(q.loc[q["code"] == c, "price"].iloc[0])
        for c, p in acc["positions"].items() if (q["code"] == c).any())
    for _, r in q.iterrows():
        code, price = r["code"], float(r["price"])
        score, trend = daily_score(code)
        if pd.isna(score):
            direction, reason = live.direction_intraday(
                float(r["pct_chg"]), price, float(r["high"]), float(r["low"]),
                float(r["pre_close"]))
        else:
            direction, reason = live.direction_for(score, float(r["pct_chg"]))
        pos = acc["positions"].get(code)
        if allow_trade:
            if direction in ("买入", "加仓") and pos is None and len(acc["positions"]) < 5:
                budget = min(acc["cash"], equity_before * 0.20)
                shares = int(budget / price / 100) * 100
                if shares >= 100:
                    acc["cash"] -= shares * price
                    acc["positions"][code] = {"shares": shares, "cost": price,
                                              "name": r["name"], "date": str(today)}
                    acc["trades"].append({"时间": _dt.now().strftime("%H:%M:%S"), "标的": r["name"],
                                          "方向": "买入", "价格": price, "数量": shares, "盈亏": None})
            elif direction in ("减仓", "回避") and pos and pos.get("date") != str(today):
                acc["cash"] += pos["shares"] * price
                pnl = (price - pos["cost"]) * pos["shares"]
                acc["trades"].append({"时间": _dt.now().strftime("%H:%M:%S"), "标的": r["name"],
                                      "方向": "卖出", "价格": price, "数量": pos["shares"], "盈亏": pnl})
                del acc["positions"][code]
        news_score = np.nan
        if news_df is not None and not news_df.empty:
            hit = news.stock_mentions(news_df, r["name"], code)
            if not hit.empty:
                news_score = float(hit["情绪分"].mean())
        rows.append({"代码": code, "标的": r["name"], "现价": price, "涨跌幅": float(r["pct_chg"]),
                     "日线评分": score, "趋势": trend, "新闻": news_score,
                     "方向": direction, "理由": reason, "来源": r.get("source", "")})
    equity = acc["cash"] + sum(
        p["shares"] * float(q.loc[q["code"] == c, "price"].iloc[0])
        for c, p in acc["positions"].items() if (q["code"] == c).any())
    if acc["bench0"] is None and (q["code"] == "000300.SH").any():
        acc["bench0"] = float(q.loc[q["code"] == "000300.SH", "price"].iloc[0])
    bench = None
    if acc["bench0"] and (q["code"] == "000300.SH").any():
        bench = 1_000_000 * float(q.loc[q["code"] == "000300.SH", "price"].iloc[0]) / acc["bench0"]
    acc["ticks"] += 1
    acc["equity"].append({"时间": _dt.now().strftime("%H:%M:%S"), "净值": equity, "基准": bench})
    acc["equity"] = acc["equity"][-500:]
    return pd.DataFrame(rows), acc


DIR_COLOR = {"买入": "#1f6f4a", "加仓": "#2e8b6f", "持有": "#2e8b6f",
             "观望": "#8a8f98", "减仓": "#c07a2c", "回避": "#a33b4a"}


def page_live():
    hero("A股实时盯盘", "手动启动后每 30 秒取一次行情：给方向，再用虚拟账户记录按这个方向买卖的结果。",
         ["同花顺 iFinD", "30 秒刷新", "模拟成交", "T+1 约束"])
    explain("这页不预测明天涨跌。它把每只股票自己的历史评分和今天的涨跌放在一起，给出买入、观望或减仓；"
            "然后用 100 万虚拟资金按这个方向买卖，记录赚亏。看的是这套规则在当前市场是否顺手，不是保证收益。")

    if "paper" not in st.session_state:
        st.session_state.paper = _paper_new()
    if "live_on" not in st.session_state:
        st.session_state.live_on = False

    options = {f"{name} ({code})": code for code, (name, _) in live.WATCH_POOL.items()}
    default = [k for k, v in options.items() if v in live.DEFAULT_WATCH]
    picked = st.multiselect("盯盘池（建议不超过 8 个，控制接口配额）", list(options), default=default)
    selected = [options[p] for p in picked] or live.DEFAULT_WATCH
    extra = st.text_input("补充任意 A 股代码（逗号分隔，例如 000001,600036）", "")
    if extra:
        known = set(data.load_all_stocks()["code"])
        for c in extra.replace("，", ",").split(","):
            c = c.strip()
            if c.isdigit() and len(c) == 6 and (not known or c in known):
                selected.append(live.to_ths_code(c))

    stat, is_open = live.market_status()
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.session_state.live_on = st.toggle("启动 30 秒自动盯盘", value=st.session_state.live_on)
        force = st.checkbox("休市时也用最新价模拟（仅演示）", value=False)
    with c2:
        st.metric("市场状态", stat, "仅交易时段自动记账" if not is_open else "自动记账中")
    with c3:
        if st.button("重置模拟盘", width="stretch"):
            st.session_state.paper = _paper_new()
            st.rerun()
    allow_trade = is_open or force

    st.caption("手动启动是为了控制同花顺接口配额；每次刷新只发 1 次批量请求。接口失败会自动降级到新浪公开行情。")

    @st.fragment(run_every=30)
    def dashboard():
        if not st.session_state.live_on:
            st.info("自动盯盘未启动。打开上方开关后，将每 30 秒刷新一次；也可点击下方按钮启动。")
            if st.button("启动并立即刷新"):
                st.session_state.live_on = True
                st.rerun()
            return
        if st.button("立即刷新", key="live_refresh"):
            st.rerun()
        try:
            try:
                ndf = load_news()
            except Exception:
                ndf = pd.DataFrame()
            sig, acc = _paper_tick(selected, allow_trade, ndf)
        except Exception as exc:  # noqa: BLE001
            st.error(f"行情获取失败：{exc}")
            return
        if sig.empty:
            st.warning("未获取到行情，请检查代码或接口。")
            return
        source = sig["来源"].iloc[0] if "来源" in sig.columns and not sig.empty else "—"
        equity = acc["equity"][-1]["净值"] if acc["equity"] else 1_000_000
        bench = acc["equity"][-1]["基准"] if acc["equity"] and acc["equity"][-1]["基准"] else 1_000_000
        wins = [t for t in acc["trades"] if t["盈亏"] is not None and t["盈亏"] > 0]
        closed = [t for t in acc["trades"] if t["盈亏"] is not None]
        nav_series = pd.Series([e["净值"] for e in acc["equity"]])
        mdd = float((nav_series / nav_series.cummax() - 1).min()) if len(nav_series) > 1 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("虚拟净值", f"{equity:,.0f}", f"{(equity/1_000_000-1)*100:+.2f}%")
        m2.metric("相对沪深300", f"{(equity-bench)/1_000_000*100:+.2f}%", f"基准 {bench:,.0f}")
        m3.metric("成交/持仓", f"{len(acc['trades'])} / {len(acc['positions'])}", "累计 / 当前")
        m4.metric("已平仓胜率", f"{len(wins)/len(closed)*100:.0f}%" if closed else "—", f"{len(closed)} 笔")
        m5.metric("盯盘状态", stat, f"第 {acc['ticks']} 次刷新 · {source}")

        st.markdown("##### 当前信号")
        cols = st.columns(3)
        for i, (_, r) in enumerate(sig.iterrows()):
            with cols[i % 3]:
                color = DIR_COLOR.get(r["方向"], "#8a8f98")
                st.markdown(
                    f"""<div class="card" style="margin-bottom:10px">
                    <h4>{r['标的']} <span class="small">{r['代码']}</span></h4>
                    <p style="font-size:22px;margin:4px 0;color:{'#a33b4a' if r['涨跌幅']<0 else '#1f6f4a'}">
                    {r['现价']:,.2f} <span style="font-size:14px">{r['涨跌幅']*100:+.2f}%</span></p>
                    <p>{pill(r['方向'], color)} <span class="small">日线评分 {r['日线评分']:.0f}</span></p>
                    <p class="small">{r['理由']}</p></div>""", unsafe_allow_html=True)

        st.markdown("##### 策略与模拟盘")
        left, right = st.columns([1.4, 1])
        with left:
            detail = sig.copy()
            detail["涨跌幅"] = detail["涨跌幅"].map(lambda x: f"{x*100:+.2f}%")
            detail["日线评分"] = detail["日线评分"].map(lambda x: "—" if pd.isna(x) else f"{x:.0f}")
            detail["趋势"] = detail["趋势"].map(lambda x: "—" if pd.isna(x) else f"{x:.0f}")
            detail["新闻"] = detail["新闻"].map(lambda x: "—" if pd.isna(x) else f"{x:.0f}")
            st.dataframe(detail, hide_index=True, width="stretch", height=260)
        with right:
            if acc["positions"]:
                pos = []
                for code, p in acc["positions"].items():
                    price = float(sig.loc[sig["代码"] == code, "现价"].iloc[0]) if (sig["代码"] == code).any() else p["cost"]
                    pos.append({"标的": p["name"], "数量": p["shares"], "成本": p["cost"],
                                "现价": price, "浮盈": (price - p["cost"]) * p["shares"]})
                st.dataframe(pd.DataFrame(pos), hide_index=True, width="stretch", height=220)
            else:
                st.info("暂无持仓。出现“买入/加仓”且处于交易时段时会自动建仓（单票 20%，最多 5 只）。")
            if acc["trades"]:
                st.download_button("导出成交记录", pd.DataFrame(acc["trades"]).to_csv(index=False).encode("utf-8-sig"),
                                   "paper_trades.csv", "text/csv")

        if len(acc["equity"]) > 1:
            eq = pd.DataFrame(acc["equity"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq["时间"], y=eq["净值"], name="策略净值",
                                     line=dict(color="#1f4e79", width=2.2)))
            if eq["基准"].notna().any():
                fig.add_trace(go.Scatter(x=eq["时间"], y=eq["基准"], name="沪深300",
                                         line=dict(color="#8a8f98", width=1.4, dash="dash")))
            fig.update_yaxes(title="虚拟权益（元）")
            st.plotly_chart(fig_base(fig, 320, "模拟盘权益曲线（每 30 秒一个点）"), width="stretch")

        if acc["ticks"] < 10:
            st.info(f"样本不足（已刷新 {acc['ticks']} 次）。至少积累 10 个刷新点后再评估策略是否适应当前市场。")
        else:
            win_txt = f"{len(wins)/len(closed)*100:.0f}%" if closed else "暂无平仓"
            st.markdown(
                f"<div class='insight'>本次盯盘累计刷新 <b>{acc['ticks']}</b> 次，虚拟账户收益 "
                f"<b>{(equity/1_000_000-1)*100:+.2f}%</b>，相对沪深300 "
                f"<b>{(equity-bench)/1_000_000*100:+.2f}%</b>，已平仓胜率 <b>{win_txt}</b>，"
                f"权益最大回撤 <b>{mdd*100:.2f}%</b>。若相对基准持续为负，说明当前日线信号不适合当下的盘中节奏，"
                f"应降低仓位或等待趋势确认；模拟结果不含滑点与手续费，实盘会更差。</div>", unsafe_allow_html=True)

        if acc["trades"]:
            with st.expander("成交明细", expanded=False):
                st.dataframe(pd.DataFrame(acc["trades"]), hide_index=True, width="stretch")

    dashboard()
    st.caption(DISCLAIMER + " 模拟盘不构成投资建议，也不会向券商发送任何真实委托。")


# ------------------------------------------------------------------ page 1b
@st.cache_data(ttl=300, show_spinner=False)
def load_news() -> pd.DataFrame:
    return news.fetch_market_news()


def page_news():
    hero("新闻情绪", "把公开新闻标题按关键词分成利好和利空，做成 0–100 的情绪分。它只衡量措辞倾向，不判断新闻真假。",
         ["东方财富 7×24 快讯", "关键词法", "5 分钟缓存"])
    explain("先取最近几个小时的财经快讯，用一份利好/利空词表给每条标题打分，再平均成市场情绪分，"
            "并按行业关键词分组。它读不懂反讽和否定，只适合当“关注度+措辞方向”的辅助指标。")

    left, right = st.columns([1, 4])
    with left:
        if st.button("刷新新闻", width="stretch"):
            st.cache_data.clear()
            news.fetch_market_news.cache_clear()
            st.rerun()
    with right:
        st.caption("每次刷新只取 1 次快讯列表（约 200 条），不会高频请求。")

    try:
        df = load_news()
    except Exception as exc:  # noqa: BLE001
        st.error(f"新闻获取失败：{exc}")
        return
    if df.empty:
        st.warning("没有取到新闻。")
        return

    snap = news.market_snapshot(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("新闻情绪", f"{snap['score']:.0f} / 100", "偏暖" if snap["score"] >= 55 else ("偏冷" if snap["score"] <= 45 else "中性"))
    c2.metric("正面新闻", f"{snap['pos']} 条", f"{snap['pos_ratio']*100:.0f}%")
    c3.metric("负面新闻", f"{snap['neg']} 条", f"{snap['neg_ratio']*100:.0f}%")
    c4.metric("样本", f"{snap['n']} 条", f"截至 {snap['latest']}")

    left, right = st.columns([1, 1.6])
    with left:
        st.plotly_chart(gauge(snap["score"], "新闻情绪（50=中性）", 280), width="stretch")
        st.caption("50 以上偏正面，50 以下偏负面。分数来自标题和摘要里的关键词，不区分反讽、否定和上下文。")
    with right:
        hs = news.hourly_sentiment(df)
        if not hs.empty:
            fig = go.Figure(go.Bar(x=hs["小时"], y=hs["情绪分"],
                                   marker_color=["#2e8b6f" if v >= 50 else "#a33b4a" for v in hs["情绪分"]]))
            fig.add_hline(y=50, line_dash="dash", line_color="#8a8f98")
            fig.update_yaxes(title="情绪分", range=[0, 100])
            st.plotly_chart(fig_base(fig, 300, "最近 24 个小时的新闻情绪"), width="stretch")

    st.markdown("##### 行业新闻热度")
    sec = news.sector_sentiment(df)
    if not sec.empty:
        show = sec.head(12).sort_values("新闻数")
        fig = go.Figure(go.Bar(x=show["新闻数"], y=show["行业"], orientation="h",
                               marker=dict(color=show["情绪分"], colorscale="RdYlGn",
                                           cmin=30, cmax=70, colorbar=dict(title="情绪"))))
        st.plotly_chart(fig_base(fig, 420, "被新闻提到最多的行业（颜色=情绪分）"), width="stretch")
        st.dataframe(table_style(sec.head(12), pct_cols=["正面占比"], num_cols=["情绪分"]),
                     hide_index=True, width="stretch")

    st.markdown("##### 今天最正面 / 最负面的标题")
    left, right = st.columns(2)
    with left:
        st.caption("最正面")
        pos = news.top_headlines(df, 6, True)
        for _, r in pos.iterrows():
            st.markdown(f"<div class='card' style='margin-bottom:8px'><p style='font-size:13px'>{r['标题']}</p>"
                        f"<p class='small'>{r['发布时间']:%H:%M} · 情绪分 {r['情绪分']:.0f}</p></div>",
                        unsafe_allow_html=True)
    with right:
        st.caption("最负面")
        neg = news.top_headlines(df, 6, False)
        for _, r in neg.iterrows():
            st.markdown(f"<div class='card' style='margin-bottom:8px'><p style='font-size:13px'>{r['标题']}</p>"
                        f"<p class='small'>{r['发布时间']:%H:%M} · 情绪分 {r['情绪分']:.0f}</p></div>",
                        unsafe_allow_html=True)

    hist_path = DATA / "news" / "cctv_daily.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path, parse_dates=["date"])
        if len(hist) >= 10:
            px = data.load_index("sh000300")["close"].pct_change()

            def _next_ret(d):
                idx = px.index[px.index > d]
                return float(px.loc[idx[0]]) if len(idx) else np.nan

            hist["次日收益"] = hist["date"].map(_next_ret)
            h = hist.dropna(subset=["次日收益"])
            if len(h) >= 8:
                corr = h["情绪分"].corr(h["次日收益"])
                st.markdown("##### 新闻联播情绪与次日沪深300（小样本）")
                fig = px.scatter(h, x="情绪分", y="次日收益",
                                 labels={"情绪分": "当日新闻联播情绪分", "次日收益": "次日沪深300收益"})
                fig.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig_base(fig, 320, f"相关系数 {corr:.2f}（{len(h)} 个交易日）"),
                                width="stretch")
                st.caption("新闻联播在晚间播出，因此和次日行情对齐。样本只有几十个交易日，相关系数不稳定，不能当预测模型。")

    st.markdown("##### 查一只股票的新闻")
    allsh = data.load_all_stocks()
    names = dict(zip(allsh["code"], allsh["name"])) if not allsh.empty else {}
    q = st.text_input("输入股票代码或名称（例如 600519 / 贵州茅台）", "")
    if q:
        name = names.get(q.strip().zfill(6), q.strip())
        hits = news.stock_mentions(df, name, q.strip())
        if hits.empty:
            st.info(f"最近的快讯里没有直接提到“{name}”。新闻样本只覆盖最近几个小时，未提到不代表没有新闻。")
        else:
            st.metric(f"{name} 相关新闻情绪", f"{hits['情绪分'].mean():.0f} / 100", f"{len(hits)} 条")
            show = hits[["发布时间", "标题", "情绪分"]].copy()
            show["发布时间"] = show["发布时间"].dt.strftime("%m-%d %H:%M")
            st.dataframe(table_style(show, num_cols=["情绪分"]), hide_index=True, width="stretch")

    with st.expander("这个指标怎么算？"):
        st.markdown("""
1. 取东方财富 7×24 快讯的标题和摘要，最多 200 条；
2. 用一份人工整理的利好/利空词表匹配，给每个词一个权重；
3. 单条新闻得分 `(正面词权重 - 负面词权重) / (正面 + 负面 + 1.5)`，再映射到 0–100；
4. 对全部新闻取平均，得到市场情绪分；按行业关键词分组，得到行业情绪。

**局限**：关键词法读不懂反讽、否定和上下文，也无法判断新闻真假；它更适合做“关注度 + 措辞方向”的辅助指标，不能单独作为买卖依据。
""")
    st.caption("新闻来自东方财富公开快讯，仅用于课程研究，不构成投资建议。")


# ------------------------------------------------------------------ page 3b
def page_finance():
    hero("财务指标", "看一只股票的盈利、成长、负债和估值，再和同池公司做横向比较。",
         ["ROE", "毛利率", "负债率", "营收增速"])
    explain("财务指标来自公司季报，反映的是已经发生的经营结果，不是股价预测。"
            "一般把高 ROE、低负债、稳定增长看作质量较好，但要结合行业和估值一起看。")

    fin = load_derived("fin_snapshot")
    if fin.empty:
        st.warning("财务快照未生成，请运行 scripts/build_snapshots.py。")
        return

    mapping = {f"{r['name']} ({r['code']})": r["code"] for _, r in fin.iterrows()}
    labels = list(mapping)
    idx = next((i for i, k in enumerate(labels) if "600519" in k), 0)
    label = st.selectbox("选择股票", labels, index=idx)
    code = mapping[label]
    row = fin[fin["code"] == code].iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ROE", f"{row['roe']:.1f}%" if pd.notna(row["roe"]) else "—")
    c2.metric("毛利率", f"{row['gross_margin']:.1f}%" if pd.notna(row["gross_margin"]) else "—")
    c3.metric("净利率", f"{row['net_margin']:.1f}%" if pd.notna(row["net_margin"]) else "—")
    c4.metric("资产负债率", f"{row['debt_ratio']:.1f}%" if pd.notna(row["debt_ratio"]) else "—")
    c5.metric("营收增速", f"{row['rev_growth']:+.1f}%" if pd.notna(row["rev_growth"]) else "—")
    c6.metric("PE / PB", f"{row['pe']:.1f} / {row['pb']:.2f}" if pd.notna(row["pe"]) and pd.notna(row["pb"]) else "—")

    h = data.load_stock_fin_hist(code)
    if not h.empty:
        h = h.tail(12)
        fig = go.Figure()
        for col, name, color in [("roe", "ROE", "#1f4e79"),
                                 ("rev_growth", "营收增速", "#2e8b6f"),
                                 ("profit_growth", "净利润增速", "#c9a227")]:
            if col in h.columns:
                fig.add_trace(go.Scatter(x=h.index, y=h[col], name=name,
                                         line=dict(color=color, width=1.8)))
        fig.add_hline(y=0, line_color="#c9d3dc", line_width=1)
        fig.update_yaxes(title="%")
        st.plotly_chart(fig_base(fig, 320, "最近 12 个季度"), width="stretch")
        st.caption("季报数据按报告期排列；图中最后一个点可能是单季或累计值，比较时注意口径。")

    st.markdown("#### 同池筛选")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        min_roe = st.slider("ROE 下限", -20, 40, 10, 1)
    with f2:
        max_debt = st.slider("负债率上限", 0, 100, 70, 5)
    with f3:
        sort_by = st.selectbox("排序", ["ROE", "营收增速", "净利润增速", "毛利率", "PE"], index=0)

    view = fin.copy()
    view = view[view["roe"].fillna(-999) >= min_roe]
    view = view[view["debt_ratio"].fillna(999) <= max_debt]
    key_map = {"ROE": "roe", "营收增速": "rev_growth", "净利润增速": "profit_growth",
               "毛利率": "gross_margin", "PE": "pe"}
    view = view.sort_values(key_map[sort_by], ascending=(sort_by == "PE")).head(60)
    show = view[["name", "code", "roe", "gross_margin", "net_margin", "debt_ratio",
                 "rev_growth", "profit_growth", "asset_growth", "pe", "pb", "mcap"]].copy()
    show.columns = ["名称", "代码", "ROE", "毛利率", "净利率", "负债率",
                    "营收增速", "净利增速", "资产增速", "PE", "PB", "市值(亿)"]
    show["市值(亿)"] = (show["市值(亿)"] / 1e8).round(0)
    for c in ["ROE", "毛利率", "净利率", "负债率", "营收增速", "净利增速", "资产增速"]:
        show[c] = show[c] / 100
    st.dataframe(table_style(show, pct_cols=["ROE", "毛利率", "净利率", "负债率",
                                             "营收增速", "净利增速", "资产增速"],
                             num_cols=["PE", "PB", "市值(亿)"]),
                 hide_index=True, width="stretch", height=420)
    st.caption("筛选结果只说明财务特征，不代表推荐。金融、地产等行业的负债率天然偏高，应和同业比。")


# ------------------------------------------------------------------ page 1
def page_xray():
    hero("个股诊断 · X-Ray", "把一只股票和它自己的历史比：估值贵不贵、趋势坏没坏、未来一年可能落在什么区间。",
         ["估值分位", "买卖区间", "蒙特卡洛", "止损与仓位"])
    explain("这里不是给目标价。它做三件事：看当前市盈率在近8年中的位置；看价格相对均线是强是弱；"
            "用过去3年的日涨跌随机组合出2000条未来一年的路径，给出一个区间。买入区间和止损只是参考。")

    qp_kind = st.query_params.get("kind", "")
    kind = st.segmented_control("资产类别", ["个股", "宽基指数", "行业"],
                                default={"index": "宽基指数", "sector": "行业",
                                         "stock": "个股"}.get(qp_kind, "个股"))
    qp_code = st.query_params.get("code", "")
    if kind == "宽基指数":
        options = {INDEX_META[k][0]: k for k in INDEX_META}
        idx = list(options.values()).index(qp_code) if qp_code in options.values() else 0
        label = st.selectbox("选择指数", list(options), index=idx)
        sym = options[label]
        pe_name = {"sh000300": "沪深300", "sh000905": "中证500", "sh000016": "上证50"}.get(sym)
        d = signals.equity_diagnosis(sym, pe_name)
        df = data.load_index(sym)
        pe_df = data.load_index_valuation(pe_name, "pe") if pe_name else pd.DataFrame()
        pe_col = ("滚动市盈率" if "滚动市盈率" in pe_df.columns else pe_df.columns[-1]) if not pe_df.empty else None
        pe_series = pe_df[pe_col].dropna() if pe_col else pd.Series(dtype=float)
    elif kind == "行业":
        options = {SECTOR_META[k][0]: k for k in SECTOR_META}
        idx = list(options.values()).index(qp_code) if qp_code in options.values() else 0
        label = st.selectbox("选择行业", list(options), index=idx)
        code = options[label]
        d = signals.sector_diagnosis(code)
        df = data.load_sector(code)
        pe_series = pd.Series(dtype=float)
    elif kind == "商品":
        options = {FUTURES_META[k][0]: k for k in FUTURES_META}
        idx = list(options.values()).index(qp_code) if qp_code in options.values() else 0
        label = st.selectbox("选择商品", list(options), index=idx)
        sym = options[label]
        d = signals.commodity_diagnosis(sym)
        df = data.load_futures(sym)
        pe_series = pd.Series(dtype=float)
    else:
        uni = data.load_universe()
        mapping = {f"{r['name']} ({r['code']})": r["code"] for _, r in uni.iterrows()}
        vals = list(mapping.values())
        idx = vals.index(qp_code) if qp_code in vals else (vals.index("600519") if "600519" in vals else 0)
        label = st.selectbox("选择个股（沪深800深度池）", list(mapping), index=idx)
        code = mapping[label]
        st.caption("深度诊断覆盖 791 只沪深300/中证500成分股；其余 4,788 只 A 股可在“实时盯盘”输入代码，使用盘中信号。")
        d = signals.stock_diagnosis(code)
        df = data.load_stock_price(code)
        val = data.load_stock_value(code)
        pe_series = val["pe_ttm"].dropna() if not val.empty else pd.Series(dtype=float)

    if not d:
        st.warning("该资产数据不足，请先运行数据抓取脚本。")
        return

    score = d.get("score", np.nan)
    action, color = d.get("action"), d.get("color")
    if not action:
        action, color = signals.action_for(score)
    buy = (signals.buy_zone_from_pe(df["close"], pe_series) if not pe_series.empty
           else signals.buy_zone_from_price(df["close"]))
    risk = signals.stop_level(df)
    mc = signals.monte_carlo(df["close"])

    val_pct = (d.get("valuation") or {}).get("pct")
    if val_pct is None:
        val_pct = d.get("price_pct", np.nan)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(label, fmt_num(d.get("price")))
    c2.metric("综合评分", f"{score:.0f}", action)
    c3.metric("估值/价格分位", f"{val_pct*100:.0f}%" if not pd.isna(val_pct) else "—")
    c4.metric("近12月", fmt_pct(d.get("ret_12m", np.nan)))
    c5.metric("距高点回撤", fmt_pct(d.get("drawdown", np.nan)))

    if kind == "个股":
        try:
            _hits = news.stock_mentions(load_news(), label.split(" (")[0], code)
            if not _hits.empty:
                st.caption(f"最近快讯里提到这只股票 {len(_hits)} 次，平均情绪分 {_hits['情绪分'].mean():.0f}/100。")
        except Exception:
            pass

    left, right = st.columns([2.2, 1])
    with left:
        st.plotly_chart(price_figure(df, buy, risk.get("stop"),
                                     f"{label} · 近三年价格与均线",
                                     zone_label="估值中枢区间" if d.get("valuation") else "价格中枢区间"),
                        width="stretch")
    with right:
        st.markdown(f"""
        <div class="card">
          <h4>结论</h4>
          <p style="font-size:20px;margin:6px 0">{action_badge(score, d.get('action'), d.get('color'))}</p>
          <p>综合评分 <b>{score:.0f}</b> / 100<br>
          评分体系：{d.get('basis','')}</p>
          <hr style="border:none;border-top:1px solid #eef2f6;margin:12px 0">
          <p><b>{'估值中枢区间（PE 30–55 分位）' if d.get('valuation') else '价格中枢区间（近5年 30–55 分位）'}</b><br>
          {fmt_num(buy.get('buy_low'))} — {fmt_num(buy.get('buy_high'))}
          （距当前 {fmt_pct(buy.get('discount_to_high', np.nan))}）</p>
          <p style="color:#5d6b7a">
          {('当前价已低于区间下沿，处于低估区，可分批建仓。' if buy and d.get('price',0) < buy.get('buy_low', 0)
            else ('当前价高于区间上沿，等待回落至区间再介入。' if buy and d.get('price',0) > buy.get('buy_high', 1e18)
                  else '当前价位于区间之内，可按计划分批。'))}</p>
          <p><b>技术止损</b><br>{fmt_num(risk.get('stop'))}（{fmt_pct(risk.get('stop_pct'))}，2×ATR）</p>
          <p><b>建议仓位上限</b><br>{max(0, min(30, (score-40)/60*30)):.0f}%（按评分线性映射）</p>
        </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["估值与前瞻收益", "概率化收益", "风险与相关性", "指标明细"])
    with tabs[0]:
        if not pe_series.empty:
            pe = pe_series.copy()
            pe.index = pd.to_datetime(pe.index)
            recent = pe[pe.index >= pe.index.max() - pd.Timedelta(days=365 * 10)]
            q30, q50, q70 = recent.quantile([0.3, 0.5, 0.7])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent.index, y=recent, name="PE(TTM)",
                                     line=dict(color="#12304f", width=1.6)))
            for v, lab, col in [(q30, "30%分位", "#2e8b6f"), (q50, "中位数", "#8a8f98"),
                                (q70, "70%分位", "#c07a2c")]:
                fig.add_hline(y=v, line_dash="dash", line_color=col,
                              annotation_text=f"{lab} {v:.1f}", annotation_font_color=col)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig_base(fig, 330, f"{label} 估值带"), width="stretch")
            vf = load_derived("valuation_forward")
            if not vf.empty and kind == "宽基指数" and sym == "sh000300":
                vf["估值区间"] = [f"{a*100:.0f}–{b*100:.0f}%" for a, b in zip(vf["估值分位下限"], vf["估值分位上限"])]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=vf["估值区间"], y=vf["未来收益中位数"],
                                     marker_color=["#2e8b6f" if v > 0 else "#a33b4a" for v in vf["未来收益中位数"]],
                                     name="未来12月中位收益"))
                fig.add_trace(go.Scatter(x=vf["估值区间"], y=vf["正收益概率"] * 0.3,
                                         yaxis="y2", name="正收益概率", mode="lines+markers",
                                         line=dict(color="#c9a227")))
                fig.update_layout(yaxis=dict(tickformat=".0%", title="未来12个月收益"),
                                  yaxis2=dict(overlaying="y", side="right", range=[0, 1],
                                              tickformat=".0%", title="胜率"))
                st.plotly_chart(fig_base(fig, 320, "沪深300：按估值分位统计的未来12个月收益"),
                                width="stretch")
            if kind == "宽基指数" and sym == "sh000300":
                cf = signals.conditional_forward(pe_series, data.load_index("sh000300")["close"].resample("ME").last())
                if cf:
                    st.markdown(
                        f"<div class='insight'>当前估值分位 <b>{cf['current_pct']*100:.0f}%</b>；"
                        f"历史上处于相近分位的 {cf['n']} 个观测，未来 12 个月收益中位数 "
                        f"<b>{cf['median']*100:+.1f}%</b>，正收益概率 <b>{cf['win']*100:.0f}%</b>，"
                        f"四分位区间 {cf['p25']*100:+.1f}% 至 {cf['p75']*100:+.1f}%。</div>",
                        unsafe_allow_html=True)
        else:
            st.info("该资产缺少长历史估值序列，使用价格分位与趋势指标替代。")
            st.plotly_chart(price_figure(df, buy, risk.get("stop"), f"{label} 价格与均线",
                                         zone_label="价格中枢区间"), width="stretch")

    with tabs[1]:
        if mc:
            terminal = 1 + np.array(mc.get("terminal", [])) if mc.get("terminal") is not None else None
            c1, c2, c3 = st.columns(3)
            c1.metric("12个月期望收益", fmt_pct(mc["exp"]))
            c2.metric("正收益概率", f"{mc['prob_up']*100:.0f}%")
            c3.metric("95%分位回撤", fmt_pct(mc["mdd_p95"]))
            if terminal is not None and len(terminal):
                fig = go.Figure(go.Histogram(x=(terminal - 1) * 100, nbinsx=50,
                                             marker_color="#1f4e79", opacity=0.85))
                fig.add_vline(x=0, line_color="#a33b4a", line_dash="dash")
                fig.update_xaxes(title="未来12个月收益率（%）")
                st.plotly_chart(fig_base(fig, 320, "蒙特卡洛：2000条路径的12个月收益分布"),
                                width="stretch")
            st.dataframe(pd.DataFrame([{
                "情景": "悲观(5%)", "收益率": fmt_pct(mc["p05"]),
                "对应价格": fmt_num(mc["prices"]["p05"])},
                {"情景": "偏弱(25%)", "收益率": fmt_pct(mc["p25"]), "对应价格": fmt_num(mc["prices"]["p25"])},
                {"情景": "中性(50%)", "收益率": fmt_pct(mc["p50"]), "对应价格": fmt_num(mc["prices"]["p50"])},
                {"情景": "偏强(75%)", "收益率": fmt_pct(mc["p75"]), "对应价格": fmt_num(mc["prices"]["p75"])},
                {"情景": "乐观(95%)", "收益率": fmt_pct(mc["p95"]), "对应价格": fmt_num(mc["prices"]["p95"])},
            ]), hide_index=True, width="stretch")
        else:
            st.info("历史样本不足，无法进行蒙特卡洛模拟。")

    with tabs[2]:
        r = returns(df["close"]).dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("年化波动率", fmt_pct(r.iloc[-252:].std() * np.sqrt(252)))
        c2.metric("近一年最大回撤", fmt_pct(drawdown(df["close"]).iloc[-252:].min()))
        c3.metric("60日波动分位", f"{d.get('vol_score', np.nan):.0f}%")
        st.plotly_chart(drawdown_fig(df["close"], label, 240), width="stretch")
        bench = data.load_index("sh000300")["close"]
        corr = returns(df["close"]).corr(returns(bench))
        st.markdown(f"<div class='insight'>与沪深300的日收益相关系数为 <b>{corr:.2f}</b>。"
                    f"组合中加入该资产时，应结合相关性判断分散效果，而非只看单一资产收益。</div>",
                    unsafe_allow_html=True)

    with tabs[3]:
        items = {
            "价格": d.get("price"), "综合评分": score, "PE(TTM)": (d.get("valuation") or {}).get("pe"),
            "PE分位": (d.get("valuation") or {}).get("pct"), "趋势评分": d.get("trend"),
            "动量分位": d.get("momentum_pct"), "低波评分": d.get("vol_score"),
            "近1月收益": d.get("ret_1m"), "近12月收益": d.get("ret_12m"),
            "ROE": d.get("roe"), "毛利率": d.get("gross_margin"),
            "库存分位": d.get("pct") if kind == "商品" else None,
        }
        tbl = pd.DataFrame([{"指标": k, "数值": v} for k, v in items.items() if v is not None])
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption(DISCLAIMER)


# ------------------------------------------------------------------ page 2
def page_sectors():
    hero("行业轮动", "31 个申万一级行业：看哪些便宜、哪些正在走强，再用一条简单规则回测。",
         ["四象限定位", "动量前 5", "沪深300 200 日线择时"])
    explain("左边是价格位置，越靠左说明相对过去5年越便宜；下面是动量，越靠上说明最近一年越强。"
            "回测规则很朴素：每月挑动量最强的5个行业，如果沪深300跌破200日线就空仓。看它过去8年是否跑赢行业等权。")
    sig = get_signals()
    sec = sig[sig["type"] == "行业指数"].copy()
    if sec.empty:
        st.warning("行业数据未就绪。")
        return

    fig = go.Figure()
    ranked = sec.sort_values("score", ascending=False).reset_index(drop=True)
    label_set = set(ranked.head(10)["name"]) | set(ranked.tail(6)["name"])
    for _, r in sec.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["price_pct"] * 100], y=[r["momentum_pct"] * 100],
            mode="markers+text", text=[r["name"] if r["name"] in label_set else ""], textposition="top center",
            marker=dict(size=18 + r["score"] * 0.25, color=ACTION_COLORS.get(r["action"], "#8a8f98"),
                        line=dict(color="white", width=1.5), opacity=0.88),
            name=r["name"], showlegend=False,
            hovertemplate=f"{r['name']}<br>价格分位 %{{x:.0f}}%<br>动量分位 %{{y:.0f}}%<br>评分 {r['score']:.0f}"))
    fig.add_vline(x=50, line_dash="dash", line_color="#c9d3dc")
    fig.add_hline(y=50, line_dash="dash", line_color="#c9d3dc")
    fig.add_annotation(x=8, y=94, text="低估 + 强动量<br>优先配置", showarrow=False,
                       font=dict(color="#1f6f4a", size=12), align="left")
    fig.add_annotation(x=92, y=94, text="高估 + 强动量<br>持有但不追", showarrow=False,
                       font=dict(color="#c07a2c", size=12), align="right")
    fig.add_annotation(x=8, y=4, text="低估 + 弱动量<br>左侧观察", showarrow=False,
                       font=dict(color="#2f7fa8", size=12), align="left")
    fig.add_annotation(x=92, y=4, text="高估 + 弱动量<br>规避", showarrow=False,
                       font=dict(color="#a33b4a", size=12), align="right")
    fig.update_xaxes(title="价格分位（近5年，越低越便宜）", range=[-2, 102])
    fig.update_yaxes(title="动量分位（近8年，越高越强）", range=[-2, 104])
    st.plotly_chart(fig_base(fig, 560, "行业轮动四象限（气泡=综合评分，仅标注两端）"), width="stretch")

    left, right = st.columns([1.1, 1])
    with left:
        d = sec.sort_values("score", ascending=False)[["name", "score", "price_pct", "trend", "ret_3m", "ret_12m", "action"]]
        st.markdown("##### 行业评分排行")
        st.dataframe(table_style(d, pct_cols=["price_pct", "ret_3m", "ret_12m"], num_cols=["score", "trend"]),
                     width="stretch", hide_index=True)
    with right:
        rot = load_derived("sector_rotation_returns")
        if not rot.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rot["date"], y=rot["nav"], name="轮动策略",
                                     line=dict(color="#1f4e79", width=2.2)))
            fig.add_trace(go.Scatter(x=rot["date"], y=rot["nav_bench"], name="行业等权",
                                     line=dict(color="#8a8f98", width=1.6, dash="dash")))
            st.plotly_chart(fig_base(fig, 330, "月度轮动：策略净值 vs 行业等权"), width="stretch")
            m1 = metrics_from_returns(rot.set_index("date")["net"], periods=12)
            m2 = metrics_from_returns(rot.set_index("date")["bench"], periods=12)
            st.dataframe(pd.DataFrame([
                {"指标": "年化收益", "轮动策略": f"{m1['年化收益']*100:.1f}%", "行业等权": f"{m2['年化收益']*100:.1f}%"},
                {"指标": "年化波动", "轮动策略": f"{m1['年化波动']*100:.1f}%", "行业等权": f"{m2['年化波动']*100:.1f}%"},
                {"指标": "夏普比率", "轮动策略": f"{m1['夏普比率']:.2f}", "行业等权": f"{m2['夏普比率']:.2f}"},
                {"指标": "最大回撤", "轮动策略": f"{m1['最大回撤']*100:.1f}%", "行业等权": f"{m2['最大回撤']*100:.1f}%"},
            ]), hide_index=True, width="stretch")

    top = sec.sort_values("score", ascending=False).iloc[0]
    st.markdown(
        f"<div class='insight'>当前排名第一的是 <b>{top['name']}</b>（{top['score']:.0f} 分），"
        f"价格处于近5年 <b>{top['price_pct']*100:.0f}%</b> 分位、趋势评分 {top['trend']:.0f}。"
        f"轮动策略每月末按 12-1 动量持有前 5 名行业，且仅在沪深300站上200日线时持仓，"
        f"否则空仓；回测已扣除单边 10bp 交易成本。</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ page 3
def page_factors():
    hero("策略实验室", "用 8 个常见因子做一次体检：它们和未来涨跌到底有没有关系。",
         ["2019–2026", "791 只成分股", "月度调仓 20bp", "均值 vs 中位数"])
    explain("因子就是选股时看的一个特征，比如“便宜”（PE低）、“最近涨得好”（动量）、“波动小”。"
            "IC 是当月因子值和下个月涨跌的横截面相关性，越接近0越没用。分层图把股票按因子从低到高分成5组，看高分组是否真的跑赢。"
            "多空组合是买高分组、卖低分组；A股融券难，所以它只用来判断因子有没有效，不是实盘策略。")
    ic = load_derived("factor_ic")
    summ = load_derived("factor_summary")
    q = load_derived("factor_quantiles")
    qmed = load_derived("factor_quantiles_median")
    ls = load_derived("factor_ls_returns")
    lo = load_derived("factor_lo_returns")
    decay = load_derived("factor_decay")
    if ic.empty:
        st.warning("因子快照未生成，请运行 scripts/build_snapshots.py。")
        return

    comp = ic["score"].dropna()
    lo_m = metrics_from_returns(lo.set_index("date")["net"], periods=12) if not lo.empty else {}
    ew_m = metrics_from_returns(lo.set_index("date")["ew"], periods=12) if not lo.empty else {}
    ls_m = metrics_from_returns(ls.set_index("date")["net"], periods=12) if not ls.empty else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("综合因子 IC 均值", f"{comp.mean():.3f}", f"ICIR {comp.mean()/comp.std():.2f}")
    c2.metric("IC 正比例", f"{(comp > 0).mean()*100:.0f}%", "月度胜率")
    c3.metric("多头组合年化", f"{lo_m.get('年化收益', float('nan'))*100:.1f}%",
              f"夏普 {lo_m.get('夏普比率', float('nan')):.2f}")
    c4.metric("股票池等权年化", f"{ew_m.get('年化收益', float('nan'))*100:.1f}%",
              "真正的机会成本")

    left, right = st.columns([1.15, 1])
    with left:
        if not summ.empty:
            sv = summ[summ["因子"] != "综合"].copy()
            fig = go.Figure(go.Bar(x=sv["IC均值"], y=sv["因子"], orientation="h",
                                   marker_color=["#2e8b6f" if v > 0 else "#a33b4a" for v in sv["IC均值"]]))
            fig.update_xaxes(title="平均 IC（Spearman）")
            st.plotly_chart(fig_base(fig, 340, "各因子平均 IC"), width="stretch")
    with right:
        if not decay.empty:
            fig = go.Figure(go.Scatter(x=decay["horizon"], y=decay["ic"], mode="lines+markers",
                                       line=dict(color="#1f4e79", width=2), marker=dict(size=8)))
            fig.update_xaxes(title="预测期（月）", tickmode="array", tickvals=[1, 2, 3, 6, 12])
            fig.update_yaxes(title="IC")
            st.plotly_chart(fig_base(fig, 340, "综合因子 IC 衰减"), width="stretch")

    st.markdown("##### 因子 IC 统计")
    st.dataframe(table_style(summ, num_cols=["IC均值", "IC标准差", "ICIR", "t值", "正IC占比"]),
                 width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        yr = load_derived("factor_ic_yearly")
        if not yr.empty:
            yr = yr.set_index("year")
            cols = [c for c in yr.columns if c.startswith("z_")]
            mat = yr[cols].rename(columns={f"z_{k}": v for k, v in factors.FACTORS.items()}).T
            fig = px.imshow(mat, text_auto=".2f", color_continuous_scale="RdYlGn",
                            color_continuous_midpoint=0, aspect="auto", labels=dict(color="IC"))
            st.plotly_chart(fig_base(fig, 340, "因子 IC 的年度轮动"), width="stretch")
    with right:
        fc = load_derived("factor_correlation")
        if not fc.empty:
            fc = fc.set_index(fc.columns[0])
            fig = px.imshow(fc, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig_base(fig, 340, "因子横截面相关性"), width="stretch")

    left, right = st.columns(2)
    with left:
        if not q.empty:
            nav = (1 + q.set_index("date")).cumprod()
            fig = go.Figure()
            for i, c in enumerate(nav.columns):
                fig.add_trace(go.Scatter(x=nav.index, y=nav[c], name=c,
                                         line=dict(color=PALETTE[i], width=2 if c in ("Q5", "Q1") else 1.2)))
            fig.update_yaxes(title="累计净值")
            st.plotly_chart(fig_base(fig, 360, "五分层累计净值（按均值）"), width="stretch")
    with right:
        if not qmed.empty:
            nav = (1 + qmed.set_index("date")).cumprod()
            fig = go.Figure()
            for i, c in enumerate(nav.columns):
                fig.add_trace(go.Scatter(x=nav.index, y=nav[c], name=c,
                                         line=dict(color=PALETTE[i], width=2 if c in ("Q5", "Q1") else 1.2)))
            fig.update_yaxes(title="累计净值")
            st.plotly_chart(fig_base(fig, 360, "五分层累计净值（按中位数）"), width="stretch")

    if not lo.empty:
        nav_lo = (1 + lo.set_index("date")["net"]).cumprod()
        nav_ew = (1 + lo.set_index("date")["ew"].fillna(0)).cumprod()
        nav_bm = (1 + lo.set_index("date")["bench"].fillna(0)).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav_lo.index, y=nav_lo, name="多头前30",
                                 line=dict(color="#1f4e79", width=2.2)))
        fig.add_trace(go.Scatter(x=nav_ew.index, y=nav_ew, name="股票池等权",
                                 line=dict(color="#2e8b6f", width=1.8)))
        fig.add_trace(go.Scatter(x=nav_bm.index, y=nav_bm, name="沪深300",
                                 line=dict(color="#8a8f98", width=1.4, dash="dash")))
        fig.update_yaxes(title="累计净值")
        st.plotly_chart(fig_base(fig, 380, "多头组合 vs 等权基准 vs 沪深300"), width="stretch")

    st.markdown(
        f"<div class='insight'><b>体检结论：</b>综合因子的 IC 均值为 {comp.mean():.3f}，"
        f"但多头组合年化 {lo_m.get('年化收益',0)*100:.1f}%，低于股票池等权的 "
        f"{ew_m.get('年化收益',0)*100:.1f}%。原因是 2019–2026 年 A 股小市值风格极强："
        f"市值因子 IC 最高，而低波、价值、低换手在多头端均被小市值 Beta 压制；"
        f"多空组合（年化 {ls_m.get('年化收益',0)*100:.1f}%）则被空头端的极端上涨个股拖累，"
        f"且 A 股融券成本高、券源有限。<b>因此平台不把因子回测包装成交易策略，只把因子作为诊断维度。</b>"
        f"分层图也提示：按均值看 Q1 反而最高，按中位数看 Q5 才单调占优——等权组合的均值被少数极端收益主导。</div>",
        unsafe_allow_html=True)

    mfr = load_derived("factor_model_returns")
    msum = load_derived("factor_model_summary")
    st.markdown("#### 经典因子模型：FF3 / FF5 / 中国四因子")
    st.caption("每个因子是“做多高分组、做空低分组”的月度收益；市场因子是股票池等权收益减去无风险利率。未扣交易成本。")
    if not mfr.empty and not msum.empty:
        tabs = st.tabs(["FF3", "FF5", "中国四因子"])
        for tab, model in zip(tabs, ["FF3", "FF5", "CH4"]):
            cols = factors.MODEL_SETS[model]
            with tab:
                sub = msum[msum["因子"].isin(cols)].copy()
                sub["名称"] = sub["因子"].map(factors.MODEL_LABELS)
                sub = sub[["名称", "年化收益", "年化波动", "夏普", "t值", "月胜率"]]
                st.dataframe(table_style(sub, pct_cols=["年化收益", "年化波动", "月胜率"],
                                         num_cols=["夏普", "t值"]),
                             hide_index=True, width="stretch")
                nav = (1 + mfr[cols]).cumprod()
                fig = go.Figure()
                for i, c in enumerate(cols):
                    fig.add_trace(go.Scatter(x=nav.index, y=nav[c],
                                             name=factors.MODEL_LABELS.get(c, c),
                                             line=dict(color=PALETTE[i], width=1.8)))
                fig.update_yaxes(title="累计净值")
                st.plotly_chart(fig_base(fig, 320), width="stretch")
        st.caption("FF3=市场+小市值+价值；FF5 再加盈利(ROE)与低投资(资产增速)；中国四因子=市场+小市值+价值+低异常换手(PMO)。"
                   "t值绝对值大于 2 才大致可认为因子收益显著不为零。")

    st.caption("新闻情绪目前只有最近几个小时的快讯，无法构成 2019 年以来的历史面板，因此暂不放进因子收益与回归；"
               "它只作为实时盯盘和今日信号的辅助列，等积累足够历史后再检验显著性。")

    st.markdown("#### 自定义因子权重")
    st.caption("给每个因子一个权重，系统按权重合成打分，买入前 30 名，每月调仓，扣 20bp 单边成本。")
    defaults = {"MOM": 60, "REV": 30, "LOWVOL": 50, "SIZE": 40, "VALUE": 50,
                "LIQUID": 30, "TREND": 60, "LOTTERY": 20, "HML": 50,
                "RMW": 50, "CMA": 30, "PMO": 30, "HIGH52": 40}
    for k, v in defaults.items():
        st.session_state.setdefault(f"w_{k}", v)
    sliders = st.columns(4)
    weights = {}
    for i, k in enumerate(defaults):
        with sliders[i % 4]:
            weights[k] = st.slider(f"{factors.FACTORS[k].split('（')[0]} · {k}", 0, 100,
                                   key=f"w_{k}", step=5) / 100
    if sum(weights.values()) == 0:
        st.warning("权重全为 0，使用等权。")
        weights = {k: 1.0 for k in weights}
    bt = factors.custom_backtest(weights)
    r = bt["returns"]
    nav = (1 + r["net"]).cumprod()
    ew_nav = (1 + r["ew"].fillna(0)).cumprod()
    bench_nav = (1 + r["bench"].fillna(0)).cumprod()
    m = metrics_from_returns(r["net"], periods=12)
    me = metrics_from_returns(r["ew"], periods=12)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("自定义组合年化", f"{m['年化收益']*100:.1f}%", f"等权 {me['年化收益']*100:.1f}%")
    c2.metric("夏普", f"{m['夏普比率']:.2f}", f"等权 {me['夏普比率']:.2f}")
    c3.metric("最大回撤", f"{m['最大回撤']*100:.1f}%", f"等权 {me['最大回撤']*100:.1f}%")
    c4.metric("月胜率", f"{m['月胜率']*100:.0f}%", "多头前 30")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="自定义权重",
                             line=dict(color="#1f4e79", width=2.2)))
    fig.add_trace(go.Scatter(x=ew_nav.index, y=ew_nav, name="股票池等权",
                             line=dict(color="#2e8b6f", width=1.6)))
    fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name="沪深300",
                             line=dict(color="#8a8f98", width=1.3, dash="dash")))
    fig.update_yaxes(title="累计净值")
    st.plotly_chart(fig_base(fig, 340), width="stretch")
    st.caption("权重会改变因子暴露；如果提高某个权重后收益没有改善，说明这个因子在当前样本里没有提供增量信息。")

    st.markdown("##### 因子定义与方向")
    defs = pd.DataFrame([
        {"因子": "动量 MOM", "构造": "过去12个月收益，跳过最近1个月", "方向": "高动量占优"},
        {"因子": "短期反转 REV", "构造": "过去1个月收益的负值", "方向": "近期超跌占优"},
        {"因子": "低波动 LOWVOL", "构造": "过去60日年化波动率的负值", "方向": "低波占优"},
        {"因子": "小市值 SIZE", "构造": "流通市值的对数取负", "方向": "小市值占优"},
        {"因子": "价值 VALUE", "构造": "盈利收益率 E/P = 1/PE(TTM)", "方向": "低估值占优"},
        {"因子": "低换手 LIQUID", "构造": "过去20日平均换手率取负", "方向": "低换手占优"},
        {"因子": "趋势 TREND", "构造": "收盘价/200日均线 - 1", "方向": "强趋势占优"},
        {"因子": "低彩票 LOTTERY", "构造": "过去20日最大单日涨幅取负", "方向": "低彩票偏好占优"},
        {"因子": "价值 HML", "构造": "账面市值比 BM = 1/PB", "方向": "高 BM 占优"},
        {"因子": "盈利 RMW", "构造": "最近一期 ROE，按公告日后 120 天对齐", "方向": "高 ROE 占优"},
        {"因子": "低投资 CMA", "构造": "股本扩张速度取负（总资产增速缺失时的代理）", "方向": "低扩张占优"},
        {"因子": "低异常换手 PMO", "构造": "20日换手率相对252日均值的偏离取负", "方向": "低异常换手占优"},
        {"因子": "接近52周高 HIGH52", "构造": "收盘价 / 过去252日最高价", "方向": "接近高点占优"},
    ])
    st.dataframe(defs, hide_index=True, width="stretch")
    st.caption("注：股票池为当前沪深300成分股，回测存在幸存者偏差；因子在每月末做 2%/98% 缩尾并标准化；"
               "组合等权持有 30 只，单边成本 20bp。")


# ------------------------------------------------------------------ page 4
def page_allocation():
    hero("资产配置", "股票、债券、黄金、商品四类资产：风险平价、双动量、60/40 与衡策组合的实证对比。",
         ["2010–2026", "月度再平衡", "风险预算 + 动量择时 + 波动率目标"])
    nav = load_derived("strategy_nav")
    summ = load_derived("strategy_summary")
    w = load_derived("strategy_weights")
    if nav.empty:
        st.warning("配置快照未生成。")
        return
    nav = nav.set_index("date")

    fig = go.Figure()
    for i, c in enumerate(nav.columns):
        fig.add_trace(go.Scatter(x=nav.index, y=nav[c], name=c,
                                 line=dict(color=[ "#1f4e79", "#2e8b6f", "#c9a227", "#b35c2e"][i % 4],
                                           width=2.4 if c == "衡策组合" else 1.6)))
    fig.update_yaxes(title="累计净值（起点=1）")
    st.plotly_chart(fig_base(fig, 430, "四类配置策略：累计净值"), width="stretch")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("##### 策略绩效")
        if not summ.empty:
            st.dataframe(table_style(summ, pct_cols=["累计收益", "年化收益", "年化波动", "最大回撤", "月胜率"],
                                     num_cols=["夏普比率", "卡玛比率"]),
                         width="stretch", hide_index=True)
    with c2:
        st.markdown("##### 衡策组合最新权重")
        if not w.empty:
            last = w[w["date"] == w["date"].max()].sort_values("weight", ascending=False)
            fig = go.Figure(go.Bar(x=last["weight"], y=last["asset"], orientation="h",
                                   marker_color=[ASSET_COLORS.get(a, "#8a8f98") for a in last["asset"]]))
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_base(fig, 300), width="stretch")

    left, right = st.columns(2)
    with left:
        if not w.empty:
            piv = w.pivot(index="date", columns="asset", values="weight").fillna(0)
            fig = go.Figure()
            for a in piv.columns:
                fig.add_trace(go.Scatter(x=piv.index, y=piv[a], stackgroup="one", name=a,
                                         line=dict(width=0.4, color=ASSET_COLORS.get(a, "#8a8f98"))))
            fig.update_yaxes(tickformat=".0%", title="权重")
            st.plotly_chart(fig_base(fig, 360, "衡策组合权重演变"), width="stretch")
    with right:
        st.plotly_chart(drawdown_fig(nav["衡策组合"], "衡策组合", 360), width="stretch")

    left, right = st.columns(2)
    with left:
        corr = load_derived("asset_correlation")
        if not corr.empty:
            corr = corr.set_index(corr.columns[0])
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig_base(fig, 340, "跨资产月度收益相关性"), width="stretch")
    with right:
        reg = load_derived("regime_performance")
        if not reg.empty:
            reg = reg.set_index(reg.columns[0])
            cols = [c for c in reg.columns if c != "月数"]
            fig = px.imshow(reg[cols], text_auto=".0%", color_continuous_scale="RdYlGn",
                            aspect="auto", labels=dict(color="年化收益"))
            st.plotly_chart(fig_base(fig, 340, "宏观状态下的资产年化收益"), width="stretch")
            st.caption("状态划分：PMI 与 CPI 的月度趋势组合（复苏/过热/滞胀/衰退）。")


# ------------------------------------------------------------------ page 4b
def page_portfolio():
    hero("组合回测", "自己配股票、债券、黄金、商品的比例，看历史收益、回撤和极端行情下会亏多少。",
         ["60/40 基准", "月度再平衡", "风险贡献", "压力测试"])
    explain("拖动五个滑块设定比例，系统按月度再平衡回测。风险贡献告诉你组合的波动主要来自谁；"
            "压力测试看2015股灾、2018熊市、2020疫情等区间里，这个组合大概会亏多少。回测不含滑点和手续费。")
    rets = allocation.portfolio_monthly_returns()
    defaults = {"沪深300": 35, "中证500": 15, "债券": 30, "黄金": 10, "商品": 10}
    cols = st.columns(5)
    raw = {}
    for (name, default), col in zip(defaults.items(), cols):
        raw[name] = col.slider(name, 0, 100, default, 5, key=f"pf_{name}") / 100.0
    total = sum(raw.values())
    if total <= 0:
        weights = {k: 1 / len(raw) for k in raw}
        st.warning("权重全为 0，已自动切换为等权。")
    else:
        weights = {k: v / total for k, v in raw.items()}
    st.caption("有效权重：" + " · ".join(f"{k} {v*100:.0f}%" for k, v in weights.items()))

    bt = allocation.portfolio_backtest(weights)["returns"]
    bench = allocation.portfolio_backtest({"沪深300": 0.6, "债券": 0.4})["returns"]
    m = metrics_from_returns(bt["net"], periods=12)
    mb = metrics_from_returns(bench["net"], periods=12)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("年化收益", f"{m['年化收益']*100:.1f}%", f"60/40 {mb['年化收益']*100:.1f}%")
    c2.metric("年化波动", f"{m['年化波动']*100:.1f}%", f"60/40 {mb['年化波动']*100:.1f}%")
    c3.metric("夏普比率", f"{m['夏普比率']:.2f}", f"60/40 {mb['夏普比率']:.2f}")
    c4.metric("最大回撤", f"{m['最大回撤']*100:.1f}%", f"60/40 {mb['最大回撤']*100:.1f}%")
    c5.metric("累计收益", f"{m['累计收益']*100:.1f}%", f"{m['样本月数']} 个月")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["nav"], name="我的组合",
                             line=dict(color="#1f4e79", width=2.4)))
    fig.add_trace(go.Scatter(x=bench.index, y=bench["nav"], name="60/40",
                             line=dict(color="#8a8f98", width=1.6, dash="dash")))
    strat = load_derived("strategy_nav")
    if not strat.empty:
        strat = strat.set_index("date")
        fig.add_trace(go.Scatter(x=strat.index, y=strat["衡策组合"], name="衡策组合",
                                 line=dict(color="#2e8b6f", width=1.8)))
    fig.update_yaxes(title="累计净值（起点=1）")
    st.plotly_chart(fig_base(fig, 400, "自定义组合：累计净值"), width="stretch")

    left, right = st.columns(2)
    with left:
        rc = allocation.risk_contribution(weights).sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=rc.index, y=rc.values,
                               marker_color=[ASSET_COLORS.get(a, "#8a8f98") for a in rc.index]))
        fig.update_yaxes(tickformat=".0%", title="风险贡献占比")
        st.plotly_chart(fig_base(fig, 340, "风险贡献：谁在主导组合波动"), width="stretch")
        st.caption("风险贡献可大于权重，也可能为负；负值代表该资产在样本期内起到分散作用。")
    with right:
        corr = rets.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig_base(fig, 340, "五类资产月度收益相关性"), width="stretch")

    st.markdown("##### 历史压力测试（区间累计收益）")
    stress = allocation.stress_test(weights)
    if not stress.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stress["区间"], y=stress["组合"], name="我的组合",
                             marker_color="#1f4e79"))
        fig.add_trace(go.Bar(x=stress["区间"], y=stress["60/40"], name="60/40",
                             marker_color="#b8c1cc"))
        fig.add_trace(go.Scatter(x=stress["区间"], y=stress["沪深300"], name="沪深300",
                                 mode="markers", marker=dict(color="#a33b4a", size=9)))
        fig.update_yaxes(tickformat=".0%", title="区间累计收益")
        fig.update_layout(barmode="group")
        st.plotly_chart(fig_base(fig, 350), width="stretch")
        st.dataframe(table_style(stress, pct_cols=["组合", "60/40", "沪深300"]),
                     hide_index=True, width="stretch")

    top = rc.index[0] if not rc.empty else "—"
    st.markdown(
        f"<div class='insight'>你的组合年化 <b>{m['年化收益']*100:.1f}%</b>、夏普 <b>{m['夏普比率']:.2f}</b>、"
        f"最大回撤 <b>{m['最大回撤']*100:.1f}%</b>；风险贡献最大的是 <b>{top}</b>。"
        f"与 60/40 相比，夏普{'更高' if m['夏普比率'] > mb['夏普比率'] else '更低'}，"
        f"最大回撤{'更小' if abs(m['最大回撤']) < abs(mb['最大回撤']) else '更大'}。"
        f"压力测试显示，组合在 2015 股灾、2018 熊市与 2022–24 成长回撤中的表现，"
        f"取决于股票与商品的权重；债券权重是控制回撤的主要工具。</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ page 5
def page_commodities():
    hero("商品研究", "20 个国内商品主力合约 + 5 个海外基准：价格分位、库存周期与季节性。",
         ["库存数据", "10年季节性", "价格位置"])
    ct = load_derived("commodity_table")
    if ct.empty:
        st.warning("商品数据未就绪。")
        return
    if "score" not in ct.columns:
        sig_all = get_signals()
        comm_sig = sig_all[sig_all["type"] == "商品"][["symbol", "score", "action", "odds", "timing"]]
        ct = ct.merge(comm_sig, on="symbol", how="left")
    ct = ct.sort_values("score", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("覆盖品种", f"{len(ct)}", "国内主力连续")
    c2.metric("平均综合评分", f"{ct['score'].mean():.0f}", "评分越高赔率越好")
    c3.metric("最高分品种", ct.iloc[0]["name"], f"{ct.iloc[0]['score']:.0f} 分")

    show = ct[["name", "category", "last", "ret_1m", "ret_3m", "ret_12m", "price_pct_5y",
               "pct", "score", "action"]].rename(columns={
        "name": "品种", "category": "类别", "last": "最新价", "ret_1m": "近1月",
        "ret_3m": "近3月", "ret_12m": "近12月", "price_pct_5y": "价格分位",
        "pct": "库存分位", "score": "评分", "action": "结论"})
    st.dataframe(table_style(show, pct_cols=["近1月", "近3月", "近12月", "价格分位", "库存分位"],
                             num_cols=["最新价", "评分"]), width="stretch", hide_index=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(price_figure(data.load_futures(ct.iloc[0]["symbol"]),
                                     None, None, f"{ct.iloc[0]['name']} · 近三年"), width="stretch")
    with right:
        seas = load_derived("seasonality")
        if not seas.empty:
            mat = seas.set_index(seas.columns[0])
            fig = px.imshow(mat, text_auto=".1%", color_continuous_scale="RdYlGn",
                            aspect="auto", labels=dict(color="平均月收益"))
            st.plotly_chart(fig_base(fig, 430, "商品季节性（近10年平均月度收益）"), width="stretch")

    st.markdown("##### 库存—价格象限")
    q = ct.dropna(subset=["pct", "price_pct_5y"]).copy()
    if not q.empty:
        fig = go.Figure()
        for _, r in q.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["price_pct_5y"] * 100], y=[(1 - r["pct"]) * 100],
                mode="markers+text", text=[r["name"]], textposition="top center",
                marker=dict(size=14 + r["score"] * 0.2, color=ACTION_COLORS.get(r["action"], "#8a8f98"),
                            line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=f"{r['name']}<br>价格分位 %{{x:.0f}}%<br>库存紧张度 %{{y:.0f}}%<br>评分 {r['score']:.0f}"))
        fig.add_vline(x=50, line_dash="dash", line_color="#c9d3dc")
        fig.add_hline(y=50, line_dash="dash", line_color="#c9d3dc")
        fig.update_xaxes(title="价格分位（低=便宜）", range=[-2, 102])
        fig.update_yaxes(title="库存紧张度（高=低库存）", range=[-2, 104])
        st.plotly_chart(fig_base(fig, 440, "低价格 + 低库存 = 潜在做多窗口（右上角）"), width="stretch")

    st.markdown(
        f"<div class='insight'>综合评分最高的品种为 <b>{ct.iloc[0]['name']}</b>（{ct.iloc[0]['score']:.0f} 分），"
        f"价格处于近5年 {ct.iloc[0]['price_pct_5y']*100:.0f}% 分位；"
        f"库存分位 {ct.iloc[0]['pct']*100:.0f}% 表示当前库存高于历史 {ct.iloc[0]['pct']*100:.0f}% 的时间。"
        f"商品评分把“低价格、低库存、强趋势、低波动”放在同一量纲，仅用于横向比较，不构成交易建议。</div>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------ page 6
def page_signals():
    sig = get_signals()
    n_a = int((sig["type"] != "商品").sum())
    hero("今日信号", f"把 {n_a} 个 A 股标的按同一套规则排序：先看相对赔率，再决定关注什么、避开什么。",
         ["A股统一评分", "可下载 CSV", "每日更新"])
    explain("这是一张排序表，不是买入清单。分数由估值位置、趋势、动量、波动等维度合成；"
            "分数高只表示“相对更值得看”，还要结合资产诊断里的买入区间和止损。")
    if sig.empty:
        st.warning("信号表未生成。")
        return
    sig = sig.sort_values("score", ascending=False)
    f1, f2 = st.columns([1, 2])
    with f1:
        types = st.multiselect("资产类型", sorted(sig["type"].unique()),
                               default=[t for t in sorted(sig["type"].unique()) if t != "商品"])
    with f2:
        min_score = st.slider("最低评分", 0, 100, 0, 5)
    view = sig[sig["type"].isin(types) & (sig["score"] >= min_score)].copy()
    try:
        ndf = load_news()
        msnap = news.market_snapshot(ndf)
        smap = news.sector_sentiment(ndf).set_index("行业")["情绪分"].to_dict()
        view["新闻情绪"] = view.apply(
            lambda r: smap.get(r["name"], msnap["score"]) if r["type"] == "行业指数" else msnap["score"], axis=1)
    except Exception:
        view["新闻情绪"] = np.nan

    cols = st.columns(5)
    for i, (_, r) in enumerate(view.head(5).iterrows()):
        with cols[i]:
            st.markdown(card(r["name"], f"{action_badge(r['score'], r.get('action'), r.get('color'))}<br><span class='small'>评分 {r['score']:.0f} · {r['type']}</span>"),
                        unsafe_allow_html=True)

    st.markdown("##### 评分前 5：操作清单")
    plans = []
    for _, r in view.head(5).iterrows():
        sym = r["symbol"]
        pe = pd.Series(dtype=float)
        if r["type"] == "宽基指数":
            df = data.load_index(sym)
            pe_name = {"sh000300": "沪深300", "sh000905": "中证500",
                       "sh000016": "上证50"}.get(sym)
            if pe_name:
                pedf = data.load_index_valuation(pe_name, "pe")
                if not pedf.empty:
                    col = "滚动市盈率" if "滚动市盈率" in pedf.columns else pedf.columns[-1]
                    pe = pedf[col].dropna()
        elif r["type"] == "行业指数":
            df = data.load_sector(sym)
        else:
            df = data.load_futures(sym)
        if df.empty:
            continue
        buy = signals.buy_zone_from_pe(df["close"], pe) if not pe.empty else signals.buy_zone_from_price(df["close"])
        risk = signals.stop_level(df)
        mc = signals.monte_carlo(df["close"])
        plans.append({
            "标的": r["name"], "结论": r["action"], "评分": f"{r['score']:.0f}",
            "参考区间": f"{buy.get('buy_low', float('nan')):,.0f} – {buy.get('buy_high', float('nan')):,.0f}",
            "止损": f"{risk.get('stop', float('nan')):,.0f} ({risk.get('stop_pct', 0)*100:.1f}%)",
            "12月正收益概率": f"{mc.get('prob_up', float('nan'))*100:.0f}%",
            "12月中位收益": f"{mc.get('p50', float('nan'))*100:+.1f}%",
        })
    if plans:
        st.dataframe(pd.DataFrame(plans), hide_index=True, width="stretch")
        st.caption("参考区间：有 PE 历史时按 30–55 分位估值映射，否则按近5年价格分位；止损为 2×ATR；"
                   "概率来自 3 年日收益的块状 bootstrap（2000 条路径）。")

    show = view[["name", "type", "price", "score", "action", "新闻情绪", "ret_1m", "ret_12m"]].copy()
    show.columns = ["标的", "类型", "最新价", "综合评分", "结论", "新闻情绪", "近1月", "近12月"]
    st.dataframe(table_style(show, pct_cols=["近1月", "近12月"], num_cols=["最新价", "综合评分", "新闻情绪"]),
                 width="stretch", hide_index=True, height=520)
    st.download_button("下载完整信号表 (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                       "hengce_signals.csv", "text/csv")

    top = view.head(3)
    weak = view.tail(3)
    st.markdown(
        f"<div class='insight'><b>组合视角：</b>评分前三是 "
        + "、".join(f"{r['name']}（{r['score']:.0f}）" for _, r in top.iterrows())
        + "；评分后三是 "
        + "、".join(f"{r['name']}（{r['score']:.0f}）" for _, r in weak.iterrows())
        + "。若把评分前 20% 作为观察池、后 20% 作为回避池，可显著降低组合的估值与趋势错配；"
          "具体买卖点请回到“资产诊断”查看估值区间与 ATR 止损。</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ page 7
def page_about():
    hero("方法与部署", "平台的架构、数据口径、复现步骤与云端分享方式。",
         ["Streamlit", "公开数据", "离线快照", "一键部署"])
    st.markdown("#### 四个差异化设计")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(card("赔率 × 胜率", "不用一个分数掩盖矛盾：估值回答“能赚多少”，趋势回答“现在买会不会挨打”，再映射到六种动作。"),
                    unsafe_allow_html=True)
    with f2:
        st.markdown(card("A股同一把尺", "791 只沪深300/中证500成分股、31 个申万行业与全 A 代码搜索，共用估值、趋势、动量、风险四维框架。"),
                    unsafe_allow_html=True)
    with f3:
        st.markdown(card("概率而非点预测", "历史条件分布 + 块状 bootstrap 蒙特卡洛，给出 P5/P50/P95 与正收益概率，不给单一目标价。"),
                    unsafe_allow_html=True)
    with f4:
        st.markdown(card("可复现、可分享", "数据快照随仓库分发，断网可跑；Streamlit Cloud / Hugging Face / Docker 部署，另有手动开启的 30 秒实时盯盘。"),
                    unsafe_allow_html=True)

    st.markdown("#### 页面地图：每一页解决什么问题")
    page_map = pd.DataFrame([
        {"页面": "实时盯盘", "解决什么问题": "今天有没有可执行的方向？策略最近顺不顺？", "怎么用": "手动打开开关，看信号和模拟盈亏；不用一直开着。"},
        {"页面": "A股总览", "解决什么问题": "现在整体贵不贵、趋势强不强、新闻冷不冷？", "怎么用": "先看温度、PE分位和ERP，再决定仓位。"},
        {"页面": "新闻情绪", "解决什么问题": "新闻措辞偏利好还是偏利空？哪个行业被讨论最多？", "怎么用": "当辅助指标，不要单独据此买卖。"},
        {"页面": "个股诊断", "解决什么问题": "这只股票相对自己的历史贵不贵？趋势坏没坏？", "怎么用": "输入代码，看估值区间、止损和概率分布。"},
        {"页面": "财务指标", "解决什么问题": "公司赚不赚钱、负债高不高、成长快不快？", "怎么用": "看单只股票的季报趋势，或按 ROE/负债率筛选。"},
        {"页面": "行业轮动", "解决什么问题": "哪些行业便宜且正在走强？轮动规则过去有效吗？", "怎么用": "看四象限位置和回测净值。"},
        {"页面": "策略实验室", "解决什么问题": "因子（选股特征）到底有没有预测力？", "怎么用": "看IC、年度轮动和分层，不要只看一条净值曲线。"},
        {"页面": "组合回测", "解决什么问题": "给定配置比例，历史上会赚多少、极端行情会亏多少？", "怎么用": "拖滑块，观察风险贡献和压力测试。"},
        {"页面": "今日信号", "解决什么问题": "把所有A股标的放在同一把尺上排序。", "怎么用": "当观察池，不是买入清单。"},
        {"页面": "扩展研究", "解决什么问题": "商品、债券、黄金对A股环境有什么参考？", "怎么用": "可选，不影响主线。"},
    ])
    st.dataframe(page_map, hide_index=True, width="stretch")

    st.markdown("#### 名词解释（给第一次接触量化的同学）")
    glossary = pd.DataFrame([
        {"名词": "PE 分位", "解释": "当前市盈率在近 8 年里，有多少时间比现在更低。80% 表示比历史上大多数时候贵。"},
        {"名词": "赔率 / 胜率", "解释": "赔率看“便宜不便宜”，胜率看“趋势强不强”。两者都高才叫好机会。"},
        {"名词": "因子", "解释": "选股时看的一个特征，比如便宜、最近涨得好、波动小、市值小。"},
        {"名词": "IC", "解释": "当月因子值和下个月涨跌的相关性。0.03 已经算弱相关，0.1 以上很少见。"},
        {"名词": "分层回测", "解释": "按因子把股票分成 5 组，看高分组的收益是不是真的比低分组高。"},
        {"名词": "多空组合", "解释": "买高分组、卖低分组。A 股融券难，所以只用来判断因子有没有效。"},
        {"名词": "夏普比率", "解释": "每承担 1 单位波动，能换来多少超额收益。越高越划算。"},
        {"名词": "最大回撤", "解释": "从历史最高点跌到最低点的最大幅度。10% 就是最多亏过一成。"},
        {"名词": "ATR", "解释": "最近 14 天的平均真实波幅，用来估计一天大概能波动多少，常用来设止损。"},
        {"名词": "蒙特卡洛", "解释": "把过去几年的日涨跌随机打乱，拼出 2000 条未来一年的路径，看收益大概落在什么区间。"},
        {"名词": "ERP", "解释": "股票盈利收益率减去 10 年国债收益率。越高说明股票相对债券越便宜。"},
        {"名词": "新闻情绪", "解释": "用利好/利空词表给新闻标题打分，只反映措辞方向，不判断真假。"},
    ])
    st.dataframe(glossary, hide_index=True, width="stretch")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("""
#### 系统架构
```
数据层   akshare / 中证指数 / 新浪财经 / 东方财富 / 巨潮
          │ 每日增量抓取 → CSV 快照（随仓库分发）
计算层   pandas + numpy + scipy
          │ 指标 · 因子 · 回测 · 风险平价 · 蒙特卡洛
展示层   Streamlit + Plotly（交互）
          │
交付     PDF 报告 + 可部署 Web 应用 + 一键复现脚本
```
#### 复现步骤
```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # 抓取公开数据（约 5–10 分钟）
python scripts/build_snapshots.py     # 预计算因子与回测快照
streamlit run app.py                  # 本地打开 http://localhost:8501
```
#### 实时盯盘（可选）
同花顺 iFinD 需要本机安装终端与 Python 接口。账号只写入 `.streamlit/secrets.toml`（已在 `.gitignore` 中），不会进入仓库：
```toml
THS_ACCOUNT = "your_account"
THS_PASSWORD = "your_password"
```
未配置或登录失败时，自动降级为新浪公开行情；每次刷新只发 1 次批量请求，默认 30 秒，手动启动。

#### 云端分享
1. 把仓库推到 GitHub；
2. 在 share.streamlit.io 选择仓库、分支与 `app.py`；
3. 无需数据库与密钥，`data/` 内已含离线快照，冷启动即可用；
4. 也可部署到 Hugging Face Spaces（SDK: streamlit）或任意支持 Python 的容器平台。
""")
        st.markdown("""
#### 设计取舍
- **不做高频**：使用日频与月度数据，规避微观结构噪声，聚焦配置与选股；
- **不做黑箱**：所有评分均可拆解为估值、趋势、动量、风险四个维度；
- **不做单点预测**：用历史条件分布与蒙特卡洛给出区间和概率，而非“目标价”；
- **可离线**：数据快照与代码同仓，断网也能复现报告。
""")
    with right:
        st.markdown("""
#### 数据口径
| 数据 | 来源 | 频率 | 用途 |
|---|---|---|---|
| A股/指数行情 | 东方财富、新浪、同花顺 iFinD | 日/实时 | 趋势、动量、回测、盯盘 |
| 沪深300/中证500成分 | 中证指数 | 月度 | 深度股票池（791只可用） |
| PE/PB/市值 | 东方财富 | 日 | 估值、规模因子 |
| 财务指标 | 新浪财经 | 季度 | 质量评分 |
| 全A股代码 | 沪深交易所 | 日 | 5579 只代码搜索 |
| 商品库存 | 东方财富 | 日 | 库存周期 |
| 国债收益率 | 中债 | 日 | 债券收益、股债性价比 |
| PMI/CPI/PPI/M2 | 国家统计局 | 月 | 宏观状态 |
| 财经快讯 | 东方财富 7×24 | 实时 | 新闻情绪指标 |

#### 已知局限
- 股票池为**当前**沪深300成分，存在幸存者偏差；
- 商品使用主力连续合约，换月点存在跳空；
- 债券收益为久期近似，未计凸性与信用利差；
- 估值分位基于历史，不代表未来中枢不变。
""")
    st.warning(DISCLAIMER)


# ------------------------------------------------------------------ sidebar
def page_extension():
    st.markdown("### 扩展研究：跨资产环境参考")
    st.caption("以下内容保留用于观察 A 股之外的环境变量，不属于平台主线；主线是 A 股盯盘与策略。")
    tab1, tab2 = st.tabs(["跨资产配置", "商品周期"])
    with tab1:
        page_allocation()
    with tab2:
        page_commodities()


def main():
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 4px 18px">
          <div style="font-size:24px;font-weight:600;letter-spacing:2px;color:#111827">衡策</div>
          <div style="font-size:11px;letter-spacing:3px;color:#6b7280">HENGCE QUANT</div>
          <div style="font-size:12px;color:#6b7280;margin-top:8px">A股量化盯盘与策略</div>
        </div>
        """, unsafe_allow_html=True)
        nav = ["实时盯盘", "A股总览", "新闻情绪", "个股诊断", "财务指标",
               "行业轮动", "策略实验室", "组合回测", "今日信号", "扩展研究", "方法与部署"]
        keys = {"live": 0, "overview": 1, "news": 2, "xray": 3, "finance": 4,
                "sectors": 5, "factors": 6, "portfolio": 7,
                "signals": 8, "extension": 9, "about": 10}
        if "nav" not in st.session_state:
            qp = st.query_params.get("page", "")
            idx = keys.get(qp, nav.index(qp) if qp in nav else 0)
            st.session_state["nav"] = nav[idx]
        page = st.radio("导航", nav, key="nav", label_visibility="collapsed")
        st.markdown("---")
        meta = data.data_freshness()
        st.caption(f"数据截至 {meta.get('fetched_at','—')[:10]}")
        bm = load_json("build_meta.json")
        if bm:
            st.caption(f"股票池 {bm.get('stocks','—')} 只 · 因子样本 {bm.get('start','')} 至 {bm.get('end','')}")
        st.caption("行情：同花顺 iFinD / 新浪 · 数据：akshare")
        st.caption("仅供课程研究，不构成投资建议")

    pages = {"A股总览": page_overview, "实时盯盘": page_live, "个股诊断": page_xray,
             "行业轮动": page_sectors, "策略实验室": page_factors,
             "组合回测": page_portfolio, "今日信号": page_signals,
             "新闻情绪": page_news, "财务指标": page_finance,
             "扩展研究": page_extension, "方法与部署": page_about}
    pages[page]()


if __name__ == "__main__":
    main()
