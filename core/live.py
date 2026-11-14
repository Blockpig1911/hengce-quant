"""A-share realtime quotes: iFinD (preferred) with Sina fallback.

Credentials are read from environment variables first, then from the local
(uncommitted) .streamlit/secrets.toml. They are never written into the repo.
"""
from __future__ import annotations

import os
import threading
import tomllib
from datetime import datetime, time as dtime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_THS_READY: bool | None = None

WATCH_POOL = {
    "000300.SH": ("沪深300", "指数"),
    "000905.SH": ("中证500", "指数"),
    "399006.SZ": ("创业板指", "指数"),
    "000001.SH": ("上证指数", "指数"),
    "000688.SH": ("科创50", "指数"),
    "399001.SZ": ("深证成指", "指数"),
    "000016.SH": ("上证50", "指数"),
    "000852.SH": ("中证1000", "指数"),
    "899050.BJ": ("北证50", "指数"),
    "000985.CSI": ("中证全指", "指数"),
    "600519.SH": ("贵州茅台", "个股"),
    "300750.SZ": ("宁德时代", "个股"),
    "601318.SH": ("中国平安", "个股"),
    "000858.SZ": ("五粮液", "个股"),
    "002594.SZ": ("比亚迪", "个股"),
    "600036.SH": ("招商银行", "个股"),
    "601899.SH": ("紫金矿业", "个股"),
    "600900.SH": ("长江电力", "个股"),
}

DEFAULT_WATCH = ["000300.SH", "000905.SH", "399006.SZ", "000016.SH",
                 "600519.SH", "300750.SZ", "601318.SH"]


def _secret(name: str) -> str:
    val = os.getenv(name)
    if val:
        return val
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            return str(data.get(name, ""))
        except Exception:
            return ""
    return ""


def to_ths_code(code: str) -> str:
    if "." in code:
        return code.upper()
    c = code.lower()
    if c.startswith(("sh", "sz", "bj")):
        return f"{c[2:]}.{c[:2].upper()}"
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    return f"{code}.BJ"


def to_sina_code(code: str) -> str:
    if "." in code:
        num, ex = code.split(".")
        return f"{ex.lower()}{num}"
    c = code.lower()
    if c.startswith(("sh", "sz", "bj")):
        return c
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def ths_login(force: bool = False) -> bool:
    global _THS_READY
    with _LOCK:
        if _THS_READY is not None and not force:
            return _THS_READY
        try:
            import iFinDPy as ths
        except Exception:
            _THS_READY = False
            return False
        user, pwd = _secret("THS_ACCOUNT"), _secret("THS_PASSWORD")
        if not user or not pwd:
            _THS_READY = False
            return False
        try:
            _THS_READY = ths.THS_iFinDLogin(user, pwd) in (0, "0")
        except Exception:
            _THS_READY = False
        return bool(_THS_READY)


def ths_quotes(codes: list[str]) -> pd.DataFrame:
    import iFinDPy as ths

    joined = ",".join(to_ths_code(c) for c in codes)
    res = ths.THS_RealtimeQuotes(
        joined, "latest;preClose;changeRatio;open;high;low;volume;amount", "")
    if res.get("errorcode") != 0:
        raise RuntimeError(res.get("errmsg", "THS error"))
    rows = []
    for table in res.get("tables", []):
        code = table["thscode"]
        t = table["table"]
        rows.append({
            "code": code,
            "name": WATCH_POOL.get(code, (all_stock_names().get(code.split(".")[0], code), ""))[0],
            "price": _first(t.get("latest")),
            "pre_close": _first(t.get("preClose")),
            "open": _first(t.get("open")),
            "high": _first(t.get("high")),
            "low": _first(t.get("low")),
            "pct_chg": _first(t.get("changeRatio")) / 100.0,
            "volume": _first(t.get("volume")),
            "amount": _first(t.get("amount")),
            "time": (table.get("time") or [""])[0],
        })
    df = pd.DataFrame(rows)
    df["source"] = "同花顺 iFinD"
    return df


@lru_cache(maxsize=1)
def all_stock_names() -> dict[str, str]:
    path = ROOT / "data" / "meta" / "all_stocks.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return dict(zip(df["code"], df["name"]))


def _first(v):
    if isinstance(v, (list, tuple)) and v:
        return float(v[0])
    if v is None:
        return float("nan")
    return float(v)


def sina_quotes(codes: list[str]) -> pd.DataFrame:
    url = "https://hq.sinajs.cn/list=" + ",".join(to_sina_code(c) for c in codes)
    r = requests.get(url, headers={
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0",
    }, timeout=10)
    r.encoding = "gbk"
    rows = []
    for line in r.text.split(";"):
        if "hq_str_" not in line or "=" not in line:
            continue
        key = line.split("hq_str_")[1].split("=")[0].strip()
        val = line.split('="')[1].rstrip('"')
        f = val.split(",")
        if len(f) < 10:
            continue
        try:
            price, pre = float(f[3]), float(f[2])
            rows.append({
                "code": to_ths_code(key),
                "name": f[0],
                "price": price,
                "pre_close": pre,
                "open": float(f[1]),
                "high": float(f[4]),
                "low": float(f[5]),
                "pct_chg": price / pre - 1 if pre else float("nan"),
                "volume": float(f[8]),
                "amount": float(f[9]),
                "time": f"{f[30]} {f[31]}" if len(f) > 31 else "",
            })
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["source"] = "新浪财经（降级）"
    return df


def fetch_quotes(codes: list[str], prefer: str = "ths") -> pd.DataFrame:
    if prefer == "ths" and ths_login():
        try:
            return ths_quotes(codes)
        except Exception:
            pass
    return sina_quotes(codes)


def market_status(now: datetime | None = None) -> tuple[str, bool]:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return "休市（周末）", False
    t = now.time()
    if dtime(9, 30) <= t <= dtime(11, 30) or dtime(13, 0) <= t <= dtime(15, 0):
        return "交易中", True
    if t < dtime(9, 30):
        return "盘前", False
    if dtime(11, 30) < t < dtime(13, 0):
        return "午间休市", False
    return "已收盘", False


def direction_intraday(pct_chg: float, price: float, high: float, low: float,
                       pre_close: float) -> tuple[str, str]:
    """Fallback for stocks without cached daily history: intraday momentum only."""
    rng = max(high - low, 1e-9)
    pos = (price - low) / rng
    if pct_chg >= 0.02 and pos >= 0.7:
        return "买入", "盘中放量走强（仅盘中信号）"
    if pct_chg <= -0.03:
        return "减仓", "盘中急跌（仅盘中信号）"
    if pct_chg < -0.01 and pos < 0.3:
        return "回避", "盘中弱势（仅盘中信号）"
    return "观望", "无深度历史，仅参考盘中"


INDEX_CODES = {"000300", "000905", "000852", "000016",
               "000688", "000985", "000922", "399006", "399001", "899050"}


def resolve(text: str) -> tuple[str, str]:
    """Resolve a stock/index code or Chinese name to a canonical THS code."""
    t = str(text).strip().replace(" ", "")
    if not t:
        return "", ""
    upper = t.upper()
    if upper in WATCH_POOL:
        return upper, WATCH_POOL[upper][0]
    for c, (n, _) in WATCH_POOL.items():
        if n == t:
            return c, n
    if "." in upper:
        return upper, WATCH_POOL.get(upper, (upper, ""))[0]
    if t in INDEX_CODES:
        suffix = "BJ" if t.startswith("899") else ("SZ" if t.startswith("399") else ("CSI" if t == "000985" else "SH"))
        code = f"{t}.{suffix}"
        return code, WATCH_POOL.get(code, (code, ""))[0]
    names = all_stock_names()
    for c, n in names.items():
        if n == t:
            return to_ths_code(c), n
    if t.isdigit() and len(t) == 6:
        return to_ths_code(t), names.get(t, t)
    return "", ""


def direction_for(score: float, pct_chg: float) -> tuple[str, str]:
    """Translate a daily score + intraday move into an A-share action."""
    if pd.isna(score):
        return "观望", "评分缺失"
    if score >= 65 and pct_chg > -0.02:
        return "买入", "日线评分高，盘中未破位"
    if score >= 55 and pct_chg > 0:
        return "加仓", "评分与盘中方向一致"
    if score < 35 or pct_chg <= -0.04:
        return "减仓", "评分偏低或盘中急跌"
    if score < 45:
        return "回避", "赔率与趋势均弱"
    return "观望", "信号不明确"
