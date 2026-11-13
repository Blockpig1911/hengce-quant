"""Fetch public market data via akshare and cache it under ../data.

Run:  python scripts/fetch_data.py [--workers 6] [--skip-stocks]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.config import SECTOR_META  # noqa: E402
DATA = ROOT / "data"
START = "20150101"
END = datetime.now().strftime("%Y%m%d")

for sub in ["indices", "sectors", "stocks/price", "stocks/value", "stocks/fin",
            "futures", "inventory", "macro", "meta"]:
    (DATA / sub).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- universe
INDEXES = [
    ("sh000001", "上证指数", "A股宽基"),
    ("sh000300", "沪深300", "A股宽基"),
    ("sh000905", "中证500", "A股宽基"),
    ("sh000852", "中证1000", "A股宽基"),
    ("sh000016", "上证50", "A股宽基"),
    ("sz399006", "创业板指", "A股宽基"),
    ("sh000688", "科创50", "A股宽基"),
    ("sz399001", "深证成指", "A股宽基"),
]
SECTORS = [(code, name) for code, (name, _) in SECTOR_META.items()]
FUTURES = [
    ("AU0", "沪金", "贵金属", "沪金"),
    ("AG0", "沪银", "贵金属", "沪银"),
    ("CU0", "沪铜", "工业金属", "沪铜"),
    ("AL0", "沪铝", "工业金属", "沪铝"),
    ("ZN0", "沪锌", "工业金属", "沪锌"),
    ("RB0", "螺纹钢", "黑色系", "螺纹钢"),
    ("I0", "铁矿石", "黑色系", "铁矿石"),
    ("J0", "焦炭", "黑色系", "焦炭"),
    ("JM0", "焦煤", "黑色系", "焦煤"),
    ("SC0", "原油", "能化", None),
    ("TA0", "PTA", "能化", "PTA"),
    ("MA0", "甲醇", "能化", "甲醇"),
    ("FG0", "玻璃", "能化", "玻璃"),
    ("SA0", "纯碱", "能化", "纯碱"),
    ("M0", "豆粕", "农产品", "豆粕"),
    ("Y0", "豆油", "农产品", "豆油"),
    ("P0", "棕榈油", "农产品", None),
    ("SR0", "白糖", "农产品", "白糖"),
    ("CF0", "棉花", "农产品", None),
    ("AP0", "苹果", "农产品", "苹果"),
]
FOREIGN = [("CL", "WTI原油"), ("GC", "COMEX黄金"), ("HG", "COMEX铜"), ("C", "CBOT玉米"), ("S", "CBOT大豆")]

STOCK_COLS = {"日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close", "最高": "high",
              "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover",
              "涨跌幅": "pct_chg"}
VALUE_COLS = {"数据日期": "date", "当日收盘价": "close", "当日涨跌幅": "pct_chg", "总市值": "mcap",
              "流通市值": "float_mcap", "总股本": "shares", "流通股本": "float_shares",
              "PE(TTM)": "pe_ttm", "PE(静)": "pe_static", "市净率": "pb", "PEG值": "peg",
              "市现率": "pcf", "市销率": "ps"}


def _retry(fn, tries=4, wait=1.2):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 network flakiness
            last = exc
            time.sleep(wait * (i + 1))
    raise last


def _write(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fetch_index(symbol: str) -> pd.DataFrame:
    df = _retry(lambda: ak.stock_zh_index_daily(symbol=symbol))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def fetch_sector(symbol: str) -> pd.DataFrame:
    df = _retry(lambda: ak.index_hist_sw(symbol=symbol, period="day"))
    df = df.rename(columns={"日期": "date", "收盘": "close", "开盘": "open",
                            "最高": "high", "最低": "low", "成交量": "volume",
                            "成交额": "amount"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def fetch_stock_price(code: str) -> pd.DataFrame:
    def east():
        return ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START,
                                  end_date=END, adjust="qfq")

    try:
        df = _retry(east, tries=3)
        df = df.rename(columns=STOCK_COLS)
        keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"] if c in df]
        df = df[keep]
    except Exception:
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        df = _retry(lambda: ak.stock_zh_a_daily(symbol=f"{prefix}{code}", start_date=START,
                                                end_date=END, adjust="qfq"))
        df = df.rename(columns={"turnover": "turnover"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def fetch_stock_value(code: str) -> pd.DataFrame:
    df = _retry(lambda: ak.stock_value_em(symbol=code))
    df = df.rename(columns=VALUE_COLS)
    df["date"] = pd.to_datetime(df["date"])
    cols = ["date", "close", "pct_chg", "mcap", "float_mcap", "shares", "float_shares",
            "pe_ttm", "pe_static", "pb", "peg", "pcf", "ps"]
    return df[[c for c in cols if c in df]].sort_values("date")


FIN_KEYS = {
    "净资产收益率(%)": "roe",
    "销售毛利率(%)": "gross_margin",
    "销售净利率(%)": "net_margin",
    "资产负债率(%)": "debt_ratio",
    "主营业务收入增长率(%)": "rev_growth",
    "净利润增长率(%)": "profit_growth",
    "流动比率": "current_ratio",
    "速动比率": "quick_ratio",
}


def fetch_stock_fin(code: str) -> dict:
    df = _retry(lambda: ak.stock_financial_analysis_indicator(symbol=code, start_year="2019"))
    if df.empty:
        return {"code": code}
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    latest = df.iloc[-1]
    out = {"code": code, "report_date": str(latest["日期"].date())}
    for cn, en in FIN_KEYS.items():
        if cn in df.columns:
            val = pd.to_numeric(latest[cn], errors="coerce")
            out[en] = None if pd.isna(val) else float(val)
    return out


def fetch_future(symbol: str) -> pd.DataFrame:
    df = _retry(lambda: ak.futures_main_sina(symbol=symbol))
    df = df.rename(columns={"日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low",
                            "收盘价": "close", "成交量": "volume", "持仓量": "open_interest",
                            "动态结算价": "settle"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def fetch_foreign(symbol: str) -> pd.DataFrame:
    df = _retry(lambda: ak.futures_foreign_hist(symbol=symbol))
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


def fetch_inventory(name: str) -> pd.DataFrame:
    df = _retry(lambda: ak.futures_inventory_em(symbol=name))
    df = df.rename(columns={"日期": "date", "库存": "inventory", "增减": "change"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-stocks", action="store_true")
    ap.add_argument("--stocks-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    meta = {"fetched_at": datetime.now().isoformat(timespec="seconds"), "errors": []}

    if args.stocks_only:
        pass
    else:
      # indices + sectors
      for symbol, name, group in INDEXES:
        try:
            df = fetch_index(symbol)
            _write(df, DATA / "indices" / f"{symbol}.csv")
            print(f"[index] {name} {len(df)}")
        except Exception as exc:
            meta["errors"].append(f"index {symbol}: {exc}")
    for code, name in SECTORS:
        try:
            df = fetch_sector(code)
            _write(df, DATA / "sectors" / f"{code}.csv")
            print(f"[sector] {name} {len(df)}")
        except Exception as exc:
            meta["errors"].append(f"sector {code}: {exc}")

    # index valuation (monthly, long history)
    for sym in ["上证50", "沪深300", "中证500", "上证180", "深证成指"]:
        try:
            pe = _retry(lambda s=sym: ak.stock_index_pe_lg(symbol=s))
            pb = _retry(lambda s=sym: ak.stock_index_pb_lg(symbol=s))
            _write(pe, DATA / "indices" / f"pe_{sym}.csv")
            _write(pb, DATA / "indices" / f"pb_{sym}.csv")
            print(f"[valuation] {sym} {len(pe)}")
        except Exception as exc:
            meta["errors"].append(f"valuation {sym}: {exc}")

    # commodities
    for symbol, name, cat, inv in FUTURES:
        try:
            df = fetch_future(symbol)
            if len(df) < 200:
                continue
            _write(df, DATA / "futures" / f"{symbol}.csv")
            print(f"[futures] {name} {len(df)}")
        except Exception as exc:
            meta["errors"].append(f"futures {symbol}: {exc}")
        if inv:
            try:
                d = fetch_inventory(inv)
                _write(d, DATA / "inventory" / f"{symbol}.csv")
                print(f"[inventory] {name} {len(d)}")
            except Exception as exc:
                meta["errors"].append(f"inventory {symbol}: {exc}")
    for symbol, name in FOREIGN:
        try:
            df = fetch_foreign(symbol)
            _write(df, DATA / "futures" / f"X{symbol}.csv")
            print(f"[foreign] {name} {len(df)}")
        except Exception as exc:
            meta["errors"].append(f"foreign {symbol}: {exc}")

    # macro / rates
    macro_jobs = {
        "bond_yield": lambda: ak.bond_zh_us_rate(start_date="20050101"),
        "cpi_yoy": lambda: ak.macro_china_cpi(),
        "ppi_yoy": lambda: ak.macro_china_ppi_yearly(),
        "pmi": lambda: ak.macro_china_pmi(),
        "m2": lambda: ak.macro_china_money_supply(),
        "social_financing": lambda: ak.macro_china_shrzgm(),
        "margin_sse": lambda: ak.stock_margin_sse(start_date="20150101", end_date=END),
        "usdcny": lambda: ak.currency_boc_sina(symbol="美元", start_date="20150101", end_date=END),
    }
    for name, fn in macro_jobs.items():
        try:
            df = _retry(fn, tries=3)
            _write(df, DATA / "macro" / f"{name}.csv")
            print(f"[macro] {name} {len(df)}")
        except Exception as exc:
            meta["errors"].append(f"macro {name}: {exc}")

    # stocks
    if not args.skip_stocks:
        try:
            frames = []
            for idx in ["000300", "000905"]:
                c = _retry(lambda i=idx: ak.index_stock_cons_csindex(symbol=i))
                c = c.rename(columns={"成分券代码": "code", "成分券名称": "name",
                                      "日期": "as_of"})
                c["index"] = idx
                frames.append(c)
            cons = pd.concat(frames, ignore_index=True)
            cons["code"] = cons["code"].astype(str).str.zfill(6)
            cons = cons.drop_duplicates("code")
            cons[["code", "name", "as_of", "index"]].to_csv(
                DATA / "meta" / "universe.csv", index=False, encoding="utf-8-sig")
            codes = cons["code"].tolist()
            print(f"[universe] CSI300+500 {len(codes)}")
            allsh = _retry(lambda: ak.stock_info_a_code_name())
            allsh = allsh.rename(columns={"code": "code", "name": "name"})
            allsh["code"] = allsh["code"].astype(str).str.zfill(6)
            allsh.drop_duplicates("code").to_csv(
                DATA / "meta" / "all_stocks.csv", index=False, encoding="utf-8-sig")
            print(f"[universe] all A-share {len(allsh)}")
        except Exception as exc:
            meta["errors"].append(f"universe: {exc}")
            codes = []

        def job(code):
            res = {"code": code}
            price_p = DATA / "stocks" / "price" / f"{code}.parquet"
            price_c = DATA / "stocks" / "price" / f"{code}.csv"
            value_p = DATA / "stocks" / "value" / f"{code}.parquet"
            value_c = DATA / "stocks" / "value" / f"{code}.csv"
            fin_p = DATA / "stocks" / "fin" / f"{code}.json"
            if not (price_p.exists() or price_c.exists()):
                try:
                    p = fetch_stock_price(code)
                    _write(p, DATA / "stocks" / "price" / f"{code}.csv")
                    res["price"] = len(p)
                except Exception as exc:
                    res["price_error"] = str(exc)[:80]
            if not (value_p.exists() or value_c.exists()):
                try:
                    v = fetch_stock_value(code)
                    _write(v, DATA / "stocks" / "value" / f"{code}.csv")
                    res["value"] = len(v)
                except Exception as exc:
                    res["value_error"] = str(exc)[:80]
            if not fin_p.exists():
                try:
                    f = fetch_stock_fin(code)
                    fin_p.write_text(json.dumps(f, ensure_ascii=False, indent=1), encoding="utf-8")
                except Exception as exc:
                    res["fin_error"] = str(exc)[:80]
            return res

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(job, c): c for c in codes}
            for fut in as_completed(futures):
                done += 1
                r = fut.result()
                if "price_error" in r or "value_error" in r:
                    meta["errors"].append(str(r))
                if done % 20 == 0:
                    print(f"[stocks] {done}/{len(codes)}  {time.time()-t0:.0f}s")
        meta["stocks"] = len(codes)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    (DATA / "meta" / "fetch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {meta['elapsed_s']}s, errors={len(meta['errors'])}")
    for e in meta["errors"][:20]:
        print("  !", e)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
