"""News headlines -> simple, transparent sentiment indicators.

The score is keyword based on purpose: it is easy to audit and explain in a
course report. It measures the tone of the wording, not whether the news is true.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

NEWS_URL = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"

POS_WORDS = {
    "利好": 2.0, "上涨": 1.2, "增长": 1.0, "突破": 1.0, "回暖": 1.2, "复苏": 1.2,
    "盈利": 1.0, "增持": 1.5, "回购": 1.5, "降准": 2.0, "降息": 2.0, "超预期": 2.0,
    "创新高": 2.0, "改善": 0.8, "提振": 1.2, "支持": 0.8, "放宽": 1.0, "减税": 1.5,
    "补贴": 1.0, "中标": 1.2, "订单": 0.8, "涨停": 1.5, "净流入": 1.2, "扩产": 1.0,
    "签约": 0.8, "获批": 1.2, "上调": 1.0, "升温": 0.8, "走强": 1.0, "新高": 1.2,
}
NEG_WORDS = {
    "利空": 2.0, "下跌": 1.2, "下滑": 1.0, "亏损": 1.5, "暴跌": 2.0, "违约": 2.0,
    "退市": 2.0, "减持": 1.5, "处罚": 1.5, "调查": 1.2, "收紧": 1.2, "加息": 2.0,
    "衰退": 1.5, "低于预期": 2.0, "创新低": 2.0, "裁员": 1.2, "债务": 1.0,
    "爆雷": 2.0, "停牌": 1.2, "跌停": 1.5, "净流出": 1.2, "下调": 1.0, "走弱": 1.0,
    "风险": 0.6, "承压": 0.8, "回落": 0.8, "缩水": 1.0, "解禁": 0.8, "问询": 0.8,
}

SECTOR_KEYWORDS = {
    "农林牧渔": ["农业", "养殖", "猪价", "种业", "饲料"],
    "基础化工": ["化工", "化肥", "农药", "树脂", "纯碱"],
    "钢铁": ["钢铁", "钢材", "铁矿石", "螺纹钢"],
    "有色金属": ["有色", "铜", "铝", "锂", "黄金", "稀土"],
    "电子": ["芯片", "半导体", "电子", "PCB", "面板", "消费电子"],
    "汽车": ["汽车", "新能源车", "整车", "零部件", "智能驾驶"],
    "家用电器": ["家电", "空调", "冰箱", "厨电"],
    "食品饮料": ["白酒", "食品", "饮料", "乳业", "调味品"],
    "纺织服饰": ["纺织", "服装", "鞋帽"],
    "轻工制造": ["轻工", "造纸", "包装", "家居"],
    "医药生物": ["医药", "创新药", "医疗", "疫苗", "CXO", "器械"],
    "公用事业": ["电力", "燃气", "水务", "公用事业"],
    "交通运输": ["航运", "航空", "物流", "港口", "铁路"],
    "房地产": ["房地产", "地产", "楼市", "房企"],
    "商贸零售": ["零售", "电商", "商超", "免税"],
    "社会服务": ["旅游", "酒店", "餐饮", "教育", "服务消费"],
    "银行": ["银行", "信贷", "存款", "净息差"],
    "非银金融": ["券商", "保险", "证券", "基金", "信托"],
    "综合": [],
    "建筑材料": ["建材", "水泥", "玻璃", "玻纤"],
    "建筑装饰": ["建筑", "基建", "工程", "装饰"],
    "电力设备": ["光伏", "风电", "储能", "电池", "电网"],
    "机械设备": ["机械", "机器人", "工程机械", "机床"],
    "国防军工": ["军工", "国防", "航天", "航空发动机"],
    "计算机": ["计算机", "软件", "云计算", "信创", "算力"],
    "传媒": ["传媒", "游戏", "影视", "广告", "出版"],
    "通信": ["通信", "5G", "光模块", "运营商"],
    "煤炭": ["煤炭", "焦煤", "动力煤"],
    "石油石化": ["石油", "原油", "石化", "天然气"],
    "环保": ["环保", "污水", "固废", "碳中和"],
    "美容护理": ["美容", "化妆品", "护理"],
}


def score_text(text: str) -> tuple[float, float, float]:
    t = str(text)
    pos = sum(w for k, w in POS_WORDS.items() if k in t)
    neg = sum(w for k, w in NEG_WORDS.items() if k in t)
    raw = (pos - neg) / (pos + neg + 1.5)
    return float(raw), float(pos), float(neg)


@lru_cache(maxsize=4)
def fetch_market_news(page_size: int = 200) -> pd.DataFrame:
    r = requests.get(NEWS_URL, params={
        "client": "web", "biz": "web_724", "fastColumn": "102",
        "sortEnd": "", "pageSize": str(page_size),
        "req_trace": str(int(datetime.now().timestamp() * 1000)),
    }, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
    r.raise_for_status()
    items = r.json().get("data", {}).get("fastNewsList", [])
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"title": "标题", "summary": "摘要", "showTime": "发布时间",
                            "code": "链接"})
    df["链接"] = "https://finance.eastmoney.com/a/" + df["链接"].astype(str) + ".html"
    df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")
    df = df.dropna(subset=["发布时间"]).sort_values("发布时间", ascending=False)
    df["文本"] = (df["标题"].fillna("") + " " + df["摘要"].fillna("")).str.slice(0, 300)
    scored = df["文本"].map(score_text)
    df["情绪"] = [x[0] for x in scored]
    df["正面词"] = [x[1] for x in scored]
    df["负面词"] = [x[2] for x in scored]
    df["情绪分"] = 50 + 50 * df["情绪"]
    df["小时"] = df["发布时间"].dt.strftime("%m-%d %H:00")
    return df.reset_index(drop=True)


def market_snapshot(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    score = float(np.clip(df["情绪分"].mean(), 0, 100))
    pos = int((df["情绪"] > 0.05).sum())
    neg = int((df["情绪"] < -0.05).sum())
    neu = len(df) - pos - neg
    # heat: news per hour vs the median hour in the batch
    hourly = df.groupby("小时").size()
    heat = float(hourly.iloc[-1] / max(hourly.median(), 1)) if len(hourly) else np.nan
    return {"score": score, "pos": pos, "neg": neg, "neu": neu,
            "n": len(df), "heat": heat,
            "latest": df["发布时间"].max().strftime("%m-%d %H:%M"),
            "pos_ratio": pos / len(df), "neg_ratio": neg / len(df)}


def hourly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("小时").agg(情绪分=("情绪分", "mean"), 条数=("情绪", "size")).reset_index()
    return g.tail(24)


def sector_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for sector, words in SECTOR_KEYWORDS.items():
        if not words:
            continue
        mask = df["文本"].map(lambda x: any(w in x for w in words))
        sub = df[mask]
        if len(sub) < 2:
            continue
        rows.append({"行业": sector, "新闻数": len(sub),
                     "情绪分": float(sub["情绪分"].mean()),
                     "正面占比": float((sub["情绪"] > 0.05).mean())})
    out = pd.DataFrame(rows)
    return out.sort_values("新闻数", ascending=False) if not out.empty else out


def stock_mentions(df: pd.DataFrame, name: str, code: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    key = str(name).replace(" ", "")
    mask = df["文本"].str.contains(key, na=False)
    if code:
        mask = mask | df["文本"].str.contains(str(code), na=False)
    return df[mask]


def top_headlines(df: pd.DataFrame, n: int = 5, positive: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    s = df.sort_values("情绪", ascending=not positive)
    s = s[s["情绪"] > 0.05] if positive else s[s["情绪"] < -0.05]
    return s.head(n)[["发布时间", "标题", "情绪分"]]
