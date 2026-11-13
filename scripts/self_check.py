"""Smoke check for the cached data and analytics layers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import allocation, commodities, data, factors, signals  # noqa: E402


def main() -> int:
    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append((name, True, detail))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    def _signals():
        s = signals.all_signals()
        assert len(s) == 59, len(s)
        assert s["score"].notna().all()
        assert s["action"].notna().all()
        return f"{len(s)} 个标的，{(s['action'] == '回避').sum()} 个回避"

    def _universe():
        u = data.load_universe()
        a = data.load_all_stocks()
        assert len(u) >= 750, len(u)
        assert len(a) >= 5000, len(a)
        n = sum(1 for c in u["code"] if not data.load_stock_price(c).empty)
        assert n >= 750, n
        return f"{len(u)} 只深度池 / {len(a)} 只全A，{n} 只有行情"

    def _factors():
        p = factors.scored_panel()
        assert len(p) > 60000, len(p)
        ic = factors.ic_series(p)
        assert len(ic) >= 90, len(ic)
        return f"{len(p):,} 行面板，{len(ic)} 个月 IC"

    def _allocation():
        back = allocation.strategy_backtests()
        nav = back["衡策组合"]
        assert len(nav) > 150
        w = back["_weights"].iloc[-1]
        assert 0 < w.sum() <= 1.01, float(w.sum())
        return f"{len(nav)} 个月，最新股票权重 {w['股票']:.0%}"

    def _portfolio():
        w = {"沪深300": 0.4, "中证500": 0.1, "债券": 0.3, "黄金": 0.1, "商品": 0.1}
        bt = allocation.portfolio_backtest(w)["returns"]
        st = allocation.stress_test(w)
        assert len(bt) > 150 and len(st) >= 4
        return f"{len(bt)} 个月回测，{len(st)} 个压力区间"

    def _commodities():
        t = commodities.commodity_table()
        assert len(t) == 20
        assert t["price_pct_5y"].notna().all()
        return f"{len(t)} 个品种"

    def _macro():
        reg = signals.current_regime()
        erp = signals.erp_snapshot()
        assert reg and erp
        return f"{reg['regime']}，ERP {erp['erp']*100:.1f}%"

    check("信号表", _signals)
    check("股票池", _universe)
    check("因子面板", _factors)
    check("配置策略", _allocation)
    check("组合构建", _portfolio)
    check("商品", _commodities)
    check("宏观/ERP", _macro)

    print(f"{'检查项':<10} {'结果':<6} 说明")
    for name, ok, detail in results:
        print(f"{name:<10} {'PASS' if ok else 'FAIL':<6} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
