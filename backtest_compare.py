#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캐시 전용 전략 비교 백테스트 (pykrx 네트워크 호출 없음)
Usage: python backtest_compare.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# pykrx 증분 다운로드 차단: 캐시 허용 범위를 90 영업일로 늘려
# get_ohlcv_one 의 _is_cache_stale 이 stale 판정을 내리지 않도록 함
import config
config.DATA_STALE_TOLERANCE_BDAYS = 90
import data_loader
data_loader.DATA_STALE_TOLERANCE_BDAYS = 90

from reports import load_data, filter_universe
from strategies import get_strategy
from utils import perf_stats

STRATEGIES = [
    "multi_allocator_plus_no_etf",          # 현재 실제 운용
    "multi_allocator_plus_safe_etf",        # 개선안A: 일반 ETF 방어 복원
    "multi_allocator_plus_safe_etf_kqm",   # 개선안B: ETF + KQM 추가
    "etf_defensive_safe",                   # etf_defensive_safe 단독 (참고용)
]

COLORS = ["#E53935", "#43A047", "#1565C0", "#FB8C00"]
LABELS = {
    "multi_allocator_plus_no_etf":        "현재 운용 (no_etf)",
    "multi_allocator_plus_safe_etf":      "개선안A (safe_etf 복원)",
    "multi_allocator_plus_safe_etf_kqm":  "개선안B (safe_etf + kqm)",
    "etf_defensive_safe":                 "etf_defensive_safe 단독 (참고)",
}

def main():
    from config import START
    import datetime
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("=" * 65)
    print("  전략 비교 백테스트")
    print(f"  기간: {START} ~ {end_date} (약 6.4년)")
    print(f"  비교: {len(STRATEGIES)}개 전략")
    print("=" * 65)

    print("\n📂 캐시 데이터 로딩 중... (네트워크 없음)")
    enriched, idx_map = load_data(use_cache=True, incremental=False, include_market_cap=False)
    idx_kosdaq = idx_map.get("KOSDAQ")
    print(f"✅ 로딩 완료: {len(enriched)}개 종목")

    results = []
    for name in STRATEGIES:
        strat = get_strategy(name)
        print(f"\n{'='*65}")
        print(f"📈 [{name}] 실행 중...")
        print("=" * 65)
        ec, trades = strat.run_backtest(enriched, market_index=idx_kosdaq, silent=True)
        results.append((name, ec, trades))
        if ec is not None and not ec.empty:
            s = perf_stats(ec["equity"])
            total = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
            print(f"  → CAGR {s.get('CAGR',0)*100:.2f}%  Sharpe {s.get('Sharpe',0):.3f}  MDD {s.get('MDD',0)*100:.2f}%  최종수익 {total:.2f}%")

    # ── 성과 비교 출력 ──
    print("\n" + "=" * 65)
    print("📊 성과 비교 요약")
    print("=" * 65)
    stats_rows = []
    for name, ec, trades in results:
        if ec is None or ec.empty:
            print(f"[WARN] {name}: 결과 없음")
            continue
        s = perf_stats(ec["equity"])
        row = {
            "전략": LABELS.get(name, name),
            "CAGR(%)":    round(s.get("CAGR", 0) * 100, 2),
            "Sharpe":     round(s.get("Sharpe", 0), 3),
            "MDD(%)":     round(s.get("MDD", 0) * 100, 2),
            "최종수익(%)": round((ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100, 2),
            "거래일수":    s.get("Days", 0),
        }
        stats_rows.append(row)

    # ── 차트 저장 ──
    from config import START
    import datetime
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    for i, (name, ec, _) in enumerate(results):
        if ec is None or ec.empty:
            continue
        color = COLORS[i % len(COLORS)]
        label = LABELS.get(name, name)
        norm = ec["equity"] / ec["equity"].iloc[0] * 100
        lw = 2.5 if "no_etf" in name else 1.8
        ls = "-" if "no_etf" in name else "--"
        axes[0].plot(norm.index, norm.values, label=label,
                     color=color, linewidth=lw, linestyle=ls)
        dd = (ec["equity"] / ec["equity"].cummax() - 1) * 100
        axes[1].fill_between(dd.index, dd.values, 0,
                              alpha=0.2 if "no_etf" not in name else 0.35, color=color)
        axes[1].plot(dd.index, dd.values, color=color,
                     linewidth=lw, linestyle=ls, label=label)

    axes[0].axhline(100, color="gray", linewidth=0.7, linestyle=":")
    axes[0].set_title(f"누적 수익률 비교 (기준=100)  |  기간: {START} ~ {end_date}", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("수익률 지수")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_title("낙폭 (Drawdown, %)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MDD (%)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"reports/strategy_compare_{ts}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ 차트 저장: {out_path}")

    # ── 통계 테이블 ──
    if stats_rows:
        df_stats = pd.DataFrame(stats_rows).set_index("전략")
        print("\n" + df_stats.to_string())
        csv_path = f"reports/strategy_compare_{ts}.csv"
        df_stats.to_csv(csv_path)
        print(f"✅ CSV 저장: {csv_path}")

if __name__ == "__main__":
    main()
