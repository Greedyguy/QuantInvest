# small_cap_strategy_comparison.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

from config import *
from utils import perf_stats
from reports import load_data  # 이미 만든 load_data 재사용

from strategies.strategy_ksms import KSMSStrategy
from strategies.ksturbo import KSTurbo
from strategies.ksmicromo import KSmicroMo
from strategies.kmr_midcap_reversion import KMRMidcapReversion

def summarize_trades(trades):
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_gain": 0.0,
            "avg_loss": 0.0,
        }
    df = pd.DataFrame(trades)
    sells = df.copy()
    wins = sells[sells["ret"] > 0]
    losses = sells[sells["ret"] < 0]
    win_rate = len(wins) / len(sells) if len(sells) > 0 else 0.0
    avg_gain = wins["ret"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["ret"].mean() if len(losses) > 0 else 0.0

    return {
        "trades": len(sells),
        "win_rate": win_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
    }


def run_and_report():
    print("\n" + "="*70)
    print("🚀 소액 전략 비교 리포트: KSMS v2 vs KSTurbo vs KSmicroMo")
    print("="*70)

    # 1) 데이터 로드
    enriched = load_data()   # reports.py에 있는 함수 그대로 사용

    # 2) 전략 리스트
    strategies = [
        KMRMidcapReversion()
    ]

    results = []

    # 3) 각 전략 실행
    for strat in strategies:
        name = strat.get_name()
        print("\n" + "-"*60)
        print(f"🔍 전략 실행: {name} ({strat.get_description()})")
        print("-"*60)

        ec, trades = strat.run_backtest(enriched, silent=False)
        if ec.empty:
            print(f"⚠️ {name}: equity curve 없음 (데이터 부족 또는 예외)")
            continue

        stats = perf_stats(ec)
        tstats = summarize_trades(trades)

        print("\n📊 성과 요약")
        print(f"  CAGR      : {stats['CAGR']:10.4f}")
        print(f"  Vol       : {stats['Vol']:10.4f}")
        print(f"  Sharpe    : {stats['Sharpe']:10.4f}")
        print(f"  MDD       : {stats['MDD']:10.4f}")
        print(f"  Days      : {stats['Days']:10.0f}")
        print(f"  승률        : {tstats['win_rate']*100:9.2f}%")
        print(f"  평균익절      : {tstats['avg_gain']*100:9.2f}%")
        print(f"  평균손절      : {tstats['avg_loss']*100:9.2f}%")
        print(f"  총거래       : {tstats['trades']:10d}회")

        results.append((name, stats, tstats))

    # 4) 전략간 간단 비교 테이블
    print("\n" + "="*70)
    print("📈 전략별 핵심 성과 비교")
    print("="*70)
    rows = []
    for name, s, t in results:
        rows.append([
            name,
            s["CAGR"],
            s["Sharpe"],
            s["MDD"],
            t["trades"],
            t["win_rate"],
        ])
    if rows:
        df_res = pd.DataFrame(
            rows,
            columns=["Strategy", "CAGR", "Sharpe", "MDD", "Trades", "WinRate"]
        )
        df_res["WinRate"] = (df_res["WinRate"] * 100).round(2)
        print(df_res.to_string(index=False))
    else:
        print("⚠️ 유효한 결과가 없습니다.")

    print("\n✅ 소액 전략 비교 리포트 완료\n")
    return results


if __name__ == "__main__":
    run_and_report()