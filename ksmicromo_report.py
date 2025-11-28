# ksmicromo_report.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KSmicroMo v2 단독 리포트 스크립트

- reports.py 의 load_data()로 enriched 로드
- KSmicroMo v2 백테스트 실행
- 전체 성과 요약 + 트레이드 통계 + (옵션) 섹터별 요약 출력
"""

import pickle
import os
import pandas as pd

from reports import load_data          # 이미 만든 함수 재사용
from utils import perf_stats
from strategies.ksmicromo import KSmicroMo


def summarize_trades(trades):
    """전체 트레이드 통계 요약"""
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_gain": 0.0,
            "avg_loss": 0.0,
            "payoff": 0.0,
            "avg_hold_days": 0.0,
        }

    df = pd.DataFrame(trades)
    sells = df[df["action"] == "SELL"].copy()
    if sells.empty:
        sells = df.copy()

    wins = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] < 0]

    win_rate = len(wins) / len(sells) if len(sells) > 0 else 0.0
    avg_gain = wins["ret"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["ret"].mean() if len(losses) > 0 else 0.0
    payoff = (avg_gain / abs(avg_loss)) if avg_loss < 0 else 0.0
    avg_hold = sells["hold_days"].mean() if "hold_days" in sells.columns else 0.0

    return {
        "trades": int(len(sells)),
        "win_rate": float(win_rate),
        "avg_gain": float(avg_gain),
        "avg_loss": float(avg_loss),
        "payoff": float(payoff),
        "avg_hold_days": float(avg_hold),
    }


def sector_report(trades, sector_map_path="./data/meta/sector_map.pkl"):
    """
    섹터별 트레이드 요약 (섹터 맵이 있을 때만)
    - trades: run_backtest 반환된 trade_log (list[dict])
    - sector_map.pkl: {ticker: sector_name} 딕셔너리
    """
    if not trades:
        print("\n⚠️ 섹터 리포트: 트레이드가 없습니다.")
        return

    try:
        with open(sector_map_path, "rb") as f:
            sector_map = pickle.load(f)
        print(f"\n✅ 섹터 매핑 로드: {len(sector_map)}개 종목")
    except FileNotFoundError:
        print("\n⚠️ sector_map.pkl 파일이 없어 섹터별 리포트는 스킵합니다.")
        return

    df = pd.DataFrame(trades)
    sells = df[df["action"] == "SELL"].copy()
    if sells.empty:
        sells = df.copy()

    sells["sector"] = sells["ticker"].map(lambda x: sector_map.get(x, "기타"))

    rows = []
    for sec, g in sells.groupby("sector"):
        wins = g[g["pnl"] > 0]
        losses = g[g["pnl"] < 0]
        win_rate = len(wins) / len(g) if len(g) > 0 else 0.0
        avg_gain = wins["ret"].mean() if len(wins) > 0 else 0.0
        avg_loss = losses["ret"].mean() if len(losses) > 0 else 0.0
        payoff = (avg_gain / abs(avg_loss)) if avg_loss < 0 else 0.0
        avg_hold = g["hold_days"].mean() if "hold_days" in g.columns else 0.0

        rows.append(
            [
                sec,
                len(g),
                win_rate,
                avg_gain,
                avg_loss,
                payoff,
                avg_hold,
            ]
        )

    if not rows:
        print("\n⚠️ 섹터별 데이터가 없습니다.")
        return

    sec_df = pd.DataFrame(
        rows,
        columns=["Sector", "Trades", "WinRate", "AvgGain", "AvgLoss", "Payoff", "AvgHoldDays"],
    ).sort_values("Trades", ascending=False)

    print("\n📊 섹터별 트레이드 성과 요약")
    print("-" * 72)
    # 퍼센트/소수 변환
    sec_df["WinRate"] = (sec_df["WinRate"] * 100).round(2)
    sec_df["AvgGain"] = (sec_df["AvgGain"] * 100).round(2)
    sec_df["AvgLoss"] = (sec_df["AvgLoss"] * 100).round(2)
    sec_df["Payoff"] = sec_df["Payoff"].round(2)
    sec_df["AvgHoldDays"] = sec_df["AvgHoldDays"].round(2)
    print(sec_df.to_string(index=False))


def main():
    print("\n" + "=" * 70)
    print("🚀 KSmicroMo v2 전략 리포트 생성 시작")
    print("=" * 70)

    # 1) 데이터 로드 (reports.load_data 재사용)
    enriched = load_data()

    # 2) 전략 인스턴스 생성
    strat = KSmicroMo()

    # 3) 백테스트 실행
    ec, trades = strat.run_backtest(enriched, silent=False)

    if ec.empty:
        print("\n⚠️ Equity curve가 비어 있습니다. 데이터/전략 조건을 확인하세요.")
        return

    # 4) 성과 지표 요약
    stats = perf_stats(ec)
    tstats = summarize_trades(trades)

    print("\n" + "=" * 70)
    print("📊 KSmicroMo v2 성과 요약")
    print("=" * 70)
    print(f"  CAGR      : {stats['CAGR']:10.4f}")
    print(f"  Vol       : {stats['Vol']:10.4f}")
    print(f"  Sharpe    : {stats['Sharpe']:10.4f}")
    print(f"  MDD       : {stats['MDD']:10.4f}")
    print(f"  Days      : {stats['Days']:10.0f}")
    print(f"  승률        : {tstats['win_rate']*100:9.2f}%")
    print(f"  평균익절      : {tstats['avg_gain']*100:9.2f}%")
    print(f"  평균손절      : {tstats['avg_loss']*100:9.2f}%")
    print(f"  Payoff    : {tstats['payoff']:10.2f}")
    print(f"  총거래       : {tstats['trades']:10d}회")
    print(f"  평균 보유일수   : {tstats['avg_hold_days']:9.2f}일")

    # 5) 섹터별 요약 (옵션)
    sector_report(trades)

    print("\n✅ KSmicroMo v2 리포트 생성 완료!\n")


if __name__ == "__main__":
    main()