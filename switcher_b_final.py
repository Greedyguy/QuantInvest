#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A-2 Switching Strategy (reports.py 기반)
- 날짜 alignment: intersection only
- no ffill / no bfill
- lookahead 제거
- 신호 중립(default) = 공격 전략
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 동일 파이프라인
from data_loader import get_universe, load_panel, get_index_close, infer_market
from cache_manager import load_enriched, save_enriched, load_index, save_index
from universe_filter import filter_universe
from signals import compute_indicators, add_rel_strength
from strategies import get_strategy
from utils import perf_stats


# =============================================
# 1) 인디케이터 + 상대강도
# =============================================
def build_signal(ticker, df, idx_map, start, end, use_cache=True):
    if df is None or df.empty:
        return ticker, None
    if use_cache:
        cached = load_enriched(ticker, start, end)
        if cached is not None:
            return ticker, cached

    df = compute_indicators(df)
    if df is None or df.empty:
        return ticker, None

    mkt = infer_market(ticker)
    idx = idx_map.get(mkt, pd.DataFrame())
    df = add_rel_strength(df, idx)

    if use_cache:
        save_enriched(ticker, df)
    return ticker, df


# =============================================
# 2) 전체 데이터 로드
# =============================================
def load_all_data(start_date, end_date, use_cache=True):

    universe = get_universe(["KOSPI", "KOSDAQ"], include_etf=False)
    panel = load_panel(universe, start_date, end_date, max_workers=6)

    idx_map = {}
    for mkt in ["KOSPI", "KOSDAQ"]:
        cached = load_index(mkt, start_date, end_date) if use_cache else None
        if cached is not None:
            idx_map[mkt] = cached
        else:
            data = get_index_close(mkt, start_date, end_date)
            idx_map[mkt] = data
            if use_cache:
                save_index(mkt, data)

    enriched = {}
    for ticker, df in tqdm(panel.items(), desc="Indicators"):
        t, edf = build_signal(ticker, df, idx_map, start_date, end_date, use_cache)
        if edf is not None:
            enriched[t] = edf

    final_uni = filter_universe(enriched)
    enriched = {t: enriched[t] for t in final_uni}

    return enriched, idx_map


# =============================================
# 3) A-2 개선 신호 생성
# =============================================
def generate_switch_signal(kosdaq):

    df = kosdaq.copy()
    df["MA60"] = df["close"].rolling(60).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MOM20"] = df["close"].pct_change(20)
    df["RSI"] = (df["close"].diff().clip(lower=0).rolling(14).mean() /
                 df["close"].diff().abs().rolling(14).mean()).fillna(0)

    cond1 = df["close"] > df["MA60"]
    cond2 = df["MOM20"] > -0.02
    cond3 = df["RSI"] >= 35

    # A-2: majority voting
    sig = ((cond1.astype(int) + cond2.astype(int) + cond3.astype(int)) >= 2)

    # 신호 중립 구간 → 공격 전략
    sig = sig.reindex(df.index).fillna(True)

    # lookahead removal
    sig = sig.shift(1).fillna(True)

    return sig


# =============================================
# 4) 스위칭 백테스트
# =============================================
def run_switch_backtest(enriched, kosdaq, strat_def, strat_agg):

    sig = generate_switch_signal(kosdaq)

    ec_def, trades_def = strat_def.run_backtest(enriched, market_index=kosdaq)
    ec_agg, trades_agg = strat_agg.run_backtest(enriched, market_index=kosdaq)

    # intersection only — A-2 핵심
    common_dates = ec_def.index.intersection(ec_agg.index)
    common_dates = common_dates.intersection(sig.index)

    ec_def = ec_def.loc[common_dates]
    ec_agg = ec_agg.loc[common_dates]
    sig = sig.loc[common_dates]

    equity = []
    for d in common_dates:
        cur = ec_agg.loc[d, "equity"] if sig.loc[d] else ec_def.loc[d, "equity"]
        equity.append((d, cur))

    sw = pd.DataFrame(equity, columns=["date", "equity"]).set_index("date")
    return ec_def, ec_agg, sw


# =============================================
# 5) 메인
# =============================================
def main(args):

    if args.start_date and args.end_date:
        start = args.start_date
        end = args.end_date
    else:
        start = "2020-01-01"
        end = datetime.today().strftime("%Y-%m-%d")

    enriched, idx_map = load_all_data(start, end, use_cache=True)
    kosdaq = idx_map["KOSDAQ"]

    strat_def = get_strategy("hybrid_portfolio")
    strat_agg = get_strategy("kqm_small_cap_v22")

    ec_def, ec_agg, ec_sw = run_switch_backtest(enriched, kosdaq, strat_def, strat_agg)

    print("\n=== 개별 전략 성과 ===")
    print("[방어]", perf_stats(ec_def))
    print("[성장]", perf_stats(ec_agg))

    print("\n=== 스위칭 전략 성과 ===")
    print(perf_stats(ec_sw))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    main(parser.parse_args())