#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US 마켓 - multi_allocator_plus_no_etf 백테스트
KR 실전과 동일한 전략으로 US 데이터를 사용해 성과를 비교한다.

실행:
    python backtest_us_strategy.py
    python backtest_us_strategy.py --start 2022-01-01
    python backtest_us_strategy.py --start 2022-01-01 --limit 30
    python backtest_us_strategy.py --compare   # KR vs US 나란히 비교
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import START, END
from data_loader import (
    get_universe_us,
    load_panel_us,
    get_index_close,
    get_universe,
    load_panel,
    infer_market,
)
from signals import compute_indicators, add_rel_strength
from strategies import get_strategy
from utils import perf_stats


# ──────────────────────────────────────────────────────────────
# 데이터 준비
# ──────────────────────────────────────────────────────────────

def _build_enriched_us(start: str, end: str, limit: int) -> tuple[dict, pd.DataFrame]:
    """US OHLCV → indicators → enriched dict + S&P500 index"""
    universe = get_universe_us(limit=limit)
    print(f"[US] 유니버스: {len(universe)}개 — {universe[:5]} ...")

    panel = load_panel_us(universe, start, end, max_workers=6)
    print(f"[US] 데이터 로드: {len(panel)}개 종목")

    idx = get_index_close("US", start, end)

    enriched = {}
    for ticker, df in panel.items():
        df = compute_indicators(df)
        if df is None or df.empty:
            continue
        df = add_rel_strength(df, idx)
        enriched[ticker] = df

    print(f"[US] 지표 계산 완료: {len(enriched)}개 종목")
    return enriched, idx


def _build_enriched_kr(start: str, end: str) -> tuple[dict, pd.DataFrame]:
    """KR enriched 캐시 직접 로딩 → universe_filter 적용 + KOSDAQ index"""
    from cache_manager import load_enriched, load_index
    from universe_filter import filter_universe
    import glob

    req_start = pd.to_datetime(start)
    req_end   = pd.to_datetime(end)

    # enriched 캐시 파일 목록
    enriched_dir = Path("data/enriched")
    files = sorted(enriched_dir.glob("*.parquet"))
    print(f"[KR] 캐시 파일 {len(files)}개 로딩 중 ...")

    # 티커별로 파일을 모으고, 요청 기간을 가장 많이 커버하는 파일 하나만 사용
    from collections import defaultdict
    ticker_files: dict = defaultdict(list)
    for fp in files:
        ticker = fp.stem.split("_")[0]
        ticker_files[ticker].append(fp)

    enriched_all: dict = {}
    for ticker, fps in ticker_files.items():
        best_df = None
        best_rows = 0
        for fp in fps:
            try:
                df = pd.read_parquet(fp)
                if df is None or df.empty:
                    continue
                df = df.sort_index()
                sliced = df[(df.index >= req_start) & (df.index <= req_end)]
                if len(sliced) > best_rows:
                    best_rows = len(sliced)
                    best_df = sliced
            except Exception:
                continue
        if best_df is not None and best_rows >= 60:
            enriched_all[ticker] = best_df

    print(f"[KR] 로드 완료: {len(enriched_all)}개")

    valid = filter_universe(enriched_all)
    enriched = {t: enriched_all[t] for t in valid if t in enriched_all}
    print(f"[KR] 필터 후 유니버스: {len(enriched)}개 종목")

    # KOSDAQ 지수 캐시
    idx = load_index("KOSDAQ", start, end)
    if idx is None or idx.empty:
        from data_loader import get_index_close
        idx = get_index_close("KOSDAQ", start, end)
    return enriched, idx


# ──────────────────────────────────────────────────────────────
# 백테스트 실행
# ──────────────────────────────────────────────────────────────

def run_backtest(enriched: dict, market_index: pd.DataFrame, label: str) -> tuple:
    strategy = get_strategy("multi_allocator_plus_no_etf")
    print(f"\n[{label}] 백테스트 시작 ...")
    ec, trades = strategy.run_backtest(enriched, market_index=market_index, silent=False)
    return ec, trades


# ──────────────────────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────────────────────

def print_stats(ec: pd.DataFrame, label: str):
    if ec is None or ec.empty:
        print(f"[{label}] 결과 없음")
        return
    stats = perf_stats(ec)
    print(f"\n{'='*50}")
    print(f"  {label}  —  multi_allocator_plus_no_etf")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k:<12}: {v:>10.4f}")
    start_v = float(ec["equity"].iloc[0])
    end_v   = float(ec["equity"].iloc[-1])
    total_r = (end_v / start_v - 1) * 100 if start_v > 0 else 0.0
    print(f"  {'총수익률':<12}: {total_r:>9.2f}%")
    print(f"  기간: {ec.index[0].date()} ~ {ec.index[-1].date()}")


def save_results(ec_us, ec_kr, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    if ec_us is not None and not ec_us.empty:
        p = out_dir / f"backtest_us_no_etf_{today}.csv"
        ec_us.to_csv(p)
        print(f"[저장] US equity curve → {p}")

    if ec_kr is not None and not ec_kr.empty:
        p = out_dir / f"backtest_kr_no_etf_{today}.csv"
        ec_kr.to_csv(p)
        print(f"[저장] KR equity curve → {p}")

    if (ec_us is not None and not ec_us.empty and
            ec_kr is not None and not ec_kr.empty):
        combined = ec_us.rename(columns={"equity": "US"}).join(
            ec_kr.rename(columns={"equity": "KR"}), how="outer"
        )
        p = out_dir / f"backtest_comparison_{today}.csv"
        combined.to_csv(p)
        print(f"[저장] 비교 테이블 → {p}")


def plot_comparison(ec_us, ec_kr):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 1, figsize=(14, 9))
        fig.suptitle("multi_allocator_plus_no_etf: US vs KR", fontsize=14)

        ax1 = axes[0]
        if ec_us is not None and not ec_us.empty:
            norm_us = ec_us["equity"] / ec_us["equity"].iloc[0]
            ax1.plot(norm_us.index, norm_us, label="US", linewidth=2, color="steelblue")
        if ec_kr is not None and not ec_kr.empty:
            norm_kr = ec_kr["equity"] / ec_kr["equity"].iloc[0]
            ax1.plot(norm_kr.index, norm_kr, label="KR", linewidth=2, color="tomato")
        ax1.set_title("누적 수익 곡선 (정규화)")
        ax1.set_ylabel("배율")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        if ec_us is not None and not ec_us.empty:
            dd_us = ec_us["equity"] / ec_us["equity"].cummax() - 1
            ax2.fill_between(dd_us.index, dd_us * 100, 0, alpha=0.4, label="US", color="steelblue")
        if ec_kr is not None and not ec_kr.empty:
            dd_kr = ec_kr["equity"] / ec_kr["equity"].cummax() - 1
            ax2.fill_between(dd_kr.index, dd_kr * 100, 0, alpha=0.4, label="KR", color="tomato")
        ax2.set_title("낙폭 (Drawdown %)")
        ax2.set_ylabel("DD (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path("reports") / f"backtest_us_kr_comparison_{date.today().isoformat()}.png"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[차트] {out}")
        plt.show()
    except ImportError:
        print("[INFO] matplotlib 없어 차트 생략")


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="US multi_allocator_plus_no_etf 백테스트")
    parser.add_argument("--start",   default=START,  help="시작일 YYYY-MM-DD")
    parser.add_argument("--end",     default=None,    help="종료일 YYYY-MM-DD (기본: 오늘)")
    parser.add_argument("--limit",   type=int, default=50, help="US 유니버스 종목 수 (기본 50)")
    parser.add_argument("--compare", action="store_true",  help="KR 백테스트도 함께 실행해 비교")
    parser.add_argument("--no-plot", action="store_true",  help="차트 출력 생략")
    args = parser.parse_args()

    start = args.start
    end   = args.end or (date.today().strftime("%Y-%m-%d") if END is None else END)

    print(f"\n기간: {start} ~ {end}")
    print("전략: multi_allocator_plus_no_etf\n")

    # ── US 백테스트
    enriched_us, idx_us = _build_enriched_us(start, end, args.limit)
    ec_us, _ = run_backtest(enriched_us, idx_us, "US")
    print_stats(ec_us, "US")

    # ── KR 백테스트 (--compare 옵션)
    ec_kr = None
    if args.compare:
        enriched_kr, idx_kr = _build_enriched_kr(start, end)
        ec_kr, _ = run_backtest(enriched_kr, idx_kr, "KR")
        print_stats(ec_kr, "KR")

    # ── 저장 & 차트
    save_results(ec_us, ec_kr, Path("reports"))
    if not args.no_plot:
        plot_comparison(ec_us, ec_kr)


if __name__ == "__main__":
    main()
