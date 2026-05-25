#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
서브전략 기여도 분해 분석
────────────────────────
multi_allocator_plus의 5개 서브전략을 각각 독립 백테스트해서
어떤 전략이 성과를 이끌고 어떤 전략이 발목을 잡는지 진단.

Usage: python backtest_substrat_decompose.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

import config
config.DATA_STALE_TOLERANCE_BDAYS = 90
import data_loader
data_loader.DATA_STALE_TOLERANCE_BDAYS = 90

from data_loader import get_universe, load_panel, get_index_close, infer_market
from signals import compute_indicators, add_rel_strength
from strategies import get_strategy
from utils import perf_stats
from config import MARKETS, BLOCKED_TICKERS

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ── multi_allocator_plus 서브전략 구성 (이름, 비중, 역할) ──
SUB_STRATEGIES = [
    ("kqm_small_cap_v22_short", 0.32, "short",     "#E53935"),  # 빨강
    ("hybrid_portfolio_v2_4",   0.22, "offensive",  "#1E88E5"),  # 파랑
    ("kqm_small_cap_v22",       0.20, "offensive",  "#43A047"),  # 초록
    ("etf_defensive",           0.18, "defensive",  "#FB8C00"),  # 주황
    ("k200_mean_rev",           0.12, "offensive",  "#8E24AA"),  # 보라
]

ENSEMBLE_NAME = "multi_allocator_plus"
ENSEMBLE_COLOR = "#000000"


def _stats(name: str, ec: pd.DataFrame) -> dict:
    if ec is None or ec.empty:
        return dict(이름=name, CAGR="-", Sharpe="-", MDD="-", 최종수익="-", 거래일="-")
    s = perf_stats(ec)
    total_ret = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
    return {
        "이름":   name,
        "CAGR(%)":  f"{s['CAGR']*100:.2f}",
        "Sharpe":   f"{s['Sharpe']:.3f}",
        "MDD(%)":   f"{s['MDD']*100:.2f}",
        "최종수익(%)": f"{total_ret:.2f}",
        "거래일":   s["Days"],
    }


def main():
    print("=" * 65)
    print("  서브전략 기여도 분해 분석")
    print("=" * 65)

    # ── 데이터 로드 (parquet 캐시 직접 사용, 네트워크 없음) ──
    print("\n📂 데이터 로드 중 (캐시 전용, market_cap 제외)...")
    from config import START
    import datetime
    end = datetime.date.today().strftime("%Y-%m-%d")

    universe = [t for t in get_universe(MARKETS) if t not in BLOCKED_TICKERS]
    panel = load_panel(universe, START, end, include_market_cap=False)

    idx_map = {
        "KOSPI":  get_index_close("KOSPI",  START, end),
        "KOSDAQ": get_index_close("KOSDAQ", START, end),
    }

    enriched = {}
    for t, df in panel.items():
        mkt = infer_market(t) or "KOSPI"
        df = compute_indicators(df)
        if df.empty:
            continue
        df = add_rel_strength(df, idx_map.get(mkt, pd.DataFrame()))
        enriched[t] = df

    idx_kosdaq = idx_map.get("KOSDAQ")
    idx_kospi  = idx_map.get("KOSPI")
    print(f"✅ {len(enriched)}개 종목 로드 완료\n")

    # ── 서브전략 개별 실행 ──
    results = {}

    for name, weight, role, color in SUB_STRATEGIES:
        print(f"{'─'*65}")
        print(f"▶ [{name}]  비중={weight*100:.0f}%  역할={role}")
        try:
            strat = get_strategy(name)
            # 지수 선택: etf/k200은 KOSPI, 나머지 KOSDAQ
            mkt_idx = idx_kospi if name in ("etf_defensive", "k200_mean_rev") else idx_kosdaq
            ec, trades = strat.run_backtest(enriched, market_index=mkt_idx, silent=True)
            if ec is not None and not ec.empty:
                s = perf_stats(ec)
                total = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
                print(f"  CAGR {s['CAGR']*100:.2f}%  Sharpe {s['Sharpe']:.3f}  "
                      f"MDD {s['MDD']*100:.2f}%  최종수익 {total:.2f}%")
            else:
                print("  ⚠️  결과 없음")
            results[name] = (ec, weight, role, color)
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            results[name] = (pd.DataFrame(columns=["equity"]), weight, role, color)

    # ── 앙상블 실행 ──
    print(f"\n{'─'*65}")
    print(f"▶ [{ENSEMBLE_NAME}]  (실제 운용 전략)")
    try:
        ens = get_strategy(ENSEMBLE_NAME)
        ec_ens, _ = ens.run_backtest(enriched, market_index=idx_kosdaq, silent=True)
        if ec_ens is not None and not ec_ens.empty:
            s = perf_stats(ec_ens)
            total = (ec_ens["equity"].iloc[-1] / ec_ens["equity"].iloc[0] - 1) * 100
            print(f"  CAGR {s['CAGR']*100:.2f}%  Sharpe {s['Sharpe']:.3f}  "
                  f"MDD {s['MDD']*100:.2f}%  최종수익 {total:.2f}%")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        ec_ens = pd.DataFrame(columns=["equity"])

    # ── 성과 테이블 ──
    print(f"\n{'='*65}")
    print("📋 성과 비교 테이블 (비중 가중 최종수익 = 해당 전략 단독 운용 시)")
    print("=" * 65)

    rows = []
    for name, weight, role, color in SUB_STRATEGIES:
        ec, _, _, _ = results[name]
        row = _stats(f"{name}  [{role}]", ec)
        row["배정비중(%)"] = f"{weight*100:.0f}"
        # 비중 가중 최종수익: "만약 전체 자산을 이 비중만큼 이 전략에만 투자했다면"
        try:
            raw_ret = float(row["최종수익(%)"])
            row["기여수익(%)"] = f"{raw_ret * weight:.2f}"
        except Exception:
            row["기여수익(%)"] = "-"
        rows.append(row)

    df_tbl = pd.DataFrame(rows).set_index("이름")
    col_order = ["배정비중(%)", "CAGR(%)", "Sharpe", "MDD(%)", "최종수익(%)", "기여수익(%)"]
    print(df_tbl[col_order].to_string())

    # 앙상블 행 추가
    row_ens = _stats(f"{ENSEMBLE_NAME}  [ensemble]", ec_ens)
    row_ens["배정비중(%)"] = "100"
    row_ens["기여수익(%)"] = row_ens["최종수익(%)"]
    print("\n앙상블:")
    print(f"  {ENSEMBLE_NAME}: "
          f"CAGR {row_ens['CAGR(%)']}%  Sharpe {row_ens['Sharpe']}  "
          f"MDD {row_ens['MDD(%)']}%  최종수익 {row_ens['최종수익(%)']}%")

    # ── 진단 메시지 ──
    print(f"\n{'='*65}")
    print("🔍 진단")
    print("=" * 65)
    try:
        contrib = {}
        for name, weight, role, color in SUB_STRATEGIES:
            ec, _, _, _ = results[name]
            if ec is not None and not ec.empty:
                r = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
                contrib[name] = (r, weight, r * weight, role)

        # 기여수익 기준 정렬
        sorted_contrib = sorted(contrib.items(), key=lambda x: x[1][2], reverse=True)
        print("\n  기여수익 순위 (단독수익 × 배정비중):")
        for n, (ret, w, contrib_ret, role) in sorted_contrib:
            sign = "✅" if contrib_ret > 0 else "⚠️ "
            print(f"  {sign} {n:<35} {ret:+7.2f}% × {w*100:.0f}% = {contrib_ret:+7.2f}%p")

        # 숏 전략 진단
        if "kqm_small_cap_v22_short" in contrib:
            short_ret, short_w, short_contrib, _ = contrib["kqm_small_cap_v22_short"]
            if short_contrib < 0:
                print(f"\n  ⚠️  숏 전략 진단: 단독 {short_ret:+.2f}% × {short_w*100:.0f}% = {short_contrib:+.2f}%p 손실")
                print(f"     → 숏 비중을 줄이거나 제거하면 앙상블 성과 개선 가능")
            else:
                print(f"\n  ✅ 숏 전략: 단독 {short_ret:+.2f}% × {short_w*100:.0f}% = {short_contrib:+.2f}%p 기여")
                print(f"     → 드래그가 아닌 실질 기여 중. 비중 유지 or 소폭 조정 검토")
    except Exception as e:
        print(f"  진단 계산 실패: {e}")

    # ── 차트 ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))

    # 상단: 누적 수익률
    for name, weight, role, color in SUB_STRATEGIES:
        ec, _, _, c = results[name]
        if ec is None or ec.empty:
            continue
        norm = ec["equity"] / ec["equity"].iloc[0] * 100
        lbl = f"{name} ({weight*100:.0f}%)"
        axes[0].plot(norm.index, norm.values, label=lbl, color=c, linewidth=1.5, alpha=0.85)

    if ec_ens is not None and not ec_ens.empty:
        norm_ens = ec_ens["equity"] / ec_ens["equity"].iloc[0] * 100
        axes[0].plot(norm_ens.index, norm_ens.values,
                     label=f"{ENSEMBLE_NAME} (앙상블)", color=ENSEMBLE_COLOR,
                     linewidth=2.5, linestyle="--")

    axes[0].axhline(100, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set_title("서브전략별 누적 수익률 (기준=100)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("수익률 지수")
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    axes[0].grid(alpha=0.3)

    # 하단: 기여수익 막대
    names_sorted = [n for n, _, _, _ in sorted(
        [(n, w, r * w, _role) for n, (r, w, rw, _role) in contrib.items()],
        key=lambda x: x[2]
    )]
    contrib_vals  = [contrib[n][2] for n in names_sorted]
    bar_colors    = [next(c for nm, _, _, c in SUB_STRATEGIES if nm == n) for n in names_sorted]

    bars = axes[1].barh(names_sorted, contrib_vals, color=bar_colors, alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1.0)
    for bar, val in zip(bars, contrib_vals):
        axes[1].text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                     f"{val:+.2f}%p", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    axes[1].set_title("비중 가중 기여수익 (단독수익 × 배정비중)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("기여수익 (%p)")
    axes[1].grid(alpha=0.3, axis="x")

    fig.tight_layout(pad=2.5)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    chart_path = out_dir / f"substrat_decompose_{ts}.png"
    csv_path   = out_dir / f"substrat_decompose_{ts}.csv"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    df_tbl[col_order].to_csv(csv_path)
    print(f"\n✅ 차트: {chart_path}")
    print(f"✅ CSV:  {csv_path}")
    print("\n완료")


if __name__ == "__main__":
    main()
