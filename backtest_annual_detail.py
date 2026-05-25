#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연도별 상세 성과 분석
- 현재 운용(no_etf) / 개선안A(safe_etf) / 최적안(kqm=25%) 비교
- 연도별 수익률, MDD, Sharpe 및 월별 히트맵 출력

Usage: python backtest_annual_detail.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

import config
config.DATA_STALE_TOLERANCE_BDAYS = 90
import data_loader
data_loader.DATA_STALE_TOLERANCE_BDAYS = 90

from reports import load_data
from strategies import get_strategy
from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus
from utils import perf_stats

# ── 비교 대상 정의 ─────────────────────────────────────────
def make_kqm25() -> MultiStrategyAllocatorPlus:
    """kqm=25% 최적 조합 전략 인스턴스"""
    BASE = [
        {"name": "kqm_small_cap_v22_short", "role": "short",     "base": 0.30},
        {"name": "hybrid_portfolio_v2_4",   "role": "offensive", "base": 0.20},
        {"name": "kqm_small_cap_v22",       "role": "offensive", "base": 0.22},
        {"name": "etf_defensive_safe",      "role": "defensive", "base": 0.18},
        {"name": "k200_mean_rev",           "role": "offensive", "base": 0.10},
    ]
    kqm_w = 0.25
    remain = 1.0 - kqm_w
    configs = [{"name": c["name"], "weight": round(c["base"] * remain, 4),
                "role": c["role"]} for c in BASE]
    configs.append({"name": "kqm", "weight": kqm_w, "role": "offensive"})

    strat = MultiStrategyAllocatorPlus()
    strat.strategy_configs     = configs
    strat.strategy_names       = [c["name"] for c in configs]
    strat.strategy_base_weight = {c["name"]: c["weight"] for c in configs}
    strat.strategy_roles       = {c["name"]: c["role"]   for c in configs}
    return strat


STRATEGIES = [
    ("multi_allocator_plus_no_etf", "현재 운용 (no_etf)",    "#E53935"),
    ("multi_allocator_plus_safe_etf", "개선안A (safe_etf)",  "#43A047"),
    (None,                           "최적안 (kqm=25%)",     "#1565C0"),
]
COLORS = [s[2] for s in STRATEGIES]
LABELS = [s[1] for s in STRATEGIES]


def annual_stats(equity: pd.Series):
    """연도별 수익률·MDD·Sharpe 계산"""
    rows = []
    for year, grp in equity.groupby(equity.index.year):
        ret   = grp.iloc[-1] / grp.iloc[0] - 1
        dd    = (grp / grp.cummax() - 1).min()
        daily = grp.pct_change().dropna()
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0
        rows.append({"연도": year, "수익률(%)": ret*100, "MDD(%)": dd*100, "Sharpe": sharpe})
    return pd.DataFrame(rows).set_index("연도")


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """월별 수익률 피벗 테이블"""
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [f"{m}월" for m in pivot.columns]
    return pivot


def main():
    from config import START
    import datetime
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("=" * 65)
    print("  연도별 상세 성과 분석")
    print(f"  기간: {START} ~ {end_date}")
    print("=" * 65)

    print("\n📂 데이터 로딩 중...")
    enriched, idx_map = load_data(use_cache=True, incremental=False,
                                  include_market_cap=False)
    idx_kosdaq = idx_map.get("KOSDAQ")
    print(f"✅ 로딩 완료: {len(enriched)}개 종목\n")

    results = []
    for name, label, color in STRATEGIES:
        if name is None:
            strat = make_kqm25()
        else:
            strat = get_strategy(name)
        print(f"▶ [{label}] 백테스트 중...")
        ec, _ = strat.run_backtest(enriched, market_index=idx_kosdaq, silent=True)
        results.append((label, color, ec))
        if ec is not None and not ec.empty:
            s = perf_stats(ec["equity"])
            print(f"  → CAGR {s['CAGR']*100:+.2f}%  Sharpe {s['Sharpe']:.3f}  MDD {s['MDD']*100:.2f}%\n")

    # ── 연도별 수익률 표 ─────────────────────────────────
    print("\n" + "=" * 65)
    print("📅 연도별 수익률 / MDD")
    print("=" * 65)

    ann_dfs = {}
    for label, color, ec in results:
        if ec is None or ec.empty:
            continue
        ann_dfs[label] = annual_stats(ec["equity"])

    # 수익률 비교
    print(f"\n{'연도':<6}", end="")
    for label in ann_dfs:
        print(f"  {label:<28}", end="")
    print()
    print("-" * (6 + 30 * len(ann_dfs)))

    years = sorted(set().union(*[df.index for df in ann_dfs.values()]))
    for yr in years:
        print(f"{yr:<6}", end="")
        for label, df in ann_dfs.items():
            if yr in df.index:
                r   = df.loc[yr, "수익률(%)"]
                mdd = df.loc[yr, "MDD(%)"]
                sign = "✅" if r >= 0 else "❌"
                print(f"  {sign} {r:+6.1f}%  MDD {mdd:5.1f}%     ", end="")
            else:
                print(f"  {'N/A':<28}", end="")
        print()

    # 전체 요약
    print("\n" + "=" * 65)
    print("📊 전체 구간 요약")
    print("=" * 65)
    for label, color, ec in results:
        if ec is None or ec.empty:
            continue
        s = perf_stats(ec["equity"])
        total = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
        loss_yrs = sum(1 for df in [ann_dfs[label]] for yr in df.index if df.loc[yr, "수익률(%)"] < 0)
        print(f"  {label:<30} CAGR {s['CAGR']*100:+.2f}%  Sharpe {s['Sharpe']:.3f}"
              f"  MDD {s['MDD']*100:.2f}%  최종 {total:+.1f}%  손실연도 {loss_yrs}년")

    # ── 월별 히트맵 (최적안만) ─────────────────────────────
    print("\n📆 월별 수익률 히트맵 (최적안 kqm=25%)")
    opt_ec = results[2][2]  # 최적안
    if opt_ec is not None and not opt_ec.empty:
        pivot = monthly_returns(opt_ec["equity"])
        print(pivot.round(1).to_string())

    # ── 차트 (4개 패널) ──────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    ax_eq  = fig.add_subplot(gs[0, :])   # 누적 수익률 (전 구간)
    ax_dd  = fig.add_subplot(gs[1, :])   # 낙폭
    ax_ann = fig.add_subplot(gs[2, 0])   # 연도별 수익률 막대
    ax_hm  = fig.add_subplot(gs[2, 1])   # 월별 히트맵

    # 1) 누적 수익률
    for label, color, ec in results:
        if ec is None or ec.empty: continue
        norm = ec["equity"] / ec["equity"].iloc[0] * 100
        lw = 2.5 if "최적" in label else 1.8
        ls = "-" if "최적" in label else "--"
        ax_eq.plot(norm.index, norm.values, label=label, color=color, lw=lw, ls=ls)
    ax_eq.axhline(100, color="gray", lw=0.7, ls=":")
    ax_eq.set_title(f"누적 수익률 비교  |  {START} ~ {end_date}", fontsize=13, fontweight="bold")
    ax_eq.set_ylabel("수익률 지수 (기준=100)")
    ax_eq.legend(fontsize=9); ax_eq.grid(alpha=0.3)

    # 2) 낙폭
    for label, color, ec in results:
        if ec is None or ec.empty: continue
        dd = (ec["equity"] / ec["equity"].cummax() - 1) * 100
        lw = 2.5 if "최적" in label else 1.8
        ls = "-" if "최적" in label else "--"
        ax_dd.plot(dd.index, dd.values, label=label, color=color, lw=lw, ls=ls)
        ax_dd.fill_between(dd.index, dd.values, 0,
                           alpha=0.25 if "최적" in label else 0.08, color=color)
    ax_dd.set_title("낙폭 (Drawdown, %)", fontsize=13, fontweight="bold")
    ax_dd.set_ylabel("MDD (%)"); ax_dd.legend(fontsize=9); ax_dd.grid(alpha=0.3)

    # 3) 연도별 수익률 막대
    years_list = list(years)
    x = np.arange(len(years_list))
    width = 0.28
    for i, (label, df) in enumerate(ann_dfs.items()):
        vals = [df.loc[yr, "수익률(%)"] if yr in df.index else 0 for yr in years_list]
        color = COLORS[i]
        bars = ax_ann.bar(x + (i - 1) * width, vals, width,
                          label=label, color=color, alpha=0.8)
    ax_ann.axhline(0, color="black", lw=0.8)
    ax_ann.set_xticks(x); ax_ann.set_xticklabels(years_list, fontsize=9)
    ax_ann.set_title("연도별 수익률 (%)", fontsize=12, fontweight="bold")
    ax_ann.set_ylabel("%"); ax_ann.legend(fontsize=8); ax_ann.grid(alpha=0.3, axis="y")

    # 4) 월별 히트맵 (최적안)
    if opt_ec is not None and not opt_ec.empty:
        pivot_vals = monthly_returns(opt_ec["equity"])
        vmax = max(abs(pivot_vals.values[~np.isnan(pivot_vals.values)]).max(), 1)
        cmap = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax_hm.imshow(pivot_vals.values, aspect="auto", cmap="RdYlGn", norm=cmap)
        ax_hm.set_xticks(range(len(pivot_vals.columns)))
        ax_hm.set_xticklabels(pivot_vals.columns, fontsize=8)
        ax_hm.set_yticks(range(len(pivot_vals.index)))
        ax_hm.set_yticklabels(pivot_vals.index, fontsize=9)
        for r in range(pivot_vals.shape[0]):
            for c in range(pivot_vals.shape[1]):
                val = pivot_vals.values[r, c]
                if not np.isnan(val):
                    ax_hm.text(c, r, f"{val:.1f}", ha="center", va="center",
                               fontsize=7, color="black")
        fig.colorbar(im, ax=ax_hm, shrink=0.8)
        ax_hm.set_title("월별 수익률 히트맵 — 최적안 (kqm=25%)", fontsize=12, fontweight="bold")

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"reports/annual_detail_{ts}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n✅ 차트 저장: {out_png}")


if __name__ == "__main__":
    main()
