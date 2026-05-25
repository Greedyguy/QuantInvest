#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kqm 비중 그리드 서치 백테스트
- safe_etf 5전략의 상대 비중을 유지하면서 kqm 비중을 0~25% 탐색
- 최적 조합을 Sharpe / CAGR / MDD 기준으로 정렬해서 출력

Usage: python backtest_grid_search.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

import config
config.DATA_STALE_TOLERANCE_BDAYS = 90
import data_loader
data_loader.DATA_STALE_TOLERANCE_BDAYS = 90

from reports import load_data
from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus
from utils import perf_stats

# ── 탐색 범위 ─────────────────────────────────────────────
KQM_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

# safe_etf 기준 5전략 상대 비중 (합산 1.0)
BASE_CONFIGS = [
    {"name": "kqm_small_cap_v22_short", "role": "short",     "base": 0.30},
    {"name": "hybrid_portfolio_v2_4",   "role": "offensive", "base": 0.20},
    {"name": "kqm_small_cap_v22",       "role": "offensive", "base": 0.22},
    {"name": "etf_defensive_safe",      "role": "defensive", "base": 0.18},
    {"name": "k200_mean_rev",           "role": "offensive", "base": 0.10},
]
BASE_TOTAL = sum(c["base"] for c in BASE_CONFIGS)  # 1.0


def make_strategy(kqm_w: float) -> MultiStrategyAllocatorPlus:
    """kqm 비중 kqm_w 로 나머지 5전략 비중을 스케일한 전략 인스턴스 반환"""
    remain = 1.0 - kqm_w
    configs = []
    for c in BASE_CONFIGS:
        w = round(c["base"] / BASE_TOTAL * remain, 4)
        configs.append({"name": c["name"], "weight": w, "role": c["role"]})
    if kqm_w > 0:
        configs.append({"name": "kqm", "weight": round(kqm_w, 4), "role": "offensive"})

    strat = MultiStrategyAllocatorPlus()
    strat.strategy_configs     = configs
    strat.strategy_names       = [c["name"] for c in configs]
    strat.strategy_base_weight = {c["name"]: c["weight"] for c in configs}
    strat.strategy_roles       = {c["name"]: c["role"]   for c in configs}
    return strat


def weight_label(kqm_w: float) -> str:
    """출력용 라벨"""
    if kqm_w == 0:
        return "kqm=0% (safe_etf 기준)"
    pct = int(kqm_w * 100)
    return f"kqm={pct}%"


def main():
    from config import START
    import datetime
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("=" * 65)
    print("  kqm 비중 그리드 서치")
    print(f"  기간: {START} ~ {end_date}")
    print(f"  탐색: kqm 비중 {[int(w*100) for w in KQM_WEIGHTS]}%")
    print("=" * 65)

    print("\n📂 데이터 로딩 중... (캐시 전용)")
    enriched, idx_map = load_data(use_cache=True, incremental=False,
                                  include_market_cap=False)
    idx_kosdaq = idx_map.get("KOSDAQ")
    print(f"✅ 로딩 완료: {len(enriched)}개 종목\n")

    results = []
    for kqm_w in KQM_WEIGHTS:
        label = weight_label(kqm_w)
        strat = make_strategy(kqm_w)

        # 비중 출력
        cfg_str = "  ".join(
            f"{c['name'].split('_')[-1] if 'kqm' not in c['name'] else c['name']}:{int(c['weight']*100)}%"
            for c in strat.strategy_configs
        )
        print(f"▶ [{label}]  {cfg_str}")

        ec, trades = strat.run_backtest(enriched, market_index=idx_kosdaq, silent=True)
        if ec is None or ec.empty:
            print(f"  → 결과 없음\n")
            continue

        s = perf_stats(ec["equity"])
        total = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
        cagr   = s.get("CAGR", 0) * 100
        sharpe = s.get("Sharpe", 0)
        mdd    = s.get("MDD", 0) * 100

        print(f"  → CAGR {cagr:+.2f}%  Sharpe {sharpe:.3f}  MDD {mdd:.2f}%  최종수익 {total:+.2f}%\n")
        results.append({
            "label":   label,
            "kqm_w":  kqm_w,
            "CAGR":   cagr,
            "Sharpe": sharpe,
            "MDD":    mdd,
            "최종수익": total,
            "ec":     ec,
        })

    if not results:
        print("[ERROR] 결과가 없습니다.")
        return

    # ── 성과 요약 테이블 ────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 그리드 서치 결과 (Sharpe 내림차순)")
    print("=" * 65)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ec"} for r in results])
    df_sorted = df.sort_values("Sharpe", ascending=False)
    print(df_sorted[["label", "CAGR", "Sharpe", "MDD", "최종수익"]].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🏆 최적 조합: {best['label']}  "
          f"CAGR {best['CAGR']:+.2f}%  Sharpe {best['Sharpe']:.3f}  MDD {best['MDD']:.2f}%")

    # ── 차트 ────────────────────────────────────────────────
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))

    for i, r in enumerate(results):
        ec   = r["ec"]
        norm = ec["equity"] / ec["equity"].iloc[0] * 100
        dd   = (ec["equity"] / ec["equity"].cummax() - 1) * 100
        lw   = 2.5 if r["kqm_w"] == best["kqm_w"] else 1.5
        ls   = "-" if r["kqm_w"] == best["kqm_w"] else "--"
        lbl  = r["label"] + (" ★" if r["kqm_w"] == best["kqm_w"] else "")
        axes[0].plot(norm.index, norm.values, label=lbl,
                     color=colors[i], linewidth=lw, linestyle=ls)
        axes[1].plot(dd.index, dd.values, label=lbl,
                     color=colors[i], linewidth=lw, linestyle=ls)
        axes[1].fill_between(dd.index, dd.values, 0, alpha=0.08, color=colors[i])

    axes[0].axhline(100, color="gray", linewidth=0.7, linestyle=":")
    axes[0].set_title(f"kqm 비중 그리드 서치 — 누적 수익률  |  {START} ~ {end_date}",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("수익률 지수 (기준=100)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_title("낙폭 (Drawdown, %)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MDD (%)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"reports/grid_search_kqm_{ts}.png"
    out_csv = f"reports/grid_search_kqm_{ts}.csv"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    df_sorted[["label", "kqm_w", "CAGR", "Sharpe", "MDD", "최종수익"]].to_csv(out_csv, index=False)
    print(f"\n✅ 차트: {out_png}")
    print(f"✅ CSV:  {out_csv}")


if __name__ == "__main__":
    main()
