#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
레짐 없이 고정 노출 비율로 운용하면 어떨까?

비교 대상:
  1. 고정 100%  — 레짐 무시, 항상 완전 투자
  2. 고정  90%  — 10% 항상 현금
  3. 고정  80%  — 20% 항상 현금
  4. 현재 운용  — 레짐별 노출 (neutral=0.82, bear=0.46, floor=0.22)
  5. 그리드 최적 — (neutral=0.75, bear=0.46, floor=0.10)

sub-strategy 동적 가중치(rolling Sharpe 재배분)는 모두 동일하게 유지.
레짐 역할 바이어스(offensive/defensive 비율 조정)는 고정 노출 버전에서 제거.

Usage: python backtest_fixed_exposure.py
"""

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
from strategies import get_strategy
from utils import perf_stats

# ── 비교 시나리오 ──────────────────────────────────────────
SCENARIOS = [
    {"label": "고정 100% (레짐 없음)",  "mode": "fixed",  "expo": 1.00, "color": "#D32F2F"},
    {"label": "고정  90% (레짐 없음)",  "mode": "fixed",  "expo": 0.90, "color": "#E64A19"},
    {"label": "고정  80% (레짐 없음)",  "mode": "fixed",  "expo": 0.80, "color": "#F57C00"},
    {"label": "현재 운용 (레짐 있음)",  "mode": "regime",
     "neutral": 0.82, "bear": 0.46, "floor": 0.22, "color": "#1565C0"},
    {"label": "그리드 최적 (레짐 있음)", "mode": "regime",
     "neutral": 0.75, "bear": 0.46, "floor": 0.10, "color": "#2E7D32"},
]


def _run_fixed(base_strat, ret_df, fixed_expo: float) -> pd.DataFrame:
    """레짐 감지 없이 고정 노출로 equity 계산 (동적 sub-weight는 유지)"""
    shared_index = ret_df.index
    base_w = pd.Series(base_strat.strategy_base_weight)
    base_w = base_w / base_w.sum()

    # rolling Sharpe 기반 서브전략 가중치 동적 조정 (유지)
    raw_weights = (base_strat._dynamic_strategy_weights(ret_df)
                   .reindex(shared_index).ffill().fillna(base_w))

    # 고정 노출 — 레짐/스트레스/모멘텀 모두 무시
    expos = pd.Series(fixed_expo, index=shared_index)

    blended     = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
    combined_ret = expos * blended
    equity       = (1.0 + combined_ret).cumprod() * 1_000_000.0
    return pd.DataFrame({"equity": equity})


def _run_regime(base_strat, ret_df, regime_df, neutral, bear, floor) -> pd.DataFrame:
    """레짐 기반 노출 (exposure_grid 스크립트와 동일 로직)"""
    BULL_EXPO = 1.12
    ULTRA_BEAR_EXPO = 0.28

    orig_re    = base_strat.regime_exposure.copy()
    orig_floor = base_strat.exposure_floor
    base_strat.regime_exposure = {
        "bull": BULL_EXPO, "neutral": neutral,
        "bear": bear, "ultra_bear": ULTRA_BEAR_EXPO,
    }
    base_strat.exposure_floor = floor

    try:
        shared_index = ret_df.index
        base_w = pd.Series(base_strat.strategy_base_weight)
        base_w = base_w / base_w.sum()
        raw_weights = (base_strat._dynamic_strategy_weights(ret_df)
                       .reindex(shared_index).ffill().fillna(base_w))

        expos = base_strat._dynamic_exposure(regime_df, shared_index)
        expos = expos.shift(1).fillna(neutral)
        expos = expos.clip(lower=floor, upper=1.2).reindex(shared_index).fillna(neutral)

        base_blended = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
        fast_signal  = base_strat._meta_fast_signal(base_blended)
        expos, stress_levels = base_strat._performance_stress(expos, base_blended)
        expos = base_strat._apply_momentum_exposure_boost(expos, fast_signal)

        sw = base_strat._apply_regime_bias(raw_weights, expos, stress_levels=stress_levels)
        sw = base_strat._apply_performance_filter(sw, ret_df)
        sw = base_strat._apply_fast_momentum_boost(sw, ret_df)
        sw = base_strat._apply_recent_acceleration(sw, fast_signal)

        blended     = (sw * ret_df.reindex(shared_index)).sum(axis=1)
        combined_ret = expos * blended

        ann_vol = combined_ret.std() * np.sqrt(252)
        target_vol_series = base_strat._vol_target_series(expos)
        desired_vol = target_vol_series.median() if not target_vol_series.empty else None
        if desired_vol and desired_vol > 0 and ann_vol > 0:
            combined_ret = combined_ret * (desired_vol / ann_vol)

        equity = (1.0 + combined_ret).cumprod() * 1_000_000.0
        return pd.DataFrame({"equity": equity})
    finally:
        base_strat.regime_exposure = orig_re
        base_strat.exposure_floor  = orig_floor


def annual_stats(equity: pd.Series) -> pd.DataFrame:
    rows = []
    for yr, grp in equity.groupby(equity.index.year):
        ret    = grp.iloc[-1] / grp.iloc[0] - 1
        dd     = (grp / grp.cummax() - 1).min()
        daily  = grp.pct_change().dropna()
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0
        rows.append({"연도": yr, "수익률(%)": round(ret*100,1),
                     "MDD(%)": round(dd*100,1), "Sharpe": round(sharpe,2)})
    return pd.DataFrame(rows).set_index("연도")


def main():
    from config import START
    import datetime
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("=" * 70)
    print("  고정 노출 vs 레짐 기반 노출 비교")
    print(f"  기간: {START} ~ {end_date}")
    print("=" * 70)

    # ── 1. 데이터 & 서브전략 수익률 ────────────────────────
    print("\n📂 데이터 로딩 중...")
    enriched, idx_map = load_data(use_cache=True, incremental=False,
                                  include_market_cap=False)
    idx_kosdaq = idx_map.get("KOSDAQ")
    print(f"✅ {len(enriched)}개 종목\n")

    print("⚙️  서브전략 수익률 계산 중... (1회만)")
    base_strat   = get_strategy("multi_allocator_plus_safe_etf_kqm")
    child_results = base_strat._run_child_strategies(
        enriched, idx_kosdaq, weights_override=None, silent=True)
    child_returns = base_strat._build_child_returns(child_results)
    ret_df    = pd.concat(child_returns.values(), axis=1).fillna(0.0)
    ret_df.columns = list(child_returns.keys())
    regime_df = base_strat._prepare_regime(idx_kosdaq)
    print(f"✅ 완료 ({ret_df.shape[1]}개 전략, {len(ret_df)}일)\n")

    # ── 2. 시나리오 실행 ────────────────────────────────────
    results = []
    for sc in SCENARIOS:
        if sc["mode"] == "fixed":
            ec = _run_fixed(base_strat, ret_df, sc["expo"])
        else:
            ec = _run_regime(base_strat, ret_df, regime_df,
                             sc["neutral"], sc["bear"], sc["floor"])
        s     = perf_stats(ec["equity"])
        total = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100
        results.append({**sc, "ec": ec, "s": s, "total": total})
        print(f"  {sc['label']:<28}  CAGR {s['CAGR']*100:+.2f}%"
              f"  Sharpe {s['Sharpe']:.3f}  MDD {s['MDD']*100:.2f}%"
              f"  최종 {total:+.1f}%")

    # ── 3. 연도별 비교 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("📅 연도별 수익률 / MDD")
    print("=" * 70)
    ann = {r["label"]: annual_stats(r["ec"]["equity"]) for r in results}

    header = f"{'연도':<6}" + "".join(f"  {lb[:14]:<22}" for lb in ann)
    print(header)
    print("-" * len(header))
    years = sorted(set().union(*[df.index for df in ann.values()]))
    for yr in years:
        line = f"{yr:<6}"
        for lb, df in ann.items():
            if yr in df.index:
                r   = df.loc[yr, "수익률(%)"]
                mdd = df.loc[yr, "MDD(%)"]
                sign = "✅" if r >= 0 else "❌"
                line += f"  {sign}{r:+5.1f}% MDD{mdd:5.1f}%  "
            else:
                line += f"  {'N/A':<22}"
        print(line)

    # ── 4. 전체 요약 ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 전체 구간 요약")
    print("=" * 70)
    for r in results:
        s     = r["s"]
        total = r["total"]
        loss  = sum(1 for df in [ann[r["label"]]]
                    for yr in df.index if df.loc[yr, "수익률(%)"] < 0)
        print(f"  {r['label']:<28}  CAGR {s['CAGR']*100:+.2f}%"
              f"  Sharpe {s['Sharpe']:.3f}  MDD {s['MDD']*100:.2f}%"
              f"  최종 {total:+.1f}%  손실연도 {loss}년")

    # ── 5. 차트 ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))

    # 누적 수익률
    for r in results:
        ec   = r["ec"]
        norm = ec["equity"] / ec["equity"].iloc[0] * 100
        lw   = 2.5 if "100%" in r["label"] or "현재" in r["label"] or "최적" in r["label"] else 1.8
        ls   = "--" if "레짐 없음" in r["label"] else "-"
        axes[0].plot(norm.index, norm.values, label=r["label"],
                     color=r["color"], lw=lw, ls=ls)
    axes[0].axhline(100, color="gray", lw=0.7, ls=":")
    axes[0].set_title(f"누적 수익률 비교  |  {START} ~ {end_date}",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("수익률 지수 (기준=100)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # 낙폭
    for r in results:
        ec = r["ec"]
        dd = (ec["equity"] / ec["equity"].cummax() - 1) * 100
        lw = 2.5 if "100%" in r["label"] or "현재" in r["label"] or "최적" in r["label"] else 1.8
        ls = "--" if "레짐 없음" in r["label"] else "-"
        axes[1].plot(dd.index, dd.values, label=r["label"],
                     color=r["color"], lw=lw, ls=ls)
        axes[1].fill_between(dd.index, dd.values, 0, alpha=0.07, color=r["color"])
    axes[1].set_title("낙폭 (Drawdown, %)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MDD (%)"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # 연도별 수익률 막대
    years_list = list(years)
    x = np.arange(len(years_list))
    w = 0.16
    for i, r in enumerate(results):
        df   = ann[r["label"]]
        vals = [df.loc[yr, "수익률(%)"] if yr in df.index else 0 for yr in years_list]
        axes[2].bar(x + (i - 2) * w, vals, w,
                    label=r["label"], color=r["color"], alpha=0.8)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(years_list, fontsize=9)
    axes[2].set_title("연도별 수익률 (%)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("%"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, axis="y")

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"reports/fixed_vs_regime_{ts}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n✅ 차트 저장: {out_png}")


if __name__ == "__main__":
    main()
