#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
레짐 barometer 비교 백테스트 (빠른 버전)
  A) KOSDAQ 단독 (기존)
  B) KOSDAQ 50% + KOSPI 50% 블렌드 (신규)

서브전략은 1회만 실행하고, 레짐 노출 계산만 두 버전으로 비교.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import date
from pathlib import Path

from reports import load_data
from strategies import get_strategy

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

STRATEGY_NAME = "multi_allocator_plus_safe_etf_kqm"
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)


def build_equity(strat, child_results, ret_df, regime_df):
    """서브전략 결과 재사용, 레짐만 교체해 equity curve 계산."""
    shared_index = ret_df.index
    base = pd.Series(strat.strategy_base_weight)
    base = base / base.sum()

    raw_weights = strat._dynamic_strategy_weights(ret_df).reindex(shared_index).ffill().fillna(base)
    expos = strat._dynamic_exposure(regime_df, ret_df.index)
    expos = expos.shift(1).fillna(strat.regime_exposure.get("neutral", 0.7))
    expos = expos.clip(lower=strat.exposure_floor, upper=1.2).reindex(shared_index).fillna(strat.regime_exposure["neutral"])

    base_blended = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
    fast_signal = strat._meta_fast_signal(base_blended)
    expos, stress_levels = strat._performance_stress(expos, base_blended)
    expos = strat._apply_momentum_exposure_boost(expos, fast_signal)
    sw = strat._apply_regime_bias(raw_weights, expos, stress_levels=stress_levels)
    sw = strat._apply_performance_filter(sw, ret_df)
    sw = strat._apply_fast_momentum_boost(sw, ret_df)
    sw = strat._apply_recent_acceleration(sw, fast_signal)

    blended = (sw * ret_df.reindex(shared_index)).sum(axis=1)
    combined = expos * blended
    eq = (1 + combined).cumprod() * 1e9
    return eq


def perf(eq, label):
    eq = eq.dropna()
    ret = eq.pct_change().dropna()
    n_years = len(ret) / 252
    cagr  = (eq.iloc[-1]/eq.iloc[0])**(1/n_years) - 1
    sr    = ret.mean()/ret.std()*np.sqrt(252)
    mdd   = (eq/eq.cummax()-1).min()
    total = eq.iloc[-1]/eq.iloc[0] - 1
    print(f"[{label}]  CAGR={cagr:+.2%}  Sharpe={sr:+.3f}  MDD={mdd:+.2%}  Total={total:+.2%}", flush=True)
    return dict(CAGR=cagr, Sharpe=sr, MDD=mdd, Total=total)


def main():
    print("데이터 로드 중...", flush=True)
    enriched, idx_map = load_data(use_cache=True)
    idx_kosdaq = idx_map.get("KOSDAQ")
    idx_kospi  = idx_map.get("KOSPI")

    strat = get_strategy(STRATEGY_NAME)

    # ── 서브전략 1회 실행 ──────────────────────────────────────
    print("서브전략 실행 중 (1회)...", flush=True)
    child_results = strat._run_child_strategies(enriched, idx_kosdaq, silent=True)
    if not child_results:
        print("서브전략 결과 없음 — 종료")
        return

    child_returns = strat._build_child_returns(child_results)
    ret_df = pd.concat(child_returns.values(), axis=1).fillna(0.0)
    ret_df.columns = list(child_returns.keys())
    print(f"ret_df: {ret_df.shape}  {ret_df.index.min().date()} ~ {ret_df.index.max().date()}", flush=True)

    # ── 레짐 A: KOSDAQ 단독 ────────────────────────────────────
    print("A) KOSDAQ 단독 레짐 계산...", flush=True)
    regime_a = strat._prepare_regime(idx_kosdaq, secondary_index=None)
    eq_a = build_equity(strat, child_results, ret_df, regime_a)

    # ── 레짐 B: KOSDAQ+KOSPI 블렌드 ───────────────────────────
    print("B) KOSDAQ+KOSPI 블렌드 레짐 계산...", flush=True)
    regime_b = strat._prepare_regime(idx_kosdaq, secondary_index=idx_kospi)
    eq_b = build_equity(strat, child_results, ret_df, regime_b)

    # ── 성과 표 ───────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    sa = perf(eq_a, "A. KOSDAQ 단독  (기존)")
    sb = perf(eq_b, "B. KOSDAQ+KOSPI 블렌드")
    print("-"*60, flush=True)
    for k in ["CAGR", "Sharpe", "MDD", "Total"]:
        diff = sb[k] - sa[k]
        fmt = f"{diff:+.3f}" if k == "Sharpe" else f"{diff:+.2%}"
        print(f"  Δ {k:<8}: {fmt}", flush=True)
    print("="*60, flush=True)

    # ── 레짐 노출 최근 30일 비교 ──────────────────────────────
    print("\n[최근 30거래일 레짐 노출 비교]", flush=True)
    dates_last = ret_df.index[-30:]
    expo_a = strat._dynamic_exposure(regime_a, dates_last)
    expo_b = strat._dynamic_exposure(regime_b, dates_last)
    comp = pd.DataFrame({"A_노출": expo_a, "B_노출": expo_b})
    comp["Δ"] = comp["B_노출"] - comp["A_노출"]
    print(comp.tail(10).round(3).to_string(), flush=True)

    # ── 시각화 ────────────────────────────────────────────────
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_png = OUT_DIR / f"regime_barometer_comparison_{ts}.png"

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    norm_a = eq_a / eq_a.iloc[0] * 100
    norm_b = eq_b / eq_b.iloc[0] * 100
    axes[0].plot(norm_a.index, norm_a.values, label="A. KOSDAQ 단독 (기존)")
    axes[0].plot(norm_b.index, norm_b.values, label="B. KOSDAQ+KOSPI 50/50 블렌드", linestyle="--")
    axes[0].set_title("레짐 Barometer 비교 — 누적 수익 (기준=100)", fontsize=13)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    dd_a = eq_a / eq_a.cummax() - 1
    dd_b = eq_b / eq_b.cummax() - 1
    axes[1].plot(dd_a.index, dd_a.values * 100, label="A. KOSDAQ 단독")
    axes[1].plot(dd_b.index, dd_b.values * 100, label="B. 블렌드", linestyle="--")
    axes[1].set_title("낙폭 (Drawdown %)", fontsize=13)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # 노출 비교
    expo_a_full = strat._dynamic_exposure(regime_a, ret_df.index)
    expo_b_full = strat._dynamic_exposure(regime_b, ret_df.index)
    axes[2].plot(expo_a_full.index, expo_a_full.values, label="A. 노출 (KOSDAQ)", alpha=0.7)
    axes[2].plot(expo_b_full.index, expo_b_full.values, label="B. 노출 (블렌드)", alpha=0.7, linestyle="--")
    axes[2].set_title("동적 레짐 노출 비교 (shift 전)", fontsize=13)
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n📊 차트 저장: {out_png}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
