#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
레짐별 현금 보유 비율 그리드 서치 백테스트

핵심 최적화:
  서브전략 수익률(child_returns)과 레짐(regime_df)을 최초 1회만 계산 후
  노출 파라미터만 교체해서 빠르게 N개 조합 탐색

탐색 파라미터:
  neutral_expo  : 0.65 ~ 1.00  (중립장 노출 / 현금 보유 0%~35%)
  bear_expo     : 0.28 ~ 0.65  (약세장 노출 / 현금 보유 35%~72%)
  exposure_floor: 0.10 ~ 0.30  (최저 보장 노출)

고정:
  bull      = 1.12
  ultra_bear = 0.28

Usage: python backtest_exposure_grid.py
"""

import copy
import itertools
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
from utils import perf_stats

# ── 탐색 범위 ──────────────────────────────────────────────
NEUTRAL_VALS     = [0.65, 0.75, 0.82, 0.90, 1.00]
BEAR_VALS        = [0.28, 0.38, 0.46, 0.55, 0.65]
FLOOR_VALS       = [0.10, 0.15, 0.22, 0.30]

# 고정 파라미터
BULL_EXPO        = 1.12
ULTRA_BEAR_EXPO  = 0.28

# ── 현재 운용 기준값 (비교용) ────────────────────────────
CURRENT = dict(neutral=0.82, bear=0.46, floor=0.22)


def _run_with_params(base_strat, ret_df, regime_df, neutral, bear, floor):
    """
    child_returns·regime_df 재활용 — 노출 파라미터만 교체해서 equity 계산
    """
    import pandas as pd
    import numpy as np

    # 파라미터 임시 교체 (원본 훼손 없이 직접 속성 변경 후 복원)
    orig_re    = base_strat.regime_exposure.copy()
    orig_floor = base_strat.exposure_floor

    base_strat.regime_exposure = {
        "bull":       BULL_EXPO,
        "neutral":    neutral,
        "bear":       bear,
        "ultra_bear": ULTRA_BEAR_EXPO,
    }
    base_strat.exposure_floor = floor

    try:
        shared_index = ret_df.index
        base_w = pd.Series(base_strat.strategy_base_weight)
        base_w = base_w / base_w.sum()

        raw_weights = base_strat._dynamic_strategy_weights(ret_df) \
                          .reindex(shared_index).ffill().fillna(base_w)

        expos = base_strat._dynamic_exposure(regime_df, shared_index)
        expos = expos.shift(1).fillna(neutral)
        expos = expos.clip(lower=floor, upper=1.2)
        expos = expos.reindex(shared_index).fillna(neutral)

        base_blended = (raw_weights * ret_df.reindex(shared_index)).sum(axis=1)
        fast_signal  = base_strat._meta_fast_signal(base_blended)

        expos, stress_levels = base_strat._performance_stress(expos, base_blended)
        expos = base_strat._apply_momentum_exposure_boost(expos, fast_signal)

        strategy_weights = base_strat._apply_regime_bias(raw_weights, expos,
                                                          stress_levels=stress_levels)
        strategy_weights = base_strat._apply_performance_filter(strategy_weights, ret_df)
        strategy_weights = base_strat._apply_fast_momentum_boost(strategy_weights, ret_df)
        strategy_weights = base_strat._apply_recent_acceleration(strategy_weights, fast_signal)

        blended     = (strategy_weights * ret_df.reindex(shared_index)).sum(axis=1)
        combined_ret = expos * blended

        # vol scaling (기존 로직 유지)
        ann_vol = combined_ret.std() * np.sqrt(252)
        target_vol_series = base_strat._vol_target_series(expos)
        desired_vol = None
        if base_strat.vol_target is not None:
            desired_vol = base_strat.vol_target
        elif not target_vol_series.empty:
            desired_vol = target_vol_series.median()
        if desired_vol and desired_vol > 0 and ann_vol > 0:
            combined_ret = combined_ret * (desired_vol / ann_vol)

        equity = (1.0 + combined_ret).cumprod() * 1_000_000.0
        return pd.DataFrame({"equity": equity})

    finally:
        # 원본 복원
        base_strat.regime_exposure = orig_re
        base_strat.exposure_floor  = orig_floor


def main():
    from config import START
    import datetime
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    print("=" * 70)
    print("  레짐별 노출 비율 그리드 서치")
    print(f"  기간: {START} ~ {end_date}")
    print(f"  탐색: neutral {NEUTRAL_VALS}  ×  bear {BEAR_VALS}  ×  floor {FLOOR_VALS}")
    print("=" * 70)

    # ── 1. 데이터 로드 ─────────────────────────────────────
    print("\n📂 데이터 로딩 중...")
    enriched, idx_map = load_data(use_cache=True, incremental=False,
                                  include_market_cap=False)
    idx_kosdaq = idx_map.get("KOSDAQ")
    print(f"✅ {len(enriched)}개 종목 로딩 완료\n")

    # ── 2. 서브전략 수익률 1회 계산 ─────────────────────────
    print("⚙️  서브전략 수익률 계산 중... (1회만 실행)")
    base_strat = get_strategy("multi_allocator_plus_safe_etf_kqm")
    child_results = base_strat._run_child_strategies(
        enriched, idx_kosdaq, weights_override=None, silent=True)
    child_returns = base_strat._build_child_returns(child_results)
    ret_df = pd.concat(child_returns.values(), axis=1).fillna(0.0)
    ret_df.columns = list(child_returns.keys())
    regime_df = base_strat._prepare_regime(idx_kosdaq)
    print(f"✅ 서브전략 수익률 계산 완료: {ret_df.shape[1]}개 전략, {len(ret_df)}일\n")

    # ── 3. 유효 조합 생성 (floor ≤ bear) ────────────────────
    combos = [
        (n, b, f)
        for n, b, f in itertools.product(NEUTRAL_VALS, BEAR_VALS, FLOOR_VALS)
        if f <= b
    ]
    print(f"🔍 유효 조합: {len(combos)}개 (floor ≤ bear 제약 적용)\n")

    # ── 4. 그리드 서치 ─────────────────────────────────────
    results = []
    for i, (neutral, bear, floor) in enumerate(combos, 1):
        is_current = (neutral == CURRENT["neutral"] and
                      bear   == CURRENT["bear"]    and
                      floor  == CURRENT["floor"])
        tag = " ← 현재값" if is_current else ""

        ec = _run_with_params(base_strat, ret_df, regime_df, neutral, bear, floor)
        s  = perf_stats(ec["equity"])
        total = (ec["equity"].iloc[-1] / ec["equity"].iloc[0] - 1) * 100

        row = {
            "neutral":  neutral,
            "bear":     bear,
            "floor":    floor,
            "cash_neutral": round((1 - neutral) * 100, 1),
            "cash_bear":    round((1 - bear)    * 100, 1),
            "CAGR":    round(s.get("CAGR",   0) * 100, 3),
            "Sharpe":  round(s.get("Sharpe", 0),       3),
            "MDD":     round(s.get("MDD",    0) * 100, 3),
            "최종수익": round(total, 2),
            "현재값":  is_current,
            "ec":      ec,
        }
        results.append(row)

        if i % 10 == 0 or i == len(combos) or is_current:
            print(f"  [{i:3d}/{len(combos)}] neutral={neutral:.2f}(현금{row['cash_neutral']}%)"
                  f"  bear={bear:.2f}(현금{row['cash_bear']}%)"
                  f"  floor={floor:.2f}"
                  f"  → CAGR {row['CAGR']:+.2f}%  Sharpe {row['Sharpe']:.3f}"
                  f"  MDD {row['MDD']:.2f}%{tag}")

    # ── 5. 결과 분석 ────────────────────────────────────────
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ec"} for r in results])

    print("\n" + "=" * 70)
    print("📊 TOP 15 조합 (Sharpe 내림차순)")
    print("=" * 70)
    top15 = df.sort_values("Sharpe", ascending=False).head(15)
    print(top15[["neutral", "cash_neutral", "bear", "cash_bear", "floor",
                 "CAGR", "Sharpe", "MDD", "최종수익", "현재값"]].to_string(index=False))

    best = df.sort_values("Sharpe", ascending=False).iloc[0]
    curr = df[df["현재값"] == True].iloc[0] if df["현재값"].any() else None

    print(f"\n🏆 최적 조합:")
    print(f"   neutral={best['neutral']:.2f} (현금 {best['cash_neutral']}%)"
          f"  bear={best['bear']:.2f} (현금 {best['cash_bear']}%)"
          f"  floor={best['floor']:.2f}")
    print(f"   CAGR {best['CAGR']:+.2f}%  Sharpe {best['Sharpe']:.3f}"
          f"  MDD {best['MDD']:.2f}%  최종 {best['최종수익']:+.1f}%")

    if curr is not None:
        print(f"\n📌 현재 운용값:")
        print(f"   neutral={curr['neutral']:.2f} (현금 {curr['cash_neutral']}%)"
              f"  bear={curr['bear']:.2f} (현금 {curr['cash_bear']}%)"
              f"  floor={curr['floor']:.2f}")
        print(f"   CAGR {curr['CAGR']:+.2f}%  Sharpe {curr['Sharpe']:.3f}"
              f"  MDD {curr['MDD']:.2f}%  최종 {curr['최종수익']:+.1f}%")
        print(f"\n   개선폭: CAGR {best['CAGR']-curr['CAGR']:+.2f}%p"
              f"  Sharpe {best['Sharpe']-curr['Sharpe']:+.3f}"
              f"  MDD {best['MDD']-curr['MDD']:+.2f}%p")

    # ── 6. 히트맵 차트 ─────────────────────────────────────
    # floor=0.22(현재값) 고정 슬라이스로 neutral×bear 히트맵 3종
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics   = ["Sharpe", "CAGR", "MDD"]
    cmaps     = ["YlGn", "YlGn", "RdYlGn_r"]

    for ax, metric, cmap in zip(axes, metrics, cmaps):
        for floor_val in [0.22]:  # 현재 floor 기준 슬라이스
            sub = df[df["floor"] == floor_val].copy()
            pivot = sub.pivot(index="bear", columns="neutral", values=metric)
            pivot = pivot.sort_index(ascending=False)

            im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2f}\n(현금{int((1-v)*100)}%)"
                                 for v in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.2f}\n(현금{int((1-v)*100)}%)"
                                 for v in pivot.index], fontsize=8)
            ax.set_xlabel("neutral 노출")
            ax.set_ylabel("bear 노출")
            ax.set_title(f"{metric}  (floor=0.22 고정)", fontsize=12, fontweight="bold")

            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.values[r, c]
                    if not np.isnan(val):
                        # 현재값 표시
                        is_curr = (pivot.columns[c] == CURRENT["neutral"] and
                                   pivot.index[r] == CURRENT["bear"])
                        txt = f"{val:.2f}" if metric == "Sharpe" else f"{val:.1f}"
                        weight = "bold" if is_curr else "normal"
                        color  = "red"  if is_curr else "black"
                        ax.text(c, r, txt, ha="center", va="center",
                                fontsize=8, fontweight=weight, color=color)
            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"레짐별 노출 비율 그리드 서치  |  {START} ~ {end_date}  (빨간숫자=현재값)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"reports/exposure_grid_{ts}.png"
    out_csv = f"reports/exposure_grid_{ts}.csv"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    df.drop(columns=["ec"], errors="ignore").to_csv(out_csv, index=False)
    print(f"\n✅ 히트맵 차트: {out_png}")
    print(f"✅ 전체 결과 CSV: {out_csv}")


if __name__ == "__main__":
    main()
