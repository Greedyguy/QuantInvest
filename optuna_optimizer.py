#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 기반 하이퍼파라미터 자동 최적화

목표:
- 팩터 가중치 최적화 (Momentum, Quality, Volatility, Value)
- 리밸런싱 주기 최적화 (5~20일)
- 과적합 방지: Train/Valid 분리
- 목표 함수: Sharpe Ratio + MDD 페널티
"""

import optuna
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from strategies.strategy_kqm_v3 import KQMStrategyV3
from reports import load_data
from config import START


def calculate_metrics(equity_curve: pd.DataFrame) -> dict:
    """성과 지표 계산"""
    if equity_curve.empty or len(equity_curve) < 2:
        return {"sharpe": -10, "mdd": -1, "cagr": -1}
    
    equity_curve = equity_curve.sort_index()
    returns = equity_curve["equity"].pct_change().dropna()
    
    if len(returns) < 2:
        return {"sharpe": -10, "mdd": -1, "cagr": -1}
    
    # Sharpe Ratio (연율화)
    mean_ret = returns.mean() * 252
    std_ret = returns.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else -10
    
    # MDD
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()
    
    # CAGR
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    total_return = equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1
    cagr = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
    
    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "cagr": cagr
    }


def objective(trial, enriched_train, enriched_valid):
    """Optuna 목표 함수"""
    # 1️⃣ 팩터 가중치 탐색
    w_mom6 = trial.suggest_float("w_mom6", 0.2, 0.5)
    w_mom3 = trial.suggest_float("w_mom3", 0.05, 0.2)
    w_quality = trial.suggest_float("w_quality", 0.1, 0.4)
    w_vol = trial.suggest_float("w_vol", 0.1, 0.4)
    w_val = trial.suggest_float("w_val", 0.0, 0.3)
    
    # 정규화
    total = w_mom6 + w_mom3 + w_quality + w_vol + w_val
    if total == 0:
        return -10
    
    factor_weights = {
        'MOM6': w_mom6 / total,
        'MOM3': w_mom3 / total,
        'QUALITY': w_quality / total,
        'VOL': w_vol / total,
        'VAL': w_val / total,
    }
    
    # 2️⃣ 리밸런싱 주기 탐색 (5~20일)
    rebal_days = trial.suggest_int("rebal_days", 5, 20)
    
    # 3️⃣ 거래대금 필터 탐색 (억원 단위)
    # min_trade_value_threshold = trial.suggest_float("min_trade_value", 10.0, 100.0)  # 10억~100억
    
    # 전략 인스턴스 생성 (v3 기반 최적화)
    strategy = KQMStrategyV3(
        rebal_days=rebal_days,
        factor_weights=factor_weights
    )
    
    # Train 백테스트
    try:
        ec_train, _ = strategy.run_backtest(enriched_train, silent=True)
        if ec_train.empty:
            return -10
        
        metrics_train = calculate_metrics(ec_train)
        
        # Valid 백테스트
        ec_valid, _ = strategy.run_backtest(enriched_valid, silent=True)
        if ec_valid.empty:
            return -10
        
        metrics_valid = calculate_metrics(ec_valid)
        
        # 목표 함수: Train/Valid 평균 Sharpe + MDD 페널티
        # MDD가 클수록 페널티 (음수이므로 더하기)
        score_train = metrics_train["sharpe"] + 0.5 * metrics_train["mdd"]
        score_valid = metrics_valid["sharpe"] + 0.5 * metrics_valid["mdd"]
        
        # Train/Valid 균형
        score = 0.6 * score_train + 0.4 * score_valid
        
        # 로깅
        trial.set_user_attr("train_sharpe", metrics_train["sharpe"])
        trial.set_user_attr("train_mdd", metrics_train["mdd"])
        trial.set_user_attr("train_cagr", metrics_train["cagr"])
        trial.set_user_attr("valid_sharpe", metrics_valid["sharpe"])
        trial.set_user_attr("valid_mdd", metrics_valid["mdd"])
        trial.set_user_attr("valid_cagr", metrics_valid["cagr"])
        trial.set_user_attr("rebal_days", rebal_days)
        
        return score
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return -10


def split_data_by_date(enriched, split_date):
    """데이터를 날짜 기준으로 분할"""
    split_pd = pd.to_datetime(split_date)
    
    enriched_before = {}
    enriched_after = {}
    
    for ticker, df in enriched.items():
        if df is None or df.empty:
            continue
        
        df_before = df[df.index < split_pd]
        df_after = df[df.index >= split_pd]
        
        if not df_before.empty:
            enriched_before[ticker] = df_before
        if not df_after.empty:
            enriched_after[ticker] = df_after
    
    return enriched_before, enriched_after


def optimize_weights(n_trials=100, train_end="2021-12-31", valid_end="2023-12-31"):
    """
    하이퍼파라미터 최적화
    
    Args:
        n_trials: Optuna 시도 횟수
        train_end: 훈련 데이터 종료일
        valid_end: 검증 데이터 종료일
    """
    print("\n" + "="*60)
    print("🔬 Optuna 하이퍼파라미터 최적화 시작")
    print("="*60)
    print(f"📊 시도 횟수: {n_trials}")
    print(f"📅 Train: {START} ~ {train_end}")
    print(f"📅 Valid: {train_end} ~ {valid_end}")
    
    # 데이터 로드
    print("\n📂 데이터 로딩 중...")
    enriched = load_data(use_cache=True)
    
    # Train/Valid 분할
    print("✂️  데이터 분할 중...")
    enriched_train, enriched_temp = split_data_by_date(enriched, train_end)
    enriched_valid, _ = split_data_by_date(enriched_temp, valid_end)
    
    print(f"✅ Train: {len(enriched_train)}개 종목")
    print(f"✅ Valid: {len(enriched_valid)}개 종목")
    
    # Optuna Study 생성
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 최적화 실행
    print("\n🔍 최적화 실행 중...")
    study.optimize(
        lambda trial: objective(trial, enriched_train, enriched_valid),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("✅ 최적화 완료")
    print("="*60)
    
    best_trial = study.best_trial
    print(f"\n🏆 Best Score: {best_trial.value:.4f}")
    print(f"\n📊 Best Parameters:")
    for key, value in best_trial.params.items():
        if key == "rebal_days":
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.4f}")
    
    print(f"\n📈 Train Metrics:")
    print(f"   Sharpe: {best_trial.user_attrs.get('train_sharpe', 0):.4f}")
    print(f"   CAGR: {best_trial.user_attrs.get('train_cagr', 0):.2%}")
    print(f"   MDD: {best_trial.user_attrs.get('train_mdd', 0):.2%}")
    
    print(f"\n📈 Valid Metrics:")
    print(f"   Sharpe: {best_trial.user_attrs.get('valid_sharpe', 0):.4f}")
    print(f"   CAGR: {best_trial.user_attrs.get('valid_cagr', 0):.2%}")
    print(f"   MDD: {best_trial.user_attrs.get('valid_mdd', 0):.2%}")
    
    # 정규화된 가중치 계산 (rebal_days 제외)
    weight_params = {k: v for k, v in best_trial.params.items() if k != "rebal_days"}
    total = sum(weight_params.values())
    normalized_weights = {
        'MOM6': weight_params['w_mom6'] / total,
        'MOM3': weight_params['w_mom3'] / total,
        'QUALITY': weight_params['w_quality'] / total,
        'VOL': weight_params['w_vol'] / total,
        'VAL': weight_params['w_val'] / total,
    }
    
    print(f"\n🎯 Normalized Weights:")
    for key, value in normalized_weights.items():
        print(f"   {key}: {value:.4f}")
    
    # 결과 저장
    output_dir = os.path.join(os.path.dirname(__file__), "data", "meta")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "kqm_optuna_weights.json")
    
    result = {
        "optimized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": n_trials,
        "best_score": best_trial.value,
        "train_period": f"{START} ~ {train_end}",
        "valid_period": f"{train_end} ~ {valid_end}",
        "raw_params": best_trial.params,
        "normalized_weights": normalized_weights,
        "rebal_days": best_trial.params.get('rebal_days', 10),
        "train_metrics": {
            "sharpe": best_trial.user_attrs.get('train_sharpe', 0),
            "cagr": best_trial.user_attrs.get('train_cagr', 0),
            "mdd": best_trial.user_attrs.get('train_mdd', 0)
        },
        "valid_metrics": {
            "sharpe": best_trial.user_attrs.get('valid_sharpe', 0),
            "cagr": best_trial.user_attrs.get('valid_cagr', 0),
            "mdd": best_trial.user_attrs.get('valid_mdd', 0)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    return normalized_weights, study


if __name__ == "__main__":
    import sys
    
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    weights, study = optimize_weights(
        n_trials=n_trials,
        train_end="2021-12-31",
        valid_end="2023-12-31"
    )
    
    print("\n✅ 최적화 완료!")
    print("💡 최적 가중치를 KQM v3.1 전략에 적용하세요:")
    print(f"   python reports.py --strategy kqm_v3_1")

