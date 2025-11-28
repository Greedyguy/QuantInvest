#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weight_optimizer.py
- 백테스트 기반 가중치 최적화
- perf_stats['Sharpe'] 기준 최대화
- 전략 시스템과 통합
- 베이지안 최적화 (Optuna) 지원
"""

import itertools
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (이미지 저장만, 화면 표시 안 함)
from reports import load_data
from utils import perf_stats
from strategies import get_strategy

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ optuna가 설치되지 않았습니다. 베이지안 최적화를 사용하려면 설치하세요:")
    print("   pip install optuna")


# --------------------------------------------------------
# ⚙️ 가중치 탐색 범위
# --------------------------------------------------------
WEIGHT_SPACE = {
    "LC": [0.0, 0.2, 0.4],
    "VS": [0.2, 0.4, 0.6],
    "BO": [0.1, 0.2],
    "RS": [0.2, 0.4, 0.6],
    "VCP": [0.1, 0.2],
    "GG": [0.05, 0.1]
}


# --------------------------------------------------------
# 🧩 베이지안 최적화 (Optuna)
# --------------------------------------------------------
def optimize_weights_bayesian(strategy, enriched, n_trials=100):
    """
    베이지안 최적화를 사용한 가중치 최적화
    
    Args:
        strategy: 전략 인스턴스
        enriched: enriched 데이터
        n_trials: 최적화 시행 횟수
    
    Returns:
        최적 가중치 딕셔너리와 결과 리스트
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna가 설치되지 않았습니다. pip install optuna")
    
    # 가중치 범위 정의 (연속값으로)
    weight_bounds = {
        "LC": (0.0, 2.0),
        "VS": (0.0, 2.0),
        "BO": (0.0, 1.5),
        "RS": (0.0, 1.5),
        "VCP": (0.0, 0.75),
        "GG": (0.0, 0.75)
    }
    
    results = []
    
    def objective(trial):
        # 가중치 제안
        W = {
            "LC": trial.suggest_float("LC", *weight_bounds["LC"]),
            "VS": trial.suggest_float("VS", *weight_bounds["VS"]),
            "BO": trial.suggest_float("BO", *weight_bounds["BO"]),
            "RS": trial.suggest_float("RS", *weight_bounds["RS"]),
            "VCP": trial.suggest_float("VCP", *weight_bounds["VCP"]),
            "GG": trial.suggest_float("GG", *weight_bounds["GG"])
        }
        
        try:
            # 백테스트 실행
            equity_curve, _ = strategy.run_backtest(enriched, weights=W, silent=True)
            
            if equity_curve is None or equity_curve.empty:
                return -999.0  # 매우 낮은 값 반환
            
            stats = perf_stats(equity_curve)
            sharpe = stats.get('Sharpe', -999.0)
            
            # 결과 저장
            results.append({**W, **stats})
            
            return sharpe
        except Exception as e:
            return -999.0
    
    # 최적화 실행
    study = optuna.create_study(
        direction='maximize',
        study_name=f"weight_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)  # TPE 알고리즘 사용
    )
    
    print(f"🔍 베이지안 최적화 시작 (TPE 알고리즘, {n_trials}회 시행)")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 최적 가중치
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n✅ 최적화 완료!")
    print(f"   최고 Sharpe: {best_value:.4f}")
    print(f"   최적 가중치: {best_params}")
    
    return best_params, results


# --------------------------------------------------------
# 🧩 그리드 서치 최적화 (기존 방식)
# --------------------------------------------------------
def optimize_weights_grid(strategy, enriched, max_combinations=None):
    """
    그리드 서치를 사용한 가중치 최적화 (기존 방식)
    
    Args:
        strategy: 전략 인스턴스
        enriched: enriched 데이터
        max_combinations: 최대 조합 수
    
    Returns:
        최적 가중치 DataFrame (TOP 5)
    """
    # 가중치 조합 생성
    combos = list(itertools.product(*WEIGHT_SPACE.values()))
    total_combos = len(combos)
    
    if max_combinations and total_combos > max_combinations:
        print(f"\n⚠️ 조합 수가 너무 많습니다 ({total_combos}개).")
        print(f"처음 {max_combinations}개만 테스트합니다.")
        combos = combos[:max_combinations]
    
    print(f"\n📊 총 {len(combos)}개 가중치 조합 테스트")
    print("="*60)
    
    results = []
    
    for vals in tqdm(combos, desc="Weight tuning"):
        W = dict(zip(WEIGHT_SPACE.keys(), vals))
        
        # 백테스트 실행 (가중치 전달, silent 모드)
        try:
            equity_curve, _ = strategy.run_backtest(enriched, weights=W, silent=True)
            
            if equity_curve is None or equity_curve.empty:
                continue
            
            stats = perf_stats(equity_curve)
            results.append({**W, **stats})
        except Exception as e:
            print(f"\n⚠️ 가중치 {W} 테스트 중 오류: {e}")
            continue
    
    if not results:
        print("⚠️ 결과가 없습니다.")
        return None
    
    df = pd.DataFrame(results)
    
    # 숫자 컬럼 타입 변환
    numeric_cols = ["Sharpe", "CAGR", "MDD", "Volatility", "MaxDD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sharpe 기준 정렬
    df = df.sort_values("Sharpe", ascending=False, na_last=True)
    
    return df


# --------------------------------------------------------
# 🧩 메인 최적화 함수
# --------------------------------------------------------
def optimize_weights(strategy_name="sector_weighted", use_cache=True, 
                     max_combinations=None, method="bayesian", n_trials=100):
    """
    전략 가중치 최적화
    
    Args:
        strategy_name: 최적화할 전략 이름 ("baseline" or "sector_weighted")
        use_cache: 데이터 캐시 사용 여부
        max_combinations: 최대 조합 수 (그리드 서치에서만 사용)
        method: 최적화 방법 ("bayesian" or "grid")
        n_trials: 베이지안 최적화 시행 횟수
    
    Returns:
        최적 가중치 DataFrame (TOP 5)
    """
    print("\n" + "="*60)
    print(f"🔍 {strategy_name} 전략 가중치 최적화 시작")
    print(f"   방법: {method.upper()}")
    print("="*60)
    
    # 전략 확인
    strategy = get_strategy(strategy_name)
    if strategy is None:
        print(f"⚠️ 전략 '{strategy_name}'을 찾을 수 없습니다.")
        print("사용 가능한 전략:")
        from strategies import list_strategies
        for s in list_strategies():
            print(f"  - {s['name']}: {s['description']}")
        return None
    
    print(f"✅ 전략 선택: {strategy_name}")
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    enriched = load_data(use_cache=use_cache)
    if not enriched:
        print("⚠️ 데이터 로드 실패")
        return None
    
    print(f"✅ 데이터 로드 완료: {len(enriched)}개 종목")
    
    # 최적화 실행
    if method == "bayesian":
        if not OPTUNA_AVAILABLE:
            print("⚠️ optuna가 설치되지 않았습니다. 그리드 서치 방식으로 전환합니다.")
            method = "grid"
        else:
            best_params, results = optimize_weights_bayesian(strategy, enriched, n_trials=n_trials)
            
            if not results:
                print("⚠️ 결과가 없습니다.")
                return None
            
            # 결과를 DataFrame으로 변환
            df = pd.DataFrame(results)
            
            # 숫자 컬럼 타입 변환
            numeric_cols = ["Sharpe", "CAGR", "MDD", "Volatility", "MaxDD"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sharpe 기준 정렬
            df = df.sort_values("Sharpe", ascending=False, na_last=True)
            
            # 최적 가중치가 이미 결과에 있으면 그대로, 없으면 추가
            best = df.head(5)
    
    if method == "grid":
        df = optimize_weights_grid(strategy, enriched, max_combinations=max_combinations)
        
        if df is None:
            return None
        
        best = df.head(5)
    
    # TOP 5 출력
    print("\n" + "="*60)
    print("🏆 최적 가중치 TOP 5 (Sharpe 기준)")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(best[["LC", "VS", "BO", "RS", "VCP", "GG", "Sharpe", "CAGR", "MDD"]].to_string(index=False))
    
    # CSV 저장
    import os
    os.makedirs("./reports", exist_ok=True)
    output_file = f"./reports/best_weights_{strategy_name}_{method}.csv"
    best.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {output_file}")
    
    # 전체 결과도 저장 (선택적)
    all_results_file = f"./reports/all_weights_{strategy_name}_{method}.csv"
    df.to_csv(all_results_file, index=False, encoding='utf-8-sig')
    print(f"💾 전체 결과 저장: {all_results_file}")
    
    print("\n" + "="*60)
    print("✅ 최적화 완료!")
    print("="*60)
    
    return best


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="전략 가중치 최적화")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sector_weighted",
        help="최적화할 전략 이름 (baseline 또는 sector_weighted)",
        choices=["baseline", "sector_weighted"]
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="최대 조합 수 (None이면 전체 탐색)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="데이터 캐시 사용 안 함"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="bayesian",
        choices=["bayesian", "grid"],
        help="최적화 방법: bayesian (베이지안) 또는 grid (그리드 서치)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="베이지안 최적화 시행 횟수 (기본값: 100)"
    )
    
    args = parser.parse_args()
    
    best = optimize_weights(
        strategy_name=args.strategy,
        use_cache=not args.no_cache,
        max_combinations=args.max_combinations,
        method=args.method,
        n_trials=args.n_trials
    )
    
    # 명시적으로 종료
    exit(0)
