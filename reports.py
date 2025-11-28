#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
백테스트 결과 리포트 생성 스크립트

주요 기능:
1. 기본 모멘텀 전략 백테스트
2. 섹터 가중 전략 백테스트
3. 전략별 성과 지표 비교 (CAGR, Sharpe, MDD, 승률 등)
4. 섹터별 성과 분석 (섹터별 CAGR, Sharpe, MDD, 승률, 평균 수익/손실)
5. 시각화 및 차트 저장
6. CSV 리포트 자동 생성

출력물:
- ./reports/backtest_comparison.png: 전략 비교 차트
- ./reports/sector_performance.png: 섹터별 성과 차트 (전략별)
- ./reports/sector_performance.csv: 섹터별 성과 데이터
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 모듈
from config import *
from data_loader import get_universe, load_panel, get_index_close, infer_market
from signals import compute_indicators, add_rel_strength, scoring
from utils import perf_stats
from cache_manager import (
    save_enriched, load_enriched, save_index, load_index,
    save_last_calc_date, get_last_calc_date,
    get_config_hash, save_backtest_result, load_backtest_result
)
from strategies import get_strategy, list_strategies
from universe_filter import filter_universe

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 📊 1️⃣ 데이터 로드 & 준비
# ============================================================
def build_signal_for_one(ticker, df, idx_map, infer_market_fn, start_date, end_date, use_cache=True):
    """한 종목의 indicator + relative strength 계산 (병렬 처리용)"""
    try:
        if df is None or df.empty:
            return ticker, None, False
        
        # 캐시 확인
        if use_cache:
            cached_df = load_enriched(ticker, start_date, end_date)
            if cached_df is not None:
                return ticker, cached_df, True  # 캐시 히트
        
        mkt = infer_market_fn(ticker) or "KOSPI"
        
        # 지표 계산
        df = compute_indicators(df)
        if df is None or df.empty:
            return ticker, None, False
        
        # 상대강도 계산 (idx_map에 해당 마켓이 없으면 빈 DataFrame 전달)
        idx_data = idx_map.get(mkt, pd.DataFrame())
        df = add_rel_strength(df, idx_data)
        
        # 캐시 저장
        if use_cache:
            save_enriched(ticker, df)
        
        return ticker, df, False  # 캐시 미스
        
    except Exception as e:
        return ticker, None, False


def load_data(use_cache=True, max_workers=8, incremental=True, start_date=None):
    """데이터 로드 및 인디케이터 계산 (캐싱 + 병렬 처리 + 증분 업데이트)"""
    print("\n" + "="*60)
    print("📂 데이터 로드 중...")
    print("="*60)
    
    start = start_date or START
    end = date.today().strftime("%Y-%m-%d") if END is None else END
    
    # 증분 업데이트 확인
    last_calc_date = None
    if use_cache and incremental:
        last_calc_date = get_last_calc_date()
        if last_calc_date:
            last_calc_pd = pd.to_datetime(last_calc_date)
            end_pd = pd.to_datetime(end)
            if last_calc_pd >= end_pd:
                print(f"✅ 데이터가 최신입니다 (마지막 계산일: {last_calc_date})")
                # 전체 캐시 로드
                print("📦 캐시에서 데이터 로드 중...")
            else:
                print(f"📅 증분 업데이트 모드 (마지막 계산일: {last_calc_date} → {end})")
    
    # 전체 유니버스 로드 (ETF 포함)
    universe = get_universe(MARKETS, include_etf=True, include_index_etf=True)
    print(f"✅ 유니버스: {len(universe)}개 종목")
    
    # OHLCV 데이터 로드
    panel = load_panel(universe, start, end, max_workers=6)
    print(f"✅ 데이터 로드 완료: {len(panel)}개 종목")
    
    # 지수 데이터 (캐시 우선)
    print("📊 지수 데이터 로드 중...")
    idx_map = {}
    for market in ["KOSPI", "KOSDAQ"]:
        cached_idx = load_index(market, start, end) if use_cache else None
        if cached_idx is not None:
            idx_map[market] = cached_idx
            print(f"  ✅ {market} (캐시)")
        else:
            idx_data = get_index_close(market, start, end)
            idx_map[market] = idx_data
            if use_cache:
                save_index(market, idx_data)
            print(f"  ✅ {market} (다운로드)")
    
    # 인디케이터 & 상대강도 계산 (병렬 처리)
    print("⚙️ 기술적 지표 계산 중...")
    enriched = {}
    cache_hits = 0
    cache_misses = 0
    incremental_updates = 0
    
    # 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                build_signal_for_one, 
                ticker, df, idx_map, infer_market, start, end, use_cache
            ): ticker
            for ticker, df in panel.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Indicators"):
            ticker = futures[future]
            try:
                t, result_df, is_cache_hit = future.result()
                if result_df is not None and not result_df.empty:
                    # 증분 업데이트: 기존 캐시가 있지만 날짜 범위가 부족한 경우
                    if use_cache and incremental and last_calc_date and not is_cache_hit:
                        cached_df = load_enriched(t, start, str(last_calc_date))
                        if cached_df is not None and not cached_df.empty:
                            # 새 데이터와 병합
                            new_data = result_df[result_df.index > pd.to_datetime(last_calc_date)]
                            if not new_data.empty:
                                combined_df = pd.concat([cached_df, new_data]).sort_index()
                                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                                enriched[t] = combined_df
                                save_enriched(t, combined_df)
                                incremental_updates += 1
                                continue
                    
                    enriched[t] = result_df
                    if is_cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
            except Exception as e:
                cache_misses += 1
                continue
    
    # 마지막 계산일 저장
    if use_cache:
        save_last_calc_date(end)
    
    print(f"✅ 지표 계산 완료: {len(enriched)}개 종목")
    if use_cache:
        print(f"   캐시 히트: {cache_hits}개, 캐시 미스: {cache_misses}개")
        if incremental_updates > 0:
            print(f"   증분 업데이트: {incremental_updates}개")
    print()
    
    # 인덱스 데이터도 함께 반환
    return enriched, idx_map


# ============================================================
# 📈 2️⃣ 백테스트 실행 (전략 기반)
# ============================================================
# 기존 run_baseline_backtest, run_sector_weighted_backtest 함수는
# strategies 모듈로 이동되었습니다.
# 이제 전략 레지스트리에서 전략을 가져와 실행합니다.


# ============================================================
# 📊 4️⃣ 결과 출력 및 비교
# ============================================================
def print_results(results: list):
    """
    결과 요약 출력
    
    Args:
        results: [(strategy_name, equity_curve, trade_log), ...] 형태의 리스트
                 최대 2개 전략 지원
    """
    print("\n" + "="*60)
    print("📊 백테스트 성과 요약")
    print("="*60)
    
    if len(results) == 0:
        print("⚠️ 결과가 없습니다.")
        return
    
    strategy_stats = []
    
    # 각 전략별 성과 출력
    for idx, (strategy_name, equity_curve, trade_log) in enumerate(results, 1):
        if equity_curve is None:
            continue
            
        print(f"\n[{idx}] {strategy_name}")
        print("-" * 40)
        stats = perf_stats(equity_curve)
        strategy_stats.append((strategy_name, stats))
        
        for k, v in stats.items():
            print(f"  {k:<10}: {v:>12.4f}")
        
        if trade_log and len(trade_log) > 0:
            trades_df = pd.DataFrame(trade_log)
            
            # 'ret' 또는 'pnl' 컬럼 사용
            return_col = 'ret' if 'ret' in trades_df.columns else 'pnl'
            
            if return_col in trades_df.columns:
                # SELL 거래만 수익률 계산
                sell_trades = trades_df[trades_df.get('action', 'SELL') == 'SELL'].copy() if 'action' in trades_df.columns else trades_df.copy()
                
                if len(sell_trades) > 0:
                    win_rate = (sell_trades[return_col] > 0).mean()
                    avg_gain = sell_trades.loc[sell_trades[return_col] > 0, return_col].mean() if len(sell_trades[sell_trades[return_col] > 0]) > 0 else 0.0
                    avg_loss = sell_trades.loc[sell_trades[return_col] < 0, return_col].mean() if len(sell_trades[sell_trades[return_col] < 0]) > 0 else 0.0
                    
                    print(f"  {'승률':<10}: {win_rate:>12.2%}")
                    
                    # pnl은 만원 단위로, ret은 비율로 표시
                    if return_col == 'pnl':
                        # pnl을 만원 단위로 변환
                        print(f"  {'평균익절':<10}: {avg_gain/10_000:>12,.1f}만원")
                        print(f"  {'평균손절':<10}: {avg_loss/10_000:>12,.1f}만원")
                    else:
                        print(f"  {'평균익절':<10}: {avg_gain:>12.2%}")
                        print(f"  {'평균손절':<10}: {avg_loss:>12.2%}")
                    
                    print(f"  {'총거래':<10}: {len(sell_trades):>12}회")
    
    # 두 전략 비교
    if len(strategy_stats) == 2:
        print("\n[비교] 전략 성과 차이")
        print("-" * 40)
        name1, stats1 = strategy_stats[0]
        name2, stats2 = strategy_stats[1]
        print(f"  {'CAGR 차이':<20}: {(stats2['CAGR'] - stats1['CAGR']):>10.4f} ({name2} - {name1})")
        print(f"  {'Sharpe 차이':<20}: {(stats2['Sharpe'] - stats1['Sharpe']):>10.4f}")
        print(f"  {'MDD 차이':<20}: {(stats2['MDD'] - stats1['MDD']):>10.4f}")
    
    print("\n" + "="*60)


# ============================================================
# 📈 5️⃣ 시각화
# ============================================================
def plot_results(results: list):
    """
    결과 시각화
    
    Args:
        results: [(strategy_name, equity_curve, trade_log), ...] 형태의 리스트
                 최대 2개 전략 지원
    """
    print("\n📈 차트 생성 중...")
    
    if len(results) == 0:
        print("⚠️ 시각화할 결과가 없습니다.")
        return
    
    # 단일 전략 또는 두 전략 비교
    if len(results) == 1:
        strategy_name, equity_curve, trade_log = results[0]
        if equity_curve is None:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'{strategy_name} 백테스트 결과', fontsize=16, y=0.995)
        
        # (1) Equity Curve
        ax1 = axes[0]
        # equity_curve가 Series인지 DataFrame인지 확인
        if isinstance(equity_curve, pd.Series):
            equity_values = equity_curve
        elif "equity" in equity_curve.columns:
            equity_values = equity_curve["equity"]
        else:
            equity_values = equity_curve.iloc[:, 0] if len(equity_curve.columns) > 0 else pd.Series()
        
        if len(equity_values) > 0:
            ax1.plot(equity_curve.index, equity_values, label=strategy_name, linewidth=2)
        ax1.set_title("누적 수익 곡선")
        ax1.set_ylabel("Equity (원)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # (2) Drawdown
        ax2 = axes[1]
        if len(equity_values) > 0:
            dd = (equity_values / equity_values.cummax() - 1) * 100
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=strategy_name)
        ax2.set_title("낙폭 (Drawdown)")
        ax2.set_ylabel("DD (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    else:  # 두 전략 비교
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('백테스트 결과 비교', fontsize=16, y=1.00)
        
        # (1) Equity Curve 비교
        ax1 = axes[0, 0]
        colors = ['blue', 'orange']
        for idx, (strategy_name, equity_curve, _) in enumerate(results):
            if equity_curve is not None:
                ax1.plot(equity_curve.index, equity_curve["equity"], 
                        label=strategy_name, linewidth=2, alpha=0.8, color=colors[idx % len(colors)])
        ax1.set_title("누적 수익 곡선")
        ax1.set_ylabel("Equity (원)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # (2) Drawdown
        ax2 = axes[0, 1]
        for idx, (strategy_name, equity_curve, _) in enumerate(results):
            if equity_curve is not None:
                dd = (equity_curve["equity"] / equity_curve["equity"].cummax() - 1) * 100
                ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, label=strategy_name, color=colors[idx % len(colors)])
        ax2.set_title("낙폭 (Drawdown)")
        ax2.set_ylabel("DD (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # (3) 첫 번째 전략 트레이드 분포
        ax3 = axes[1, 0]
        if len(results) > 0 and results[0][2] and len(results[0][2]) > 0:
            strategy_name, _, trade_log = results[0]
            trades_df = pd.DataFrame(trade_log)
            
            # 'ret' 또는 'pnl' 컬럼 사용
            return_col = 'ret' if 'ret' in trades_df.columns else 'pnl'
            
            if return_col in trades_df.columns:
                # SELL 거래만 표시
                if 'action' in trades_df.columns:
                    plot_trades = trades_df[trades_df['action'] == 'SELL']
                else:
                    plot_trades = trades_df
                
                if len(plot_trades) > 0:
                    if return_col == 'pnl':
                        values = plot_trades[return_col] / 100_000_000  # 억원
                        ax3.hist(values, bins=30, alpha=0.7, edgecolor='black')
                        ax3.set_xlabel("수익/손실 (억원)")
                    else:
                        values = plot_trades[return_col] * 100  # %
                        ax3.hist(values, bins=30, alpha=0.7, edgecolor='black')
                        ax3.set_xlabel("수익률 (%)")
                    
                    ax3.axvline(0, color='red', linestyle='--', linewidth=1)
                    ax3.set_title(f"{strategy_name} - 트레이드 수익률 분포")
                    ax3.set_ylabel("빈도")
                    ax3.grid(True, alpha=0.3)
        
        # (4) 두 번째 전략 트레이드 분포
        ax4 = axes[1, 1]
        if len(results) > 1 and results[1][2] and len(results[1][2]) > 0:
            strategy_name, _, trade_log = results[1]
            trades_df = pd.DataFrame(trade_log)
            
            # 'ret' 또는 'pnl' 컬럼 사용
            return_col = 'ret' if 'ret' in trades_df.columns else 'pnl'
            
            if return_col in trades_df.columns:
                # SELL 거래만 표시
                if 'action' in trades_df.columns:
                    plot_trades = trades_df[trades_df['action'] == 'SELL']
                else:
                    plot_trades = trades_df
                
                if len(plot_trades) > 0:
                    if return_col == 'pnl':
                        values = plot_trades[return_col] / 100_000_000  # 억원
                        ax4.hist(values, bins=30, alpha=0.7, edgecolor='black', color='orange')
                        ax4.set_xlabel("수익/손실 (억원)")
                    else:
                        values = plot_trades[return_col] * 100  # %
                        ax4.hist(values, bins=30, alpha=0.7, edgecolor='black', color='orange')
                        ax4.set_xlabel("수익률 (%)")
                    
                    ax4.axvline(0, color='red', linestyle='--', linewidth=1)
                    ax4.set_title(f"{strategy_name} - 트레이드 수익률 분포")
                    ax4.set_ylabel("빈도")
                    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = "./reports/backtest_comparison.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 차트 저장 완료: {output_path}")
    
    plt.show()


# ============================================================
# 🧭 6️⃣ 섹터별 성과 리포트
# ============================================================
def sector_performance_report(trades, equity_curve, title="섹터별 성과 리포트"):
    """
    섹터별 CAGR, Sharpe, MDD, 승률, 평균익절/손절, 거래 횟수 계산 및 시각화
    
    Args:
        trades: 거래 로그 (list of dict) - ticker, ret, date 필수
        equity_curve: Equity curve DataFrame (date index, equity column)
        title: 리포트 제목 (CSV/차트 파일명에 사용)
    """
    print("\n" + "="*60)
    print(f"🧭 {title}")
    print("="*60)
    
    if trades is None or len(trades) == 0:
        print("⚠️ 거래 내역이 없어 섹터 리포트를 생성할 수 없습니다.")
        return None
    
    # 섹터 매핑 로드
    try:
        with open("./data/meta/sector_map.pkl", "rb") as f:
            sector_map = pickle.load(f)
        print(f"✅ 섹터 매핑 로드: {len(sector_map)}개 종목")
    except FileNotFoundError:
        print("⚠️ sector_map.pkl 파일을 찾을 수 없습니다.")
        return None
    
    df_trades = pd.DataFrame(trades)
    if "ticker" not in df_trades.columns or "ret" not in df_trades.columns:
        print("⚠️ trade_log 형식이 올바르지 않습니다. (ticker, ret 필수)")
        return None
    
    # 섹터 매핑 추가
    df_trades["sector"] = df_trades["ticker"].map(lambda x: sector_map.get(x, "기타"))
    
    # Equity curve 기반 전체 기간 계산
    total_days = len(equity_curve) if equity_curve is not None and len(equity_curve) > 0 else 0
    total_cagr = 0
    if total_days > 0:
        start_val = equity_curve["equity"].iloc[0]
        end_val = equity_curve["equity"].iloc[-1]
        if start_val > 0:
            total_cagr = (end_val / start_val) ** (252 / total_days) - 1
    
    # 섹터별 성과 집계
    sector_stats = []
    
    for sector, grp in df_trades.groupby("sector"):
        if len(grp) == 0:
            continue
        
        # 기본 통계
        n_trades = len(grp)
        win_rate = (grp["ret"] > 0).mean()
        
        # 평균 수익/손실
        winners = grp[grp["ret"] > 0]
        losers = grp[grp["ret"] < 0]
        avg_gain = winners["ret"].mean() if len(winners) > 0 else 0.0
        avg_loss = losers["ret"].mean() if len(losers) > 0 else 0.0
        payoff = abs(avg_gain / avg_loss) if avg_loss != 0 else np.nan
        
        # 평균 보유 기간 계산 (날짜 기반)
        avg_hold_days = np.nan
        if "date" in grp.columns and len(grp) > 0:
            try:
                dates = pd.to_datetime(grp["date"])
                if len(dates) > 1:
                    # 거래 간격의 평균 (대략적인 보유 기간 추정)
                    date_diffs = dates.sort_values().diff().dt.days.dropna()
                    if len(date_diffs) > 0:
                        avg_hold_days = date_diffs.mean()
            except Exception:
                pass
        
        # 섹터별 수익률 시계열로 CAGR, Sharpe, MDD 계산
        sector_returns = grp["ret"].values
        
        # CAGR 계산 (거래 수익률 기반)
        if n_trades > 0 and len(sector_returns) > 0:
            # 총 수익률
            total_ret = (1 + sector_returns).prod() - 1
            # 거래 기간 기반 연율화 (전체 기간 대비 거래 비율 사용)
            if total_days > 0:
                # 섹터 거래가 전체 기간의 몇 %인지 추정
                sector_trade_ratio = n_trades / max(len(df_trades), 1)
                sector_days = total_days * sector_trade_ratio
                years = max(sector_days / 252.0, 0.1)
                cagr = (1 + total_ret) ** (1 / years) - 1
            else:
                # 전체 기간 정보 없으면 거래 횟수 기반 추정 (연간 50회 가정)
                years = n_trades / 50.0
                cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1
        else:
            cagr = 0.0
        
        # Sharpe Ratio 계산
        if len(sector_returns) > 1 and sector_returns.std() > 0:
            # 연율화된 Sharpe (거래당 수익률 기준)
            sharpe = (sector_returns.mean() / sector_returns.std()) * np.sqrt(252 / max(avg_hold_days, 1))
        else:
            sharpe = 0.0
        
        # MDD 계산 (누적 수익 기준)
        if len(sector_returns) > 0:
            cumret = (1 + pd.Series(sector_returns)).cumprod()
            mdd = (cumret / cumret.cummax() - 1).min()
        else:
            mdd = 0.0
        
        sector_stats.append({
            "Sector": sector,
            "Trades": n_trades,
            "WinRate": win_rate,
            "AvgGain": avg_gain,
            "AvgLoss": avg_loss,
            "Payoff": payoff,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MDD": mdd,
            "AvgHoldDays": avg_hold_days
        })
    
    df_sector = pd.DataFrame(sector_stats).sort_values("CAGR", ascending=False)
    
    # 콘솔 출력
    print("\n📊 섹터별 성과 요약")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(df_sector.to_string(index=False))
    
    # CSV 저장 (제목 기반 파일명 생성)
    os.makedirs("./reports", exist_ok=True)
    # 제목에서 파일명 안전하게 추출
    safe_title = title.replace(" ", "_").replace("-", "_").replace("/", "_")
    csv_path = f"./reports/sector_performance_{safe_title}.csv"
    df_sector.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 섹터 리포트 저장: {csv_path}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, y=0.995)
    
    # (1) 섹터별 승률
    ax1 = axes[0, 0]
    df_plot = df_sector.sort_values("WinRate", ascending=True).tail(10)
    if len(df_plot) > 0:
        ax1.barh(df_plot["Sector"], df_plot["WinRate"], color='steelblue', alpha=0.7)
        ax1.set_xlabel("승률")
        ax1.set_title("섹터별 승률 (Top 10)")
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # (2) 섹터별 CAGR
    ax2 = axes[0, 1]
    df_plot = df_sector.sort_values("CAGR", ascending=True).tail(10)
    if len(df_plot) > 0:
        colors = ['green' if x > 0 else 'red' for x in df_plot["CAGR"]]
        ax2.barh(df_plot["Sector"], df_plot["CAGR"] * 100, color=colors, alpha=0.7)
        ax2.set_xlabel("CAGR (%)")
        ax2.set_title("섹터별 CAGR (Top 10)")
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(0, color='black', linewidth=1)
    
    # (3) 섹터별 Sharpe
    ax3 = axes[1, 0]
    df_plot = df_sector.sort_values("Sharpe", ascending=True).tail(10)
    if len(df_plot) > 0:
        ax3.barh(df_plot["Sector"], df_plot["Sharpe"], color='darkorange', alpha=0.7)
        ax3.set_xlabel("Sharpe Ratio")
        ax3.set_title("섹터별 Sharpe (Top 10)")
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(0, color='black', linewidth=1)
    
    # (4) 섹터별 거래 횟수
    ax4 = axes[1, 1]
    df_plot = df_sector.sort_values("Trades", ascending=True).tail(10)
    if len(df_plot) > 0:
        ax4.barh(df_plot["Sector"], df_plot["Trades"], color='purple', alpha=0.7)
        ax4.set_xlabel("거래 횟수")
        ax4.set_title("섹터별 거래 횟수 (Top 10)")
        ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # 차트 저장 (제목 기반 파일명)
    chart_path = f"./reports/sector_performance_{safe_title}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"📈 섹터 차트 저장: {chart_path}\n")
    
    plt.show()
    
    return df_sector


# ============================================================
# 🚀 메인 실행
# ============================================================
def main(strategies=None, use_cache=True, use_backtest_cache=True, custom_weights=None, start_date=None):
    """
    메인 실행 함수
    
    Args:
        strategies: 실행할 전략 이름 리스트 (최대 2개)
                   None이면 기본적으로 ["baseline", "sector_weighted"] 실행
        use_cache: 데이터 캐시 사용 여부
        use_backtest_cache: 백테스트 결과 캐시 사용 여부
        custom_weights: 사용할 가중치 딕셔너리 (None이면 기본 가중치 사용)
    
    Returns:
        결과 딕셔너리
    """
    print("\n" + "="*60)
    print("🚀 백테스트 리포트 생성 시작")
    print("="*60)
    
    # 전략 선택
    if strategies is None:
        strategies = ["baseline", "sector_weighted"]
    
    if len(strategies) > 2:
        print("⚠️ 최대 2개 전략만 비교할 수 있습니다. 처음 2개만 사용합니다.")
        strategies = strategies[:2]
    
    if len(strategies) == 0:
        print("⚠️ 실행할 전략이 없습니다.")
        return {}
    
    # 전략 검증
    valid_strategies = []
    for strategy_name in strategies:
        strategy = get_strategy(strategy_name)
        if strategy is None:
            print(f"⚠️ 전략 '{strategy_name}'을 찾을 수 없습니다. 건너뜁니다.")
            continue
        valid_strategies.append((strategy_name, strategy))
    
    if len(valid_strategies) == 0:
        print("⚠️ 유효한 전략이 없습니다.")
        print("사용 가능한 전략 목록:")
        for s in list_strategies():
            print(f"  - {s['name']}: {s['description']}")
        return {}
    
    print(f"\n📋 선택된 전략: {[name for name, _ in valid_strategies]}")
    
    # Config 해시 생성 (백테스트 결과 캐싱용)
    config_hash = None
    if use_backtest_cache:
        try:
            config_hash = get_config_hash()
            print(f"📋 Config 해시: {config_hash[:8]}...")
        except Exception:
            pass
    
    # 1. 데이터 로드 (인덱스 포함)
    enriched, idx_map = load_data(use_cache=use_cache, start_date=start_date)
    valid_universe = filter_universe(enriched)
    print(f"✅ 필터링 후 유니버스: {len(valid_universe)}개 종목")
    
    # 필터링된 enriched 딕셔너리 생성 (전략에 전달용)
    filtered_enriched = {ticker: enriched[ticker] for ticker in valid_universe if ticker in enriched}
    
    # KOSDAQ 인덱스 추출
    idx_kosdaq = idx_map.get("KOSDAQ", None)
    
    if idx_kosdaq is not None and len(idx_kosdaq) > 0:
        print(f"\n✅ KOSDAQ 인덱스: {len(idx_kosdaq)}개 데이터 (날짜: {idx_kosdaq.index[0]} ~ {idx_kosdaq.index[-1]})")
    else:
        print("\n⚠️ KOSDAQ 인덱스 없음")
        idx_kosdaq = None

    # 2. 백테스트 실행
    results = []
    for strategy_name, strategy in valid_strategies:
        print(f"\n{'='*60}")
        print(f"📈 {strategy_name} 전략 실행 중...")
        print("="*60)
        # 최적화된 가중치가 있으면 사용, 없으면 기본 가중치
        weights_to_use = custom_weights if custom_weights else None
        if strategy_name == "multi_allocator":
            enriched_for_strategy = enriched
        else:
            enriched_for_strategy = filtered_enriched
        equity_curve, trade_log = strategy.run_backtest(enriched_for_strategy, market_index=idx_kosdaq, weights=weights_to_use)
        results.append((strategy_name, equity_curve, trade_log))
    
    # 3. 결과 출력
    print_results(results)
    
    # 4. 시각화
    plot_results(results)
    
    # 5. 섹터별 성과 리포트 (각 전략별로)
    sector_reports = {}
    for strategy_name, equity_curve, trade_log in results:
        if equity_curve is not None and trade_log and len(trade_log) > 0:
            report = sector_performance_report(
                trade_log,
                equity_curve,
                title=f"{strategy_name}_섹터별_성과"
            )
            sector_reports[strategy_name] = report
    
    print("\n✅ 모든 리포트 생성 완료!")
    print("="*60 + "\n")
    
    # 결과 딕셔너리 구성
    result_dict = {}
    for strategy_name, equity_curve, trade_log in results:
        result_dict[f"{strategy_name}_ec"] = equity_curve
        result_dict[f"{strategy_name}_trades"] = trade_log
        if strategy_name in sector_reports:
            result_dict[f"{strategy_name}_sector_report"] = sector_reports[strategy_name]
    
    return result_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="백테스트 리포트 생성")
    parser.add_argument(
        "--strategy",
        nargs="+",
        help="실행할 전략 이름 (예: baseline 또는 baseline sector_weighted)",
        default=None
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("STRATEGY1", "STRATEGY2"),
        help="두 전략 비교 (예: --compare baseline sector_weighted)",
        default=None
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 전략 목록 출력"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="데이터 캐시 사용 안 함"
    )
    parser.add_argument(
        "--no-backtest-cache",
        action="store_true",
        help="백테스트 결과 캐시 사용 안 함"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="가중치 최적화 실행 (weight_optimizer.py 실행)"
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="가중치 최적화 시 최대 조합 수 (--optimize와 함께 사용)"
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default=None,
        help="최적화된 가중치 JSON 파일 경로 (예: ./reports/best_weights_sector_weighted_bayesian.json)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="데이터 로드 시작일(YYYY-MM-DD). 지정하지 않으면 config.START 사용"
    )
    
    args = parser.parse_args()
    
    # 가중치 최적화 실행
    if args.optimize:
        from weight_optimizer import optimize_weights
        
        # 전략 선택 (기본값: sector_weighted)
        strategy_to_optimize = args.strategy[0] if args.strategy else "sector_weighted"
        if strategy_to_optimize not in ["baseline", "sector_weighted"]:
            print(f"⚠️ 최적화는 'baseline' 또는 'sector_weighted'만 지원합니다.")
            print(f"   '{strategy_to_optimize}' 대신 'sector_weighted'를 사용합니다.")
            strategy_to_optimize = "sector_weighted"
        
        print("\n" + "="*60)
        print("🔍 가중치 최적화 모드")
        print("="*60)
        
        best = optimize_weights(
            strategy_name=strategy_to_optimize,
            use_cache=not args.no_cache,
            max_combinations=args.max_combinations
        )
        
        if best is not None:
            print("\n✅ 가중치 최적화 완료!")
            print("="*60)
        
        # 최적화만 실행하는 경우 여기서 종료
        if not args.strategy and not args.compare:
            exit(0)
    
    # 전략 목록 출력
    if args.list:
        print("\n사용 가능한 전략:")
        print("-" * 60)
        for name, description in list_strategies():
            print(f"  {name:<20} : {description}")
        print()
        exit(0)
    
    # 전략 선택
    strategies = None
    if args.compare:
        strategies = args.compare
    elif args.strategy:
        strategies = args.strategy
    
    # 최적화된 가중치 로드
    custom_weights = None
    if args.weights_file:
        import json
        try:
            with open(args.weights_file, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
                custom_weights = weights_data.get("weights")
                if custom_weights:
                    print(f"\n📋 최적화된 가중치 로드: {args.weights_file}")
                    print(f"   Sharpe: {weights_data.get('best_sharpe', 'N/A'):.4f}")
                    print(f"   가중치: {custom_weights}")
        except Exception as e:
            print(f"⚠️ 가중치 파일 로드 실패: {e}")
            custom_weights = None
    
    # 실행
    results = main(
        strategies=strategies,
        use_cache=not args.no_cache,
        use_backtest_cache=not args.no_backtest_cache,
        custom_weights=custom_weights,
        start_date=args.start_date
    )
