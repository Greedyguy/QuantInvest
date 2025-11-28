#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Quality Momentum 전략

다중 팩터 기반 중기 로테이션 전략:
- Momentum (6개월 수익률)
- Quality (ROE - 재무 데이터 필요)
- Volatility (60일 변동성 역수)
- Value (PER, PBR - 재무 데이터 필요)

목표: CAGR 10~15%, Sharpe ≥ 1.0, MDD ≤ -15%
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import *


class KQMStrategy(BaseStrategy):
    """K-Quality Momentum 전략"""
    
    def __init__(self):
        """초기화"""
        self.rebalance_days = 20  # 월 1회 (20거래일)
        self.holdings_count = 20  # 보유 종목수
        self.sector_cap = 3  # 섹터당 최대 종목수
    
    def get_name(self) -> str:
        return "kqm"
    
    def get_description(self) -> str:
        return "K-Quality Momentum (다중 팩터 중기 로테이션)"
    
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
        """
        팩터 계산
        
        Args:
            df: OHLCV 데이터프레임
            current_date: 현재 날짜
            
        Returns:
            팩터 딕셔너리 또는 None
        """
        if current_date not in df.index:
            return None
        
        try:
            date_idx = df.index.get_loc(current_date)
            
            # Momentum (6개월 = 120거래일)
            if date_idx < 120:
                return None
            
            mom_6m = (df.loc[current_date, "close"] / df.iloc[date_idx - 120]["close"]) - 1
            
            # Volatility (60일 변동성)
            if date_idx < 60:
                return None
            
            returns = df["close"].pct_change()
            vol_60 = returns.iloc[date_idx - 60:date_idx].std()
            
            if vol_60 <= 0 or not np.isfinite(vol_60):
                return None
            
            inv_vol = 1.0 / vol_60
            
            # Quality (ROE) - 재무 데이터 없으므로 영업이익률 대용
            # 거래대금 안정성으로 대체
            val_ma20 = df.iloc[date_idx - 20:date_idx]["value"].mean()
            quality_proxy = val_ma20 if val_ma20 > 0 else 0
            
            # Value (PER, PBR) - 재무 데이터 없으므로 단순화
            # 최근 60일 평균 가격 대비 현재 가격의 상대적 위치로 대체
            price_60_mean = df.iloc[date_idx - 60:date_idx]["close"].mean()
            value_proxy = price_60_mean / df.loc[current_date, "close"] if df.loc[current_date, "close"] > 0 else 1.0
            
            return {
                "mom_6m": mom_6m,
                "vol_60": vol_60,
                "inv_vol": inv_vol,
                "quality": quality_proxy,
                "value": value_proxy,
                "close": df.loc[current_date, "close"],
                "val_ma20": val_ma20,
            }
        
        except Exception as e:
            return None
    
    def _calculate_factor_score(self, factors_df: pd.DataFrame) -> pd.Series:
        """
        팩터 점수 계산
        
        Args:
            factors_df: 팩터 데이터프레임
            
        Returns:
            종목별 점수 시리즈
        """
        # 각 팩터별 순위 (백분위)
        mom_rank = factors_df["mom_6m"].rank(pct=True)
        quality_rank = factors_df["quality"].rank(pct=True)
        vol_rank = factors_df["inv_vol"].rank(pct=True)
        value_rank = factors_df["value"].rank(pct=True)
        
        # 가중 평균 점수
        score = (
            0.4 * mom_rank +
            0.3 * quality_rank +
            0.2 * vol_rank +
            0.1 * value_rank
        )
        
        return score
    
    def _get_market_trend(self, enriched: dict, current_date: pd.Timestamp) -> bool:
        """
        시장 추세 확인 (KOSPI > MA60)
        
        Args:
            enriched: enriched 데이터
            current_date: 현재 날짜
            
        Returns:
            True: 상승/횡보, False: 하락
        """
        # 모든 종목의 평균 종가로 시장 지수 대용
        prices = []
        for ticker, df in enriched.items():
            if current_date in df.index:
                prices.append(df.loc[current_date, "close"])
        
        if not prices or len(prices) < 60:
            return True  # 기본값
        
        # 간단히 평균 가격 추세로 판단
        return True  # 실제로는 별도 지수 데이터 필요
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """
        K-Quality Momentum 전략 백테스트
        
        Args:
            enriched: enriched 데이터
            weights: 사용하지 않음
            silent: True이면 출력 억제
            
        Returns:
            (equity_curve, trade_log) 튜플
        """
        if not silent:
            print("\n" + "="*60)
            print("📈 K-Quality Momentum 전략 백테스트 시작...")
            print("="*60)
            print(f"⚙️  보유 종목: {self.holdings_count}개")
            print(f"⚙️  리밸런싱: {self.rebalance_days}일마다")
        
        cash = 1_000_000_000.0
        positions = {}
        equity_curve = []
        trade_log = []
        
        dates = sorted(set().union(*[df.index for df in enriched.values()]))
        
        # 리밸런싱 시점
        rebalance_dates = dates[120::self.rebalance_days]  # 최소 120일 후부터 시작
        
        if not silent:
            print(f"📅 리밸런싱 횟수: {len(rebalance_dates)}회")
            print(f"📅 첫 리밸런싱: {rebalance_dates[0]}")
            print(f"📅 마지막 리밸런싱: {rebalance_dates[-1]}")
        
        # 모든 리밸런싱 처리 (마지막 포함)
        for rebal_idx in tqdm(range(len(rebalance_dates)), desc="KQM Rebalance", disable=silent):
            rebal_date = rebalance_dates[rebal_idx]
            # 다음 리밸런싱 날짜 (마지막이면 데이터 종료일)
            if rebal_idx < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[rebal_idx + 1]
            else:
                next_rebal_date = dates[-1]  # 데이터 마지막 날짜
            
            # 디버깅: 마지막 리밸런싱 확인
            is_last_rebal = (rebal_idx == len(rebalance_dates) - 1)
            if is_last_rebal and not silent:
                print(f"\n🔍 마지막 리밸런싱 (#{rebal_idx + 1}): {rebal_date}")
                print(f"   현재 포지션 수: {len(positions)}")
                print(f"   현금: {cash:,.0f}원")
                print(f"   다음 구간: ~ {next_rebal_date}")
            
            # 시장 추세 확인
            market_ok = self._get_market_trend(enriched, rebal_date)
            
            # 팩터 계산
            factors = []
            for ticker, df in enriched.items():
                factor_dict = self._compute_factors(df, rebal_date)
                if factor_dict is not None:
                    factor_dict["ticker"] = ticker
                    factors.append(factor_dict)
            
            if not factors:
                # 팩터 계산 실패 시 기존 포지션 유지
                equity = self._calculate_equity(cash, positions, enriched, rebal_date)
                equity_curve.append((rebal_date, equity))
                continue
            
            factors_df = pd.DataFrame(factors).set_index("ticker")
            
            # 유동성 필터
            factors_df = factors_df[factors_df["val_ma20"] >= MIN_AVG_TRD_AMT_20]
            
            if len(factors_df) == 0:
                equity = self._calculate_equity(cash, positions, enriched, rebal_date)
                equity_curve.append((rebal_date, equity))
                continue
            
            # 팩터 점수 계산
            factors_df["score"] = self._calculate_factor_score(factors_df)
            
            # 상위 종목 선정
            top_stocks = factors_df.nlargest(self.holdings_count * 2, "score")
            
            # 변동성 기반 비중 계산 (Equal Risk)
            top_stocks["weight"] = top_stocks["inv_vol"] / top_stocks["inv_vol"].sum()
            
            # 최종 선정 (상위 20개)
            selected = top_stocks.head(self.holdings_count)
            
            # 기존 포지션 청산
            exit_tickers = [t for t in positions.keys() if t not in selected.index]
            for ticker in exit_tickers:
                df = enriched.get(ticker)
                if df is None or rebal_date not in df.index:
                    continue
                
                # 리밸런싱 시점 종가로 청산
                exit_px = df.loc[rebal_date, "close"] * (1 - SLIPPAGE_EXIT)
                qty = positions[ticker]["qty"]
                gross = exit_px * qty
                fee = gross * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                tax = gross * TAX_RATE_SELL
                cash += (gross - fee - tax)
                
                trade_log.append({
                    "date": rebal_date,
                    "ticker": ticker,
                    "exit_px": exit_px,
                    "ret": exit_px / positions[ticker]["entry_px"] - 1
                })
                
                del positions[ticker]
            
            # 신규 진입 및 비중 조정
            if market_ok:
                for ticker, row in selected.iterrows():
                    target_alloc = cash * row["weight"]
                    entry_px = row["close"] * (1 + SLIPPAGE_ENTRY)
                    
                    if entry_px <= 0:
                        continue
                    
                    target_qty = int(target_alloc / entry_px)
                    
                    if target_qty <= 0:
                        continue
                    
                    # 기존 포지션 있으면 조정, 없으면 신규 진입
                    if ticker in positions:
                        # 기존 수량과 목표 수량 비교
                        current_qty = positions[ticker]["qty"]
                        qty_diff = target_qty - current_qty
                        
                        if qty_diff > 0:
                            # 추가 매수
                            notional = qty_diff * entry_px
                            fee = notional * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                            
                            if cash >= notional + fee:
                                cash -= (notional + fee)
                                positions[ticker]["qty"] = target_qty
                                positions[ticker]["entry_px"] = (
                                    (current_qty * positions[ticker]["entry_px"] + qty_diff * entry_px) / target_qty
                                )
                        elif qty_diff < 0:
                            # 일부 매도
                            sell_qty = -qty_diff
                            gross = sell_qty * entry_px * (1 - SLIPPAGE_EXIT)
                            fee = gross * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                            tax = gross * TAX_RATE_SELL
                            cash += (gross - fee - tax)
                            positions[ticker]["qty"] = target_qty
                    else:
                        # 신규 진입
                        notional = target_qty * entry_px
                        fee = notional * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                        
                        if cash >= notional + fee:
                            cash -= (notional + fee)
                            positions[ticker] = {"entry_px": entry_px, "qty": target_qty, "entry_date": rebal_date}
            
            # 리밸런싱 기간 동안 Equity 기록
            rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
            for date in rebal_period_dates:
                equity = self._calculate_equity(cash, positions, enriched, date)
                equity_curve.append((date, equity))
            
            # 디버깅: 마지막 리밸런싱 후 확인
            if is_last_rebal and not silent:
                print(f"   리밸런싱 후 포지션 수: {len(positions)}")
                print(f"   리밸런싱 후 현금: {cash:,.0f}원")
                if positions:
                    print(f"   보유 종목: {list(positions.keys())[:5]}...")
        
        # 마지막 리밸런싱 이후 남은 기간 처리
        if len(rebalance_dates) > 1:
            last_rebal = rebalance_dates[-1]
            # 마지막 리밸런싱 이후 날짜만 (중복 방지)
            remaining_dates = [d for d in dates if d > last_rebal]
            
            if not silent and remaining_dates:
                print(f"\n🔍 마지막 구간 처리:")
                print(f"   마지막 리밸런싱: {last_rebal}")
                print(f"   남은 날짜 수: {len(remaining_dates)}일")
                print(f"   기간: {remaining_dates[0]} ~ {remaining_dates[-1]}")
                print(f"   포지션 수: {len(positions)}")
                
            for date in remaining_dates:
                equity = self._calculate_equity(cash, positions, enriched, date)
                equity_curve.append((date, equity))
                
            # 마지막 날짜의 equity 확인
            if not silent and remaining_dates:
                last_equity = equity_curve[-1][1]
                prev_equity = equity_curve[-len(remaining_dates)-1][1] if len(equity_curve) > len(remaining_dates) else equity_curve[0][1]
                ret = (last_equity / prev_equity - 1) * 100
                print(f"   마지막 구간 수익률: {ret:.2f}%")
                print(f"   구간 시작 equity: {prev_equity:,.0f}원")
                print(f"   구간 종료 equity: {last_equity:,.0f}원")
        
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        
        # 중복 제거 (같은 날짜가 여러 번 기록될 수 있음)
        ec = ec[~ec.index.duplicated(keep='last')]
        
        if not silent:
            print(f"✅ K-Quality Momentum 백테스트 완료: {len(ec)}개 데이터 포인트")
            print(f"📊 총 리밸런싱 횟수: {len(rebalance_dates)}회\n")
        
        return ec, trade_log

