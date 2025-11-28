#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Quality Momentum 전략 v2

개선사항:
1. 리밸런싱 주기: 월 1회(20일) → 2주(10일)
2. 보유 종목수: 20개 → 30개
3. 섹터 제한: 3종 → 5종
4. 3개월 모멘텀 추가
5. 변동성 EWMA 스무딩
6. Risk-On/Off 필터 (KOSPI MA60 + 5/20 MA)
7. 팩터 기반 가중치
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import START, END, FEE_PER_SIDE, VENUE_FEE_PER_SIDE, TAX_RATE_SELL

INIT_CASH = 1_000_000_000
TAX = TAX_RATE_SELL


class KQMStrategyV2(BaseStrategy):
    """K-Quality Momentum 전략 v2"""
    
    def __init__(self):
        self.rebalance_days = 10  # 2주 (10거래일)
        self.holdings_count = 30  # 보유 종목수 확대
        self.sector_cap = 5       # 섹터당 최대 종목수 완화
    
    def get_name(self) -> str:
        return "kqm_v2"
    
    def get_description(self) -> str:
        return "KQM v2 (Enhanced: 10d rebal, 30 stocks, Risk Filter)"
    
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
        """단일 종목의 팩터 계산"""
        if df is None or df.empty:
            return None
        
        if current_date not in df.index:
            return None
        
        date_idx = df.index.get_loc(current_date)
        
        # 최소 데이터 길이 확인
        if date_idx < 120:
            return None
        
        # Momentum (6개월)
        mom_6m = (df.loc[current_date, "close"] / df.iloc[date_idx - 120]["close"]) - 1
        
        # Momentum (3개월) 추가
        if date_idx < 60:
            return None
        mom_3m = (df.loc[current_date, "close"] / df.iloc[date_idx - 60]["close"]) - 1
        
        # Quality (ROE proxy: value의 안정성)
        if date_idx < 60:
            return None
        value_mean = df["value"].iloc[date_idx - 60:date_idx].mean()
        value_std = df["value"].iloc[date_idx - 60:date_idx].std()
        quality_proxy = value_mean / value_std if value_std > 0 else 0
        
        # Volatility (역수) - EWMA 스무딩
        returns = df["close"].pct_change()
        returns_ewm = returns.ewm(halflife=30).std()
        vol_smooth = returns_ewm.iloc[date_idx]
        if vol_smooth <= 0 or not np.isfinite(vol_smooth):
            return None
        inv_vol_smooth = 1.0 / vol_smooth
        
        # Value (PER/PBR proxy: 평균가 대비 현재가)
        avg_price = df["close"].iloc[date_idx - 60:date_idx].mean()
        value_proxy = avg_price / df.loc[current_date, "close"]
        
        return {
            "mom6m": mom_6m,
            "mom3m": mom_3m,
            "roe_proxy": quality_proxy,
            "inv_vol_smooth": inv_vol_smooth,
            "val_proxy": value_proxy,
        }
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """백테스트 실행"""
        if not silent:
            print("\n" + "="*60)
            print("📈 K-Quality Momentum v2 백테스트 시작...")
            print("="*60)
            print(f"⚙️  리밸런싱 주기: {self.rebalance_days}일")
            print(f"⚙️  보유 종목: {self.holdings_count}개")
            print(f"⚙️  섹터 제한: {self.sector_cap}개/섹터")
        
        # 날짜 리스트
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        
        if len(dates) < 120:
            return pd.DataFrame(), []
        
        # 리밸런싱 날짜 (120일 이후부터 시작, 이후 10일마다)
        rebalance_dates = dates[120::self.rebalance_days]
        
        if not silent:
            print(f"📅 리밸런싱 횟수: {len(rebalance_dates)}회")
        
        # 초기화
        cash = INIT_CASH
        positions = {}
        equity_curve = []
        trade_log = []
        
        # 리밸런싱 루프
        for rebal_idx in tqdm(range(len(rebalance_dates)), desc="KQM v2", disable=silent):
            rebal_date = rebalance_dates[rebal_idx]
            
            if rebal_idx < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[rebal_idx + 1]
            else:
                next_rebal_date = dates[-1]
            
            # 기존 포지션 청산
            if positions:
                for ticker in list(positions.keys()):
                    pos = positions[ticker]
                    df = enriched.get(ticker)
                    
                    if df is not None and rebal_date in df.index:
                        exit_px = df.loc[rebal_date, "close"]
                        qty = pos["qty"]
                        
                        notional = exit_px * qty
                        fee = notional * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                        tax = notional * TAX
                        
                        cash += notional - fee - tax
                        
                        ret = (exit_px / pos["entry_px"]) - 1
                        
                        trade_log.append({
                            "date": rebal_date,
                            "ticker": ticker,
                            "side": "sell",
                            "qty": qty,
                            "price": exit_px,
                            "ret": ret
                        })
                
                positions = {}
            
            # 팩터 계산
            factor_data = []
            for ticker, df in enriched.items():
                factors = self._compute_factors(df, rebal_date)
                if factors is not None:
                    factors["ticker"] = ticker
                    factors["close"] = df.loc[rebal_date, "close"]
                    factor_data.append(factors)
            
            if not factor_data:
                # Equity 기록
                rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
                for date in rebal_period_dates:
                    equity = self._calculate_equity(cash, positions, enriched, date)
                    equity_curve.append((date, equity))
                continue
            
            day = pd.DataFrame(factor_data)
            
            # 음수 모멘텀 제외
            day = day[(day["mom6m"] > 0) & (day["mom3m"] > 0)]
            
            if len(day) < self.holdings_count:
                rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
                for date in rebal_period_dates:
                    equity = self._calculate_equity(cash, positions, enriched, date)
                    equity_curve.append((date, equity))
                continue
            
            # 팩터 순위 계산
            for col in ["mom6m", "mom3m", "roe_proxy", "inv_vol_smooth", "val_proxy"]:
                day[f"{col}_rank"] = day[col].rank(pct=True)
            
            # 종합 점수
            day["score"] = (
                day["mom6m_rank"] * 0.30 +
                day["mom3m_rank"] * 0.20 +
                day["roe_proxy_rank"] * 0.20 +
                day["inv_vol_smooth_rank"] * 0.20 +
                day["val_proxy_rank"] * 0.10
            )
            
            # 점수 순 정렬
            day = day.sort_values("score", ascending=False)
            
            # 섹터 정보 추가
            day["sector"] = day["ticker"].apply(self.get_sector)
            
            # 섹터 제한 적용하여 상위 종목 선정
            selected_tickers = []
            sector_counts = {}
            
            for idx, row in day.iterrows():
                ticker = row["ticker"]
                sector = row["sector"]
                
                if sector_counts.get(sector, 0) >= self.sector_cap:
                    continue
                
                selected_tickers.append(ticker)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                
                if len(selected_tickers) >= self.holdings_count:
                    break
            
            if not selected_tickers:
                rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
                for date in rebal_period_dates:
                    equity = self._calculate_equity(cash, positions, enriched, date)
                    equity_curve.append((date, equity))
                continue
            
            top_n_stocks = day[day["ticker"].isin(selected_tickers)].copy()
            
            # Risk-On/Off 필터 (간단 버전: 전체 시장 평균 가격으로 근사)
            market_prices = []
            for d in dates[max(0, dates.index(rebal_date) - 60):dates.index(rebal_date) + 1]:
                daily_prices = []
                for ticker, df in enriched.items():
                    if df is not None and d in df.index:
                        daily_prices.append(df.loc[d, "close"])
                if daily_prices:
                    market_prices.append({"date": d, "price": np.mean(daily_prices)})
            
            market_df = pd.DataFrame(market_prices).set_index("date")
            market_df["ma60"] = market_df["price"].rolling(60).mean()
            market_df["ma5"] = market_df["price"].rolling(5).mean()
            market_df["ma20"] = market_df["price"].rolling(20).mean()
            
            # Risk-On 조건
            current_cash_allocation = 1.0
            if rebal_date in market_df.index:
                if pd.notna(market_df.loc[rebal_date, "ma60"]) and pd.notna(market_df.loc[rebal_date, "ma5"]) and pd.notna(market_df.loc[rebal_date, "ma20"]):
                    risk_on_condition = (
                        (market_df.loc[rebal_date, "price"] > market_df.loc[rebal_date, "ma60"]) and
                        (market_df.loc[rebal_date, "ma5"] > market_df.loc[rebal_date, "ma20"])
                    )
                    if not risk_on_condition:
                        current_cash_allocation = 0.7  # 30% 현금 유지
            
            # 팩터 기반 비중 조정
            top_n_stocks["w"] = top_n_stocks["score"] / top_n_stocks["score"].sum()
            
            # 투자
            if cash > 0:
                for idx, row in top_n_stocks.iterrows():
                    ticker = row["ticker"]
                    weight = row["w"]
                    entry_px = row["close"]
                    
                    # 현금 비중 반영
                    target_notional = cash * weight * current_cash_allocation
                    target_qty = int(target_notional / entry_px)
                    
                    if target_qty <= 0:
                        continue
                    
                    notional = target_qty * entry_px
                    fee = notional * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                    
                    if cash >= notional + fee:
                        cash -= (notional + fee)
                        positions[ticker] = {
                            "entry_px": entry_px,
                            "qty": target_qty,
                            "entry_date": rebal_date
                        }
            
            # Equity 기록
            rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
            for date in rebal_period_dates:
                equity = self._calculate_equity(cash, positions, enriched, date)
                if equity > 0:
                    equity_curve.append((date, equity))
        
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        ec = ec[~ec.index.duplicated(keep='last')]
        
        if not silent:
            print(f"✅ KQM v2 백테스트 완료: {len(ec)}개 데이터 포인트\n")
        
        return ec, trade_log

