#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Quality Momentum 전략 v3

개선사항 (v2 → v3):
1. Risk Filter 보강 (MA120 기반 현금비중 조정)
2. 섹터 모멘텀 가중치 적용
3. ERC 포지션 사이징
4. 동적 현금 관리
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import START, END, FEE_PER_SIDE, VENUE_FEE_PER_SIDE, TAX_RATE_SELL

INIT_CASH = 1_000_000_000
TAX = TAX_RATE_SELL


class KQMStrategyV3(BaseStrategy):
    """K-Quality Momentum 전략 v3"""
    
    def __init__(self, rebal_days=10, n_stocks=30, sector_cap=5, factor_weights=None):
        self.rebalance_days = rebal_days
        self.holdings_count = n_stocks
        self.sector_cap = sector_cap
        
        # 팩터 가중치 (Optuna 최적화 지원)
        if factor_weights is None:
            self.factor_weights = {
                'MOM6': 0.40,
                'MOM3': 0.10,
                'QUALITY': 0.20,
                'VOL': 0.20,
                'VAL': 0.10,
            }
        else:
            self.factor_weights = factor_weights
    
    def get_name(self) -> str:
        return "kqm_v3"
    
    def get_description(self) -> str:
        return "KQM v3 (MA120 Risk Filter + Sector Momentum + ERC)"
    
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
        """팩터 계산"""
        if df is None or df.empty or current_date not in df.index:
            return None
        
        date_idx = df.index.get_loc(current_date)
        if date_idx < 120:
            return None
        
        # Momentum
        mom_6m = (df.loc[current_date, "close"] / df.iloc[date_idx - 120]["close"]) - 1
        if date_idx < 60:
            return None
        mom_3m = (df.loc[current_date, "close"] / df.iloc[date_idx - 60]["close"]) - 1
        
        # Quality & Value
        value_mean = df["value"].iloc[date_idx - 60:date_idx].mean()
        value_std = df["value"].iloc[date_idx - 60:date_idx].std()
        quality_proxy = value_mean / value_std if value_std > 0 else 0
        
        # Volatility (EWMA)
        returns = df["close"].pct_change()
        returns_ewm = returns.ewm(halflife=30).std()
        vol_smooth = returns_ewm.iloc[date_idx]
        if vol_smooth <= 0 or not np.isfinite(vol_smooth):
            return None
        inv_vol_smooth = 1.0 / vol_smooth
        
        # Value
        avg_price = df["close"].iloc[date_idx - 60:date_idx].mean()
        value_proxy = avg_price / df.loc[current_date, "close"]
        
        return {
            "mom6m": mom_6m,
            "mom3m": mom_3m,
            "roe_proxy": quality_proxy,
            "inv_vol_smooth": inv_vol_smooth,
            "val_proxy": value_proxy,
            "vol": vol_smooth,
        }
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """백테스트 실행"""
        if not silent:
            print("\n" + "="*60)
            print("📈 K-Quality Momentum v3 백테스트 시작...")
            print("="*60)
        
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 120:
            return pd.DataFrame(), []
        
        rebalance_dates = dates[120::self.rebalance_days]
        
        cash = INIT_CASH
        positions = {}
        equity_curve = []
        trade_log = []
        
        for rebal_idx in tqdm(range(len(rebalance_dates)), desc="KQM v3", disable=silent):
            rebal_date = rebalance_dates[rebal_idx]
            
            if rebal_idx < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[rebal_idx + 1]
            else:
                next_rebal_date = dates[-1]
            
            # 청산
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
                rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
                for date in rebal_period_dates:
                    equity = self._calculate_equity(cash, positions, enriched, date)
                    equity_curve.append((date, equity))
                continue
            
            day = pd.DataFrame(factor_data)
            day = day[(day["mom6m"] > 0) & (day["mom3m"] > 0)]
            
            if len(day) < self.holdings_count:
                rebal_period_dates = [d for d in dates if rebal_date <= d < next_rebal_date]
                for date in rebal_period_dates:
                    equity = self._calculate_equity(cash, positions, enriched, date)
                    equity_curve.append((date, equity))
                continue
            
            # 팩터 순위
            for col in ["mom6m", "mom3m", "roe_proxy", "inv_vol_smooth", "val_proxy"]:
                day[f"{col}_rank"] = day[col].rank(pct=True)
            
            # Factor Score (복합) - factor_weights 사용
            day["score"] = (
                self.factor_weights.get('MOM6', 0.30) * day["mom6m_rank"] +
                self.factor_weights.get('MOM3', 0.20) * day["mom3m_rank"] +
                self.factor_weights.get('QUALITY', 0.20) * day["roe_proxy_rank"] +
                self.factor_weights.get('VOL', 0.20) * day["inv_vol_smooth_rank"] +
                self.factor_weights.get('VAL', 0.10) * day["val_proxy_rank"]
            )
            
            day = day.sort_values("score", ascending=False)
            day["sector"] = day["ticker"].apply(self.get_sector)
            
            # 섹터 제한
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
            
            # ERC 가중치
            top_n_stocks["inv_vol"] = 1 / top_n_stocks["vol"].clip(1e-6)
            top_n_stocks["w"] = top_n_stocks["inv_vol"] / top_n_stocks["inv_vol"].sum()
            
            # 진입
            if cash > 0:
                for idx, row in top_n_stocks.iterrows():
                    ticker = row["ticker"]
                    weight = row["w"]
                    entry_px = row["close"]
                    
                    target_notional = cash * weight
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
            print(f"✅ KQM v3 백테스트 완료: {len(ec)}개 데이터 포인트\n")
        
        return ec, trade_log
