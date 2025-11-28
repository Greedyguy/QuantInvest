#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Quality Momentum 소액 투자 전략 (100만원 기준)

특징:
- 초기 자본: 100만원
- 보유 종목: 3~5개 (집중 투자)
- 대상: 주가 5만원 이하 종목
- 종목당 투자: 20~33만원
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMSmallCapStrategy(BaseStrategy):
    """
    K-Quality Momentum 소액 투자 전략
    
    100만원 실계좌 운용을 위한 조정:
    1. 보유 종목 3~5개로 축소
    2. 주가 5만원 이하 종목만 선별
    3. Equal weight 투자
    """
    
    def __init__(self, rebal_days=10, n_stocks=5, max_price=50000, factor_weights=None):
        """
        Args:
            rebal_days: 리밸런싱 주기 (일)
            n_stocks: 보유 종목 수 (3~5개 권장)
            max_price: 최대 주가 (원, 기본 5만원)
            factor_weights: 팩터 가중치 dict
        """
        self.rebal_days = rebal_days
        self.n_stocks = n_stocks
        self.max_price = max_price
        
        # 팩터 가중치
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
        
        # 거래 비용
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = 0.001
    
    def get_name(self) -> str:
        return "kqm_small_cap"
    
    def get_description(self) -> str:
        return f"KQM Small Cap (100만원, {self.n_stocks}stocks, max price {self.max_price:,}원)"
    
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
        """팩터 계산 (KQM v3와 동일)"""
        if df is None or len(df) < 120:
            return None
        
        subset = df[df.index <= current_date]
        if len(subset) < 120:
            return None
        
        close = subset["close"].values
        
        # 1) Momentum 6개월
        mom_6m = (close[-1] / close[-120]) - 1.0
        
        # 2) Momentum 3개월
        if len(close) < 60:
            return None
        mom_3m = (close[-1] / close[-60]) - 1.0
        
        # 3) Quality Proxy (가격 안정성)
        ret_60 = pd.Series(close[-60:]).pct_change().dropna()
        quality_proxy = ret_60.mean() / (ret_60.std() + 1e-9) if len(ret_60) > 0 else 0.0
        
        # 4) Inverse Volatility (Smoothed)
        vol_20 = pd.Series(close[-20:]).pct_change().ewm(halflife=10).std().iloc[-1]
        inv_vol_smooth = 1.0 / (vol_20 + 1e-9)
        
        # 5) Value Proxy
        ret_120 = pd.Series(close[-120:]).pct_change().dropna()
        value_proxy = ret_120.mean() if len(ret_120) > 0 else 0.0
        
        return {
            "mom6m": mom_6m,
            "mom3m": mom_3m,
            "quality": quality_proxy,
            "inv_vol_smooth": inv_vol_smooth,
            "val_proxy": value_proxy,
            "price": close[-1],  # 현재 주가 추가
        }
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """백테스트 실행"""
        if not silent:
            print("\n" + "="*60)
            print("📈 K-Quality Momentum Small Cap 백테스트 시작...")
            print("="*60)
            print(f"⚙️  초기 자본: 100만원")
            print(f"⚙️  리밸런싱 주기: {self.rebal_days}일")
            print(f"⚙️  보유 종목: {self.n_stocks}개")
            print(f"⚙️  최대 주가: {self.max_price:,}원")
        
        # 날짜 리스트
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        
        if len(dates) < 120:
            return pd.DataFrame(), []
        
        # 리밸런싱 날짜
        rebalance_dates = dates[120::self.rebal_days]
        
        # 초기 설정
        init_cash = 1_000_000  # 100만원
        cash = init_cash
        positions = {}
        equity_curve = []
        trade_log = []
        
        # 리밸런싱 루프
        for rebal_idx in tqdm(range(len(rebalance_dates)), desc="KQM Small", disable=silent):
            d0 = rebalance_dates[rebal_idx]
            
            # 다음 리밸런싱 날짜
            if rebal_idx < len(rebalance_dates) - 1:
                next_rebal_date = rebalance_dates[rebal_idx + 1]
            else:
                next_rebal_date = dates[-1]
            
            # === 1) 팩터 스냅샷 (주가 필터 추가) ===
            rows = []
            for ticker, df in enriched.items():
                if df is None or len(df) == 0:
                    continue
                
                factors = self._compute_factors(df, d0)
                if factors is None:
                    continue
                
                # 주가 필터: max_price 이하만
                if factors["price"] > self.max_price:
                    continue
                
                # 음수 모멘텀 제외
                if factors["mom6m"] <= 0 or factors["mom3m"] <= 0:
                    continue
                
                rows.append({
                    "ticker": ticker,
                    "mom6m": factors["mom6m"],
                    "mom3m": factors["mom3m"],
                    "quality": factors["quality"],
                    "inv_vol_smooth": factors["inv_vol_smooth"],
                    "val_proxy": factors["val_proxy"],
                    "price": factors["price"],
                })
            
            if len(rows) == 0:
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            day = pd.DataFrame(rows)
            
            # === 2) 팩터 랭킹 ===
            for col in ["mom6m", "mom3m", "quality", "inv_vol_smooth", "val_proxy"]:
                day[f"{col}_rank"] = day[col].rank(pct=True)
            
            # === 3) 복합 팩터 스코어 ===
            day["score"] = (
                self.factor_weights.get('MOM6', 0.40) * day["mom6m_rank"] +
                self.factor_weights.get('MOM3', 0.10) * day["mom3m_rank"] +
                self.factor_weights.get('QUALITY', 0.20) * day["quality_rank"] +
                self.factor_weights.get('VOL', 0.20) * day["inv_vol_smooth_rank"] +
                self.factor_weights.get('VAL', 0.10) * day["val_proxy_rank"]
            )
            
            # === 4) 상위 종목 선택 ===
            day_sorted = day.sort_values("score", ascending=False)
            selected = day_sorted.head(self.n_stocks)["ticker"].tolist()
            
            if len(selected) == 0:
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            # === 5) Equal Weight ===
            target_weight = 1.0 / len(selected)
            
            # === 6) 포지션 청산 (보유 중인데 선택 안 된 종목) ===
            for ticker in list(positions.keys()):
                if ticker not in selected:
                    pos = positions.pop(ticker)
                    
                    df_t = enriched.get(ticker)
                    if df_t is None or d0 not in df_t.index:
                        continue
                    
                    exit_px = df_t.loc[d0, "close"] * (1 - self.slippage)
                    qty = pos["qty"]
                    entry_px = pos["entry_px"]
                    
                    proceeds = exit_px * qty
                    cost = entry_px * qty
                    pnl = proceeds - cost
                    
                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0
                    net_proceeds = proceeds - fee_out - tax
                    
                    cash += net_proceeds
                    
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_px,
                        "qty": qty,
                        "pnl": pnl,
                        "cash_after": cash,
                    })
            
            # === 7) 신규 진입 & 비중 조정 ===
            portfolio_value = cash + sum(
                enriched[t].loc[d0, "close"] * pos["qty"]
                for t, pos in positions.items()
                if enriched.get(t) is not None and d0 in enriched[t].index
            )
            
            for ticker in selected:
                df_t = enriched.get(ticker)
                if df_t is None or d0 not in df_t.index:
                    continue
                
                entry_px = df_t.loc[d0, "close"] * (1 + self.slippage)
                target_value = portfolio_value * target_weight
                target_qty = int(target_value / entry_px)
                
                if target_qty <= 0:
                    continue
                
                current_qty = positions.get(ticker, {}).get("qty", 0)
                delta_qty = target_qty - current_qty
                
                if delta_qty == 0:
                    continue
                
                # 매수
                if delta_qty > 0:
                    cost = entry_px * delta_qty
                    fee_in = cost * self.fee
                    total_cost = cost + fee_in
                    
                    if total_cost > cash:
                        # 현금 부족 시 가능한 만큼만 매수
                        affordable_qty = int((cash / (1 + self.fee)) / entry_px)
                        if affordable_qty > 0:
                            delta_qty = affordable_qty
                            cost = entry_px * delta_qty
                            fee_in = cost * self.fee
                            total_cost = cost + fee_in
                        else:
                            continue
                    
                    cash -= total_cost
                    
                    if ticker in positions:
                        old_qty = positions[ticker]["qty"]
                        old_px = positions[ticker]["entry_px"]
                        new_qty = old_qty + delta_qty
                        new_avg_px = (old_px * old_qty + entry_px * delta_qty) / new_qty
                        positions[ticker] = {"qty": new_qty, "entry_px": new_avg_px}
                    else:
                        positions[ticker] = {"qty": delta_qty, "entry_px": entry_px}
                    
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "action": "BUY",
                        "price": entry_px,
                        "qty": delta_qty,
                        "pnl": 0,
                        "cash_after": cash,
                    })
                
                # 매도 (비중 줄이기)
                elif delta_qty < 0:
                    sell_qty = -delta_qty
                    pos = positions[ticker]
                    
                    exit_px = entry_px * (1 - self.slippage)
                    proceeds = exit_px * sell_qty
                    cost = pos["entry_px"] * sell_qty
                    pnl = proceeds - cost
                    
                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0
                    net_proceeds = proceeds - fee_out - tax
                    
                    cash += net_proceeds
                    
                    pos["qty"] -= sell_qty
                    if pos["qty"] <= 0:
                        positions.pop(ticker)
                    
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "action": "SELL",
                        "price": exit_px,
                        "qty": sell_qty,
                        "pnl": pnl,
                        "cash_after": cash,
                    })
            
            # === 8) equity 기록 ===
            period_dates = [d for d in dates if d0 <= d < next_rebal_date]
            for eval_date in period_dates:
                equity = self._calculate_equity(cash, positions, enriched, eval_date)
                equity_curve.append((eval_date, equity))
        
        # === 9) 마지막 날 청산 ===
        final_date = dates[-1]
        for ticker in list(positions.keys()):
            pos = positions.pop(ticker)
            df_t = enriched.get(ticker)
            
            if df_t is None or final_date not in df_t.index:
                continue
            
            exit_px = df_t.loc[final_date, "close"] * (1 - self.slippage)
            qty = pos["qty"]
            proceeds = exit_px * qty
            fee_out = proceeds * self.fee
            pnl = proceeds - (pos["entry_px"] * qty)
            tax = proceeds * self.tax if pnl > 0 else 0
            
            cash += proceeds - fee_out - tax
            
            trade_log.append({
                "date": final_date,
                "ticker": ticker,
                "action": "SELL",
                "price": exit_px,
                "qty": qty,
                "pnl": pnl,
                "cash_after": cash,
            })
        
        equity = cash
        equity_curve.append((final_date, equity))
        
        # === 10) 결과 반환 ===
        ec_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        
        if not silent:
            print(f"✅ KQM Small Cap 백테스트 완료: {len(ec_df)}개 데이터 포인트")
            print(f"   최종 자산: {equity:,.0f}원 (수익률: {(equity/init_cash-1)*100:.2f}%)")
        
        return ec_df, trade_log

