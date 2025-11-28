#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reversal 전략 v2

단기 리버설(반전) 신호를 포착하는 전략입니다.
- RSI 과매도 구간 진입
- 거래량 확인
- 변동성 기반 포지션 사이징
- 동적 손절/익절
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import *


class ReversalStrategy(BaseStrategy):
    """Reversal 전략 v2"""
    
    def get_name(self) -> str:
        return "reversal"
    
    def get_description(self) -> str:
        return "단기 리버설 전략 v2 (RSI 과매도 + 동적 손절익절)"
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """Reversal 전략 백테스트"""
        if not silent:
            print("\n" + "="*60)
            print("📈 Reversal v2 백테스트 시작...")
            print("="*60)
        
        cash = 1_000_000_000.0
        positions = {}
        equity_curve = []
        trade_log = []
        
        dates = sorted(set().union(*[df.index for df in enriched.values()]))
        
        # 동적 파라미터
        RSI_ENTRY = 35
        RSI_EXIT = 60
        STOP_LOSS = -0.05  # -5%
        TAKE_PROFIT = 0.10  # +10%
        MAX_HOLD_DAYS = 15
        
        config = {
            "SLIPPAGE_EXIT": SLIPPAGE_EXIT,
            "FEE_PER_SIDE": FEE_PER_SIDE,
            "VENUE_FEE_PER_SIDE": VENUE_FEE_PER_SIDE,
            "TAX_RATE_SELL": TAX_RATE_SELL,
        }
        
        for i in tqdm(range(60, len(dates)-1), desc="Reversal v2", disable=silent):
            d0, d1 = dates[i], dates[i+1]
            
            # 기존 포지션 청산
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                df = enriched.get(ticker)
                
                if df is None or d0 not in df.index:
                    continue
                
                current_px = df.loc[d0, "close"]
                ret = (current_px / pos["entry_px"]) - 1
                held_days = i - pos["entry_date"]
                rsi14 = df.loc[d0, "rsi14"]
                ma5 = df.loc[d0, "ma5"]
                
                exit_flag = False
                
                # 손절/익절
                if ret <= STOP_LOSS or ret >= TAKE_PROFIT:
                    exit_flag = True
                
                # 시그널 기반 청산
                if rsi14 > RSI_EXIT or current_px < ma5:
                    exit_flag = True
                
                # 최대 보유 기간
                if held_days >= MAX_HOLD_DAYS:
                    exit_flag = True
                
                if exit_flag:
                    exit_px = current_px * (1 - config["SLIPPAGE_EXIT"])
                    qty = pos["qty"]
                    notional = exit_px * qty
                    
                    fee = notional * (config["FEE_PER_SIDE"] + config["VENUE_FEE_PER_SIDE"])
                    tax = notional * config["TAX_RATE_SELL"]
                    
                    cash += notional - fee - tax
                    
                    trade_log.append({
                        "date": d0,
                        "ticker": ticker,
                        "side": "sell",
                        "qty": qty,
                        "price": exit_px,
                        "ret": ret
                    })
                    
                    del positions[ticker]
            
            # 신규 진입
            candidates = []
            
            for t, df in enriched.items():
                if df is None or d0 not in df.index or t in positions:
                    continue
                
                r = df.loc[d0]
                
                # 리버설 시그널
                if (r["rsi14"] < RSI_ENTRY and 
                    r["close"] > r["ma5"] and 
                    r["volume"] > r["volume_20d_avg"] * 1.2):
                    
                    # RSI 가중치 (더 과매도일수록 높은 비중)
                    rsi_weight = (40 - r["rsi14"]) / 15 if r["rsi14"] < 40 else 0
                    
                    candidates.append({
                        "ticker": t,
                        "close": r["close"],
                        "rsi_weight": rsi_weight
                    })
            
            if not candidates:
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            # 상위 5개 선정
            cand_df = pd.DataFrame(candidates)
            cand_df = cand_df.sort_values("rsi_weight", ascending=False).head(5)
            
            # 가중치 정규화
            total_weight = cand_df["rsi_weight"].sum()
            if total_weight > 0:
                cand_df["weight"] = cand_df["rsi_weight"] / total_weight
            else:
                cand_df["weight"] = 1.0 / len(cand_df)
            
            # 진입
            if cash > 0:
                for idx, row in cand_df.iterrows():
                    t = row["ticker"]
                    weight = row["weight"]
                    
                    df = enriched.get(t)
                    if df is None or d1 not in df.index:
                        continue
                    
                    entry_px = df.loc[d1, "open"]
                    alloc = cash * weight
                    qty = int(alloc / entry_px)
                    
                    if qty <= 0:
                        continue
                    
                    notional = qty * entry_px
                    fee = notional * (config["FEE_PER_SIDE"] + config["VENUE_FEE_PER_SIDE"])
                    
                    if cash >= notional + fee:
                        cash -= (notional + fee)
                        positions[t] = {
                            "entry_px": entry_px,
                            "qty": qty,
                            "entry_date": i
                        }
            
            # Equity 기록
            equity = self._calculate_equity(cash, positions, enriched, d0)
            equity_curve.append((d0, equity))
        
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        
        if not silent:
            print(f"✅ Reversal v2 백테스트 완료: {len(ec)}개 데이터 포인트\n")
        
        return ec, trade_log

