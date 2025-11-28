#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 전략

baseline 및 sector_weighted 전략의 개선 버전입니다.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import *
from signals import scoring


class ImprovedStrategy(BaseStrategy):
    """개선된 전략"""
    
    def get_name(self) -> str:
        return "improved"
    
    def get_description(self) -> str:
        return "개선된 통합 전략"
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """개선된 전략 백테스트"""
        if not silent:
            print("\n" + "="*60)
            print("📈 개선된 전략 백테스트 시작...")
            print("="*60)
        
        if weights is None:
            weights = {
                "mom": 1.2,
                "rsi": 0.8,
                "volume": 1.5,
                "sector": 1.0,
            }
        
        cash = 1_000_000_000.0
        positions = {}
        equity_curve = []
        trade_log = []
        
        dates = sorted(set().union(*[df.index for df in enriched.values()]))
        
        config = {
            "STOP_LOSS": STOP_LOSS,
            "TAKE_PROFIT": TAKE_PROFIT,
            "MAX_HOLD_DAYS": MAX_HOLD_DAYS,
            "SLIPPAGE_EXIT": SLIPPAGE_EXIT,
            "FEE_PER_SIDE": FEE_PER_SIDE,
            "VENUE_FEE_PER_SIDE": VENUE_FEE_PER_SIDE,
            "TAX_RATE_SELL": TAX_RATE_SELL,
        }
        
        for i in tqdm(range(60, len(dates)-1), desc="Improved", disable=silent):
            d0, d1 = dates[i], dates[i+1]
            
            # 기존 포지션 청산 조건 확인
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                df = enriched.get(ticker)
                
                if df is None or d0 not in df.index:
                    continue
                
                current_px = df.loc[d0, "close"]
                ret = (current_px / pos["entry_px"]) - 1
                held_days = i - pos["entry_date"]
                
                exit_flag = False
                if ret <= config["STOP_LOSS"] or ret >= config["TAKE_PROFIT"] or held_days >= config["MAX_HOLD_DAYS"]:
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
            
            # 신규 진입 기회 스캔
            rows = {}
            for t, df in enriched.items():
                if df is None or d0 not in df.index or t in positions:
                    continue
                
                if df.loc[d0, "lc"]:  # 상한가 제외
                    continue
                
                r = df.loc[d0]
                
                sector = self.get_sector(t)
                
                rows[t] = {
                    "close": r["close"],
                    "rsi14": r["rsi14"],
                    "vol20": r["volume_20d_avg"],
                    "open": r["open"],
                    "ma5": r["ma5"],
                    "ma20": r["ma20"],
                    "ma60": r["ma60"],
                    "lc": r["lc"],
                    "sector": sector,
                }
            
            if not rows:
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            day = pd.DataFrame(rows).T
            
            # 섹터 모멘텀
            if i >= 90:
                d_30d_ago = dates[i - 30]
                sector_returns = {}
                
                for sector in day["sector"].unique():
                    sector_tickers = day[day["sector"] == sector].index
                    sector_rets = []
                    
                    for t in sector_tickers:
                        df = enriched.get(t)
                        if df is not None and d0 in df.index and d_30d_ago in df.index:
                            ret_30d = (df.loc[d0, "close"] / df.loc[d_30d_ago, "close"]) - 1
                            sector_rets.append(ret_30d)
                    
                    if sector_rets:
                        sector_returns[sector] = np.mean(sector_rets)
                    else:
                        sector_returns[sector] = 0
                
                day["sector_mom"] = day["sector"].map(sector_returns)
            else:
                day["sector_mom"] = 0
            
            # 시그널 스코어 계산
            day["score"] = day.apply(lambda r: scoring(r, d0, weights), axis=1)
            day = day.sort_values("score", ascending=False)
            
            # 상위 종목 선정
            top_n = day.head(min(7, len(day)))
            
            # 진입
            if cash > 0 and len(top_n) > 0:
                alloc = cash / len(top_n)
                
                for t in top_n.index:
                    df = enriched.get(t)
                    if df is None or d1 not in df.index:
                        continue
                    
                    entry_px = df.loc[d1, "open"]
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
            print(f"✅ 개선 전략 백테스트 완료: {len(ec)}개 데이터 포인트\n")
        
        return ec, trade_log

