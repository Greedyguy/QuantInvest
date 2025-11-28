#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 모멘텀 전략

기본적인 모멘텀 기반 백테스트 전략입니다.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import *
from signals import scoring


class BaselineStrategy(BaseStrategy):
    """기본 모멘텀 전략"""
    
    def get_name(self) -> str:
        return "baseline"
    
    def get_description(self) -> str:
        return "기본 모멘텀 전략"
    
    def _close_positions(self, positions, enriched, d0, d1, trade_log, config):
        """포지션 청산 로직"""
        to_close = []
        for t, pos in positions.items():
            df = enriched.get(t)
            if df is None or d0 not in df.index or d1 not in df.index:
                continue
            
            r0 = df.loc[d0]
            entry_px = pos["entry_px"]
            ret = (r0["close"] - entry_px) / entry_px
            held = (d0 - pos["entry_date"]).days
            
            # 청산 조건
            if ret <= config["STOP_LOSS"] or ret >= config["TAKE_PROFIT"] or held >= config["MAX_HOLD_DAYS"]:
                to_close.append(t)
        
        cash_from_close = 0.0
        for t in to_close:
            pos = positions.pop(t)
            df = enriched[t]
            exit_px = df.loc[d1, "open"] * (1 - config["SLIPPAGE_EXIT"])
            qty = pos["qty"]
            notional = qty * exit_px
            fee = notional * (config["FEE_PER_SIDE"] + config["VENUE_FEE_PER_SIDE"])
            tax = notional * config["TAX_RATE_SELL"]
            cash_from_close += notional - fee - tax
            
            ret = (exit_px - pos["entry_px"]) / pos["entry_px"]
            trade_log.append({
                "ticker": t,
                "entry_date": pos["entry_date"],
                "exit_date": d1,
                "entry_px": pos["entry_px"],
                "exit_px": exit_px,
                "qty": qty,
                "ret": ret
            })
        
        return positions, cash_from_close, trade_log
    
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """
        기본 모멘텀 전략 백테스트
        
        Args:
            enriched: 종목별 데이터
            market_index: 사용하지 않음 (다른 전략과의 호환성을 위해 존재)
            weights: 사용하지 않음 (다른 전략과의 호환성을 위해 존재)
            silent: 출력 억제 여부
        """
        print("\n" + "="*60)
        print("📈 기본 백테스트 시작...")
        print("="*60)
        
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
        
        for i in tqdm(range(60, len(dates)-1), desc="Baseline"):
            d0, d1 = dates[i], dates[i+1]
            
            # 당일 단면 데이터
            rows = []
            for t, df in enriched.items():
                if d0 not in df.index or d1 not in df.index:
                    continue
                r = df.loc[d0]
                price_ok = True
                if FORBID_LIMIT_UP_ENTRY and r["lc"] == 1:
                    price_ok = False
                
                rows.append({
                    "ticker": t,
                    "close": r["close"],
                    "open_next": df.loc[d1, "open"],
                    "val_ma20": r["val_ma20"],
                    "v_mult": r["v_mult"],
                    "bo": r["bo"],
                    "vcp": r["vcp"],
                    "gg": r["gg"],
                    "lc": r["lc"],
                    "rs_raw": r.get("rs_raw", np.nan),
                    "price_ok": price_ok
                })
            
            if not rows:
                equity = self._calculate_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity))
                continue
            
            day = pd.DataFrame(rows).set_index("ticker")
            day["score"] = scoring(day, W)
            
            # 청산
            positions, cash_from_close, trade_log = self._close_positions(
                positions, enriched, d0, d1, trade_log, config
            )
            cash += cash_from_close
            
            # 신규 진입
            picks = day.sort_values("score", ascending=False)
            picks = picks[picks["val_ma20"] >= MIN_AVG_TRD_AMT_20]
            picks = picks[picks["price_ok"]].head(MAX_HOLDINGS)
            
            slots = max(0, MAX_HOLDINGS - len(positions))
            picks = picks.head(slots)
            
            if len(picks) > 0 and slots > 0:
                target_w = min(1 / max(1, len(positions) + len(picks)), MAX_WEIGHT_PER_NAME)
                alloc_cash = cash * target_w
                for t, r in picks.iterrows():
                    px = r["open_next"] * (1 + SLIPPAGE_ENTRY)
                    
                    # 안전 체크
                    if px <= 0 or not np.isfinite(px):
                        continue
                    
                    qty = int(alloc_cash / px)
                    if qty <= 0:
                        continue
                    notional = qty * px
                    fee = notional * (FEE_PER_SIDE + VENUE_FEE_PER_SIDE)
                    cash_post = cash - notional - fee
                    if cash_post < 0:
                        continue
                    cash = cash_post
                    positions[t] = {"entry_px": px, "qty": qty, "entry_date": d1}
            
            # Equity 기록
            equity = self._calculate_equity(cash, positions, enriched, d0)
            equity_curve.append((d0, equity))
        
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        print(f"✅ 기본 백테스트 완료: {len(ec)}개 데이터 포인트\n")
        return ec, trade_log

