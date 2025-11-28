#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract base class for backtesting strategies
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """백테스팅 전략 추상 베이스 클래스"""

    def __init__(self):
        self._target_weight_history = []
    
    @abstractmethod
    def get_name(self) -> str:
        """전략 이름 반환"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """전략 설명 반환"""
        pass
    
    @abstractmethod
    def run_backtest(self, enriched: dict, market_index=None, weights: dict = None, silent: bool = False) -> tuple:
        """
        백테스트 실행
        
        Args:
            enriched: 종목별 enriched 데이터 딕셔너리
            weights: 가중치 파라미터 (선택)
            silent: True면 출력 억제
            
        Returns:
            (equity_curve, trade_log) 튜플
            - equity_curve: pd.DataFrame with columns ['date', 'equity']
            - trade_log: list of trade dictionaries
        """
        pass

    def _ensure_weight_history(self):
        if not hasattr(self, "_target_weight_history"):
            self._target_weight_history = []

    def _reset_weight_history(self):
        self._ensure_weight_history()
        self._target_weight_history = []

    def _record_weights(self, date: pd.Timestamp, cash: float, positions: dict, enriched: dict):
        if date is None:
            return
        equity = cash
        values = {}
        for t, pos in positions.items():
            df = enriched.get(t)
            if df is None or len(df) == 0:
                px = pos.get("entry_px", 0)
            else:
                if date in df.index:
                    px = df.loc[date, "close"]
                else:
                    prev = df[df.index <= date]
                    if len(prev) == 0:
                        px = pos.get("entry_px", 0)
                    else:
                        px = prev.iloc[-1]["close"]
            if px is None or px <= 0:
                px = pos.get("entry_px", 0)
            val = px * pos.get("qty", 0)
            if val > 0:
                values[t] = val
                equity += val

        row = {}
        if equity > 0:
            for t, val in values.items():
                row[t] = val / equity
            row["__CASH__"] = max(cash / equity, 0.0)
        else:
            row["__CASH__"] = 1.0
        self._target_weight_history.append((pd.to_datetime(date), row))

    def get_target_weight_history(self) -> pd.DataFrame:
        self._ensure_weight_history()
        if not self._target_weight_history:
            return pd.DataFrame()
        records = {ts: row for ts, row in self._target_weight_history}
        df = pd.DataFrame(records).T.fillna(0.0)
        df.index.name = "date"
        return df.sort_index()
    
    def _calculate_equity(self, cash: float, positions: dict, enriched: dict, date: pd.Timestamp) -> float:
        """
        특정 날짜의 총 자산 계산
        
        Args:
            cash: 현금
            positions: 현재 포지션 딕셔너리
            enriched: enriched 데이터
            date: 평가 날짜
            
        Returns:
            총 자산 (현금 + 포지션 가치)
        """
        import pandas as pd
        import numpy as np
        
        equity = cash
        for t, pos in positions.items():
            df = enriched.get(t)
            
            # 데이터 없으면 진입가로 평가
            if df is None or len(df) == 0:
                equity += pos["entry_px"] * pos["qty"]
                continue
            
            # 해당 날짜 데이터 있으면 사용
            if date in df.index:
                px = df.loc[date, "close"]
                if not pd.isna(px) and px > 0:
                    equity += px * pos["qty"]
                else:
                    # NaN이면 진입가 사용
                    equity += pos["entry_px"] * pos["qty"]
            else:
                # 해당 날짜 데이터 없으면 직전 유효 가격 사용
                prev_data = df[df.index <= date]
                if len(prev_data) > 0:
                    last_px = prev_data.iloc[-1]["close"]
                    if not pd.isna(last_px) and last_px > 0:
                        equity += last_px * pos["qty"]
                    else:
                        equity += pos["entry_px"] * pos["qty"]
                else:
                    # 해당 날짜 이전 데이터 없으면 진입가 사용
                    equity += pos["entry_px"] * pos["qty"]
        
        return equity
    
    def get_sector(self, ticker: str) -> str:
        """종목의 섹터 반환"""
        try:
            import pickle
            import os
            sector_map_path = os.path.join(os.path.dirname(__file__), "..", "data", "meta", "sector_map.pkl")
            if os.path.exists(sector_map_path):
                with open(sector_map_path, "rb") as f:
                    sector_map = pickle.load(f)
                return sector_map.get(ticker, "기타")
        except:
            pass
        return "기타"
