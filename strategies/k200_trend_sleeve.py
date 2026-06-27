#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KOSPI200 trend sleeve.

Uses a broad-market ETF so the allocator can participate when leadership is
concentrated in KOSPI large caps instead of KOSDAQ/small-cap names.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import FEE_PER_SIDE, FEE_PER_SIDE_US, SLIPPAGE_ENTRY, SLIPPAGE_EXIT, TAX_RATE_SELL, US_TAX_RATE_SELL
from strategies.base_strategy import BaseStrategy


class K200TrendSleeve(BaseStrategy):
    def __init__(
        self,
        ticker: str = "069500",  # KODEX 200
        strong_exposure: float = 0.95,
        trend_exposure: float = 0.70,
        recovery_exposure: float = 0.40,
        max_ret_abs: float = 0.20,
    ):
        self.ticker = ticker
        self.strong_exposure = strong_exposure
        self.trend_exposure = trend_exposure
        self.recovery_exposure = recovery_exposure
        self.max_ret_abs = max_ret_abs
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage_entry = SLIPPAGE_ENTRY
        self.slippage_exit = SLIPPAGE_EXIT
        self._market_profile_set = False

    def get_name(self):
        return "k200_trend_sleeve"

    def get_description(self):
        return "KOSPI200 trend sleeve using KODEX 200 / SPY fallback"

    def _is_us_market(self, tickers):
        sample = list(tickers)[:30]
        return bool(sample) and any(not str(t).isdigit() for t in sample)

    def _set_market_profile(self, enriched):
        if self._market_profile_set:
            return
        if self._is_us_market(enriched.keys()):
            if self.ticker == "069500":
                self.ticker = "SPY"
            self.fee = FEE_PER_SIDE_US
            self.tax = US_TAX_RATE_SELL
            self.slippage_entry = 0.0008
            self.slippage_exit = 0.0008
        self._market_profile_set = True

    def _indicators(self, df):
        out = df.copy()
        close = out["close"].astype(float)
        out["ma20"] = close.rolling(20).mean()
        out["ma60"] = close.rolling(60).mean()
        out["mom20"] = close.pct_change(20)
        out["mom60"] = close.pct_change(60)
        return out

    def _target_exposure(self, row):
        close = row.get("close", np.nan)
        ma20 = row.get("ma20", np.nan)
        ma60 = row.get("ma60", np.nan)
        mom20 = row.get("mom20", np.nan)
        mom60 = row.get("mom60", np.nan)
        if not all(np.isfinite(v) for v in [close, ma20, ma60, mom20, mom60]):
            return 0.0
        if close > ma20 > ma60 and mom20 > 0.03 and mom60 > 0.08:
            return self.strong_exposure
        if close > ma60 and mom20 > 0 and mom60 > 0:
            return self.trend_exposure
        if close > ma60 and mom20 > 0:
            return self.recovery_exposure
        return 0.0

    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent: bool = False):
        self._reset_weight_history()
        self._set_market_profile(enriched)
        df = enriched.get(self.ticker)
        if df is None or df.empty or len(df) < 80:
            if not silent:
                print(f"[k200_trend_sleeve] missing data: {self.ticker}")
            return pd.DataFrame(), []

        df = self._indicators(df).dropna(subset=["open", "close", "ma20", "ma60", "mom20", "mom60"])
        if df.empty:
            return pd.DataFrame(), []

        dates = list(df.index)
        cash = 1_000_000.0
        qty = 0
        entry_px = 0.0
        equity_curve = []
        trades = []

        for i in tqdm(range(1, len(dates)), disable=silent, desc=self.get_name()):
            signal_date = dates[i - 1]
            current_date = dates[i]
            signal = df.loc[signal_date]
            current = df.loc[current_date]

            close_px = float(current["close"])
            open_px = float(current["open"])
            if not np.isfinite(open_px) or open_px <= 0:
                open_px = close_px
            if not np.isfinite(close_px) or close_px <= 0:
                close_px = open_px

            equity = cash + qty * close_px
            target_exposure = self._target_exposure(signal)
            target_value = equity * target_exposure
            target_qty = int(target_value / open_px) if open_px > 0 else 0
            delta = target_qty - qty

            if delta < 0:
                sell_qty = min(qty, -delta)
                if sell_qty > 0:
                    exec_px = open_px * (1 - self.slippage_exit)
                    gross = sell_qty * exec_px
                    cost = sell_qty * entry_px
                    pnl = gross - cost
                    fee = gross * self.fee
                    tax = gross * self.tax if pnl > 0 else 0.0
                    cash += gross - fee - tax
                    qty -= sell_qty
                    if qty <= 0:
                        qty = 0
                        entry_px = 0.0
                    trades.append({
                        "date": current_date,
                        "ticker": self.ticker,
                        "action": "SELL",
                        "price": exec_px,
                        "qty": sell_qty,
                        "pnl": pnl,
                        "cash_after": cash,
                        "reason": "trend_reduce",
                    })
            elif delta > 0:
                exec_px = open_px * (1 + self.slippage_entry)
                cash_per_share = exec_px * (1 + self.fee)
                buy_qty = min(delta, int(cash / cash_per_share) if cash_per_share > 0 else 0)
                if buy_qty > 0:
                    gross = buy_qty * exec_px
                    fee = gross * self.fee
                    cash -= gross + fee
                    if qty > 0:
                        entry_px = (entry_px * qty + exec_px * buy_qty) / (qty + buy_qty)
                    else:
                        entry_px = exec_px
                    qty += buy_qty
                    trades.append({
                        "date": current_date,
                        "ticker": self.ticker,
                        "action": "BUY",
                        "price": exec_px,
                        "qty": buy_qty,
                        "pnl": 0.0,
                        "cash_after": cash,
                        "reason": "trend_follow",
                    })

            equity = cash + qty * close_px
            equity_curve.append((current_date, equity))
            positions = {}
            if qty > 0:
                positions[self.ticker] = {"qty": qty, "entry_px": entry_px}
            self._record_weights(current_date, cash, positions, {self.ticker: df})

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        return ec, trades
