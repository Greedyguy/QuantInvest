#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class ETFRiskOverlayStrategy(BaseStrategy):
    """
    단순 ETF 리스크 관리 전략
    - KODEX 200 / 성장 / 인버스 ETF 조합
    - 시장 레짐에 따라 노출 조절 및 헤지 ETF 사용
    """

    def __init__(
        self,
        etf_universe: Dict[str, Dict[str, str]] = None,
        rebal_days: int = 5,
        lookback: int = 120,
        slippage: float = 0.0005,
        regime_buffer_days: int = 3,
    ):
        self.etf_universe = etf_universe or {
            "069500": {"role": "beta"},     # KODEX 200
            "233740": {"role": "growth"},   # KODEX 코스닥150 레버리지
            "305720": {"role": "balanced"}, # TIGER 미국채
            "114800": {"role": "hedge"},    # KODEX 인버스
        }
        self.rebal_days = rebal_days
        self.lookback = lookback
        self.slippage = slippage
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.regime_buffer_days = regime_buffer_days

        self.regime_exposure = {
            "bull": 0.95,
            "neutral": 0.65,
            "bear": 0.4,
            "ultra_bear": 0.25,
        }

    def get_name(self) -> str:
        return "etf_defensive"

    def get_description(self) -> str:
        return "ETF Defensive Overlay (risk-aware single position ETF)"

    # ------------------------------------------------------------------
    def _prepare_universe(self, enriched: dict):
        universe = {}
        for ticker, meta in self.etf_universe.items():
            df = enriched.get(ticker)
            if df is None or df.empty:
                continue
            universe[ticker] = {
                "df": df.sort_index(),
                "role": meta.get("role", "beta"),
            }
        return universe

    def _get_price(self, df, date):
        if date in df.index:
            px = df.loc[date, "close"]
        else:
            prev = df[df.index <= date]
            px = prev.iloc[-1]["close"] if len(prev) else np.nan
        if not np.isfinite(px) or px <= 0:
            return np.nan
        return float(px)

    def _calc_equity(self, cash, positions, universe, date):
        total = cash
        for ticker, pos in positions.items():
            df = universe.get(ticker, {}).get("df")
            if df is None or df.empty:
                px = pos["entry_px"]
            else:
                px = self._get_price(df, date)
                if np.isnan(px):
                    px = pos["entry_px"]
            total += px * pos["qty"]
        return total

    def _compute_regime_states(self, market_index):
        if market_index is None or "close" not in market_index.columns:
            return {}
        idx = market_index.copy().sort_index()
        idx = idx[idx["close"].notna()]
        if len(idx) < 50:
            return {}
        idx["ma50"] = idx["close"].rolling(50).mean()
        idx["ma120"] = idx["close"].rolling(120).mean()
        idx["mom20"] = idx["close"].pct_change(20)
        idx["mom60"] = idx["close"].pct_change(60)
        flags = {}
        buffer = self.regime_buffer_days
        raw = []
        for dt, row in idx.iterrows():
            state = "neutral"
            if row["close"] > row["ma50"] and row["close"] > row["ma120"] and row["mom20"] > 0:
                state = "bull"
            elif row["close"] >= row["ma120"]:
                state = "neutral"
            elif row["mom20"] < -0.03 or row["close"] < row["ma120"]:
                state = "bear"
                if row["mom60"] < -0.08:
                    state = "ultra_bear"
            raw.append((dt, state))
        states = {}
        last_state = "neutral"
        count_same = 0
        for dt, state in raw:
            if state == last_state:
                count_same += 1
            else:
                count_same = 1
            if count_same >= buffer:
                last_state = state
            states[dt] = last_state
        return states

    def _score_etf(self, df, date):
        subset = df[df.index <= date]
        if len(subset) < 80:
            return None
        close = subset["close"]
        mom20 = close.iloc[-1] / close.iloc[-20] - 1
        mom60 = close.iloc[-1] / close.iloc[-60] - 1
        vol20 = close.pct_change().tail(20).std()
        score = 0.6 * mom20 + 0.4 * mom60 - 0.5 * (vol20 if np.isfinite(vol20) else 0)
        return score

    def _select_target(self, universe, date, regime):
        role_priority = {
            "bull": ["growth", "beta", "balanced"],
            "neutral": ["beta", "balanced", "hedge"],
            "bear": ["hedge", "balanced"],
            "ultra_bear": ["hedge"],
        }
        allowed_roles = role_priority.get(regime, ["beta"])
        best_ticker = None
        best_score = None
        for ticker, meta in universe.items():
            role = meta["role"]
            if role not in allowed_roles:
                continue
            score = self._score_etf(meta["df"], date)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_ticker = ticker
        # fallback: allow any ETF when none scored but regime asks for hedge
        if best_ticker is None and allowed_roles == ["hedge"]:
            for ticker, meta in universe.items():
                if meta["role"] != "hedge":
                    continue
                score = self._score_etf(meta["df"], date)
                if score is None:
                    continue
                best_ticker = ticker
                break
        return best_ticker

    # ------------------------------------------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):
        self._reset_weight_history()
        universe = self._prepare_universe(enriched)
        if not universe:
            return pd.DataFrame(), []

        date_set = set()
        for meta in universe.values():
            date_set.update(meta["df"].index)
        dates = sorted(date_set)
        if len(dates) < self.lookback + 5:
            return pd.DataFrame(), []

        rebalance_dates = dates[self.lookback::self.rebal_days]
        if not rebalance_dates:
            return pd.DataFrame(), []

        regime_states = self._compute_regime_states(market_index)

        cash = 1_000_000.0
        positions = {}
        equity_curve = []
        trade_log = []
        df_map = {t: meta["df"] for t, meta in universe.items()}

        for idx, d0 in enumerate(rebalance_dates):
            next_rebal = rebalance_dates[idx + 1] if idx < len(rebalance_dates) - 1 else dates[-1]
            state = regime_states.get(d0, "neutral")
            target_expo = self.regime_exposure.get(state, 0.6)
            target_ticker = self._select_target(universe, d0, state)

            # 1) Sell positions not targeted
            for ticker in list(positions.keys()):
                if ticker == target_ticker:
                    continue
                df = df_map.get(ticker)
                if df is None:
                    continue
                px = self._get_price(df, d0)
                if np.isnan(px):
                    continue
                exit_px = px * (1 - self.slippage)
                qty = positions[ticker]["qty"]
                proceeds = qty * exit_px
                fee = proceeds * self.fee
                tax = proceeds * self.tax
                net = proceeds - fee - tax
                cash += net
                trade_log.append({
                    "date": d0,
                    "ticker": ticker,
                    "action": "SELL",
                    "price": exit_px,
                    "qty": qty,
                    "pnl": net - positions[ticker]["entry_px"] * qty,
                })
                positions.pop(ticker, None)

            # 2) Buy/adjust target ETF
            equity = self._calc_equity(cash, positions, universe, d0)
            target_value = equity * target_expo
            if target_ticker is not None:
                df = df_map.get(target_ticker)
                if df is not None:
                    px = self._get_price(df, d0)
                    if not np.isnan(px):
                        entry_px = px * (1 + self.slippage)
                        target_qty = int(target_value / entry_px)
                        cur_qty = positions.get(target_ticker, {}).get("qty", 0)
                        delta = target_qty - cur_qty
                        if delta < 0 and cur_qty > 0:
                            sell_qty = -delta
                            exit_px = px * (1 - self.slippage)
                            proceeds = sell_qty * exit_px
                            fee = proceeds * self.fee
                            tax = proceeds * self.tax
                            net = proceeds - fee - tax
                            cash += net
                            positions[target_ticker]["qty"] -= sell_qty
                            if positions[target_ticker]["qty"] <= 0:
                                positions.pop(target_ticker, None)
                            trade_log.append({
                                "date": d0,
                                "ticker": target_ticker,
                                "action": "SELL",
                                "price": exit_px,
                                "qty": sell_qty,
                                "pnl": net - positions.get(target_ticker, {}).get("entry_px", entry_px) * sell_qty,
                            })
                        elif delta > 0:
                            buy_cost = delta * entry_px
                            fee = buy_cost * self.fee
                            total_cost = buy_cost + fee
                            if total_cost <= cash and delta > 0:
                                cash -= total_cost
                                if target_ticker in positions:
                                    old = positions[target_ticker]
                                    new_qty = old["qty"] + delta
                                    avg_px = (old["entry_px"] * old["qty"] + entry_px * delta) / new_qty
                                    positions[target_ticker] = {"qty": new_qty, "entry_px": avg_px}
                                else:
                                    positions[target_ticker] = {"qty": delta, "entry_px": entry_px}
                                trade_log.append({
                                    "date": d0,
                                    "ticker": target_ticker,
                                    "action": "BUY",
                                    "price": entry_px,
                                    "qty": delta,
                                    "pnl": 0,
                                })

            # 3) Daily equity/weights until next rebalance
            interval = [dt for dt in dates if d0 <= dt < next_rebal]
            for dt in interval:
                eq = self._calc_equity(cash, positions, universe, dt)
                equity_curve.append((dt, eq))
                self._record_weights(dt, cash, positions, df_map)

        # Final liquidation
        final_date = dates[-1]
        for ticker, pos in list(positions.items()):
            df = df_map.get(ticker)
            if df is None:
                continue
            px = self._get_price(df, final_date)
            if np.isnan(px):
                px = pos["entry_px"]
            exit_px = px * (1 - self.slippage)
            qty = pos["qty"]
            proceeds = qty * exit_px
            fee = proceeds * self.fee
            tax = proceeds * self.tax
            net = proceeds - fee - tax
            cash += net
            trade_log.append({
                "date": final_date,
                "ticker": ticker,
                "action": "SELL",
                "price": exit_px,
                "qty": qty,
                "pnl": net - pos["entry_px"] * qty,
                "reason": "final_liq",
            })
            positions.pop(ticker, None)

        equity_curve.append((final_date, cash))
        self._record_weights(final_date, cash, positions, df_map)
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        return ec.sort_index(), trade_log
