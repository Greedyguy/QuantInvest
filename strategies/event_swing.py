# strategies/event_swing.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class EventSwingStrategy(BaseStrategy):
    """
    Event-Swing v1
    - 의사 이벤트(갭업 + 거래량 급증)를 기반으로 하는 단기 스윙 전략
    """

    def __init__(
        self,
        stop_loss: float = -0.05,
        take_profit: float = 0.10,
        max_hold_days: int = 7,
        min_price: int = 5_000,
        max_price: int = 80_000,
        min_trade_value_20d: float = 5e8,
        vol_surge_th: float = 3.0,    # 거래대금 3배
        gap_th: float = 0.05,         # 시가 갭업 5%
        day_ret_th: float = 0.07,     # 또는 종가 수익률 7%
        pullback_min: float = 0.03,   # 이벤트 종가 대비 3%~8% 눌림
        pullback_max: float = 0.08,
        slippage: float = 0.002,
    ):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.min_price = min_price
        self.max_price = max_price
        self.min_trade_value_20d = min_trade_value_20d
        self.vol_surge_th = vol_surge_th
        self.gap_th = gap_th
        self.day_ret_th = day_ret_th
        self.pullback_min = pullback_min
        self.pullback_max = pullback_max
        self.slippage = slippage

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    def get_name(self):
        return "event_swing_v1"

    def get_description(self):
        return "Event-Swing v1 (갭 + 거래량 급증 후 눌림 스윙)"

    # ------------------------------------------------------------------
    @staticmethod
    def _vwap_proxy(row: pd.Series) -> float:
        h, l, c = row["high"], row["low"], row["close"]
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
            return np.nan
        return (h + l + c) / 3.0

    @staticmethod
    def _get_last_valid_price(df: pd.DataFrame, date: pd.Timestamp) -> float:
        if df is None or len(df) == 0:
            return np.nan
        if date in df.index:
            return float(df.loc[date, "close"])
        valid = df.index[df.index <= date]
        if len(valid) == 0:
            return np.nan
        return float(df.loc[valid.max(), "close"])

    # ------------------------------------------------------------------
    def _is_event_day(self, df: pd.DataFrame, idx: int) -> bool:
        """
        의사 이벤트 day:
        - 시가 갭업 >= gap_th (open / prev_close - 1 >= gap_th) OR
        - 당일 종가 수익률 >= day_ret_th
        - 거래대금 ≥ 20일 평균 * vol_surge_th
        """
        if idx < 20:
            return False

        sub = df.iloc[idx - 20: idx + 1]
        row = sub.iloc[-1]

        c = row["close"]
        o = row["open"]
        if c < self.min_price or c > self.max_price:
            return False

        prev_c = sub["close"].iloc[-2]
        if prev_c <= 0:
            return False

        gap = o / prev_c - 1.0
        day_ret = c / prev_c - 1.0

        # 갭업 또는 당일 급등
        if (gap < self.gap_th) and (day_ret < self.day_ret_th):
            return False

        # 거래대금 급증
        if "volume" in sub.columns:
            tv_20 = (sub["close"][:-1] * sub["volume"][:-1]).mean()
            tv_today = row["close"] * row["volume"]
            if tv_20 <= 0 or tv_today <= 0:
                return False
            if tv_20 < self.min_trade_value_20d:
                return False
            if tv_today / tv_20 < self.vol_surge_th:
                return False

        return True

    # ------------------------------------------------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent: bool = False):
        if not silent:
            print("\n" + "=" * 60)
            print("📈 Event-Swing v1 백테스트 시작...")
            print("=" * 60)

        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 60:
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        position = None
        equity_curve = []
        trade_log = []

        # 종목별 이벤트일 리스트 계산
        event_map = {}  # ticker -> list of (event_date, event_close)
        for t, df in enriched.items():
            if df is None or len(df) < 40:
                event_map[t] = []
                continue
            ev_list = []
            for i in range(len(df)):
                if self._is_event_day(df, i):
                    d = df.index[i]
                    ev_close = df.iloc[i]["close"]
                    ev_list.append((d, ev_close))
            event_map[t] = ev_list

        for d in tqdm(dates, desc="EventSwing", disable=silent):
            # 1) 기존 포지션 관리
            if position is not None:
                t = position["ticker"]
                df_t = enriched.get(t)
                if df_t is not None and d in df_t.index:
                    row = df_t.loc[d]
                    vwap = self._vwap_proxy(row)
                    if not np.isfinite(vwap) or vwap <= 0:
                        vwap = row["close"]

                    ret = vwap / position["entry_px"] - 1.0
                    hold_days = (d - position["entry_date"]).days
                    reason = None

                    if ret <= self.stop_loss:
                        reason = "STOP"
                    elif ret >= self.take_profit:
                        reason = "TAKE"
                    elif hold_days >= self.max_hold_days:
                        reason = "MAX_HOLD"

                    if reason is not None:
                        exit_px = vwap * (1 - self.slippage)
                        qty = position["qty"]
                        proceeds = exit_px * qty
                        pnl = proceeds - position["entry_px"] * qty
                        fee = proceeds * self.fee
                        tax = proceeds * self.tax if pnl > 0 else 0.0
                        net = proceeds - fee - tax
                        cash += net

                        trade_log.append({
                            "date": d,
                            "ticker": t,
                            "action": "SELL",
                            "price": exit_px,
                            "qty": qty,
                            "pnl": pnl,
                            "reason": reason,
                            "cash_after": cash,
                            "hold_days": hold_days,
                        })
                        position = None

            # 2) 신규 진입 (포지션 없을 때만)
            if position is None:
                candidates = []
                for t, df in enriched.items():
                    if df is None or d not in df.index:
                        continue
                    ev_list = event_map.get(t, [])
                    # 이벤트 이후 1~3일 사이 눌림 구간인지
                    recent_ev = [ev for ev in ev_list if 0 < (d - ev[0]).days <= 3]
                    if not recent_ev:
                        continue
                    ev_date, ev_close = sorted(recent_ev, key=lambda x: x[0])[-1]
                    c_today = df.loc[d, "close"]
                    dd = 1.0 - (c_today / ev_close)  # 이벤트 종가 대비 하락률
                    if self.pullback_min <= dd <= self.pullback_max:
                        candidates.append((t, dd, ev_date, ev_close))

                if candidates:
                    # 중간 정도 눌림 종목 선택
                    mid_dd = (self.pullback_min + self.pullback_max) / 2
                    candidates.sort(key=lambda x: abs(x[1] - mid_dd))
                    best_t, best_dd, ev_date, ev_close = candidates[0]

                    df_t = enriched[best_t]
                    row = df_t.loc[d]
                    vwap = self._vwap_proxy(row)
                    if not np.isfinite(vwap) or vwap <= 0:
                        vwap = row["close"]

                    entry_px = vwap * (1 + self.slippage)
                    if entry_px <= 0:
                        equity = cash
                        equity_curve.append((d, equity))
                        continue

                    qty = int(cash / (entry_px * (1 + self.fee)))
                    if qty > 0:
                        cost = entry_px * qty
                        fee = cost * self.fee
                        total_cost = cost + fee
                        if total_cost <= cash:
                            cash -= total_cost
                            position = {
                                "ticker": best_t,
                                "qty": qty,
                                "entry_px": entry_px,
                                "entry_date": d,
                                "event_date": ev_date,
                                "event_close": ev_close,
                            }
                            trade_log.append({
                                "date": d,
                                "ticker": best_t,
                                "action": "BUY",
                                "price": entry_px,
                                "qty": qty,
                                "pnl": 0.0,
                                "reason": "EVENT_PULLBACK",
                                "cash_after": cash,
                            })

            # 3) Equity 기록
            equity = cash
            if position is not None:
                df_t = enriched.get(position["ticker"])
                if df_t is not None:
                    px = self._get_last_valid_price(df_t, d)
                    if not np.isfinite(px) or px <= 0:
                        px = position["entry_px"]
                    equity += px * position["qty"]
            equity_curve.append((d, equity))

        # 마지막 청산
        final_d = dates[-1]
        if position is not None:
            t = position["ticker"]
            df_t = enriched.get(t)
            if df_t is not None:
                px = self._get_last_valid_price(df_t, final_d)
                exit_px = px * (1 - self.slippage)
                qty = position["qty"]
                proceeds = exit_px * qty
                pnl = proceeds - position["entry_px"] * qty
                fee = proceeds * self.fee
                tax = proceeds * self.tax if pnl > 0 else 0.0
                net = proceeds - fee - tax
                cash += net
                trade_log.append({
                    "date": final_d,
                    "ticker": t,
                    "action": "SELL",
                    "price": exit_px,
                    "qty": qty,
                    "pnl": pnl,
                    "reason": "END",
                    "cash_after": cash,
                })

        equity_curve[-1] = (final_d, cash)
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        if not silent:
            print(f"✅ Event-Swing v1 완료: {len(ec)} 포인트, 최종 자산 {cash:,.0f}원")
        return ec, trade_log