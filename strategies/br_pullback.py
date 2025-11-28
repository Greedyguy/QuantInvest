# strategies/br_pullback.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class BreakoutPullbackStrategy(BaseStrategy):
    """
    BR-Pullback v1
    - 돌파(20일 신고가 + 거래량 급증) 이후 2~6% 눌림에서 매수
    - 1종목 집중 스윙
    """

    def __init__(
        self,
        stop_loss: float = -0.04,
        take_profit: float = 0.08,
        max_hold_days: int = 10,
        min_price: int = 5_000,
        max_price: int = 80_000,
        min_trade_value_20d: float = 5e8,
        vol_surge_th: float = 2.0,  # 거래대금 2배 이상
        pullback_min: float = 0.02,
        pullback_max: float = 0.06,
        slippage: float = 0.0015,
    ):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.min_price = min_price
        self.max_price = max_price
        self.min_trade_value_20d = min_trade_value_20d
        self.vol_surge_th = vol_surge_th
        self.pullback_min = pullback_min
        self.pullback_max = pullback_max
        self.slippage = slippage

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    def get_name(self):
        return "br_pullback_v1"

    def get_description(self):
        return "Breakout Pullback v1 (20일 신고가 + 눌림 매수)"

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
    def _is_breakout_day(self, df: pd.DataFrame, idx: int) -> bool:
        """
        idx: 돌파일
        조건:
        - close[idx] == 지난 20일 중 최고가
        - 당일 수익률 > +3%
        - 거래대금이 20일 평균의 vol_surge_th 배 이상
        """
        if idx < 20:
            return False
        sub = df.iloc[idx - 20: idx + 1]
        if len(sub) < 21:
            return False
        row = sub.iloc[-1]
        c = row["close"]
        if c < self.min_price or c > self.max_price:
            return False

        # 유동성 필터
        if "volume" in sub.columns:
            tv_20 = (sub["close"][:-1] * sub["volume"][:-1]).mean()
            tv_today = row["close"] * row["volume"]
            if tv_20 <= 0 or tv_today <= 0:
                return False
            if tv_20 < self.min_trade_value_20d:
                return False
            if tv_today / tv_20 < self.vol_surge_th:
                return False

        # 20일 고가 돌파
        if c < sub["high"][:-1].max():
            return False

        # 당일 수익률
        prev_c = sub["close"].iloc[-2]
        ret = c / prev_c - 1.0
        if ret < 0.03:  # 3% 이상 상승
            return False

        return True

    # ------------------------------------------------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent: bool = False):
        self._reset_weight_history()
        if not silent:
            print("\n" + "=" * 60)
            print("📈 Breakout Pullback v1 백테스트 시작...")
            print("=" * 60)

        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 60:
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        position = None
        equity_curve = []
        trade_log = []

        # 각 종목별 breakout 후보 날짜 미리 표시
        breakout_map = {}  # ticker -> list of (breakout_date, breakout_close)
        for t, df in enriched.items():
            if df is None or len(df) < 40:
                continue
            br_list = []
            for i in range(len(df)):
                if self._is_breakout_day(df, i):
                    d = df.index[i]
                    br_list.append((d, df.iloc[i]["close"]))
            breakout_map[t] = br_list

        for d in tqdm(dates, desc="BR-Pullback", disable=silent):
            # 1) 기존 포지션 청산/관리
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

            # 2) 신규 진입 (포지션 없을 때)
            if position is None:
                candidates = []
                for t, df in enriched.items():
                    if df is None or d not in df.index:
                        continue
                    # 이 종목에 대해 직전 며칠 내 break-out이 있었는지 찾기
                    br_list = breakout_map.get(t, [])
                    # 최근 3일 내 돌파일
                    recent_br = [br for br in br_list if 0 < (d - br[0]).days <= 3]
                    if not recent_br:
                        continue
                    # 가장 최근 돌파일 기준
                    br_date, br_close = sorted(recent_br, key=lambda x: x[0])[-1]
                    c_today = df.loc[d, "close"]
                    # 돌파 종가 대비 2~6% 눌림
                    dd = 1.0 - (c_today / br_close)
                    if self.pullback_min <= dd <= self.pullback_max:
                        candidates.append((t, dd, br_date, br_close))

                if candidates:
                    # 더 "적당히 눌린" (dd 중간값 근처) 종목 선택
                    candidates.sort(key=lambda x: abs(x[1] - (self.pullback_min + self.pullback_max) / 2))
                    best_t, best_dd, br_date, br_c = candidates[0]

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
                                "br_date": br_date,
                                "br_close": br_c,
                            }
                            trade_log.append({
                                "date": d,
                                "ticker": best_t,
                                "action": "BUY",
                                "price": entry_px,
                                "qty": qty,
                                "pnl": 0.0,
                                "reason": "PULLBACK",
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
            positions_snapshot = {}
            if position is not None:
                positions_snapshot[position["ticker"]] = position
            self._record_weights(d, cash, positions_snapshot, enriched)

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
        self._record_weights(final_d, cash, {}, enriched)
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        if not silent:
            print(f"✅ BR-Pullback v1 완료: {len(ec)} 포인트, 최종 자산 {cash:,.0f}원")
        return ec, trade_log
