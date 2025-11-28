# strategies/mr_fast.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class MRFastStrategy(BaseStrategy):
    """
    MR-FAST v1
    - 단기 과매도 반등(mean reversion) 전략
    - 1종목 집중, 최대 3일 보유
    """

    def __init__(
        self,
        stop_loss: float = -0.02,    # -2%
        take_profit: float = 0.03,   # +3%
        max_hold_days: int = 3,
        min_price: int = 5_000,
        max_price: int = 50_000,
        min_trade_value_20d: float = 3e8,  # 20일 평균 거래대금 ≥ 3억
        slippage: float = 0.0015,
        z_th: float = -1.5,          # 볼린저 z-score 임계값
        min_drop: float = -0.04,     # 하루 수익률 ≤ -4% 이상 급락만
    ):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.min_price = min_price
        self.max_price = max_price
        self.min_trade_value_20d = min_trade_value_20d
        self.slippage = slippage
        self.z_th = z_th
        self.min_drop = min_drop

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    def get_name(self) -> str:
        return "mr_fast_v1"

    def get_description(self) -> str:
        return "MR-FAST v1 (단기 과매도 반등 1종목 집중)"

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
    def _compute_signal_today(self, df: pd.DataFrame, d: pd.Timestamp) -> bool:
        """
        오늘이 진입 후보(과매도 반등 신호)인지 판단.
        - 20일 추세 완전 붕괴는 아니고 (20일 수익률 > -10%)
        - 당일 수익률 ≤ -4%
        - 볼린저 밴드 하단(z-score < -1.5) 이탈
        """
        if df is None or d not in df.index:
            return False
        idx = df.index.get_loc(d)
        if idx < 20:
            return False

        sub = df.iloc[idx - 20: idx + 1]
        close = sub["close"].values
        if len(close) < 21:
            return False

        # 가격 필터
        price = close[-1]
        if price < self.min_price or price > self.max_price:
            return False

        # 유동성 필터: 20일 평균 거래대금
        if "volume" in sub.columns:
            trade_val = (sub["close"] * sub["volume"]).mean()
            if not np.isfinite(trade_val) or trade_val < self.min_trade_value_20d:
                return False

        # 20일 추세 수익률
        ret_20 = close[-1] / close[0] - 1.0
        if ret_20 < -0.10:  # 너무 무너진 추세는 피함
            return False

        # 당일 수익률
        c_y, c_t = close[-2], close[-1]
        daily_ret = c_t / c_y - 1.0
        if daily_ret > self.min_drop:  # 예: -4% 보다 덜 빠지면 제외
            return False

        # 볼린저 z-score
        mu = close[:-1].mean()
        sigma = close[:-1].std() + 1e-9
        z = (c_t - mu) / sigma
        if z > self.z_th:
            return False

        return True

    # ------------------------------------------------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent: bool = False):
        self._reset_weight_history()
        if not silent:
            print("\n" + "=" * 60)
            print("📈 MR-FAST v1 백테스트 시작...")
            print("=" * 60)

        # 전체 날짜
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 40:
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        position = None  # {"ticker", "qty", "entry_px", "entry_date"}
        equity_curve = []
        trade_log = []

        for d in tqdm(dates, desc="MR-FAST", disable=silent):
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
                    should_exit = False
                    reason = None

                    if ret <= self.stop_loss:
                        should_exit = True
                        reason = "STOP"
                    elif ret >= self.take_profit:
                        should_exit = True
                        reason = "TAKE"
                    elif hold_days >= self.max_hold_days:
                        should_exit = True
                        reason = "MAX_HOLD"

                    if should_exit:
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
                # 과매도 후보 스캔
                candidates = []
                for t, df in enriched.items():
                    if df is None or d not in df.index:
                        continue
                    if self._compute_signal_today(df, d):
                        candidates.append(t)

                if candidates:
                    # 간단히: 랜덤 대신 알파 가장 높은 후보 선택 → 20일 수익률이 가장 좋은 것
                    best_t = None
                    best_ret20 = -999
                    for t in candidates:
                        df = enriched[t]
                        idx = df.index.get_loc(d)
                        sub = df.iloc[idx - 20: idx + 1]
                        close = sub["close"].values
                        r20 = close[-1] / close[0] - 1.0
                        if r20 > best_ret20:
                            best_ret20 = r20
                            best_t = t

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
                            }
                            trade_log.append({
                                "date": d,
                                "ticker": best_t,
                                "action": "BUY",
                                "price": entry_px,
                                "qty": qty,
                                "pnl": 0.0,
                                "reason": "SIGNAL",
                                "cash_after": cash,
                            })

            # 3) Daily equity 기록
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

        # 마지막 포지션 청산
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
            print(f"✅ MR-FAST v1 완료: {len(ec)} 포인트, 최종 자산 {cash:,.0f}원")

        return ec, trade_log
