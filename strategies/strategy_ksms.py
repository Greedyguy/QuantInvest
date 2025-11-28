#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KSMS v2 (BUGFIXED & CONSERVATIVE VERSION)

- 소액(100만) 스윙 전략
- 유동성 제약 + 계좌 상한 + 시그널/진입 시점 분리 + 수익률 이상치 방어
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KSMSStrategy(BaseStrategy):
    def __init__(
        self,
        stop_loss: float = -0.04,
        take_profit: float = 0.12,
        max_hold_days: int = 7,
        min_market_cap: int = 300,    # 억
        max_market_cap: int = 3000,   # 억
        min_price: int = 500,
        max_price: int = 30000,
        min_trade_value: int = 5,     # 억
        max_trade_value: int = 80,    # 억
        adv_participation: float = 0.10,  # ADV20의 최대 참여 비율 (10%)
        max_trade_risk: float = 0.20,     # 계좌의 20%까지만 한 종목에
        max_equity_limit: int = 20_000_000,  # KSMS 전략 유효 구간 상한 (2천만)
        max_ret_abs: float = 1.0,     # 1회 거래 수익률 절대값 상한 (100%)
        use_next_open_entry: bool = True,  # True: 전일 시그널 → 익일 시가 진입
    ):
        # 리스크 파라미터
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days

        # 단위 변환 (억 → 원)
        self.min_market_cap = min_market_cap * 100_000_000
        self.max_market_cap = max_market_cap * 100_000_000
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.min_trade_value = min_trade_value * 100_000_000
        self.max_trade_value = max_trade_value * 100_000_000

        # 비용/슬리피지
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = 0.003  # 0.3%

        # 유동성 & 계좌 제약
        self.adv_participation = adv_participation
        self.max_trade_risk = max_trade_risk
        self.max_equity_limit = max_equity_limit

        # 이상치 방어
        self.max_ret_abs = max_ret_abs  # 예: 1.0 → ±100% 이상은 클리핑
        self.max_qty = 1_000_000        # 100만주는 비정상으로 간주

        # 시그널/진입 시점 설정
        self.use_next_open_entry = use_next_open_entry

    def get_name(self):
        return "ksms_v2"

    def get_description(self):
        mode = "D-1 signal / D open" if self.use_next_open_entry else "same-day close"
        return f"KSMS v2 (Small-Mo Swing, {mode})"

    # ---------------------------------------------------------
    # 내부 유틸
    # ---------------------------------------------------------
    def _safe_slice(self, df: pd.DataFrame, end_date: pd.Timestamp, window: int) -> pd.DataFrame:
        if df is None or end_date not in df.index:
            return pd.DataFrame()
        loc = df.index.get_loc(end_date)
        if isinstance(loc, slice):
            loc = df.index.tolist().index(end_date)
        if loc + 1 < window:
            return pd.DataFrame()
        return df.iloc[loc + 1 - window : loc + 1]

    def _get_position_value(self, position, enriched, current_date: pd.Timestamp) -> float:
        """현재 날짜 기준 포지션의 평가 금액."""
        if position is None:
            return 0.0
        tkr = position["ticker"]
        df_t = enriched.get(tkr)
        if df_t is None or len(df_t) == 0:
            return position["entry_px"] * position["qty"]
        # current_date 기준 가장 가까운 과거 가격 사용
        sub = df_t[df_t.index <= current_date]
        if len(sub) == 0:
            px = position["entry_px"]
        else:
            px = float(sub["close"].iloc[-1])
        if not np.isfinite(px) or px <= 0:
            px = position["entry_px"]
        return px * position["qty"]

    # ---------------------------------------------------------
    # Universe 필터 & 시그널
    # ---------------------------------------------------------
    def _check_universe_filter(self, df: pd.DataFrame, current_date: pd.Timestamp) -> bool:
        if df is None or current_date not in df.index:
            return False

        row = df.loc[current_date]
        price = float(row["close"])
        if (not np.isfinite(price)) or price <= 0:
            return False
        if price < self.min_price or price > self.max_price:
            return False

        if "volume" not in df.columns:
            return False

        recent20 = self._safe_slice(df, current_date, 20)
        if recent20.empty:
            return False

        tv20 = (recent20["close"] * recent20["volume"]).mean()
        if (not np.isfinite(tv20)) or tv20 <= 0:
            return False
        if tv20 < self.min_trade_value or tv20 > self.max_trade_value:
            return False

        if "market_cap" in df.columns:
            mc = float(row.get("market_cap", 0))
            if np.isfinite(mc) and mc > 0:
                if mc < self.min_market_cap or mc > self.max_market_cap:
                    return False

        return True

    def _compute_signal(self, df: pd.DataFrame, signal_date: pd.Timestamp):
        """신호는 signal_date 기준 (전일 or 당일)"""
        if df is None or signal_date not in df.index:
            return None

        if not self._check_universe_filter(df, signal_date):
            return None

        recent5 = self._safe_slice(df, signal_date, 5)
        recent20 = self._safe_slice(df, signal_date, 20)
        if recent5.empty or recent20.empty:
            return None

        p_now = float(recent5["close"].iloc[-1])
        p_5ago = float(recent5["close"].iloc[0])
        if p_5ago <= 0 or not np.isfinite(p_5ago) or not np.isfinite(p_now):
            return None

        ret5 = p_now / p_5ago - 1.0
        if abs(ret5) > 2.0:  # 5일 동안 ±200%는 데이터 이상 가능성
            return None

        tv5 = (recent5["close"] * recent5["volume"]).mean()
        tv20 = (recent20["close"] * recent20["volume"]).mean()
        if tv20 <= 0 or (not np.isfinite(tv20)):
            return None
        volume_surge = tv5 / tv20

        if "high" in df.columns:
            high20 = float(recent20["high"].max())
        else:
            high20 = float(recent20["close"].max())
        if not np.isfinite(high20) or high20 <= 0:
            return None

        is_breakout = p_now >= high20 * 0.999  # float 여유

        return {
            "ret_5d": ret5,
            "volume_surge": volume_surge,
            "is_breakout": is_breakout,
            "price": p_now,
            "adv20": tv20,
        }

    # ---------------------------------------------------------
    # 백테스트 실행
    # ---------------------------------------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights: dict | None = None, silent: bool = False):
        if not silent:
            print("\n" + "=" * 60)
            print("📈 KSMS v2 백테스트 (BUGFIXED) 시작...")
            print("=" * 60)

        dates = sorted(
            set().union(*[df.index for df in enriched.values() if df is not None and len(df) > 0])
        )
        if len(dates) < 60:
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        position = None
        equity_curve = []
        trades = []

        # 메인 루프: 시그널 날짜와 진입 날짜 분리
        if self.use_next_open_entry:
            # D-1 signal / D open 구조
            iterable = range(1, len(dates))  # i: 1..N-1
        else:
            # same-day close 구조
            iterable = range(0, len(dates))

        for i in tqdm(iterable, disable=silent, desc=self.get_name()):
            if self.use_next_open_entry:
                signal_date = dates[i - 1]
                current_date = dates[i]  # 진입 & 포지션 평가 기준
            else:
                signal_date = dates[i]
                current_date = dates[i]

            # 0) 현재 equity 계산 & 계좌 상한 체크 (❗ cash → equity로 수정)
            pos_val = self._get_position_value(position, enriched, current_date)
            equity_now = cash + pos_val
            if equity_now > self.max_equity_limit:
                if not silent:
                    print(f"\n🎯 Equity {equity_now:,.0f}원 > limit {self.max_equity_limit:,.0f}원 → KSMS 종료")
                break

            # 1) 기존 포지션 관리/청산
            if position is not None:
                tkr = position["ticker"]
                df_t = enriched.get(tkr)
                if df_t is not None and current_date in df_t.index:
                    px = float(df_t.loc[current_date, "close"])
                    if not np.isfinite(px) or px <= 0:
                        px = position["entry_px"]
                    entry_px = position["entry_px"]
                    qty = position["qty"]
                    hold_days = (current_date - position["entry_date"]).days

                    ret = px / entry_px - 1.0
                    # 이상치 수익률 방어
                    if abs(ret) > self.max_ret_abs:
                        if not silent:
                            print(f"[WARN] 이상 수익률 클리핑: {tkr} {current_date.date()} ret={ret:.2f}")
                        ret = np.sign(ret) * self.max_ret_abs
                        px = entry_px * (1 + ret)

                    # 트레일링용 high 업데이트
                    position["high"] = max(position["high"], px)

                    reason = None
                    if ret <= self.stop_loss:
                        reason = "STOP_LOSS"
                    elif ret >= self.take_profit:
                        reason = "TAKE_PROFIT"
                    elif hold_days >= self.max_hold_days:
                        reason = "MAX_HOLD"
                    else:
                        if ret > 0.05:
                            trail = px / position["high"] - 1.0
                            if trail <= -0.03:
                                reason = "TRAIL"

                    if reason is not None:
                        exit_px = px * (1 - self.slippage)
                        gross = exit_px * qty
                        pnl = gross - entry_px * qty
                        fee = gross * self.fee
                        tax = gross * self.tax if pnl > 0 else 0.0
                        cash += (gross - fee - tax)

                        trades.append({
                            "date": current_date,
                            "ticker": tkr,
                            "ret": pnl / (entry_px * qty),
                            "pnl": pnl,
                            "reason": reason,
                        })
                        position = None

            # 2) 신규 진입
            if position is None:
                candidates = []
                for tkr, df in enriched.items():
                    sig = self._compute_signal(df, signal_date)
                    if sig is None:
                        continue
                    if sig["volume_surge"] < 2.0:
                        continue
                    if not sig["is_breakout"]:
                        continue
                    candidates.append({**sig, "ticker": tkr})

                if len(candidates) > 0:
                    df_c = pd.DataFrame(candidates).sort_values("ret_5d", ascending=False)
                    top_n = max(1, int(len(df_c) * 0.03))
                    top_c = df_c.head(top_n)
                    best = top_c.sort_values("volume_surge", ascending=False).iloc[0]

                    tkr = best["ticker"]
                    df_t = enriched[tkr]

                    # 진입 가격: D open (또는 same-day close)
                    if self.use_next_open_entry:
                        if current_date not in df_t.index:
                            # 진입 불가
                            pass
                        else:
                            raw_px = float(df_t.loc[current_date, "open"])
                    else:
                        # same-day close 진입 모드
                            # 주의: 실전 가능성은 낮지만 선택 옵션
                        if current_date not in df_t.index:
                            raw_px = None
                        else:
                            raw_px = float(df_t.loc[current_date, "close"])

                    if raw_px is not None and raw_px > 0 and np.isfinite(raw_px):
                        entry_px = raw_px * (1 + self.slippage)
                        adv20 = float(best["adv20"])
                        # 유동성 제약
                        max_notional_liq = adv20 * self.adv_participation
                        max_notional_eq = cash * self.max_trade_risk
                        max_notional = min(max_notional_liq, max_notional_eq)
                        qty = int(max_notional / (entry_px * (1 + self.fee)))

                        if qty > 0 and qty <= self.max_qty:
                            cost = entry_px * qty
                            fee = cost * self.fee
                            total = cost + fee
                            if total <= cash:
                                cash -= total
                                position = {
                                    "ticker": tkr,
                                    "qty": qty,
                                    "entry_px": entry_px,
                                    "entry_date": current_date,
                                    "high": entry_px,
                                }

            # 3) Equity 기록
            pos_val = self._get_position_value(position, enriched, current_date)
            equity = cash + pos_val
            equity_curve.append((current_date, equity))

        ec_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        return ec_df, trades