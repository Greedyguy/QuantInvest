# strategies/k200_mean_reversion.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K200 Mean Reversion Strategy (소액 계좌용)

- 대상: KODEX 200 (069500) 1종목
- 컨셉: 일봉 기반 단기 mean-reversion
  * 전일 기준 20일 z-score 과매도 + 단기 하락 → 다음날 VWAP 근사치 진입
  * 손절 -3%, 익절 +4%, 최대 보유 5일, z-score 0 회귀시 청산
- 데이터: enriched["069500"]에 OHLCV가 있다고 가정
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class K200MeanReversion(BaseStrategy):
    def __init__(
        self,
        ticker: str = "069500",        # KODEX 200
        stop_loss: float = -0.03,      # -3%
        take_profit: float = 0.04,     # +4%
        max_hold_days: int = 5,
        z_entry: float = -1.0,         # 진입 z-score 기준
        z_exit: float = 0.0,           # 청산 z-score 기준
        min_ret5: float = -0.03,       # 최근 5일 수익률 <= -3%
        slippage_entry: float = 0.001, # 0.1%
        slippage_exit: float = 0.001,  # 0.1%
        max_ret_abs: float = 0.20,     # ±20% 이상은 클리핑
        max_equity_limit: float = 100_000_000.0,  # 1억 넘으면 전략 비활성 (원하면 조정)
    ):
        self.ticker = ticker
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.min_ret5 = min_ret5
        self.slippage_entry = slippage_entry
        self.slippage_exit = slippage_exit
        self.max_ret_abs = max_ret_abs
        self.max_equity_limit = float(max_equity_limit)

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    def get_name(self):
        return "k200_mean_rev"

    def get_description(self):
        return (
            f"K200 Mean Reversion: {self.ticker}, "
            f"z<={self.z_entry}, ret5<={self.min_ret5:.1%}, "
            f"SL {self.stop_loss:.1%} / TP {self.take_profit:.1%}, "
            f"max {self.max_hold_days}일"
        )

    # ---------------------------
    # 내부 유틸
    # ---------------------------
    def _vwap_proxy(self, row: pd.Series) -> float:
        """일봉 기준 VWAP 근사치: (H + L + C) / 3"""
        h = float(row.get("high", np.nan))
        l = float(row.get("low", np.nan))
        c = float(row.get("close", np.nan))
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
            return np.nan
        return (h + l + c) / 3.0

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """z-score, 5일 수익률 등 계산"""
        out = df.copy()
        c = out["close"].astype(float)

        # 20일 이동평균/표준편차
        out["ma20"] = c.rolling(20).mean()
        out["std20"] = c.rolling(20).std()

        # z-score
        out["z20"] = (c - out["ma20"]) / out["std20"]
        # 5일 수익률
        out["ret5"] = c / c.shift(5) - 1.0

        return out

    # ---------------------------
    # 백테스트 본체
    # ---------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent: bool = False):
        """
        enriched: {ticker: DataFrame(OHLCV...)} 구조
        - 이 전략은 self.ticker (기본: 069500)만 사용
        - 전일(signal_date) 기준 시그널 → 익일(current_date) VWAP 진입
        
        Args:
            enriched: 종목별 데이터
            market_index: 사용하지 않음 (다른 전략과의 호환성을 위해 존재)
            weights: 사용하지 않음 (다른 전략과의 호환성을 위해 존재)
            silent: 출력 억제 여부
        """
        if not silent:
            print("\n" + "=" * 60)
            print("📈 K200 Mean Reversion 백테스트 시작")
            print("=" * 60)

        df = enriched.get(self.ticker)
        if df is None or len(df) < 60:
            if not silent:
                print(f"⚠️ {self.ticker} 데이터가 부족하거나 없습니다.")
            return pd.DataFrame(), []

        # 인디케이터 계산
        df = self._compute_indicators(df)
        df = df.dropna(subset=["ma20", "std20", "z20", "ret5"]).copy()
        if df.empty:
            if not silent:
                print("⚠️ 유효한 인디케이터 데이터가 없습니다.")
            return pd.DataFrame(), []

        dates = list(df.index)
        init_cash = 1_000_000.0
        cash = init_cash
        position = None  # {"qty", "entry_px", "entry_date"}
        equity_curve = []
        trades = []

        for i in tqdm(range(1, len(dates)), disable=silent, desc=self.get_name()):
            signal_date = dates[i - 1]
            current_date = dates[i]

            row_sig = df.loc[signal_date]
            row_cur = df.loc[current_date]

            # 현재 equity 계산
            equity = cash
            if position is not None:
                # 현재 가격은 종가 기준 (혹은 VWAP)
                cpx = float(row_cur["close"])
                if not np.isfinite(cpx) or cpx <= 0:
                    cpx = position["entry_px"]
                equity += cpx * position["qty"]

            # 자산이 너무 커지면 전략 종료 (소액 전략 보장용)
            if equity > self.max_equity_limit:
                if not silent:
                    print(
                        f"\n🎯 Equity {equity:,.0f} > limit {self.max_equity_limit:,.0f} → 전략 종료"
                    )
                break

            # -----------------------
            # 1) 기존 포지션 관리/청산
            # -----------------------
            if position is not None:
                entry_px = position["entry_px"]
                qty = position["qty"]
                hold_days = (current_date - position["entry_date"]).days

                vwap_today = self._vwap_proxy(row_cur)
                if not np.isfinite(vwap_today) or vwap_today <= 0:
                    vwap_today = float(row_cur["close"])

                # 수익률 계산 + 클리핑
                raw_ret = vwap_today / entry_px - 1.0
                if abs(raw_ret) > self.max_ret_abs:
                    if not silent:
                        print(
                            f"[WARN] {current_date.date()} ret={raw_ret:.2f} → 클리핑"
                        )
                    raw_ret = np.sign(raw_ret) * self.max_ret_abs
                    vwap_today = entry_px * (1 + raw_ret)

                z_today = float(row_cur["z20"])
                exit_reason = None

                if raw_ret <= self.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif raw_ret >= self.take_profit:
                    exit_reason = "TAKE_PROFIT"
                elif z_today >= self.z_exit:
                    exit_reason = "Z_REVERT"
                elif hold_days >= self.max_hold_days:
                    exit_reason = "MAX_HOLD"

                if exit_reason is not None:
                    exit_px = vwap_today * (1 - self.slippage_exit)
                    gross = exit_px * qty
                    cost = entry_px * qty
                    pnl = gross - cost
                    fee_out = gross * self.fee
                    tax = gross * self.tax if pnl > 0 else 0.0
                    net = gross - fee_out - tax
                    cash += net

                    trades.append(
                        {
                            "date": current_date,
                            "ticker": self.ticker,
                            "action": "SELL",
                            "entry_px": entry_px,
                            "exit_px": exit_px,
                            "qty": qty,
                            "pnl": pnl,
                            "ret": (net - cost) / cost if cost > 0 else 0.0,
                            "reason": exit_reason,
                            "hold_days": hold_days,
                        }
                    )
                    position = None

            # -----------------------
            # 2) 신규 진입 (포지션 없을 때만)
            # -----------------------
            if position is None:
                z_sig = float(row_sig["z20"])
                ret5_sig = float(row_sig["ret5"])
                # 과매도 + 단기 하락 조건
                if z_sig <= self.z_entry and ret5_sig <= self.min_ret5:
                    vwap_next = self._vwap_proxy(row_cur)
                    if not np.isfinite(vwap_next) or vwap_next <= 0:
                        vwap_next = float(row_cur["close"])
                    entry_px = vwap_next * (1 + self.slippage_entry)
                    if np.isfinite(entry_px) and entry_px > 0:
                        # 풀 포지션 (단, 수수료 고려)
                        qty = int(cash / (entry_px * (1 + self.fee)))
                        if qty > 0:
                            cost = entry_px * qty
                            fee_in = cost * self.fee
                            total_in = cost + fee_in
                            if total_in <= cash:
                                cash -= total_in
                                position = {
                                    "qty": qty,
                                    "entry_px": entry_px,
                                    "entry_date": current_date,
                                }
                                trades.append(
                                    {
                                        "date": current_date,
                                        "ticker": self.ticker,
                                        "action": "BUY",
                                        "entry_px": entry_px,
                                        "qty": qty,
                                        "pnl": 0.0,
                                        "ret": 0.0,
                                        "reason": "SIGNAL",
                                        "hold_days": 0,
                                    }
                                )

            # -----------------------
            # 3) 일별 equity 기록
            # -----------------------
            equity = cash
            if position is not None:
                cpx = float(row_cur["close"])
                if not np.isfinite(cpx) or cpx <= 0:
                    cpx = position["entry_px"]
                equity += cpx * position["qty"]

            equity_curve.append((current_date, equity))

        # ---------------------------
        # 마지막 포지션 강제 청산 (마지막 날 VWAP 기준)
        # ---------------------------
        if position is not None:
            final_date = dates[-1]
            row_fd = df.loc[final_date]
            vwap_fd = self._vwap_proxy(row_fd)
            if not np.isfinite(vwap_fd) or vwap_fd <= 0:
                vwap_fd = float(row_fd["close"])

            entry_px = position["entry_px"]
            qty = position["qty"]

            raw_ret = vwap_fd / entry_px - 1.0
            if abs(raw_ret) > self.max_ret_abs:
                if not silent:
                    print(
                        f"[WARN] FINAL {final_date.date()} ret={raw_ret:.2f} → 클리핑"
                    )
                raw_ret = np.sign(raw_ret) * self.max_ret_abs
                vwap_fd = entry_px * (1 + raw_ret)

            exit_px = vwap_fd * (1 - self.slippage_exit)
            gross = exit_px * qty
            cost = entry_px * qty
            pnl = gross - cost
            fee_out = gross * self.fee
            tax = gross * self.tax if pnl > 0 else 0.0
            net = gross - fee_out - tax
            cash += net

            hold_days = (final_date - position["entry_date"]).days
            trades.append(
                {
                    "date": final_date,
                    "ticker": self.ticker,
                    "action": "SELL",
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "qty": qty,
                    "pnl": pnl,
                    "ret": (net - cost) / cost if cost > 0 else 0.0,
                    "reason": "FORCE_END",
                    "hold_days": hold_days,
                }
            )
            position = None

            # 마지막 equity 수정
            if len(equity_curve) > 0:
                equity_curve[-1] = (equity_curve[-1][0], cash)

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(
                f"✅ K200 Mean Reversion 백테스트 완료: {len(ec)}포인트, "
                f"최종 자산: {cash:,.0f}원 (수익률: {(cash/init_cash-1)*100:.2f}%)"
            )

        return ec, trades