# strategies/kmr_midcap_reversion.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KMR (Korea Midcap Reversion) 전략

- Universe: KOSPI/KOSDAQ 중형주 + 고유동성
- Logic: 상승추세(60d up) 내 단기 과매도(3d 하락 + MA20 하단 이탈) mean-reversion
- Entry: 전일 신호 → 익일 VWAP proxy 진입 (시가 매매 금지)
- Exit: -7% 손절 / +12% 익절 / MA20 재돌파 / 7일 경과
- Capital: 100만원 기준, 최대 3종목, 종목당 최대 30%
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class KMRMidcapReversion(BaseStrategy):
    def __init__(
        self,
        stop_loss: float = -0.07,
        take_profit: float = 0.12,
        max_hold_days: int = 7,
        min_price: int = 10_000,
        max_price: int = 150_000,
        min_adv20: float = 20e8,   # 20억
        max_adv20: float = 300e8,  # 300억
        adv_participation: float = 0.10,  # ADV20의 10%까지
        max_weight_per_name: float = 0.30,
        max_holdings: int = 3,
        max_equity_limit: float = 30_000_000,  # 3천만 넘으면 전략 비활성
        slippage_entry: float = 0.002,
        slippage_exit: float = 0.002,
        max_ret_abs: float = 1.0,
        max_qty: int = 200_000,
    ):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.min_adv20 = float(min_adv20)
        self.max_adv20 = float(max_adv20)
        self.adv_participation = adv_participation
        self.max_weight_per_name = max_weight_per_name
        self.max_holdings = max_holdings
        self.max_equity_limit = float(max_equity_limit)
        self.slippage_entry = slippage_entry
        self.slippage_exit = slippage_exit
        self.max_ret_abs = max_ret_abs
        self.max_qty = max_qty

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    def get_name(self):
        return "kmr_midcap"

    def get_description(self):
        return (
            "KMR Midcap Reversion: 상승추세 중형주 단기 과매도 mean-reversion, "
            "전일 signal → 익일 VWAP 진입"
        )

    # -------------------------
    # 유틸
    # -------------------------
    def _safe_slice(self, df: pd.DataFrame, end_date, window: int) -> pd.DataFrame:
        if df is None or end_date not in df.index:
            return pd.DataFrame()
        loc = df.index.get_loc(end_date)
        if isinstance(loc, slice):
            loc = df.index.tolist().index(end_date)
        if loc + 1 < window:
            return pd.DataFrame()
        return df.iloc[loc + 1 - window : loc + 1]

    def _vwap_proxy(self, row: pd.Series) -> float:
        # 시가 매매 회피: (H+L+C)/3 기준
        h = float(row.get("high", np.nan))
        l = float(row.get("low", np.nan))
        c = float(row.get("close", np.nan))
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
            return np.nan
        return (h + l + c) / 3.0

    def _compute_adv20(self, df: pd.DataFrame, date):
        if "volume" not in df.columns or "close" not in df.columns:
            return np.nan
        sub = self._safe_slice(df, date, 20)
        if sub.empty:
            return np.nan
        tv = sub["close"] * sub["volume"]
        adv20 = tv.mean()
        return float(adv20) if np.isfinite(adv20) else np.nan

    # -------------------------
    # 시그널: 상승추세 + 단기 과매도
    # -------------------------
    def _compute_signal(self, df: pd.DataFrame, signal_date):
        if df is None or signal_date not in df.index:
            return None

        # 가격/유동성 필터
        row = df.loc[signal_date]
        c = float(row["close"])
        if not np.isfinite(c) or c <= 0:
            return None
        if c < self.min_price or c > self.max_price:
            return None

        adv20 = self._compute_adv20(df, signal_date)
        if not np.isfinite(adv20) or adv20 < self.min_adv20 or adv20 > self.max_adv20:
            return None

        # 지표 계산용 슬라이스
        hist_60 = self._safe_slice(df, signal_date, 60)
        if len(hist_60) < 60:
            return None

        close = hist_60["close"].astype(float)
        if (close <= 0).any():
            return None

        # 60일 수익률 (추세)
        c_now = close.iloc[-1]
        c_60ago = close.iloc[0]
        ret60 = c_now / c_60ago - 1.0
        long_up = ret60 > 0.10

        # MA20 / MA60
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        if not np.isfinite(ma20) or not np.isfinite(ma60):
            return None
        trend_ok = ma20 > ma60

        # 단기 과매도: 3일 수익률
        c_3ago = close.iloc[-4]  # 60-window slice에서 -4가 3일 전
        if c_3ago <= 0:
            return None
        ret3 = c_now / c_3ago - 1.0
        short_oversold = ret3 < -0.04

        # MA20 대비 이탈
        dist_ma20 = c_now / ma20 - 1.0
        dist_ok = dist_ma20 < -0.03

        if not (long_up and trend_ok and short_oversold and dist_ok):
            return None

        # 스코어: 더 과매도일수록 점수 높게 (절대값)
        score = (-ret3) + (-dist_ma20)

        return {
            "score": float(score),
            "adv20": adv20,
            "price": c_now,
        }

    # -------------------------
    # 백테스트
    # -------------------------
    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent: bool = False):
        if not silent:
            print("\n" + "="*60)
            print("📈 KMR Midcap Reversion 백테스트 시작")
            print("="*60)

        # 날짜 축 생성
        dates = sorted(
            set().union(*[df.index for df in enriched.values() if df is not None and len(df) > 0])
        )
        if len(dates) < 120:  # 60d 지표 + 버퍼
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        positions = {}  # ticker -> {qty, entry_px, entry_date}
        equity_curve = []
        trades = []

        # D-1 signal → D VWAP entry 구조
        for i in tqdm(range(1, len(dates)), disable=silent, desc=self.get_name()):
            signal_date = dates[i - 1]
            current_date = dates[i]

            # 현재 equity 계산
            equity = cash
            for tkr, pos in positions.items():
                df_t = enriched.get(tkr)
                if df_t is not None and current_date in df_t.index:
                    px = float(df_t.loc[current_date, "close"])
                    if not np.isfinite(px) or px <= 0:
                        px = pos["entry_px"]
                else:
                    px = pos["entry_px"]
                equity += px * pos["qty"]

            if equity > self.max_equity_limit:
                if not silent:
                    print(
                        f"\n🎯 Equity {equity:,.0f} > limit {self.max_equity_limit:,.0f} → 전략 종료"
                    )
                break

            # 1) 기존 포지션 관리/청산
            to_close = []
            for tkr, pos in positions.items():
                df_t = enriched.get(tkr)
                if df_t is None or current_date not in df_t.index:
                    continue
                row = df_t.loc[current_date]
                vwap_today = self._vwap_proxy(row)
                if not np.isfinite(vwap_today) or vwap_today <= 0:
                    continue

                entry_px = pos["entry_px"]
                qty = pos["qty"]
                hold_days = (current_date - pos["entry_date"]).days

                ret = vwap_today / entry_px - 1.0
                if abs(ret) > self.max_ret_abs:
                    if not silent:
                        print(f"[WARN] 이상 수익률 클리핑: {tkr} {current_date.date()} ret={ret:.2f}")
                    ret = np.sign(ret) * self.max_ret_abs
                    vwap_today = entry_px * (1 + ret)

                reason = None
                if ret <= self.stop_loss:
                    reason = "STOP_LOSS"
                elif ret >= self.take_profit:
                    reason = "TAKE_PROFIT"
                else:
                    # MA20 재돌파 체크
                    hist = df_t[df_t.index <= current_date].tail(20)
                    if len(hist) >= 20:
                        ma20_today = hist["close"].mean()
                        if np.isfinite(ma20_today) and ma20_today > 0:
                            if vwap_today >= ma20_today * 1.01:
                                reason = "MA20_REVERT"
                    # 최대 보유일
                    if reason is None and hold_days >= self.max_hold_days:
                        reason = "MAX_HOLD"

                if reason is not None:
                    exit_px = vwap_today * (1 - self.slippage_exit)
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
                    to_close.append(tkr)

            for tkr in to_close:
                del positions[tkr]

            # 2) 신규 진입 (슬롯 남았을 때만)
            slots = max(0, self.max_holdings - len(positions))
            if slots > 0:
                candidates = []
                for tkr, df in enriched.items():
                    sig = self._compute_signal(df, signal_date)
                    if sig is None:
                        continue
                    candidates.append({**sig, "ticker": tkr})

                if len(candidates) > 0:
                    df_c = pd.DataFrame(candidates).sort_values("score", ascending=False)
                    picks = df_c.head(slots)

                    for _, row_c in picks.iterrows():
                        tkr = row_c["ticker"]
                        df_t = enriched.get(tkr)
                        if df_t is None or current_date not in df_t.index:
                            continue
                        row_px = df_t.loc[current_date]
                        vwap_today = self._vwap_proxy(row_px)
                        if not np.isfinite(vwap_today) or vwap_today <= 0:
                            continue

                        entry_px = vwap_today * (1 + self.slippage_entry)
                        adv20 = float(row_c["adv20"])
                        max_notional_liq = adv20 * self.adv_participation
                        max_notional_eq = equity * self.max_weight_per_name
                        max_notional = min(max_notional_liq, max_notional_eq)
                        if max_notional <= 0:
                            continue

                        qty = int(max_notional / (entry_px * (1 + self.fee)))
                        if qty <= 0 or qty > self.max_qty:
                            continue

                        cost = entry_px * qty
                        fee_in = cost * self.fee
                        total_in = cost + fee_in
                        if total_in > cash:
                            continue

                        cash -= total_in
                        positions[tkr] = {
                            "qty": qty,
                            "entry_px": entry_px,
                            "entry_date": current_date,
                        }

            # 3) 하루 equity 기록
            equity = cash
            for tkr, pos in positions.items():
                df_t = enriched.get(tkr)
                if df_t is not None and current_date in df_t.index:
                    px = float(df_t.loc[current_date, "close"])
                    if not np.isfinite(px) or px <= 0:
                        px = pos["entry_px"]
                else:
                    px = pos["entry_px"]
                equity += px * pos["qty"]

            equity_curve.append((current_date, equity))

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        if not silent:
            print(f"✅ KMR Midcap Reversion 백테스트 완료: {len(ec)}개 포인트, 최종 자산: {cash:,.0f}원")
        return ec, trades