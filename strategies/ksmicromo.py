# strategies/ksmicromo.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KSmicroMo v2 - 소액 계좌용 초소형주 마이크로 모멘텀 스윙 전략 (버그 패치 & 리스크 컨트롤 강화)

핵심 아이디어:
- Universe: 상대적으로 작은 거래대금(2억 ~ 15억) 구간의 KOSDAQ/소형주 위주
- 전일 종가 기준 시그널 → 익일 VWAP 근사 가격으로 진입 (시가 매매 금지)
- 조건:
    1) 최근 5일 모멘텀 양수 (최소 +6% 이상)
    2) 최근 3일 거래대금 연속 증가
    3) 20일 고점 근처(고점 대비 -2% 이내)
    4) 가격 500 ~ 20,000원
    5) ADV20 (20일 평균 거래대금) 2억 ~ 15억
- 포지션:
    - 1종목만 보유 (소액 계좌용 집중)
    - 종목당 최대 자산 30% + ADV20의 15% 참여 한도 중 최소값
- 청산:
    - 손절: -6%
    - 익절: +18%
    - 최대 보유: 5거래일
    - + 수익 +5% 이상부터 최대 낙폭 -8% 트레일링 스탑
- 안전장치:
    - 하루/트레이드 수익률 절대값 max_ret_abs(예: ±80%)로 클리핑
    - 슬리피지: 진입/청산 각각 0.4%
    - max_qty: 100,000주 (하드캡)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class KSmicroMo(BaseStrategy):
    def __init__(
        self,
        stop_loss: float = -0.06,          # 손절 -6%
        take_profit: float = 0.18,        # 익절 +18%
        max_hold_days: int = 5,           # 최대 보유 5일
        min_price: int = 500,
        max_price: int = 20_000,
        min_adv20: float = 2e8,          # 2억
        max_adv20: float = 15e8,         # 15억
        adv_participation: float = 0.15, # ADV20의 15%까지만 참여
        max_weight_per_name: float = 0.30,   # 자산의 최대 30%
        max_equity_limit: float = 30_000_000, # 3천만원 넘으면 전략 비활성
        slippage_entry: float = 0.004,   # 0.4% 진입 슬리피지
        slippage_exit: float = 0.004,    # 0.4% 청산 슬리피지
        max_ret_abs: float = 0.80,       # 트레이드별 최대 허용 수익률 절대값 (±80%)
        max_qty: int = 100_000,          # 절대 수량 상한
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
        self.max_equity_limit = float(max_equity_limit)
        self.slippage_entry = slippage_entry
        self.slippage_exit = slippage_exit
        self.max_ret_abs = max_ret_abs
        self.max_qty = max_qty

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    # --------------------------------------------------
    # 메타 정보
    # --------------------------------------------------
    def get_name(self):
        return "ksmicromo_v2"

    def get_description(self):
        return (
            "KSmicroMo v2: 초소형주 마이크로 모멘텀 스윙 "
            "(ADV20 2~15억, 3일 연속 거래대금 증가, 5일 모멘텀, "
            "VWAP 진입/청산, 1종목 집중)"
        )

    # --------------------------------------------------
    # 내부 유틸
    # --------------------------------------------------
    def _safe_slice(self, df: pd.DataFrame, end_date, window: int) -> pd.DataFrame:
        """end_date까지 포함하여 과거 window개 row를 안전하게 슬라이스"""
        if df is None or end_date not in df.index:
            return pd.DataFrame()
        loc = df.index.get_loc(end_date)
        if isinstance(loc, slice):
            # 혹시 모를 slice 반환 케이스 방어
            loc = df.index.tolist().index(end_date)
        if loc + 1 < window:
            return pd.DataFrame()
        return df.iloc[loc + 1 - window : loc + 1]

    def _vwap_proxy(self, row: pd.Series) -> float:
        """일봉 기준 VWAP 근사치: (H + L + C) / 3"""
        h = float(row.get("high", np.nan))
        l = float(row.get("low", np.nan))
        c = float(row.get("close", np.nan))
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
            return np.nan
        return (h + l + c) / 3.0

    def _compute_adv20(self, df: pd.DataFrame, date) -> float:
        """20일 평균 거래대금(ADV20) 계산"""
        if "volume" not in df.columns or "close" not in df.columns:
            return np.nan
        sub = self._safe_slice(df, date, 20)
        if sub.empty:
            return np.nan
        tv = sub["close"].astype(float) * sub["volume"].astype(float)
        adv20 = tv.mean()
        return float(adv20) if np.isfinite(adv20) else np.nan

    # --------------------------------------------------
    # 시그널 계산 (전일 종가 기준)
    # --------------------------------------------------
    def _compute_signal(self, df: pd.DataFrame, signal_date):
        """
        signal_date(전일 종가 기준) 시그널:
        - 가격 필터: 500 ~ 20,000원
        - ADV20: 2억 ~ 15억
        - 최근 3일 거래대금 연속 증가
        - 5일 모멘텀 >= +6%
        - 20일 고점 대비 -2% 이내
        """
        if df is None or signal_date not in df.index:
            return None
        if "close" not in df.columns or "volume" not in df.columns:
            return None

        row = df.loc[signal_date]
        c = float(row["close"])
        if not np.isfinite(c) or c <= 0:
            return None
        if c < self.min_price or c > self.max_price:
            return None

        adv20 = self._compute_adv20(df, signal_date)
        if not np.isfinite(adv20) or adv20 < self.min_adv20 or adv20 > self.max_adv20:
            return None

        # 20일 구간
        hist20 = self._safe_slice(df, signal_date, 20)
        if len(hist20) < 20:
            return None

        close20 = hist20["close"].astype(float)
        vol20 = hist20["volume"].astype(float)
        tv20 = close20 * vol20

        # 최근 5일
        hist5 = hist20.tail(5)
        if len(hist5) < 5:
            return None
        close5 = hist5["close"].astype(float)
        tv5 = close5 * hist5["volume"].astype(float)

        # 3일 연속 거래대금 증가
        tv3 = tv5.tail(3)
        if len(tv3) < 3:
            return None
        if not (tv3.iloc[-1] > tv3.iloc[-2] > tv3.iloc[-3]):
            return None

        # 5일 모멘텀
        c_now = float(close5.iloc[-1])
        c_5ago = float(close5.iloc[0])
        if c_5ago <= 0 or not np.isfinite(c_5ago):
            return None
        ret5 = c_now / c_5ago - 1.0
        if ret5 < 0.06:  # 최소 +6%
            return None

        # 20일 고점 대비 -2% 이내
        if "high" in hist20.columns:
            high20 = float(hist20["high"].max())
        else:
            high20 = float(close20.max())
        if not np.isfinite(high20) or high20 <= 0:
            return None
        if c_now < high20 * 0.98:
            return None

        # 스코어: 모멘텀 + 거래대금
        score = ret5 + (tv3.iloc[-1] / (tv20.mean() + 1e-9)) * 0.1

        return {
            "score": float(score),
            "adv20": float(adv20),
            "ret5": float(ret5),
            "price": float(c_now),
        }

    # --------------------------------------------------
    # 백테스트
    # --------------------------------------------------
    def run_backtest(self, enriched: dict, weights=None, silent: bool = False):
        """
        enriched: {ticker: DataFrame(OHLCV ...)} 구조
        - signal_date = D-1, trade_date = D
        - 1포지션만 운영
        """
        if not silent:
            print("\n" + "="*60)
            print("📈 KSmicroMo v2 백테스트 시작")
            print("="*60)

        # 날짜 축
        dates = sorted(
            set().union(
                *[df.index for df in enriched.values() if df is not None and len(df) > 0]
            )
        )
        if len(dates) < 60:
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        position = None  # {"ticker", "qty", "entry_px", "entry_date", "max_price"}
        equity_curve = []
        trades = []

        for i in tqdm(range(1, len(dates)), disable=silent, desc=self.get_name()):
            signal_date = dates[i - 1]
            current_date = dates[i]

            # 현재 equity 계산
            equity = cash
            if position is not None:
                tkr = position["ticker"]
                df_t = enriched.get(tkr)
                if df_t is not None and current_date in df_t.index:
                    cpx = float(df_t.loc[current_date, "close"])
                    if not np.isfinite(cpx) or cpx <= 0:
                        cpx = position["entry_px"]
                else:
                    cpx = position["entry_px"]
                equity += cpx * position["qty"]

            # 일정 자산 이상이면 전략 종료 (소액 전략 강제)
            if equity > self.max_equity_limit:
                if not silent:
                    print(
                        f"\n🎯 Equity {equity:,.0f} > limit {self.max_equity_limit:,.0f} → 전략 종료"
                    )
                break

            # 1) 기존 포지션 관리 (청산)
            if position is not None:
                tkr = position["ticker"]
                df_t = enriched.get(tkr)
                if df_t is not None and current_date in df_t.index:
                    row = df_t.loc[current_date]
                    vwap_today = self._vwap_proxy(row)
                    if not np.isfinite(vwap_today) or vwap_today <= 0:
                        vwap_today = float(row["close"])
                    entry_px = position["entry_px"]
                    qty = position["qty"]
                    hold_days = (current_date - position["entry_date"]).days

                    # 수익률 계산 + 클리핑
                    ret = vwap_today / entry_px - 1.0
                    if abs(ret) > self.max_ret_abs:
                        if not silent:
                            print(
                                f"[WARN] {tkr} {current_date.date()} ret={ret:.2f} → 클리핑"
                            )
                        ret = np.sign(ret) * self.max_ret_abs
                        vwap_today = entry_px * (1 + ret)

                    # max_price (트레일링용)
                    position["max_price"] = max(position.get("max_price", entry_px), vwap_today)
                    max_px = position["max_price"]
                    trail_ret = vwap_today / max_px - 1.0

                    # 청산 조건
                    exit_reason = None
                    if ret <= self.stop_loss:
                        exit_reason = "STOP_LOSS"
                    elif ret >= self.take_profit:
                        exit_reason = "TAKE_PROFIT"
                    elif hold_days >= self.max_hold_days:
                        exit_reason = "MAX_HOLD"
                    elif ret > 0.05 and trail_ret <= -0.08:
                        # +5% 이상 수익 구간에서 고점 대비 -8% 이상 되돌림
                        exit_reason = "TRAILING_STOP"

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
                                "ticker": tkr,
                                "action": "SELL",
                                "entry_px": entry_px,
                                "exit_px": exit_px,
                                "qty": qty,
                                "pnl": pnl,
                                "ret": (net - cost) / cost,
                                "reason": exit_reason,
                                "hold_days": hold_days,
                            }
                        )
                        position = None

            # 2) 신규 진입 (포지션 없을 때만)
            if position is None:
                candidates = []
                for tkr, df in enriched.items():
                    sig = self._compute_signal(df, signal_date)
                    if sig is None:
                        continue
                    candidates.append({**sig, "ticker": tkr})

                if len(candidates) > 0:
                    df_c = pd.DataFrame(candidates).sort_values("score", ascending=False)
                    best = df_c.iloc[0]
                    tkr = best["ticker"]
                    df_t = enriched.get(tkr)
                    if df_t is not None and current_date in df_t.index:
                        row_td = df_t.loc[current_date]
                        vwap_td = self._vwap_proxy(row_td)
                        if not np.isfinite(vwap_td) or vwap_td <= 0:
                            vwap_td = float(row_td["close"])
                        entry_px = vwap_td * (1 + self.slippage_entry)
                        if not np.isfinite(entry_px) or entry_px <= 0:
                            # 가격이 이상하면 스킵
                            pass
                        else:
                            adv20 = float(best["adv20"])
                            max_notional_liq = adv20 * self.adv_participation
                            max_notional_eq = equity * self.max_weight_per_name
                            max_notional = min(max_notional_liq, max_notional_eq)
                            if max_notional > 0:
                                qty = int(max_notional / (entry_px * (1 + self.fee)))
                                if 0 < qty <= self.max_qty:
                                    cost = entry_px * qty
                                    fee_in = cost * self.fee
                                    total_in = cost + fee_in
                                    if total_in <= cash:
                                        cash -= total_in
                                        position = {
                                            "ticker": tkr,
                                            "qty": qty,
                                            "entry_px": entry_px,
                                            "entry_date": current_date,
                                            "max_price": entry_px,
                                        }
                                        trades.append(
                                            {
                                                "date": current_date,
                                                "ticker": tkr,
                                                "action": "BUY",
                                                "entry_px": entry_px,
                                                "qty": qty,
                                                "pnl": 0.0,
                                                "ret": 0.0,
                                                "reason": "SIGNAL",
                                                "hold_days": 0,
                                            }
                                        )

            # 3) 일별 equity 기록
            equity = cash
            if position is not None:
                tkr = position["ticker"]
                df_t = enriched.get(tkr)
                if df_t is not None and current_date in df_t.index:
                    cpx = float(df_t.loc[current_date, "close"])
                    if not np.isfinite(cpx) or cpx <= 0:
                        cpx = position["entry_px"]
                else:
                    cpx = position["entry_px"]
                equity += cpx * position["qty"]

            equity_curve.append((current_date, equity))

        # 마지막 포지션 강제 청산 (마지막 날짜 VWAP)
        if position is not None:
            tkr = position["ticker"]
            df_t = enriched.get(tkr)
            final_date = dates[min(len(dates) - 1, len(equity_curve) - 1)]
            if df_t is not None and final_date in df_t.index:
                row_fd = df_t.loc[final_date]
                vwap_fd = self._vwap_proxy(row_fd)
                if not np.isfinite(vwap_fd) or vwap_fd <= 0:
                    vwap_fd = float(row_fd["close"])
                entry_px = position["entry_px"]
                qty = position["qty"]
                ret = vwap_fd / entry_px - 1.0
                if abs(ret) > self.max_ret_abs:
                    if not silent:
                        print(
                            f"[WARN] FINAL {tkr} {final_date.date()} ret={ret:.2f} → 클리핑"
                        )
                    ret = np.sign(ret) * self.max_ret_abs
                    vwap_fd = entry_px * (1 + ret)

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
                        "ticker": tkr,
                        "action": "SELL",
                        "entry_px": entry_px,
                        "exit_px": exit_px,
                        "qty": qty,
                        "pnl": pnl,
                        "ret": (net - cost) / cost,
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
                f"✅ KSmicroMo v2 백테스트 완료: {len(ec)}개 포인트, "
                f"최종 자산: {cash:,.0f}원 (수익률: {(cash/init_cash-1)*100:.2f}%)"
            )

        return ec, trades