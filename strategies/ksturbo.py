# strategies/ksturbo.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class KSTurbo(BaseStrategy):
    """
    KSTurbo: 소액 전용 일간 단타 전략
    - 전일 장대양봉 + 거래대금 급증 + 20일 고점 돌파
    - 다음날 시가 매수 → 당일 종가 매도 (1일 보유)
    """

    def __init__(
        self,
        min_price: int = 1_000,
        max_price: int = 30_000,
        min_trade_value: float = 5e8,     # 5억
        max_trade_value: float = 100e8,   # 100억
        body_thr: float = 0.10,           # 전일 양봉 몸통 10% 이상
        vol_surge_thr: float = 3.0,       # 거래대금 3배 이상
        adv_participation: float = 0.10,  # ADV20의 10%까지만 사용
        max_trade_risk: float = 0.25,     # 계좌의 25%까지만 1회 매매에 사용
        max_equity_limit: float = 20_000_000, # 2천만원 넘으면 전략 stop (소액 전략으로 한정)
        slippage_entry: float = 0.002,    # 0.2% 진입 슬리피지
        slippage_exit: float = 0.002,     # 0.2% 청산 슬리피지
    ):
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.min_trade_value = float(min_trade_value)
        self.max_trade_value = float(max_trade_value)
        self.body_thr = body_thr
        self.vol_surge_thr = vol_surge_thr
        self.adv_participation = adv_participation
        self.max_trade_risk = max_trade_risk
        self.max_equity_limit = float(max_equity_limit)
        self.slippage_entry = slippage_entry
        self.slippage_exit = slippage_exit

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

    def get_name(self):
        return "ksturbo"

    def get_description(self):
        return "KSTurbo: 전일 장대양봉 + 거래대금 급증 + 돌파, 익일 시가 진입/당일 종가 청산"

    # -------------------------
    # 내부 유틸 & 시그널
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

    def _trigger_signal(self, df: pd.DataFrame, signal_date):
        """
        signal_date 기준으로:
        - 전일 양봉 몸통 10% 이상
        - 5일 평균 거래대금 / 20일 평균 거래대금 >= vol_surge_thr
        - 20일 고점 돌파
        """
        if df is None or signal_date not in df.index:
            return None

        if "close" not in df.columns or "open" not in df.columns or "volume" not in df.columns:
            return None

        # 가격 필터
        row = df.loc[signal_date]
        c = float(row["close"])
        o = float(row["open"])
        if not np.isfinite(c) or not np.isfinite(o) or o <= 0:
            return None
        if c < self.min_price or c > self.max_price:
            return None

        recent5 = self._safe_slice(df, signal_date, 5)
        recent20 = self._safe_slice(df, signal_date, 20)
        if recent5.empty or recent20.empty:
            return None

        # 몸통 비율
        body = (c - o) / o
        if body < self.body_thr:
            return None

        # 거래대금 급증
        tv5 = (recent5["close"] * recent5["volume"]).mean()
        tv20 = (recent20["close"] * recent20["volume"]).mean()
        if tv20 <= 0 or not np.isfinite(tv20):
            return None
        if tv5 < self.min_trade_value or tv5 > self.max_trade_value:
            return None
        vol_surge = tv5 / tv20
        if vol_surge < self.vol_surge_thr:
            return None

        # 20일 고점 돌파 여부
        if "high" in df.columns:
            high20 = float(recent20["high"].max())
        else:
            high20 = float(recent20["close"].max())
        if not np.isfinite(high20) or high20 <= 0:
            return None
        is_breakout = c >= high20 * 0.999

        if not is_breakout:
            return None

        return {
            "price": c,
            "adv20": tv20,
            "vol_surge": vol_surge,
            "body": body,
        }

    # -------------------------
    # 백테스트
    # -------------------------
    def run_backtest(self, enriched: dict, weights=None, silent: bool = False):
        if not silent:
            print("\n" + "="*60)
            print("📈 KSTurbo 백테스트 시작")
            print("="*60)

        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None and len(df) > 0]))
        if len(dates) < 40:
            return pd.DataFrame(), []

        init_cash = 1_000_000.0
        cash = init_cash
        equity_curve = []
        trades = []

        # D-1 시그널 → D 진입/청산
        for i in tqdm(range(1, len(dates)), disable=silent, desc=self.get_name()):
            signal_date = dates[i - 1]
            trade_date = dates[i]

            # 현재 equity (전략 상한 체크용)
            equity = cash
            if equity > self.max_equity_limit:
                if not silent:
                    print(f"\n🎯 Equity {equity:,.0f} > limit {self.max_equity_limit:,.0f} → 전략 종료")
                break

            # 시그널 스캔
            cands = []
            for tkr, df in enriched.items():
                sig = self._trigger_signal(df, signal_date)
                if sig is None:
                    continue
                cands.append({**sig, "ticker": tkr})

            if len(cands) > 0:
                df_c = pd.DataFrame(cands).sort_values(["vol_surge", "body"], ascending=False)
                best = df_c.iloc[0]
                tkr = best["ticker"]
                df_t = enriched[tkr]

                if trade_date in df_t.index:
                    o = float(df_t.loc[trade_date, "open"])
                    c = float(df_t.loc[trade_date, "close"])
                    if not np.isfinite(o) or not np.isfinite(c) or o <= 0:
                        # 가격 이상
                        equity_curve.append((trade_date, cash))
                        continue

                    entry_px = o * (1 + self.slippage_entry)
                    exit_px = c * (1 - self.slippage_exit)

                    adv20 = float(best["adv20"])
                    max_notional_liq = adv20 * self.adv_participation
                    max_notional_eq = cash * self.max_trade_risk
                    max_notional = min(max_notional_liq, max_notional_eq)

                    qty = int(max_notional / (entry_px * (1 + self.fee)))
                    if qty > 0:
                        # 매수
                        cost = entry_px * qty
                        fee_in = cost * self.fee
                        total_in = cost + fee_in
                        if total_in <= cash:
                            cash -= total_in

                            # 매도 (당일 종가)
                            gross = exit_px * qty
                            pnl = gross - cost
                            fee_out = gross * self.fee
                            tax = gross * self.tax if pnl > 0 else 0.0
                            net = gross - fee_out - tax
                            cash += net

                            ret = (net - total_in) / total_in

                            trades.append({
                                "date": trade_date,
                                "ticker": tkr,
                                "ret": ret,
                                "entry_px": entry_px,
                                "exit_px": exit_px,
                                "pnl": net - total_in
                            })

            # 하루 equity 기록
            equity_curve.append((trade_date, cash))

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
        if not silent:
            print(f"✅ KSTurbo 백테스트 완료: {len(ec)}개 데이터 포인트, 최종 자산: {cash:,.0f}원")
        return ec, trades