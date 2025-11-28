# strategies/kqm_small_cap_v3.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import FEE_PER_SIDE, TAX_RATE_SELL


class KQMSmallCapStrategyV3(BaseStrategy):
    """
    K-Quality Momentum Small Cap v3.0

    - 100만원 소액 계좌용
    - 팩터 기반 + 리스크 관리(손절/익절/트레일링) 결합
    - 2개 종목 집중 투자, 20일 리밸런싱
    """

    def __init__(
        self,
        rebal_days: int = 20,
        n_stocks: int = 2,
        max_price: int = 50_000,
        min_price: int = 2_000,
        min_vol20: float = 5e8,   # 20일 평균 거래대금 5억 이상
        min_vol5: float = 3e8,    # 5일 평균 거래대금 3억 이상
        slippage: float = 0.001,  # 0.1%
        stop_loss: float = -0.12,  # -12%
        take_profit: float = 0.20, # +20%
        trailing_trigger: float = 0.10,  # +10% 이상 구간에서
        trailing_step: float = -0.07,    # 고점 대비 -7% 하락 시 청산
        max_hold_days: int = 40
    ):
        self.rebal_days = rebal_days
        self.n_stocks = n_stocks
        self.max_price = max_price
        self.min_price = min_price
        self.min_vol20 = min_vol20
        self.min_vol5 = min_vol5

        # 리스크 파라미터
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_trigger = trailing_trigger
        self.trailing_step = trailing_step
        self.max_hold_days = max_hold_days

        # 비용
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = slippage

        # 팩터 가중치
        self.factor_weights = {
            "mom3": 0.30,
            "mom1": 0.45,
            "quality": 0.15,
            "inv_vol": 0.10,
        }

    # ----------------------------------------------------
    # Helper: 이름/설명
    # ----------------------------------------------------
    def get_name(self) -> str:
        return "kqm_small_cap_v3"

    def get_description(self) -> str:
        return (
            f"KQM Small Cap v3 (100만, {self.n_stocks} stocks, "
            f"rebal {self.rebal_days}d, SL {self.stop_loss:.0%}, "
            f"TP {self.take_profit:.0%}, max_hold {self.max_hold_days}d)"
        )

    # ----------------------------------------------------
    # Helper: VWAP 근사치
    # ----------------------------------------------------
    @staticmethod
    def _vwap_proxy(row: pd.Series) -> float:
        h = float(row.get("high", np.nan))
        l = float(row.get("low", np.nan))
        c = float(row.get("close", np.nan))
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
            return np.nan
        return (h + l + c) / 3.0

    # ----------------------------------------------------
    # Helper: date 기준 종가 fallback (NaN 방지)
    # ----------------------------------------------------
    @staticmethod
    def _get_price(df: pd.DataFrame, date: pd.Timestamp) -> float:
        if df is None or len(df) == 0:
            return np.nan
        if date in df.index:
            px = df.loc[date, "close"]
        else:
            valid = df.index[df.index <= date]
            if len(valid) == 0:
                return np.nan
            px = df.loc[valid.max(), "close"]
        return float(px)

    # ----------------------------------------------------
    # 팩터 계산
    # ----------------------------------------------------
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp):
        if df is None or current_date not in df.index:
            return None

        subset = df[df.index <= current_date]
        if len(subset) < 60:
            return None

        close = subset["close"].values
        price = float(close[-1])

        # 가격 필터
        if price < self.min_price or price > self.max_price:
            return None

        # 유동성 필터
        if "volume" in subset.columns:
            trade_val = subset["close"] * subset["volume"]
            vol20 = trade_val.tail(20).mean()
            vol5 = trade_val.tail(5).mean()
            if not np.isfinite(vol20) or not np.isfinite(vol5):
                return None
            if vol20 < self.min_vol20 or vol5 < self.min_vol5:
                return None

        # 모멘텀
        if len(close) < 60:
            return None
        mom3 = close[-1] / close[-60] - 1.0   # 3개월
        mom1 = close[-1] / close[-20] - 1.0   # 1개월

        # Quality
        ret60 = pd.Series(close[-60:]).pct_change().dropna()
        if len(ret60) < 10:
            return None
        quality = ret60.mean() / (ret60.std() + 1e-9)

        # Inverse volatility
        vol20 = pd.Series(close[-20:]).pct_change().ewm(halflife=10).std().iloc[-1]
        inv_vol = 1.0 / (vol20 + 1e-9)

        return {
            "mom3": mom3,
            "mom1": mom1,
            "quality": quality,
            "inv_vol": inv_vol,
            "price": price,
        }

    # ----------------------------------------------------
    # Equity 계산 (NaN 방어)
    # ----------------------------------------------------
    def _calc_equity(self, cash, positions, enriched, date):
        total = cash
        for t, pos in positions.items():
            df = enriched.get(t)
            if df is None:
                continue
            px = self._get_price(df, date)
            if not np.isfinite(px) or px <= 0:
                px = pos["entry_px"]
            total += px * pos["qty"]
        return total

    # ----------------------------------------------------
    # 메인 백테스트
    # ----------------------------------------------------
    def run_backtest(self, enriched: dict, market_index: pd.Series = None, weights: dict = None, silent: bool = False):
        # 전체 거래일
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 120:
            return pd.DataFrame(), []

        dates = pd.to_datetime(pd.Index(dates)).sort_values()

        # 🔥 레짐 필터 준비 (KOSDAQ 인덱스가 넘어온 경우)
        regime = None
        if market_index is not None:
            idx = market_index.sort_index().reindex(dates).ffill()

            # 200일 이동평균 & 100일 모멘텀
            idx_ma200 = idx.rolling(200, min_periods=200).mean()
            idx_mom100 = idx / idx.shift(100) - 1.0

            # bull regime 정의
            regime = (idx > idx_ma200) & (idx_mom100 > 0)
            
        if not silent:
            print("\n" + "=" * 60)
            print("📈 KQM Small Cap v3.0 백테스트 시작...")
            print("=" * 60)

        # 전체 거래일
        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 120:
            return pd.DataFrame(), []

        # 리밸런싱 날짜 (120일 워밍업 이후)
        rebalance_dates = set(dates[120::self.rebal_days])

        init_cash = 1_000_000.0
        cash = init_cash
        positions = {}  # ticker -> {qty, entry_px, entry_date, high_px}
        equity_curve = []
        trade_log = []

        # 일별 루프
        for current_date in tqdm(dates, desc="KQM v3", disable=silent):

            # 1) 기존 포지션 리스크 관리 (SL/TP/트레일링/최대 보유일)
            to_close = []
            for t, pos in positions.items():
                df_t = enriched.get(t)
                if df_t is None:
                    continue

                price_today = self._get_price(df_t, current_date)
                if not np.isfinite(price_today) or price_today <= 0:
                    continue

                entry_px = pos["entry_px"]
                qty = pos["qty"]
                hold_days = (current_date - pos["entry_date"]).days

                # VWAP 기준 exit price 추정
                row_cur = df_t.loc[df_t.index[df_t.index <= current_date].max()]
                vwap_today = self._vwap_proxy(row_cur)
                if not np.isfinite(vwap_today) or vwap_today <= 0:
                    vwap_today = price_today

                ret = vwap_today / entry_px - 1.0

                # high_px 업데이트
                pos["high_px"] = max(pos.get("high_px", entry_px), vwap_today)

                exit_reason = None

                # 손절
                if ret <= self.stop_loss:
                    exit_reason = "STOP_LOSS"
                # 익절
                elif ret >= self.take_profit:
                    exit_reason = "TAKE_PROFIT"
                # 트레일링
                elif ret >= self.trailing_trigger:
                    dd_from_high = vwap_today / pos["high_px"] - 1.0
                    if dd_from_high <= self.trailing_step:
                        exit_reason = "TRAILING"
                # 최대 보유일
                elif hold_days >= self.max_hold_days:
                    exit_reason = "MAX_HOLD"

                if exit_reason is not None:
                    exit_px = vwap_today * (1 - self.slippage)
                    proceeds = exit_px * qty
                    cost = entry_px * qty
                    pnl = proceeds - cost

                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0.0
                    net = proceeds - fee_out - tax

                    cash += net
                    to_close.append(t)

                    trade_log.append({
                        "date": current_date,
                        "ticker": t,
                        "action": "SELL",
                        "price": exit_px,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "cash_after": cash,
                        "hold_days": hold_days,
                    })

            for t in to_close:
                positions.pop(t, None)

            # 2) 리밸런싱 날짜면 팩터 기반 포트폴리오 재구성
            if current_date in rebalance_dates:

                # 2-1) 팩터 스냅샷
                rows = []
                for t, df_t in enriched.items():
                    fac = self._compute_factors(df_t, current_date)
                    if fac is None:
                        continue
                    rows.append({"ticker": t, **fac})

                if len(rows) > 0:
                    day = pd.DataFrame(rows)

                    # 팩터 랭킹
                    for f in ["mom3", "mom1", "quality", "inv_vol"]:
                        day[f"{f}_rank"] = day[f].rank(pct=True)

                    W = self.factor_weights
                    day["score"] = (
                        W["mom3"] * day["mom3_rank"] +
                        W["mom1"] * day["mom1_rank"] +
                        W["quality"] * day["quality_rank"] +
                        W["inv_vol"] * day["inv_vol_rank"]
                    )

                    day_sorted = day.sort_values("score", ascending=False)
                    selected = day_sorted.head(self.n_stocks)["ticker"].tolist()

                    # 2-2) 리밸런싱 시점에서 탈락 종목 일부 정리
                    for t in list(positions.keys()):
                        if t not in selected:
                            pos = positions.pop(t)
                            df_t = enriched.get(t)
                            if df_t is None:
                                continue
                            px = self._get_price(df_t, current_date)
                            if not np.isfinite(px) or px <= 0:
                                px = pos["entry_px"]
                            exit_px = px * (1 - self.slippage)
                            qty = pos["qty"]
                            proceeds = exit_px * qty
                            cost = pos["entry_px"] * qty
                            pnl = proceeds - cost

                            fee_out = proceeds * self.fee
                            tax = proceeds * self.tax if pnl > 0 else 0.0
                            net = proceeds - fee_out - tax
                            cash += net

                            trade_log.append({
                                "date": current_date,
                                "ticker": t,
                                "action": "SELL",
                                "price": exit_px,
                                "qty": qty,
                                "pnl": pnl,
                                "reason": "REBAL_DROP",
                                "cash_after": cash,
                            })

                    # 2-3) 새 포트폴리오로 비중 조절 + 신규 진입
                    if len(selected) > 0:
                        equity_val = self._calc_equity(cash, positions, enriched, current_date)
                        target_val = equity_val / len(selected)

                        for t in selected:
                            df_t = enriched.get(t)
                            if df_t is None:
                                continue

                            # 🔥 레짐 필터: bull이 아닐 때는 신규 진입/증액 스킵
                            if regime is not None and len(regime) > 0 and current_date in regime.index:
                                regime_value = regime.loc[current_date]
                                # Series인 경우 첫 번째 값 추출
                                if isinstance(regime_value, pd.Series):
                                    if len(regime_value) > 0:
                                        regime_value = regime_value.iloc[0]
                                    else:
                                        regime_value = True  # 데이터 없으면 진입 허용
                                # Scalar 값이면 그대로 사용
                                if not bool(regime_value):
                                    # 기존 포지션 비중 줄이거나 손절 규칙은 위에서 이미 처리됨
                                    continue

                            px_close = self._get_price(df_t, current_date)
                            if not np.isfinite(px_close) or px_close <= 0:
                                continue

                            # Entry timing 필터: close > SMA5 & > VWAP
                            subset = df_t[df_t.index <= current_date]
                            if len(subset) < 5:
                                continue
                            sma5 = subset["close"].tail(5).mean()

                            row_cur = subset.iloc[-1]
                            vwap_today = self._vwap_proxy(row_cur)
                            if not np.isfinite(vwap_today) or vwap_today <= 0:
                                vwap_today = px_close

                            if not (px_close > sma5 and px_close > vwap_today):
                                # 타이밍 안 좋으면 신규 진입/증액 스킵
                                continue

                            entry_px = vwap_today * (1 + self.slippage)
                            if not np.isfinite(entry_px) or entry_px <= 0:
                                continue

                            target_qty = int(target_val / entry_px)
                            if target_qty <= 0:
                                continue

                            cur_qty = positions.get(t, {}).get("qty", 0)
                            delta = target_qty - cur_qty

                            # BUY
                            if delta > 0:
                                cost = entry_px * delta
                                fee_in = cost * self.fee
                                total_cost = cost + fee_in

                                if total_cost > cash:
                                    continue  # 소액 계좌: 오버매수 금지

                                cash -= total_cost
                                if t in positions:
                                    old = positions[t]
                                    old_q = old["qty"]
                                    old_px = old["entry_px"]
                                    new_q = old_q + delta
                                    new_px = (old_px * old_q + entry_px * delta) / new_q
                                    positions[t] = {
                                        "qty": new_q,
                                        "entry_px": new_px,
                                        "entry_date": old["entry_date"],
                                        "high_px": max(old.get("high_px", new_px), new_px),
                                    }
                                else:
                                    positions[t] = {
                                        "qty": delta,
                                        "entry_px": entry_px,
                                        "entry_date": current_date,
                                        "high_px": entry_px,
                                    }

                                trade_log.append({
                                    "date": current_date,
                                    "ticker": t,
                                    "action": "BUY",
                                    "price": entry_px,
                                    "qty": delta,
                                    "pnl": 0.0,
                                    "reason": "REBAL_BUY",
                                    "cash_after": cash,
                                })

                            # SELL (비중 축소)
                            elif delta < 0 and t in positions:
                                sell_qty = -delta
                                pos = positions[t]
                                exit_px = vwap_today * (1 - self.slippage)
                                proceeds = exit_px * sell_qty
                                cost = pos["entry_px"] * sell_qty
                                pnl = proceeds - cost

                                fee_out = proceeds * self.fee
                                tax = proceeds * self.tax if pnl > 0 else 0.0
                                net = proceeds - fee_out - tax
                                cash += net

                                pos["qty"] -= sell_qty
                                if pos["qty"] <= 0:
                                    positions.pop(t)

                                trade_log.append({
                                    "date": current_date,
                                    "ticker": t,
                                    "action": "SELL",
                                    "price": exit_px,
                                    "qty": sell_qty,
                                    "pnl": pnl,
                                    "reason": "REBAL_TRIM",
                                    "cash_after": cash,
                                })

            # 3) 일별 equity 기록
            equity = self._calc_equity(cash, positions, enriched, current_date)
            equity_curve.append((current_date, equity))

        # 4) 마지막 날 강제 청산 (포지션 남으면)
        final_date = dates[-1]
        for t, pos in list(positions.items()):
            df_t = enriched.get(t)
            if df_t is None:
                continue
            px = self._get_price(df_t, final_date)
            if not np.isfinite(px) or px <= 0:
                px = pos["entry_px"]
            exit_px = px * (1 - self.slippage)
            qty = pos["qty"]
            proceeds = exit_px * qty
            cost = pos["entry_px"] * qty
            pnl = proceeds - cost
            fee_out = proceeds * self.fee
            tax = proceeds * self.tax if pnl > 0 else 0.0
            net = proceeds - fee_out - tax
            cash += net

            trade_log.append({
                "date": final_date,
                "ticker": t,
                "action": "SELL",
                "price": exit_px,
                "qty": qty,
                "pnl": pnl,
                "reason": "FORCE_END",
                "cash_after": cash,
            })

        # 마지막 equity를 cash로 고정
        if len(equity_curve) == 0 or equity_curve[-1][0] != final_date:
            equity_curve.append((final_date, cash))
        else:
            equity_curve[-1] = (final_date, cash)

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(
                f"✅ KQM Small Cap v3.0 백테스트 완료: "
                f"{len(ec)} 포인트, 최종 자산: {cash:,.0f}원 "
                f"(수익률 {(cash/init_cash-1)*100:.2f}%)"
            )

        return ec, trade_log