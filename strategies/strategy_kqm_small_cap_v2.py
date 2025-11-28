#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMSmallCapStrategyV21(BaseStrategy):
    """
    K-Quality Momentum Small Cap v2.1
    - 100만원 소액 계좌 최적화
    - 3개 종목 equal-weight 기반
    - 데이터 누락/NaN/체결불가 방지
    - 팩터 개선 (3M/1M 모멘텀 기반)
    """

    # ---------------------------------------------------------
    # 초기 세팅
    # ---------------------------------------------------------
    def __init__(self,
                 rebal_days=10,
                 n_stocks=3,
                 max_price=50000,
                 min_price=2000,
                 min_vol20=5e8,   # 20일 평균 거래대금 5억 이상
                 min_vol5=3e8,    # 5일 평균 거래대금 3억 이상
                 slippage=0.001):

        self.rebal_days = rebal_days
        self.n_stocks = n_stocks
        self.max_price = max_price
        self.min_price = min_price
        self.min_vol20 = min_vol20
        self.min_vol5 = min_vol5

        # 거래 비용
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = slippage

        # 개선된 팩터 가중치
        self.factor_weights = {
            "mom3": 0.45,
            "mom1": 0.25,
            "quality": 0.20,
            "inv_vol": 0.10,
        }

    def get_name(self):
        return "kqm_small_cap_v21"

    def get_description(self):
        return f"KQM Small Cap v2.1 (100만원, {self.n_stocks}stocks, {self.min_price:,}~{self.max_price:,}원, {self.rebal_days}d rebal)"

    # ---------------------------------------------------------
    # 팩터 계산 (개선 버전)
    # ---------------------------------------------------------
    def _compute_factors(self, df, current_date):
        if df is None or current_date not in df.index:
            return None

        subset = df[df.index <= current_date]
        if len(subset) < 60:
            return None

        close = subset["close"].values

        # Price filter
        price = close[-1]
        if price < self.min_price or price > self.max_price:
            return None

        # Volume filters
        if "volume" in df.columns:
            vol20 = (subset["close"] * subset["volume"]).tail(20).mean()
            vol5 = (subset["close"] * subset["volume"]).tail(5).mean()
            if vol20 < self.min_vol20 or vol5 < self.min_vol5:
                return None

        # Momentum factors
        if len(close) < 60:
            return None
        mom3 = close[-1] / close[-60] - 1
        mom1 = close[-1] / close[-20] - 1

        # Quality: 안정적 수익률
        ret60 = pd.Series(close[-60:]).pct_change().dropna()
        if len(ret60) < 10: return None
        quality = ret60.mean() / (ret60.std() + 1e-9)

        # Low volatility (inverse)
        vol20 = pd.Series(close[-20:]).pct_change().ewm(halflife=10).std().iloc[-1]
        inv_vol = 1 / (vol20 + 1e-9)

        return {
            "mom3": mom3,
            "mom1": mom1,
            "quality": quality,
            "inv_vol": inv_vol,
            "price": price,
        }

    # ---------------------------------------------------------
    # Equity 계산 (NaN 보정 포함)
    # ---------------------------------------------------------
    def _calc_equity(self, cash, positions, enriched, date):
        total = cash
        for t, pos in positions.items():
            df = enriched.get(t)
            if df is None:
                continue

            # price fallback logic
            if date in df.index:
                px = df.loc[date, "close"]
            else:
                valid = df.index[df.index <= date]
                if len(valid) == 0:
                    continue
                px = df.loc[valid.max(), "close"]

            if not np.isfinite(px) or px <= 0:
                px = pos["entry_px"]

            total += px * pos["qty"]
        return total

    # ---------------------------------------------------------
    # 백테스트 메인
    # ---------------------------------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):

        if not silent:
            print("\n===========================================================")
            print("📈 KQM Small Cap v2.1 백테스트 시작...")
            print("===========================================================\n")

        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 120:
            return pd.DataFrame(), []

        rebalance_dates = dates[120::self.rebal_days]

        init_cash = 1_000_000
        cash = init_cash
        positions = {}
        equity_curve = []
        trade_log = []

        for idx in tqdm(range(len(rebalance_dates)), desc="KQM Small v2.1", disable=silent):

            d0 = rebalance_dates[idx]

            # Next rebalance date
            next_rebal = rebalance_dates[idx+1] if idx < len(rebalance_dates)-1 else dates[-1]

            # --------------------------
            # Factor snapshot
            # --------------------------
            rows = []
            for ticker, df in enriched.items():
                fac = self._compute_factors(df, d0)
                if fac is None:
                    continue
                rows.append({"ticker": ticker, **fac})

            if len(rows) == 0:
                equity_curve.append((d0, self._calc_equity(cash, positions, enriched, d0)))
                continue

            day = pd.DataFrame(rows)

            # Rankings
            for f in ["mom3", "mom1", "quality", "inv_vol"]:
                day[f"{f}_rank"] = day[f].rank(pct=True)

            # Score
            W = self.factor_weights
            day["score"] = (
                W["mom3"] * day["mom3_rank"] +
                W["mom1"] * day["mom1_rank"] +
                W["quality"] * day["quality_rank"] +
                W["inv_vol"] * day["inv_vol_rank"]
            )

            # --------------------------
            # Top N selection
            # --------------------------
            selected = day.sort_values("score", ascending=False).head(self.n_stocks)["ticker"].tolist()

            # --------------------------
            # Sell (dropouts)
            # --------------------------
            for t in list(positions.keys()):
                if t not in selected:
                    pos = positions.pop(t)
                    df = enriched[t]

                    # price fallback
                    if d0 in df.index:
                        px = df.loc[d0, "close"]
                    else:
                        valid = df.index[df.index <= d0]
                        if len(valid) == 0:
                            continue
                        px = df.loc[valid.max(), "close"]

                    if not np.isfinite(px) or px <= 0:
                        px = pos["entry_px"]

                    exit_px = px * (1 - self.slippage)
                    qty = pos["qty"]
                    proceeds = exit_px * qty
                    cost = pos["entry_px"] * qty
                    pnl = proceeds - cost

                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0

                    cash += proceeds - fee_out - tax

                    trade_log.append({
                        "date": d0, "ticker": t, "action": "SELL",
                        "price": exit_px, "qty": qty, "pnl": pnl, "cash_after": cash
                    })

            # --------------------------
            # Adjust / Buy
            # --------------------------
            equity_val = self._calc_equity(cash, positions, enriched, d0)
            target_val = equity_val / len(selected)

            for t in selected:
                df = enriched[t]
                if df is None:
                    continue

                # price fallback
                if d0 in df.index:
                    px = df.loc[d0, "close"]
                else:
                    valid = df.index[df.index <= d0]
                    if len(valid) == 0:
                        continue
                    px = df.loc[valid.max(), "close"]

                if not np.isfinite(px) or px <= 0:
                    continue

                entry_px = px * (1 + self.slippage)
                target_qty = int((target_val / entry_px))

                if target_qty <= 0:
                    continue

                cur_qty = positions.get(t, {}).get("qty", 0)
                delta = target_qty - cur_qty

                # BUY
                if delta > 0:
                    cost = entry_px * delta
                    fee_in = cost * self.fee
                    total_cost = cost + fee_in

                    # 소액 계좌 → 무리한 매수 금지
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    if t in positions:
                        old_q = positions[t]["qty"]
                        old_px = positions[t]["entry_px"]
                        new_q = old_q + delta
                        new_px = (old_px * old_q + entry_px * delta) / new_q
                        positions[t] = {"qty": new_q, "entry_px": new_px}
                    else:
                        positions[t] = {"qty": delta, "entry_px": entry_px}

                    trade_log.append({
                        "date": d0, "ticker": t, "action": "BUY",
                        "price": entry_px, "qty": delta, "pnl": 0, "cash_after": cash
                    })

                # SELL (비중 축소)
                elif delta < 0:
                    sell_qty = -delta
                    pos = positions[t]

                    exit_px = px * (1 - self.slippage)
                    proceeds = exit_px * sell_qty
                    pnl = proceeds - (pos["entry_px"] * sell_qty)

                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0

                    cash += proceeds - fee_out - tax
                    pos["qty"] -= sell_qty

                    if pos["qty"] <= 0:
                        positions.pop(t)

                    trade_log.append({
                        "date": d0, "ticker": t, "action": "SELL",
                        "price": exit_px, "qty": sell_qty, "pnl": pnl, "cash_after": cash
                    })

            # --------------------------
            # Equity record
            # --------------------------
            for dt in [d for d in dates if d0 <= d < next_rebal]:
                equity_curve.append((dt, self._calc_equity(cash, positions, enriched, dt)))

        # --------------------------
        # Final day cleanup
        # --------------------------
        final_date = dates[-1]

        for t in list(positions.keys()):
            df = enriched[t]
            pos = positions.pop(t)

            # fallback logic
            if final_date in df.index:
                px = df.loc[final_date, "close"]
            else:
                valid = df.index[df.index <= final_date]
                if len(valid) == 0:
                    continue
                px = df.loc[valid.max(), "close"]

            if not np.isfinite(px) or px <= 0:
                px = pos["entry_px"]

            exit_px = px * (1 - self.slippage)
            qty = pos["qty"]
            proceeds = exit_px * qty
            pnl = proceeds - (pos["entry_px"] * qty)
            fee_out = proceeds * self.fee
            tax = proceeds * self.tax if pnl > 0 else 0

            cash += proceeds - fee_out - tax

            trade_log.append({
                "date": final_date, "ticker": t, "action": "SELL",
                "price": exit_px, "qty": qty, "pnl": pnl, "cash_after": cash
            })

        equity_curve.append((final_date, cash))

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(f"✅ KQM Small Cap v2.1 백테스트 완료! 최종 자산: {cash:,.0f}원")

        return ec, trade_log