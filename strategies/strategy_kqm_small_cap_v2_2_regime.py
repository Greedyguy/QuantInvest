#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMSmallCapStrategyV22Regime(BaseStrategy):
    """
    K-Quality Momentum Small Cap v2.2 (Regime-aware)
    - KOSDAQ 레짐(추세) 기반으로 익스포저 조절
    - target_exposure를 실제 비중 계산에 반영
    - 점수 개선폭이 작으면 기존 보유 종목을 유지(최소 교체 임계치)
    """

    def __init__(
        self,
        rebal_days=10,
        n_stocks=8,
        max_price=50000,
        min_price=2000,
        min_vol20=5e8,
        min_vol5=3e8,
        slippage=0.001,
        replace_threshold=0.05,
        exposure_by_regime=None,
    ):
        self.rebal_days = rebal_days
        self.n_stocks = n_stocks
        self.max_price = max_price
        self.min_price = min_price
        self.min_vol20 = min_vol20
        self.min_vol5 = min_vol5
        self.replace_threshold = replace_threshold
        self.exposure_by_regime = exposure_by_regime or {
            "bull": 0.8,
            "neutral": 0.55,
            "bear": 0.35,
        }

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = slippage

        self.factor_weights = {
            "mom3": 0.45,
            "mom1": 0.25,
            "quality": 0.20,
            "inv_vol": 0.10,
        }

    def get_name(self):
        return "kqm_small_cap_v22_regime"

    def get_description(self):
        return (
            "KQM Small Cap v2.2 Regime (레짐 기반 익스포저/최소 교체 임계치 적용)"
        )

    # ---------------------------------------------------------
    # 팩터 계산
    # ---------------------------------------------------------
    def _compute_factors(self, df, current_date):
        if df is None or current_date not in df.index:
            return None

        subset = df[df.index <= current_date]
        if len(subset) < 60:
            return None

        close = subset["close"].values
        price = close[-1]
        if price < self.min_price or price > self.max_price:
            return None

        if "volume" in df.columns:
            vol20 = (subset["close"] * subset["volume"]).tail(20).mean()
            vol5 = (subset["close"] * subset["volume"]).tail(5).mean()
            if vol20 < self.min_vol20 or vol5 < self.min_vol5:
                return None

        if len(close) < 60:
            return None
        mom3 = close[-1] / close[-60] - 1
        mom1 = close[-1] / close[-20] - 1

        ret60 = pd.Series(close[-60:]).pct_change().dropna()
        if len(ret60) < 10:
            return None
        quality = ret60.mean() / (ret60.std() + 1e-9)

        vol20 = (
            pd.Series(close[-20:])
            .pct_change()
            .ewm(halflife=10)
            .std()
            .iloc[-1]
        )
        inv_vol = 1 / (vol20 + 1e-9)

        return {
            "mom3": mom3,
            "mom1": mom1,
            "quality": quality,
            "inv_vol": inv_vol,
            "price": price,
        }

    def _calc_equity(self, cash, positions, enriched, date):
        total = cash
        for t, pos in positions.items():
            df = enriched.get(t)
            if df is None:
                continue

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
    # 레짐 계산
    # ---------------------------------------------------------
    def _prepare_regime(self, market_index):
        if market_index is None or "close" not in market_index.columns:
            return None
        idx = market_index.copy().sort_index()
        idx = idx[["close"]].astype(float)
        idx["ma60"] = idx["close"].rolling(60).mean()
        idx["mom20"] = idx["close"].pct_change(20)
        return idx

    def _get_regime(self, regime_df, date):
        if regime_df is None or len(regime_df) == 0:
            return "neutral"
        if date not in regime_df.index:
            valid = regime_df.index[regime_df.index <= date]
            if len(valid) == 0:
                return "neutral"
            row = regime_df.loc[valid.max()]
        else:
            row = regime_df.loc[date]

        close = row.get("close")
        ma60 = row.get("ma60")
        mom20 = row.get("mom20")
        if pd.isna(ma60) or pd.isna(mom20):
            return "neutral"

        if close > ma60 and mom20 > 0:
            return "bull"
        if close < ma60 or mom20 < -0.02:
            return "bear"
        return "neutral"

    # ---------------------------------------------------------
    # 최소 교체 임계치 적용
    # ---------------------------------------------------------
    def _apply_replace_threshold(self, ranked_df, selected, positions):
        if self.replace_threshold <= 0 or not positions:
            return selected

        score_map = (
            ranked_df.set_index("ticker")["score"].to_dict()
        )

        selected_ordered = selected.copy()
        selected_set = set(selected_ordered)

        for t in positions.keys():
            if t in selected_set:
                continue
            ticker_score = score_map.get(t)
            if ticker_score is None or not np.isfinite(ticker_score):
                continue

            best_diff = None
            replace_target = None
            for cand in selected_ordered:
                if cand in positions:
                    continue
                cand_score = score_map.get(cand, -np.inf)
                diff = cand_score - ticker_score
                if best_diff is None or diff > best_diff:
                    best_diff = diff
                    replace_target = cand

            if best_diff is None or best_diff < self.replace_threshold:
                if replace_target and replace_target in selected_set:
                    selected_set.remove(replace_target)
                    selected_ordered.remove(replace_target)
                selected_set.add(t)
                selected_ordered.append(t)

        ordered = sorted(
            selected_set,
            key=lambda x: score_map.get(x, -np.inf),
            reverse=True,
        )
        return ordered[: self.n_stocks]

    # ---------------------------------------------------------
    # 백테스트 본체
    # ---------------------------------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):
        if not silent:
            print("\n===========================================================")
            print("📈 KQM Small Cap v2.2 Regime 백테스트 시작...")
            print("===========================================================\n")

        dates = sorted(
            set().union(*[df.index for df in enriched.values() if df is not None])
        )
        if len(dates) < 120:
            return pd.DataFrame(), []

        rebalance_dates = dates[120:: self.rebal_days]
        regime_df = self._prepare_regime(market_index)

        init_cash = 1_000_000
        cash = init_cash
        positions = {}
        equity_curve = []
        trade_log = []

        for idx in tqdm(range(len(rebalance_dates)), desc="KQM Small v2.2 Regime", disable=silent):
            d0 = rebalance_dates[idx]
            next_rebal = rebalance_dates[idx + 1] if idx < len(rebalance_dates) - 1 else dates[-1]

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
            day = day[
                (day["mom3"] > 0.0) &
                (day["mom1"] > -0.02) &
                (day["quality"] > 0.0)
            ]

            if len(day) == 0:
                equity_curve.append((d0, self._calc_equity(cash, positions, enriched, d0)))
                continue

            for f in ["mom3", "mom1", "quality", "inv_vol"]:
                day[f"{f}_rank"] = day[f].rank(pct=True)

            W = self.factor_weights
            day["score"] = (
                W["mom3"] * day["mom3_rank"]
                + W["mom1"] * day["mom1_rank"]
                + W["quality"] * day["quality_rank"]
                + W["inv_vol"] * day["inv_vol_rank"]
            )

            ranked = day.sort_values("score", ascending=False)
            selected = ranked.head(self.n_stocks)["ticker"].tolist()
            selected = self._apply_replace_threshold(ranked, selected, positions)

            score_map = ranked.set_index("ticker")["score"].to_dict()

            current_regime = self._get_regime(regime_df, d0)
            target_exposure = self.exposure_by_regime.get(
                current_regime,
                self.exposure_by_regime.get("neutral", 0.5),
            )

            for t in list(positions.keys()):
                if t not in selected:
                    df = enriched.get(t)
                    if df is None:
                        positions.pop(t)
                        continue

                    if d0 in df.index:
                        px = df.loc[d0, "close"]
                    else:
                        valid = df.index[df.index <= d0]
                        if len(valid) == 0:
                            positions.pop(t)
                            continue
                        px = df.loc[valid.max(), "close"]

                    if not np.isfinite(px) or px <= 0:
                        px = positions[t]["entry_px"]

                    exit_px = px * (1 - self.slippage)
                    qty = positions[t]["qty"]
                    proceeds = exit_px * qty
                    cost = positions[t]["entry_px"] * qty
                    pnl = proceeds - cost

                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0

                    cash += proceeds - fee_out - tax
                    positions.pop(t)

                    trade_log.append(
                        {
                            "date": d0,
                            "ticker": t,
                            "action": "SELL",
                            "price": exit_px,
                            "qty": qty,
                            "pnl": pnl,
                            "cash_after": cash,
                        }
                    )

            equity_val = self._calc_equity(cash, positions, enriched, d0)
            if len(selected) == 0:
                equity_curve.append((d0, equity_val))
                continue

            target_val = equity_val * target_exposure / max(len(selected), 1)

            for t in selected:
                df = enriched.get(t)
                if df is None:
                    continue

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
                target_qty = int(target_val / entry_px)
                if target_qty <= 0:
                    continue

                cur_qty = positions.get(t, {}).get("qty", 0)
                delta = target_qty - cur_qty

                if delta > 0:
                    cost = entry_px * delta
                    fee_in = cost * self.fee
                    total_cost = cost + fee_in
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

                    trade_log.append(
                        {
                            "date": d0,
                            "ticker": t,
                            "action": "BUY",
                            "price": entry_px,
                            "qty": delta,
                            "pnl": 0,
                            "cash_after": cash,
                        }
                    )

                elif delta < 0:
                    sell_qty = -delta
                    pos = positions.get(t)
                    if pos is None or sell_qty <= 0:
                        continue

                    exit_px = px * (1 - self.slippage)
                    proceeds = exit_px * sell_qty
                    pnl = proceeds - (pos["entry_px"] * sell_qty)

                    fee_out = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0

                    cash += proceeds - fee_out - tax
                    pos["qty"] -= sell_qty

                    if pos["qty"] <= 0:
                        positions.pop(t, None)

                    trade_log.append(
                        {
                            "date": d0,
                            "ticker": t,
                            "action": "SELL",
                            "price": exit_px,
                            "qty": sell_qty,
                            "pnl": pnl,
                            "cash_after": cash,
                        }
                    )

            for dt in [d for d in dates if d0 <= d < next_rebal]:
                equity_curve.append((dt, self._calc_equity(cash, positions, enriched, dt)))

        final_date = dates[-1]
        for t in list(positions.keys()):
            df = enriched.get(t)
            pos = positions.pop(t)
            if df is None:
                continue

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

            trade_log.append(
                {
                    "date": final_date,
                    "ticker": t,
                    "action": "SELL",
                    "price": exit_px,
                    "qty": qty,
                    "pnl": pnl,
                    "cash_after": cash,
                }
            )

        equity_curve.append((final_date, cash))
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(f"✅ KQM Small Cap v2.2 Regime 백테스트 완료! 최종 자산: {cash:,.0f}원")

        return ec, trade_log
