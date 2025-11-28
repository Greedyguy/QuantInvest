#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMSmallCapStrategyV22Short(BaseStrategy):
    """
    K-Quality Momentum Small Cap v2.2 (단기 최적화 버전)

    - 레짐 기반 익스포저 + 초약세(ultra-bear) 방어
    - 강화된 진입 게이트 (Breakout/VCP/RS 조건)
    - 리밸런싱 주기 15일, 최소 교체 임계치 상향
    - 역변동성 비중 + 포지션별 스탑/익절/최대보유일 적용
    """

    def __init__(
        self,
        rebal_days=15,
        max_price=50000,
        min_price=2000,
        min_vol20=5e8,
        min_vol5=3e8,
        slippage=0.0015,
        replace_threshold=0.10,
        gating_rs_rank=0.60,
        stop_loss=-0.10,
        take_profit=0.15,
        max_hold_days=20,
        exposure_by_regime=None,
        nstocks_by_regime=None,
    ):
        self.rebal_days = rebal_days
        self.max_price = max_price
        self.min_price = min_price
        self.min_vol20 = min_vol20
        self.min_vol5 = min_vol5
        self.slippage = slippage
        self.replace_threshold = replace_threshold
        self.gating_rs_rank = gating_rs_rank
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_days = max_hold_days
        self.exposure_by_regime = exposure_by_regime or {
            "bull": 0.80,
            "neutral": 0.50,
            "bear": 0.35,
            "ultra_bear": 0.15,
        }
        self.nstocks_by_regime = nstocks_by_regime or {
            "bull": 5,
            "neutral": 5,
            "bear": 4,
            "ultra_bear": 4,
        }

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

        self.factor_weights = {
            "mom3": 0.40,
            "mom1": 0.20,
            "quality": 0.25,
            "inv_vol": 0.15,
        }

    def get_name(self):
        return "kqm_small_cap_v22_short"

    def get_description(self):
        return "KQM Small Cap v2.2 (단기 최적화, 레짐+게이트+리스크 제어)"

    # ---------------------------------------------------------
    # 팩터 + 게이트 계산
    # ---------------------------------------------------------
    def _get_row_value(self, df, date, col, default=np.nan):
        if df is None or col not in df.columns:
            return default
        if date in df.index:
            return df.loc[date, col]
        valid = df.index[df.index <= date]
        if len(valid) == 0:
            return default
        return df.loc[valid.max(), col]

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

        bo = self._get_row_value(df, current_date, "bo", 0)
        vcp = self._get_row_value(df, current_date, "vcp", 0)
        rs_raw = self._get_row_value(df, current_date, "rs_raw", 0)

        return {
            "mom3": mom3,
            "mom1": mom1,
            "quality": quality,
            "inv_vol": inv_vol,
            "price": price,
            "bo": bo,
            "vcp": vcp,
            "rs_raw": rs_raw,
        }

    # ---------------------------------------------------------
    # 레짐 파악
    # ---------------------------------------------------------
    def _prepare_regime(self, market_index):
        if market_index is None or "close" not in market_index.columns:
            return None
        idx = market_index.copy().sort_index()
        idx = idx[["close"]].astype(float)
        idx["ma60"] = idx["close"].rolling(60).mean()
        idx["mom20"] = idx["close"].pct_change(20)
        idx["mom5"] = idx["close"].pct_change(5)
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
        mom5 = row.get("mom5")
        if pd.isna(ma60) or pd.isna(mom20):
            return "neutral"

        if close > ma60 and mom20 > 0:
            return "bull"
        if close < ma60 and mom20 < -0.01 and mom5 < 0:
            return "ultra_bear"
        if close < ma60 or mom20 < -0.02:
            return "bear"
        return "neutral"

    # ---------------------------------------------------------
    # 매매 헬퍼
    # ---------------------------------------------------------
    def _sell_position(self, ticker, date, positions, enriched, cash, trade_log, qty=None, reason=None):
        pos = positions.get(ticker)
        if pos is None:
            return cash
        df = enriched.get(ticker)
        if df is None:
            return cash

        if date in df.index:
            px = df.loc[date, "close"]
        else:
            valid = df.index[df.index <= date]
            if len(valid) == 0:
                return cash
            px = df.loc[valid.max(), "close"]

        if not np.isfinite(px) or px <= 0:
            px = pos["entry_px"]

        if qty is None or qty > pos["qty"]:
            qty = pos["qty"]

        exit_px = px * (1 - self.slippage)
        proceeds = exit_px * qty
        pnl = proceeds - (pos["entry_px"] * qty)

        fee_out = proceeds * self.fee
        tax = proceeds * self.tax if pnl > 0 else 0
        cash += proceeds - fee_out - tax

        pos["qty"] -= qty
        if pos["qty"] <= 0:
            positions.pop(ticker, None)

        trade_log.append(
            {
                "date": date,
                "ticker": ticker,
                "action": "SELL",
                "price": exit_px,
                "qty": qty,
                "pnl": pnl,
                "cash_after": cash,
                "reason": reason or "rebalance",
            }
        )
        return cash

    def _enforce_risk_controls(self, date, positions, enriched, cash, trade_log):
        if not positions:
            return cash
        for ticker in list(positions.keys()):
            pos = positions.get(ticker)
            if pos is None:
                continue

            df = enriched.get(ticker)
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
                continue

            ret = px / pos["entry_px"] - 1
            hold_days = (date - pos["entry_date"]).days

            trigger = None
            if ret <= self.stop_loss:
                trigger = "stop_loss"
            elif ret >= self.take_profit:
                trigger = "take_profit"
            elif hold_days >= self.max_hold_days:
                trigger = "max_hold_days"

            if trigger:
                cash = self._sell_position(
                    ticker,
                    date,
                    positions,
                    enriched,
                    cash,
                    trade_log,
                    reason=trigger,
                )
        return cash

    # ---------------------------------------------------------
    # 최소 교체 임계치
    # ---------------------------------------------------------
    def _apply_replace_threshold(self, ranked_df, selected, positions, target_n):
        if self.replace_threshold <= 0 or not positions:
            return selected[:target_n]

        score_map = ranked_df.set_index("ticker")["score"].to_dict()
        selected_ordered = selected.copy()
        selected_set = set(selected_ordered)

        for t in positions.keys():
            if t in selected_set:
                continue
            ticker_score = score_map.get(t)
            if ticker_score is None or not np.isfinite(ticker_score):
                continue

            diffs = [
                (cand, score_map.get(cand, -np.inf) - ticker_score)
                for cand in selected_ordered
            ]
            diffs.sort(key=lambda x: x[1], reverse=True)

            if diffs and diffs[0][1] < self.replace_threshold:
                replace_target = diffs[0][0]
                if replace_target in selected_set:
                    selected_set.remove(replace_target)
                    selected_ordered.remove(replace_target)
                selected_set.add(t)
                selected_ordered.append(t)

        ordered = sorted(
            selected_set,
            key=lambda x: score_map.get(x, -np.inf),
            reverse=True,
        )
        return ordered[:target_n]

    # ---------------------------------------------------------
    # Equity 계산
    # ---------------------------------------------------------
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
    # 백테스트
    # ---------------------------------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):
        self._reset_weight_history()
        if not silent:
            print("\n===========================================================")
            print("📈 KQM Small Cap v2.2 Short-term 백테스트 시작...")
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

        for idx in tqdm(range(len(rebalance_dates)), desc="KQM Small v2.2 Short", disable=silent):
            d0 = rebalance_dates[idx]
            next_rebal = (
                rebalance_dates[idx + 1]
                if idx < len(rebalance_dates) - 1
                else dates[-1]
            )

            cash = self._enforce_risk_controls(d0, positions, enriched, cash, trade_log)

            rows = []
            for ticker, df in enriched.items():
                fac = self._compute_factors(df, d0)
                if fac is None:
                    continue
                rows.append({"ticker": ticker, **fac})

            if len(rows) == 0:
                equity_val = self._calc_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity_val))
                self._record_weights(d0, cash, positions, enriched)
                continue

            day = pd.DataFrame(rows)
            day = day[
                (day["mom3"] > 0.0)
                & (day["mom1"] > -0.02)
                & (day["quality"] > 0.0)
            ]

            if len(day) == 0:
                equity_val = self._calc_equity(cash, positions, enriched, d0)
                equity_curve.append((d0, equity_val))
                self._record_weights(d0, cash, positions, enriched)
                continue

            day["rs_rank"] = day["rs_raw"].rank(pct=True)
            gating_mask = (
                (day["bo"] >= 1)
                | (day["vcp"] >= 1)
                | (day["rs_rank"] >= self.gating_rs_rank)
            )
            day = day[gating_mask]

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

            current_regime = self._get_regime(regime_df, d0)
            target_n = self.nstocks_by_regime.get(current_regime, self.nstocks_by_regime["neutral"])
            selected = ranked.head(target_n)["ticker"].tolist()
            selected = self._apply_replace_threshold(ranked, selected, positions, target_n)

            target_exposure = self.exposure_by_regime.get(
                current_regime,
                self.exposure_by_regime.get("neutral", 0.5),
            )

            for t in list(positions.keys()):
                if t not in selected:
                    cash = self._sell_position(
                        t, d0, positions, enriched, cash, trade_log, reason="rebalance"
                    )

            equity_val = self._calc_equity(cash, positions, enriched, d0)
            investable = equity_val * target_exposure

            if len(selected) == 0 or investable <= 0:
                equity_curve.append((d0, equity_val))
                continue

            inv_weights = []
            for t in selected:
                row = ranked[ranked["ticker"] == t]
                inv_vol = row["inv_vol"].iloc[0] if not row.empty else 1.0
                inv_weights.append(max(inv_vol, 1e-6))
            weight_sum = sum(inv_weights)
            if weight_sum <= 0:
                weight_sum = len(inv_weights)
                inv_weights = [1.0 for _ in inv_weights]

            for t, inv_w in zip(selected, inv_weights):
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
                allocation = investable * (inv_w / weight_sum)
                target_qty = int(allocation / entry_px)
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
                        new_px = (old_q * old_px + entry_px * delta) / new_q
                        positions[t]["qty"] = new_q
                        positions[t]["entry_px"] = new_px
                    else:
                        positions[t] = {
                            "qty": delta,
                            "entry_px": entry_px,
                            "entry_date": d0,
                        }

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
                    cash = self._sell_position(
                        t,
                        d0,
                        positions,
                        enriched,
                        cash,
                        trade_log,
                        qty=-delta,
                        reason="trim",
                    )

            for dt in [d for d in dates if d0 <= d < next_rebal]:
                cash = self._enforce_risk_controls(dt, positions, enriched, cash, trade_log)
                equity_val = self._calc_equity(cash, positions, enriched, dt)
                equity_curve.append((dt, equity_val))
                self._record_weights(dt, cash, positions, enriched)

        final_date = dates[-1]
        for t in list(positions.keys()):
            cash = self._sell_position(t, final_date, positions, enriched, cash, trade_log, reason="final")

        equity_curve.append((final_date, cash))
        self._record_weights(final_date, cash, positions, enriched)
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(f"✅ KQM Small Cap v2.2 Short-term 백테스트 완료! 최종 자산: {cash:,.0f}원")

        return ec, trade_log
