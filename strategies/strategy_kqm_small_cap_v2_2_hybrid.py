#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMSmallCapStrategyV22Hybrid(BaseStrategy):
    """
    Regime + Short 하이브리드 버전 (실거래 대응)
    - 하나의 포트폴리오에서 레짐에 따라 룰셋 전환
    - 상승장: 기존 Regime 전략과 유사한 완화 조건
    - 약세장: Short 전략의 보수적 조건 + 리스크 제어
    """

    def __init__(
        self,
        rebal_days_bull=10,
        rebal_days_bear=16,
        max_price=50000,
        min_price=2000,
        min_vol20=5e8,
        min_vol5=3e8,
        slippage=0.001,
        replace_threshold_bull=0.035,
        replace_threshold_bear=0.08,
        stop_loss_bull=-0.12,
        stop_loss_bear=-0.08,
        take_profit_bull=0.25,
        take_profit_bear=0.15,
        max_hold_bull=45,
        max_hold_bear=20,
        exposure_map=None,
        nstocks_map=None,
    ):
        self.rebal_days_bull = rebal_days_bull
        self.rebal_days_bear = rebal_days_bear
        self.max_price = max_price
        self.min_price = min_price
        self.min_vol20 = min_vol20
        self.min_vol5 = min_vol5
        self.slippage = slippage
        self.replace_threshold_bull = replace_threshold_bull
        self.replace_threshold_bear = replace_threshold_bear
        self.stop_loss_bull = stop_loss_bull
        self.stop_loss_bear = stop_loss_bear
        self.take_profit_bull = take_profit_bull
        self.take_profit_bear = take_profit_bear
        self.max_hold_bull = max_hold_bull
        self.max_hold_bear = max_hold_bear

        self.exposure_map = exposure_map or {
            "bull": 0.95,
            "neutral": 0.70,
            "bear": 0.35,
            "ultra_bear": 0.20,
        }
        self.nstocks_map = nstocks_map or {
            "bull": 8,
            "neutral": 6,
            "bear": 4,
            "ultra_bear": 3,
        }

        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL

        # 두 레짐별 팩터 가중치/조건 세트
        self.factor_weights_bull = {
            "mom3": 0.45,
            "mom1": 0.25,
            "quality": 0.20,
            "inv_vol": 0.10,
        }
        self.factor_weights_bear = {
            "mom3": 0.30,
            "mom1": 0.15,
            "quality": 0.35,
            "inv_vol": 0.20,
        }

    def get_name(self):
        return "kqm_small_cap_v22_hybrid"

    def get_description(self):
        return "KQM Small Cap v2.2 Hybrid (Regime ↔ Short 단일 포트폴리오 전환)"

    # ---------------------------------------------------------
    # 데이터 접근 유틸
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
    # KOSDAQ 레짐 판별
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
    # 매도/리스크 관리 헬퍼
    # ---------------------------------------------------------
    def _sell(self, ticker, date, positions, enriched, cash, trade_log, qty=None, reason="rebalance"):
        pos = positions.get(ticker)
        if pos is None:
            return cash
        df = enriched.get(ticker)
        if df is None:
            positions.pop(ticker, None)
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

        trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": "SELL",
            "price": exit_px,
            "qty": qty,
            "pnl": pnl,
            "cash_after": cash,
            "reason": reason,
        })
        return cash

    def _risk_control(self, date, positions, enriched, cash, trade_log, regime):
        for ticker in list(positions.keys()):
            pos = positions.get(ticker)
            df = enriched.get(ticker)
            if pos is None or df is None:
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

            if regime in ("bear", "ultra_bear"):
                stop_loss = self.stop_loss_bear
                take_profit = self.take_profit_bear
                max_hold = self.max_hold_bear
            else:
                stop_loss = self.stop_loss_bull
                take_profit = self.take_profit_bull
                max_hold = self.max_hold_bull

            reason = None
            if ret <= stop_loss:
                reason = "stop_loss"
            elif ret >= take_profit:
                reason = "take_profit"
            elif hold_days >= max_hold:
                reason = "max_hold"

            if reason:
                cash = self._sell(ticker, date, positions, enriched, cash, trade_log, reason=reason)
        return cash

    # ---------------------------------------------------------
    # 최소 교체 임계치 적용
    # ---------------------------------------------------------
    def _apply_replace_threshold(self, ranked_df, selected, positions, threshold):
        if threshold <= 0 or not positions:
            return selected
        score_map = ranked_df.set_index("ticker")["score"].to_dict()
        selected_ordered = selected.copy()
        selected_set = set(selected_ordered)

        for t in positions.keys():
            if t in selected_set:
                continue
            ticker_score = score_map.get(t)
            if ticker_score is None:
                continue

            best_diff = None
            replace_target = None
            for cand in selected_ordered:
                cand_score = score_map.get(cand, -np.inf)
                diff = cand_score - ticker_score
                if best_diff is None or diff > best_diff:
                    best_diff = diff
                    replace_target = cand

            if best_diff is None or best_diff < threshold:
                if replace_target in selected_set:
                    selected_set.remove(replace_target)
                    selected_ordered.remove(replace_target)
                selected_set.add(t)
                selected_ordered.append(t)

        ordered = sorted(selected_set, key=lambda x: score_map.get(x, -np.inf), reverse=True)
        return ordered[: len(selected)]

    # ---------------------------------------------------------
    # 메인 백테스트 루프
    # ---------------------------------------------------------
    def run_backtest(self, enriched, market_index=None, weights=None, silent=False):
        if not silent:
            print("\n===========================================================")
            print("📈 KQM Small Cap v2.2 Hybrid 백테스트 시작...")
            print("===========================================================\n")

        dates = sorted(
            set().union(*[df.index for df in enriched.values() if df is not None])
        )
        if len(dates) < 120:
            return pd.DataFrame(), []

        start_idx = 120
        if start_idx >= len(dates):
            return pd.DataFrame(), []
        regime_df = self._prepare_regime(market_index)

        init_cash = 1_000_000
        cash = init_cash
        positions = {}
        trade_log = []
        equity_curve = []

        cur_idx = start_idx
        step_guard = 0
        while cur_idx < len(dates) - 1:
            d0 = dates[cur_idx]
            regime = self._get_regime(regime_df, d0)
            rebal_gap = self.rebal_days_bear if regime in ("bear", "ultra_bear") else self.rebal_days_bull
            gap = max(int(rebal_gap), 1)
            next_idx = min(cur_idx + gap, len(dates) - 1)
            next_rebal = dates[next_idx]

            regime = self._get_regime(regime_df, d0)
            target_exposure = self.exposure_map.get(regime, self.exposure_map["neutral"])
            target_n = self.nstocks_map.get(regime, self.nstocks_map["neutral"])
            gating_rs = 0.65 if regime in ("bear", "ultra_bear") else 0.45

            cash = self._risk_control(d0, positions, enriched, cash, trade_log, regime)

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
            if regime in ("bear", "ultra_bear"):
                day = day[
                    (day["mom3"] > 0)
                    & (day["mom1"] > -0.01)
                    & (day["quality"] > 0)
                ]
                day["rs_rank"] = day["rs_raw"].rank(pct=True)
                gating = (day["bo"] >= 1) | (day["vcp"] >= 1) | (day["rs_rank"] >= gating_rs)
                day = day[gating]
                W = self.factor_weights_bear
            else:
                day = day[
                    (day["mom3"] > -0.02)
                    & (day["mom1"] > -0.05)
                    & (day["quality"] > -0.05)
                ]
                W = self.factor_weights_bull

            if len(day) == 0:
                equity_curve.append((d0, self._calc_equity(cash, positions, enriched, d0)))
                continue

            for f in ["mom3", "mom1", "quality", "inv_vol"]:
                day[f"{f}_rank"] = day[f].rank(pct=True)
            day["score"] = (
                W["mom3"] * day["mom3_rank"]
                + W["mom1"] * day["mom1_rank"]
                + W["quality"] * day["quality_rank"]
                + W["inv_vol"] * day["inv_vol_rank"]
            )

            ranked = day.sort_values("score", ascending=False)
            selected = ranked.head(target_n)["ticker"].tolist()
            threshold = self.replace_threshold_bear if regime in ("bear", "ultra_bear") else self.replace_threshold_bull
            selected = self._apply_replace_threshold(ranked, selected, positions, threshold)

            for t in list(positions.keys()):
                if t not in selected:
                    cash = self._sell(t, d0, positions, enriched, cash, trade_log)

            equity_val = self._calc_equity(cash, positions, enriched, d0)
            investable = equity_val * target_exposure
            if len(selected) == 0 or investable <= 0:
                equity_curve.append((d0, equity_val))
                continue

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
                target_val = investable / len(selected)
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
                        positions[t] = {"qty": new_q, "entry_px": new_px, "entry_date": positions[t]["entry_date"]}
                    else:
                        positions[t] = {"qty": delta, "entry_px": entry_px, "entry_date": d0}

                    trade_log.append({
                        "date": d0,
                        "ticker": t,
                        "action": "BUY",
                        "price": entry_px,
                        "qty": delta,
                        "pnl": 0,
                        "cash_after": cash,
                    })

                elif delta < 0:
                    cash = self._sell(t, d0, positions, enriched, cash, trade_log, qty=-delta, reason="trim")

            for dt in dates[cur_idx:next_idx]:
                cash = self._risk_control(dt, positions, enriched, cash, trade_log, regime)
                equity_curve.append((dt, self._calc_equity(cash, positions, enriched, dt)))

            if next_idx == cur_idx:
                cur_idx += 1
            else:
                cur_idx = next_idx
            step_guard += 1
            if step_guard > len(dates):
                break

        final_date = dates[-1]
        for t in list(positions.keys()):
            cash = self._sell(t, final_date, positions, enriched, cash, trade_log, reason="final")

        equity_curve.append((final_date, cash))
        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(f"✅ KQM Small Cap v2.2 Hybrid 백테스트 완료! 최종 자산: {cash:,.0f}원")

        return ec, trade_log
