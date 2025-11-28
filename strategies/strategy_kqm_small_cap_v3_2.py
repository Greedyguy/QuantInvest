#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Quality Momentum Small Cap (v3.2)
- 소액 투자(100만원) 특화 안정형 팩터 전략
- 유동성 강화 + 모멘텀 강화 버전
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.base_strategy import BaseStrategy
from config import TAX_RATE_SELL, FEE_PER_SIDE


class KQMSmallCapStrategyV32(BaseStrategy):
    def __init__(
        self,
        rebal_days=10,
        n_stocks=5,
        min_price=10000,     # 🔥 가격 하한 강화 (1만원 이상)
        max_price=50000,
        factor_weights=None,
    ):
        self.rebal_days = rebal_days
        self.n_stocks = n_stocks
        self.min_price = min_price
        self.max_price = max_price

        # 🔥 모멘텀 강화 버전
        self.factor_weights = factor_weights or {
            'MOM6': 0.50,
            'MOM3': 0.20,
            'QUALITY': 0.20,
            'VOL': 0.05,
            'VAL': 0.05,
        }

        # 거래 비용 & 슬리피지
        self.fee = FEE_PER_SIDE
        self.tax = TAX_RATE_SELL
        self.slippage = 0.001

    def get_name(self):
        return "kqm_small_cap_v3_2"

    def get_description(self):
        return f"KQM Small Cap v3_2 (Liquidity + Momentum 강화)"

    # =====================================================================================
    # 팩터 계산
    # =====================================================================================
    def _compute_factors(self, df: pd.DataFrame, current_date: pd.Timestamp):
        if df is None or len(df) < 120:
            return None
        if current_date not in df.index:
            return None

        subset = df[df.index <= current_date]
        if len(subset) < 120:
            return None

        close = subset["close"].values

        # Momentum
        mom6 = close[-1] / close[-120] - 1
        if len(close) < 60:
            return None
        mom3 = close[-1] / close[-60] - 1

        # Quality Proxy
        ret_60 = pd.Series(close[-60:]).pct_change().dropna()
        quality = ret_60.mean() / (ret_60.std() + 1e-9)

        # Inverse Volatility
        vol20 = pd.Series(close[-20:]).pct_change().ewm(halflife=10).std().iloc[-1]
        inv_vol = 1 / (vol20 + 1e-9)

        # Value Proxy
        ret_120 = pd.Series(close[-120:]).pct_change().dropna()
        val = ret_120.mean()

        # 🔥 유동성 필터 (가장 중요)
        if "volume" not in subset.columns:
            return None

        trade_value_20 = (subset["close"][-20:] * subset["volume"][-20:]).mean()
        trade_value_3 = (subset["close"][-3:] * subset["volume"][-3:]).mean()

        if trade_value_20 < 5_000_000_000:    # 20일 평균 50억
            return None
        if trade_value_3 < 3_000_000_000:     # 3일 평균 30억
            return None

        price = subset["close"].iloc[-1]

        # 🔥 가격 필터 강화
        if price < self.min_price or price > self.max_price:
            return None

        return {
            "mom6": mom6,
            "mom3": mom3,
            "quality": quality,
            "inv_vol": inv_vol,
            "val_proxy": val,
            "price": price,
        }

    # =====================================================================================
    # 백테스트 실행
    # =====================================================================================
    def run_backtest(self, enriched: dict, market_index=None, weights=None, silent=False):
        if not silent:
            print("\n" + "="*60)
            print("📈 KQM Small Cap v3.2 백테스트 시작...")
            print("="*60)

        dates = sorted(set().union(*[df.index for df in enriched.values() if df is not None]))
        if len(dates) < 120:
            return pd.DataFrame(), []

        rebalance_dates = dates[120::self.rebal_days]

        init_cash = 1_000_000
        cash = init_cash
        positions = {}
        equity_curve = []
        trade_log = []

        for ridx in tqdm(range(len(rebalance_dates)), desc="KQM v3.2", disable=silent):
            d0 = rebalance_dates[ridx]
            next_d = rebalance_dates[ridx + 1] if ridx < len(rebalance_dates) - 1 else dates[-1]

            # =================================================================================
            # 1) 팩터 스냅샷
            # =================================================================================
            rows = []
            for ticker, df in enriched.items():
                f = self._compute_factors(df, d0)
                if f is None:
                    continue

                rows.append({
                    "ticker": ticker,
                    **f
                })

            if len(rows) == 0:
                # 기록만 진행
                for d in [x for x in dates if d0 <= x < next_d]:
                    equity = self._calculate_equity(cash, positions, enriched, d)
                    equity_curve.append((d, equity))
                continue

            day = pd.DataFrame(rows)

            # =================================================================================
            # 2) 팩터 랭킹
            # =================================================================================
            for col in ["mom6", "mom3", "quality", "inv_vol", "val_proxy"]:
                day[f"{col}_rank"] = day[col].rank(pct=True)

            # 복합 스코어
            w = self.factor_weights
            day["score"] = (
                w["MOM6"] * day["mom6_rank"] +
                w["MOM3"] * day["mom3_rank"] +
                w["QUALITY"] * day["quality_rank"] +
                w["VOL"] * day["inv_vol_rank"] +
                w["VAL"] * day["val_proxy_rank"]
            )

            selected = day.sort_values("score", ascending=False).head(self.n_stocks)["ticker"].tolist()

            # =================================================================================
            # 3) 기존 포지션 청산
            # =================================================================================
            for t in list(positions.keys()):
                if t not in selected:
                    df_t = enriched[t]
                    if df_t is None or d0 not in df_t.index:
                        continue

                    exit_px = df_t.loc[d0, "close"] * (1 - self.slippage)
                    qty = positions[t]["qty"]
                    entry_px = positions[t]["entry_px"]

                    proceeds = exit_px * qty
                    pnl = proceeds - entry_px * qty

                    fee = proceeds * self.fee
                    tax = proceeds * self.tax if pnl > 0 else 0

                    cash += proceeds - fee - tax
                    del positions[t]

            # =================================================================================
            # 4) 신규 진입
            # =================================================================================
            port_val = cash + sum(
                enriched[t].loc[d0, "close"] * pos["qty"]
                for t, pos in positions.items()
                if d0 in enriched[t].index
            )

            target_w = 1 / len(selected)

            for t in selected:
                df_t = enriched[t]
                if df_t is None or d0 not in df_t.index:
                    continue

                entry_px = df_t.loc[d0, "close"] * (1 + self.slippage)
                target_val = port_val * target_w
                target_qty = int(target_val / entry_px)

                if target_qty <= 0:
                    continue

                cur_qty = positions.get(t, {}).get("qty", 0)
                delta = target_qty - cur_qty

                if delta > 0:
                    cost = entry_px * delta
                    fee = cost * self.fee
                    total_cost = cost + fee

                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    if t in positions:
                        old_qty = positions[t]["qty"]
                        old_px = positions[t]["entry_px"]
                        new_qty = old_qty + delta
                        new_px = (old_px * old_qty + entry_px * delta) / new_qty
                        positions[t] = {"qty": new_qty, "entry_px": new_px}
                    else:
                        positions[t] = {"qty": delta, "entry_px": entry_px}

            # =================================================================================
            # 5) 구간 Equity 기록
            # =================================================================================
            for d in [x for x in dates if d0 <= x < next_d]:
                equity = self._calculate_equity(cash, positions, enriched, d)
                equity_curve.append((d, equity))

        # =================================================================================
        # 6) 마지막 현금 반영
        # =================================================================================
        final_date = dates[-1]
        equity_curve.append((final_date, cash))

        ec = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")

        if not silent:
            print(f"✅ KQM v3.2 완료 → Final {cash:,.0f}원")

        return ec, trade_log