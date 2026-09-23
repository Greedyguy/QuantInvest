"""KODEX 200 core with a liquid-stock relative-strength satellite."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from config import BLOCKED_TICKERS, TAX_RATE_SELL
from strategies.base_strategy import BaseStrategy
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


class K200RelativeStrengthSatellite(BaseStrategy):
    """Seek independent alpha without discarding the market core.

    Signals use the prior close and orders execute at the following open.  The
    model rebalances in January/March/May/July/September/November, holds a 40%
    KODEX 200 core and six liquid relative-strength stocks, and inherits the
    separately tested low-turnover KODEX 200 risk-on/risk-off state.

    This is a research candidate.  A later-snapshot stock universe can create
    survivorship bias, so promotion requires point-in-time or live shadow data.
    """

    CASH = "__CASH__"

    def __init__(
        self,
        core_ticker: str = "069500",
        core_weight: float = 0.40,
        satellite_weight: float = 0.55,
        top_n: int = 6,
        rebalance_months: Iterable[int] = (1, 3, 5, 7, 9, 11),
        regime_mode: str = "low_turnover",
        regime_window: int = 100,
        momentum_long_window: int = 252,
        momentum_skip_window: int = 21,
        trend_window: int = 200,
        volatility_window: int = 120,
        volatility_penalty: float = 0.25,
        min_value_20: float = 20_000_000_000.0,
        min_price: float = 3_000.0,
        max_price: float = 180_000.0,
        initial_cash: float = 2_100_000.0,
        min_trade: int = 50_000,
        price_band_pct: float = 3.0,
        universe_tickers: Iterable[str] | None = None,
        distribution_events_by_ticker: Mapping[str, pd.DataFrame] | None = None,
    ):
        super().__init__()
        if core_weight < 0 or satellite_weight < 0:
            raise ValueError("portfolio weights must be non-negative")
        if core_weight + satellite_weight > 1.0:
            raise ValueError("core and satellite weights cannot exceed 100%")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.core_ticker = str(core_ticker)
        self.core_weight = float(core_weight)
        self.satellite_weight = float(satellite_weight)
        self.top_n = int(top_n)
        self.rebalance_months = frozenset(int(month) for month in rebalance_months)
        if regime_mode not in {"low_turnover", "ma100"}:
            raise ValueError("regime_mode must be 'low_turnover' or 'ma100'")
        self.regime_mode = regime_mode
        self.regime_window = int(regime_window)
        self.momentum_long_window = int(momentum_long_window)
        self.momentum_skip_window = int(momentum_skip_window)
        self.trend_window = int(trend_window)
        self.volatility_window = int(volatility_window)
        self.volatility_penalty = float(volatility_penalty)
        self.min_value_20 = float(min_value_20)
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.initial_cash = float(initial_cash)
        self.min_trade = int(min_trade)
        self.price_band_pct = float(price_band_pct)
        self.universe_tickers = (
            frozenset(str(ticker) for ticker in universe_tickers)
            if universe_tickers is not None
            else None
        )
        self.distribution_events_by_ticker = dict(
            distribution_events_by_ticker or {}
        )
        self.latest_selection_history = pd.DataFrame()

    def get_name(self) -> str:
        return "k200_relative_strength_satellite"

    def get_description(self) -> str:
        return (
            "KODEX 200 40% core + six-stock 12-1M relative-strength/low-vol "
            "satellite, bimonthly"
        )

    def _eligible_tickers(self, enriched: dict[str, pd.DataFrame]) -> list[str]:
        if self.universe_tickers is not None:
            requested = self.universe_tickers
        else:
            requested = {
                str(ticker)
                for ticker in enriched
                if len(str(ticker)) == 6
                and str(ticker).isdigit()
                and str(ticker).endswith("0")
            }
        return sorted(
            ticker
            for ticker in requested
            if ticker != self.core_ticker
            and ticker not in BLOCKED_TICKERS
            and ticker in enriched
        )

    @staticmethod
    def _normalise_frame(frame: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
        out = frame.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index().loc[lambda df: ~df.index.duplicated(keep="last")]
        return out.reindex(dates)

    def compute_security_targets(
        self,
        enriched: dict[str, pd.DataFrame],
        market_index=None,
        secondary_index=None,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        core = enriched.get(self.core_ticker)
        if core is None or core.empty or "close" not in core:
            return pd.DataFrame()
        core = core.copy()
        core.index = pd.to_datetime(core.index)
        core = core.sort_index().loc[lambda df: ~df.index.duplicated(keep="last")]
        dates = core.index
        tickers = self._eligible_tickers(enriched)
        if not tickers:
            return pd.DataFrame({self.CASH: 1.0}, index=dates)

        close_columns: dict[str, pd.Series] = {}
        value_columns: dict[str, pd.Series] = {}
        valid_tickers: list[str] = []
        for ticker in tickers:
            frame = self._normalise_frame(enriched[ticker], dates)
            if not {"close", "value"}.issubset(frame.columns):
                continue
            close_columns[ticker] = pd.to_numeric(frame["close"], errors="coerce")
            value_columns[ticker] = pd.to_numeric(frame["value"], errors="coerce")
            valid_tickers.append(ticker)
        if not valid_tickers:
            return pd.DataFrame({self.CASH: 1.0}, index=dates)
        close = pd.DataFrame(close_columns, index=dates)
        value = pd.DataFrame(value_columns, index=dates)

        daily_returns = close.pct_change(fill_method=None)
        momentum = (
            close.shift(self.momentum_skip_window)
            / close.shift(self.momentum_long_window)
            - 1.0
        )
        volatility = daily_returns.rolling(
            self.volatility_window,
            min_periods=max(40, self.volatility_window * 2 // 3),
        ).std()
        score = momentum.rank(axis=1, pct=True) - self.volatility_penalty * volatility.rank(
            axis=1, pct=True
        )
        trend = close.rolling(
            self.trend_window, min_periods=max(120, self.trend_window * 9 // 10)
        ).mean()
        value_20 = value.rolling(20, min_periods=15).mean()
        core_close = pd.to_numeric(core["close"], errors="coerce").reindex(dates)
        core_regime = core_close.rolling(self.regime_window).mean()
        if self.regime_mode == "low_turnover":
            low_turnover = K200LowTurnoverReentry(ticker=self.core_ticker)
            state_history = low_turnover.compute_state_history(core)
            next_session_risk_on = state_history["state"].eq(low_turnover.RISK_ON)
        else:
            next_session_risk_on = pd.Series(False, index=dates)
            next_session_risk_on.iloc[1:] = (
                core_close.iloc[:-1].to_numpy() > core_regime.iloc[:-1].to_numpy()
            )

        current = {self.CASH: 1.0}
        current_risk_on = False
        rows: list[dict] = []
        selection_rows: list[dict] = []
        for position, signal_date in enumerate(dates):
            next_is_rebalance = False
            if position + 1 < len(dates):
                execution_date = dates[position + 1]
                next_is_rebalance = (
                    execution_date.month != signal_date.month
                    and execution_date.month in self.rebalance_months
                )
            if position + 1 < len(dates):
                risk_on = bool(next_session_risk_on.reindex(dates).iloc[position + 1])
            else:
                risk_on = current_risk_on
            regime_changed = risk_on != current_risk_on
            if next_is_rebalance or regime_changed:
                selected: list[str] = []
                if risk_on:
                    eligible = (
                        close.loc[signal_date].between(self.min_price, self.max_price)
                        & value_20.loc[signal_date].ge(self.min_value_20)
                        & close.loc[signal_date].gt(trend.loc[signal_date])
                        & score.loc[signal_date].notna()
                    )
                    ranking = pd.DataFrame(
                        {
                            "ticker": score.columns[eligible],
                            "score": score.loc[signal_date, eligible].to_numpy(),
                        }
                    ).sort_values(
                        ["score", "ticker"], ascending=[False, True], kind="mergesort"
                    )
                    selected = ranking.head(self.top_n)["ticker"].tolist()

                current = {}
                if risk_on and selected:
                    current[self.core_ticker] = self.core_weight
                    per_name = self.satellite_weight / len(selected)
                    current.update({ticker: per_name for ticker in selected})
                current[self.CASH] = max(1.0 - sum(current.values()), 0.0)
                selection_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": dates[position + 1],
                        "risk_on": risk_on,
                        "reason": (
                            "regime_change" if regime_changed else "scheduled_rebalance"
                        ),
                        "selected": ",".join(selected),
                    }
                )
                current_risk_on = risk_on
            rows.append({"date": signal_date, **current})

        targets = pd.DataFrame(rows).set_index("date").fillna(0.0)
        targets = targets.reindex(columns=sorted(targets.columns)).fillna(0.0)
        self.latest_selection_history = pd.DataFrame(selection_rows)
        return targets

    def run_backtest(
        self,
        enriched: dict,
        market_index=None,
        weights=None,
        silent: bool = False,
    ) -> tuple[pd.DataFrame, list[dict]]:
        from backtest_live_execution import simulate

        targets = self.compute_security_targets(
            enriched,
            market_index=market_index,
            silent=silent,
        )
        if targets.empty:
            return pd.DataFrame(), []
        tax_rates = {ticker: TAX_RATE_SELL for ticker in enriched}
        tax_rates[self.core_ticker] = 0.0
        return simulate(
            targets,
            enriched,
            initial_cash=self.initial_cash,
            min_trade=self.min_trade,
            price_band_pct=self.price_band_pct,
            sell_tax_rate_by_ticker=tax_rates,
            rebalance_only_on_target_change=True,
            distribution_events_by_ticker=self.distribution_events_by_ticker,
        )
