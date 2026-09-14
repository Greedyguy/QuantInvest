"""Low-turnover KODEX core plus domestic sector-ETF relative strength."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


class KodexSectorRotation(BaseStrategy):
    """Combine the defensive KODEX 200 state with two leading sector ETFs.

    The sector score is fixed at the average of 26-week and 52-week total
    returns.  Signals are observed before the first trading session of each
    month and orders execute on that next session.  No leveraged or inverse ETF
    is used.
    """

    CASH = "__CASH__"

    def __init__(
        self,
        total_return_indices: pd.DataFrame | None = None,
        signal_prices: pd.DataFrame | None = None,
        distribution_events_by_ticker: Mapping[str, pd.DataFrame] | None = None,
        core_ticker: str = "069500",
        core_weight: float = 0.40,
        satellite_weight: float = 0.55,
        top_n: int = 2,
        short_momentum_weeks: int = 26,
        long_momentum_weeks: int = 52,
        initial_cash: float = 2_100_000.0,
        min_trade: int = 50_000,
        price_band_pct: float = 3.0,
    ):
        super().__init__()
        if core_weight < 0 or satellite_weight < 0:
            raise ValueError("portfolio weights must be non-negative")
        if core_weight + satellite_weight > 1.0:
            raise ValueError("core and satellite weights cannot exceed 100%")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.total_return_indices = (
            total_return_indices.copy()
            if total_return_indices is not None
            else None
        )
        self.signal_prices = signal_prices.copy() if signal_prices is not None else None
        self.distribution_events_by_ticker = dict(
            distribution_events_by_ticker or {}
        )
        self.core_ticker = str(core_ticker)
        self.core_weight = float(core_weight)
        self.satellite_weight = float(satellite_weight)
        self.top_n = int(top_n)
        self.short_momentum_weeks = int(short_momentum_weeks)
        self.long_momentum_weeks = int(long_momentum_weeks)
        self.initial_cash = float(initial_cash)
        self.min_trade = int(min_trade)
        self.price_band_pct = float(price_band_pct)
        self.latest_selection_history = pd.DataFrame()

    def get_name(self) -> str:
        return "kodex_sector_rotation"

    def get_description(self) -> str:
        return (
            "Low-turnover KODEX 200 core + top-two domestic sector ETF "
            "6/12-month relative strength"
        )

    def _scores(self) -> pd.DataFrame:
        if self.total_return_indices is None or self.total_return_indices.empty:
            return pd.DataFrame()
        values = self.total_return_indices.copy().astype(float).sort_index()
        values.index = pd.to_datetime(values.index)
        sectors = values.drop(columns=self.core_ticker, errors="ignore")
        short = sectors / sectors.shift(self.short_momentum_weeks) - 1.0
        long = sectors / sectors.shift(self.long_momentum_weeks) - 1.0
        return 0.5 * short + 0.5 * long

    def compute_security_targets(
        self,
        enriched: dict[str, pd.DataFrame],
        market_index=None,
        secondary_index=None,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        execution_core = enriched.get(self.core_ticker)
        signal_core = self.signal_prices if self.signal_prices is not None else execution_core
        scores = self._scores()
        if (
            execution_core is None
            or execution_core.empty
            or signal_core is None
            or signal_core.empty
            or scores.empty
        ):
            return pd.DataFrame()

        dates = pd.DatetimeIndex(pd.to_datetime(execution_core.index)).sort_values().unique()
        signal_core = signal_core.copy()
        signal_core.index = pd.to_datetime(signal_core.index)
        signal_core = signal_core.sort_index().reindex(dates)
        risk_model = K200LowTurnoverReentry(ticker=self.core_ticker)
        state_history = risk_model.compute_state_history(signal_core)
        risk_next = state_history["state"].eq(risk_model.RISK_ON).reindex(
            dates, fill_value=False
        )

        current = {self.CASH: 1.0}
        current_risk_on = False
        selected: list[str] = []
        rows: list[dict] = []
        selections: list[dict] = []
        for position, signal_date in enumerate(dates):
            if position + 1 >= len(dates):
                rows.append({"date": signal_date, **current})
                continue
            execution_date = dates[position + 1]
            first_session_next_month = execution_date.month != signal_date.month
            risk_on = bool(risk_next.iloc[position + 1])
            regime_changed = risk_on != current_risk_on
            if first_session_next_month or regime_changed:
                selected = []
                if risk_on:
                    observable_scores = scores.loc[
                        scores.index <= signal_date
                    ].tail(1)
                    if not observable_scores.empty:
                        available = [
                            ticker
                            for ticker in observable_scores.columns
                            if ticker in enriched
                        ]
                        selected = (
                            observable_scores.iloc[0]
                            .reindex(available)
                            .dropna()
                            .sort_values(ascending=False, kind="mergesort")
                            .head(self.top_n)
                            .index.tolist()
                        )
                current = {}
                if risk_on and selected:
                    current[self.core_ticker] = self.core_weight
                    per_sector = self.satellite_weight / len(selected)
                    current.update({ticker: per_sector for ticker in selected})
                current[self.CASH] = max(1.0 - sum(current.values()), 0.0)
                selections.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "risk_on": risk_on,
                        "reason": (
                            "regime_change" if regime_changed else "monthly_rebalance"
                        ),
                        "selected": ",".join(selected),
                    }
                )
                current_risk_on = risk_on
            rows.append({"date": signal_date, **current})

        targets = pd.DataFrame(rows).set_index("date").fillna(0.0)
        targets = targets.reindex(columns=sorted(targets.columns)).fillna(0.0)
        self.latest_selection_history = pd.DataFrame(selections)
        return targets

    def run_backtest(
        self,
        enriched: dict,
        market_index=None,
        weights=None,
        silent: bool = False,
    ) -> tuple[pd.DataFrame, list[dict]]:
        from backtest_live_execution import simulate

        targets = self.compute_security_targets(enriched, silent=silent)
        if targets.empty:
            return pd.DataFrame(), []
        return simulate(
            targets,
            enriched,
            initial_cash=self.initial_cash,
            min_trade=self.min_trade,
            price_band_pct=self.price_band_pct,
            sell_tax_rate_by_ticker={ticker: 0.0 for ticker in enriched},
            rebalance_only_on_target_change=True,
            distribution_events_by_ticker=self.distribution_events_by_ticker,
        )
