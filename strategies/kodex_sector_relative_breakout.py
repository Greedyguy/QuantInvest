"""KODEX sector sleeve that requires strength over the market core."""

from __future__ import annotations

import pandas as pd

from strategies.kodex_sector_rotation import KodexSectorRotation


class KodexSectorRelativeBreakout(KodexSectorRotation):
    """Use sector ETFs only when they beat KODEX 200 at 26 and 52 weeks.

    Risk-on capital falls back to a 95% KODEX 200 position when no sector
    passes both horizons.  This avoids forcing a relative bet among uniformly
    weak sectors while retaining the frozen low-turnover risk state.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.require_core_outperformance = True
        self.fallback_to_core = True

    def _scores(self) -> pd.DataFrame:
        if self.total_return_indices is None or self.total_return_indices.empty:
            return pd.DataFrame()
        values = self.total_return_indices.copy().astype(float).sort_index()
        values.index = pd.to_datetime(values.index)
        if self.core_ticker not in values:
            return pd.DataFrame()
        sectors = values.drop(columns=self.core_ticker)
        sector_short = sectors / sectors.shift(self.short_momentum_weeks) - 1.0
        sector_long = sectors / sectors.shift(self.long_momentum_weeks) - 1.0
        core = values[self.core_ticker]
        core_short = core / core.shift(self.short_momentum_weeks) - 1.0
        core_long = core / core.shift(self.long_momentum_weeks) - 1.0
        short_excess = sector_short.sub(core_short, axis=0)
        long_excess = sector_long.sub(core_long, axis=0)
        score = 0.5 * short_excess + 0.5 * long_excess
        return score.where(short_excess.gt(0.0) & long_excess.gt(0.0))

    def compute_security_targets(self, enriched, *args, **kwargs) -> pd.DataFrame:
        targets = super().compute_security_targets(enriched, *args, **kwargs)
        if targets.empty or self.latest_selection_history.empty:
            return targets
        selections = self.latest_selection_history.reset_index(drop=True)
        for position, selection in selections.iterrows():
            if not bool(selection["risk_on"]) or str(selection["selected"]):
                continue
            start = pd.Timestamp(selection["signal_date"])
            end = (
                pd.Timestamp(selections.iloc[position + 1]["signal_date"])
                if position + 1 < len(selections)
                else None
            )
            active = targets.index >= start
            if end is not None:
                active &= targets.index < end
            targets.loc[active, :] = 0.0
            targets.loc[active, self.core_ticker] = (
                self.core_weight + self.satellite_weight
            )
            targets.loc[active, self.CASH] = 1.0 - (
                self.core_weight + self.satellite_weight
            )
        return targets

    def get_name(self) -> str:
        return "kodex_sector_relative_breakout"

    def get_description(self) -> str:
        return (
            "KODEX 200 core with a top-two sector sleeve only when both "
            "6/12-month returns beat KODEX 200"
        )
