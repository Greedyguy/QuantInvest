import numpy as np
import pandas as pd

from scripts.backtest_kodex200_volatility_budget import (
    CASH,
    CORE_TICKER,
    HIGH_LEVERAGE,
    LEVERAGE_TICKER,
    MEDIUM_LEVERAGE,
    UNLEVERED,
    _tier,
    apply_volatility_budget,
    targets_from_budget_history,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


def test_pre_registered_volatility_tiers_are_exact():
    assert _tier(0.10, 0.15) == HIGH_LEVERAGE
    assert _tier(0.10, 0.150001) == MEDIUM_LEVERAGE
    assert _tier(0.10, 0.22) == MEDIUM_LEVERAGE
    assert _tier(0.10, 0.220001) == UNLEVERED
    assert _tier(0.0, 0.10) == UNLEVERED


def test_overlay_is_prefix_invariant_and_changes_only_on_month_boundaries():
    dates = pd.bdate_range("2021-01-04", periods=100)
    close = 100.0 * np.cumprod(np.repeat(1.001, len(dates)))
    prices = pd.DataFrame({"close": close}, index=dates)
    states = pd.DataFrame(
        {"state": K200LowTurnoverReentry.RISK_ON}, index=dates
    )

    full = apply_volatility_budget(prices, states)
    prefix = apply_volatility_budget(prices.iloc[:80], states.iloc[:80])

    pd.testing.assert_frame_equal(full.iloc[:80], prefix)
    changed = full.index[full["tier"].ne(full["tier"].shift())]
    assert all(
        date == dates[0] or date.month != dates[dates.get_loc(date) - 1].month
        for date in changed
    )
    reviewed = full.index[full["reviewed"]]
    assert all(date.month != dates[dates.get_loc(date) - 1].month for date in reviewed)


def test_emergency_cash_exit_overrides_monthly_leverage_tier():
    dates = pd.bdate_range("2021-01-04", periods=80)
    prices = pd.DataFrame(
        {"close": 100.0 * np.cumprod(np.repeat(1.001, len(dates)))},
        index=dates,
    )
    states = pd.DataFrame(
        {"state": K200LowTurnoverReentry.RISK_ON}, index=dates
    )
    states.loc[dates[-2]:, "state"] = K200LowTurnoverReentry.CASH

    result = apply_volatility_budget(prices, states)

    assert result.loc[dates[-2], "tier"] == CASH
    assert result.loc[dates[-1], "tier"] == CASH


def test_budget_targets_shift_effective_open_tier_to_prior_close():
    dates = pd.bdate_range("2022-01-03", periods=3)
    budget = pd.DataFrame(
        {
            "state": [
                K200LowTurnoverReentry.CASH,
                K200LowTurnoverReentry.RISK_ON,
                K200LowTurnoverReentry.RISK_ON,
            ],
            "tier": [CASH, HIGH_LEVERAGE, MEDIUM_LEVERAGE],
        },
        index=dates,
    )

    candidate, same, _ = targets_from_budget_history(budget)

    assert candidate.loc[dates[0], [CORE_TICKER, LEVERAGE_TICKER]].tolist() == [
        0.60,
        0.35,
    ]
    assert candidate.loc[dates[1], [CORE_TICKER, LEVERAGE_TICKER]].tolist() == [
        0.75,
        0.20,
    ]
    assert same.loc[dates[0], CORE_TICKER] == 0.95
