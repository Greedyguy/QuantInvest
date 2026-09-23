import numpy as np
import pandas as pd

from scripts.backtest_kodex200_bear_participation import (
    BEAR,
    CASH,
    CORE_TICKER,
    INVERSE_TICKER,
    RISK_ON,
    _bear_confirmed,
    apply_bear_participation,
    targets_from_bear_history,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


def test_bear_confirmation_is_strictly_below_both_thresholds():
    assert _bear_confirmed(99.0, 100.0, -0.01)
    assert not _bear_confirmed(100.0, 100.0, -0.01)
    assert not _bear_confirmed(99.0, 100.0, 0.0)


def test_bear_overlay_is_prefix_invariant_and_reviewed_monthly():
    dates = pd.bdate_range("2021-01-04", periods=100)
    prices = pd.DataFrame(
        {"close": 100.0 * np.cumprod(np.repeat(0.999, len(dates)))},
        index=dates,
    )
    states = pd.DataFrame(
        {"state": K200LowTurnoverReentry.CASH}, index=dates
    )

    full = apply_bear_participation(prices, states)
    prefix = apply_bear_participation(prices.iloc[:80], states.iloc[:80])

    pd.testing.assert_frame_equal(full.iloc[:80], prefix)
    reviewed = full.index[full["reviewed"]]
    assert all(date.month != dates[dates.get_loc(date) - 1].month for date in reviewed)


def test_base_state_transition_is_reviewed_and_risk_on_always_wins():
    dates = pd.bdate_range("2021-01-04", periods=80)
    prices = pd.DataFrame(
        {"close": 100.0 * np.cumprod(np.repeat(0.999, len(dates)))},
        index=dates,
    )
    states = pd.DataFrame(
        {"state": K200LowTurnoverReentry.RISK_ON}, index=dates
    )
    states.loc[dates[-3]:, "state"] = K200LowTurnoverReentry.CASH
    states.loc[dates[-1], "state"] = K200LowTurnoverReentry.RISK_ON

    result = apply_bear_participation(prices, states)

    assert result.loc[dates[-3], "reviewed"]
    assert result.loc[dates[-3], "tier"] == BEAR
    assert result.loc[dates[-1], "tier"] == RISK_ON


def test_bear_targets_shift_effective_open_tier_to_prior_close():
    dates = pd.bdate_range("2022-01-03", periods=4)
    budget = pd.DataFrame(
        {
            "state": [
                K200LowTurnoverReentry.CASH,
                K200LowTurnoverReentry.CASH,
                K200LowTurnoverReentry.RISK_ON,
                K200LowTurnoverReentry.CASH,
            ],
            "tier": [CASH, BEAR, RISK_ON, CASH],
        },
        index=dates,
    )

    candidate, same, continuous = targets_from_bear_history(budget)

    assert candidate.loc[dates[0], [INVERSE_TICKER, "__CASH__"]].tolist() == [
        0.30,
        0.70,
    ]
    assert candidate.loc[dates[1], [CORE_TICKER, "__CASH__"]].tolist() == [
        0.95,
        0.05,
    ]
    assert candidate.loc[dates[2], "__CASH__"] == 1.0
    assert same.loc[dates[0], "__CASH__"] == 1.0
    assert same.loc[dates[1], CORE_TICKER] == 0.95
    assert continuous[CORE_TICKER].eq(1.0).all()
