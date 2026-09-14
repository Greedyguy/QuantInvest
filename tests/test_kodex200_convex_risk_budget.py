import pandas as pd

from scripts.backtest_kodex200_convex_risk_budget import (
    CORE_TICKER,
    LEVERAGE_TICKER,
    registered_gates,
    targets_from_state_history,
)
from strategies.k200_low_turnover_reentry import K200LowTurnoverReentry


def test_targets_execute_next_open_state_without_daily_rebalancing():
    dates = pd.bdate_range("2022-01-03", periods=4)
    states = pd.DataFrame(
        {
            "state": [
                K200LowTurnoverReentry.CASH,
                K200LowTurnoverReentry.RISK_ON,
                K200LowTurnoverReentry.RISK_ON,
                K200LowTurnoverReentry.CASH,
            ]
        },
        index=dates,
    )

    candidate, same, continuous = targets_from_state_history(states)

    assert candidate.loc[dates[0], [CORE_TICKER, LEVERAGE_TICKER]].tolist() == [
        0.70,
        0.25,
    ]
    assert candidate.loc[dates[1]].equals(candidate.loc[dates[0]])
    assert candidate.loc[dates[2], "__CASH__"] == 1.0
    assert same.loc[dates[0], CORE_TICKER] == 0.95
    assert continuous[CORE_TICKER].eq(1.0).all()


def test_registered_gates_do_not_add_continuous_mdd_requirement():
    continuous = {
        "annualised_excess_return_pct_point": 2.1,
        "rolling_12m_beat_rate_pct": 61.0,
        "mdd_disadvantage_pct_point": 99.0,
        "positive_excess_year_concentration_pct": 59.0,
    }
    same = {
        "annualised_excess_return_pct_point": 1.6,
        "rolling_12m_beat_rate_pct": 60.0,
        "mdd_disadvantage_pct_point": 5.0,
        "positive_excess_year_concentration_pct": 60.0,
    }

    gates = registered_gates(continuous, same)

    assert all(gates.values())
    assert "mdd_disadvantage_vs_continuous" not in gates
