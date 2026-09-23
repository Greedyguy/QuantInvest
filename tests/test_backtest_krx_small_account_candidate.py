import pandas as pd
import pytest

from scripts.backtest_krx_small_account_candidate import (
    assert_development_coverage,
    continuous_kodex200_targets,
    same_timing_kodex200_targets,
)


def test_continuous_benchmark_is_fully_invested_kodex200():
    dates = pd.bdate_range("2020-01-02", periods=3)

    target = continuous_kodex200_targets(dates)

    assert target["069500"].eq(1.0).all()
    assert target["__CASH__"].eq(0.0).all()


def test_same_timing_benchmark_keeps_cash_regime_and_risk_budget():
    dates = pd.bdate_range("2020-01-02", periods=2)
    candidate = pd.DataFrame(
        {
            "069500": [0.0, 0.4],
            "005930": [0.0, 0.55],
            "__CASH__": [1.0, 0.05],
        },
        index=dates,
    )

    target = same_timing_kodex200_targets(candidate)

    assert target.iloc[0].to_dict() == {"069500": 0.0, "__CASH__": 1.0}
    assert target.iloc[1].to_dict() == {"069500": 0.95, "__CASH__": 0.05}


def test_coverage_gate_rejects_misaligned_or_thin_snapshot():
    decisions = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-04-01", "2020-07-01"]),
            "risk_on": [True, True],
            "reason": ["krx_snapshot", "krx_snapshot"],
            "snapshots_aligned": [True, False],
            "fundamental_members": [23, 25],
            "index_weight_coverage_pct": [65.0, 70.0],
        }
    )

    with pytest.raises(RuntimeError, match="coverage gate failed"):
        assert_development_coverage(decisions)


def test_coverage_gate_accepts_exact_registered_boundaries():
    decisions = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-04-01"]),
            "risk_on": [True],
            "reason": ["krx_snapshot"],
            "snapshots_aligned": [True],
            "fundamental_members": [23],
            "index_weight_coverage_pct": [65.0],
        }
    )

    assert_development_coverage(decisions)
