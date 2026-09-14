import pandas as pd
import pytest

from scripts.backtest_dart_quality_value_candidate import (
    assert_development_coverage,
    same_timing_kodex200_targets,
)


def test_same_timing_benchmark_replaces_stock_sleeve_with_kodex200():
    dates = pd.bdate_range("2020-01-02", periods=3)
    targets = pd.DataFrame(
        {
            "069500": [0.0, 0.4, 0.4],
            "005930": [0.0, 0.55, 0.55],
            "__CASH__": [1.0, 0.05, 0.05],
        },
        index=dates,
    )

    benchmark = same_timing_kodex200_targets(targets)

    assert benchmark.iloc[0].to_dict() == {"069500": 0.0, "__CASH__": 1.0}
    assert benchmark.iloc[1].to_dict() == {"069500": 0.95, "__CASH__": 0.05}


def test_development_coverage_gate_rejects_incomplete_panel():
    decisions = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2019-06-28"]),
            "risk_on": [True],
            "reason": ["quarterly_snapshot"],
            "fundamental_members": [17.0],
            "index_weight_coverage_pct": [54.9],
        }
    )

    with pytest.raises(RuntimeError, match="coverage gate failed"):
        assert_development_coverage(decisions)


def test_development_coverage_gate_accepts_pre_registered_thresholds():
    decisions = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2019-06-28"]),
            "risk_on": [True],
            "reason": ["quarterly_snapshot"],
            "fundamental_members": [18.0],
            "index_weight_coverage_pct": [55.0],
        }
    )

    assert_development_coverage(decisions)
