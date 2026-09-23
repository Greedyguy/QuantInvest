import numpy as np
import pandas as pd

from strategies.strategy_multi_allocator_plus import MultiStrategyAllocatorPlus


def test_performance_stress_is_prefix_invariant():
    strategy = MultiStrategyAllocatorPlus()
    dates = pd.bdate_range("2025-01-02", periods=220)
    exposures = pd.Series(0.82, index=dates)
    returns = pd.Series(
        np.r_[np.full(70, 0.001), np.full(70, -0.0025), np.full(80, 0.0015)],
        index=dates,
    )

    prefix_exposure, prefix_stress = strategy._performance_stress(
        exposures.iloc[:120], returns.iloc[:120]
    )
    full_exposure, full_stress = strategy._performance_stress(exposures, returns)

    pd.testing.assert_series_equal(prefix_exposure, full_exposure.iloc[:120])
    pd.testing.assert_series_equal(prefix_stress, full_stress.iloc[:120])


def test_stress_recovery_is_gradual_and_shock_resets_immediately():
    strategy = MultiStrategyAllocatorPlus()
    dates = pd.bdate_range("2025-01-02", periods=41)
    returns = pd.Series(
        np.r_[
            np.full(10, -0.006),
            [0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003],
            np.full(20, 0.002),
            -0.03,
        ],
        index=dates,
    )
    base = pd.Series(0.50, index=dates)
    stressed = pd.Series(strategy.exposure_floor + 0.03, index=dates)
    levels = pd.Series(2, index=dates)

    candidate, context = strategy._stress_recovery_candidate(
        base,
        stressed,
        levels,
        returns,
    )

    assert context["recovery_rung"].max() >= 2
    positive_steps = candidate.diff().dropna()
    assert positive_steps.max() <= strategy.stress_recovery_step + 1e-12
    assert context.iloc[-1]["shock"]
    assert context.iloc[-1]["recovery_rung"] == 0
    assert candidate.iloc[-1] == strategy.exposure_floor + 0.03


def test_recovery_candidate_is_prefix_invariant():
    strategy = MultiStrategyAllocatorPlus()
    dates = pd.bdate_range("2025-01-02", periods=80)
    returns = pd.Series(
        np.r_[np.full(20, -0.003), np.full(30, 0.004), np.full(30, -0.001)],
        index=dates,
    )
    base = pd.Series(0.50, index=dates)
    stressed = pd.Series(strategy.exposure_floor + 0.03, index=dates)
    levels = pd.Series(2, index=dates)

    prefix, prefix_context = strategy._stress_recovery_candidate(
        base.iloc[:55], stressed.iloc[:55], levels.iloc[:55], returns.iloc[:55]
    )
    full, full_context = strategy._stress_recovery_candidate(
        base, stressed, levels, returns
    )

    pd.testing.assert_series_equal(prefix, full.iloc[:55])
    pd.testing.assert_frame_equal(prefix_context, full_context.iloc[:55])


def test_recovery_overlay_never_reduces_baseline_exposure():
    strategy = MultiStrategyAllocatorPlus()
    dates = pd.bdate_range("2025-01-02", periods=30)
    base = pd.Series(1.12, index=dates)
    stressed = pd.Series(1.12, index=dates)
    levels = pd.Series(0, index=dates)
    returns = pd.Series(0.002, index=dates)

    candidate, _ = strategy._stress_recovery_candidate(
        base,
        stressed,
        levels,
        returns,
    )

    pd.testing.assert_series_equal(candidate, stressed)
