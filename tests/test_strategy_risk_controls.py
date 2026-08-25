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
