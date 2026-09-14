import numpy as np
import pandas as pd
import pytest

from strategies import get_strategy
from strategies.k200_relative_strength_satellite import K200RelativeStrengthSatellite


def _synthetic_universe(periods: int = 430) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    universe = {}
    core_close = 30_000.0 * np.exp(np.arange(periods) * 0.0005)
    universe["069500"] = pd.DataFrame(
        {
            "open": core_close,
            "close": core_close,
            "value": 100_000_000_000.0,
        },
        index=dates,
    )
    for rank in range(1, 9):
        ticker = f"{rank * 10:06d}"
        close = (20_000.0 + rank * 100.0) * np.exp(
            np.arange(periods) * (0.0002 + rank * 0.00008)
        )
        universe[ticker] = pd.DataFrame(
            {
                "open": close,
                "close": close,
                "value": 30_000_000_000.0,
            },
            index=dates,
        )
    return universe


def test_relative_strength_targets_are_prefix_invariant():
    universe = _synthetic_universe()
    strategy = K200RelativeStrengthSatellite()
    prefix_universe = {ticker: frame.iloc[:370] for ticker, frame in universe.items()}

    prefix = strategy.compute_security_targets(prefix_universe)
    full = strategy.compute_security_targets(universe)

    pd.testing.assert_frame_equal(
        prefix,
        full.loc[prefix.index].reindex(columns=prefix.columns),
    )


def test_relative_strength_candidate_uses_core_six_stocks_and_cash_buffer():
    strategy = K200RelativeStrengthSatellite()
    targets = strategy.compute_security_targets(_synthetic_universe())
    invested = targets.loc[targets["069500"].gt(0)].iloc[-1]

    stock_weights = invested.drop(["069500", "__CASH__"])
    assert stock_weights.gt(0).sum() == 6
    assert invested["069500"] == 0.40
    assert invested["__CASH__"] == pytest.approx(0.05)
    assert invested.sum() == pytest.approx(1.0)


def test_relative_strength_candidate_is_registered():
    assert isinstance(
        get_strategy("k200_relative_strength_satellite"),
        K200RelativeStrengthSatellite,
    )
