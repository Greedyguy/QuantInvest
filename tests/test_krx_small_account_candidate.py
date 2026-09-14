import numpy as np
import pandas as pd

from krx_small_account_candidate import (
    compute_krx_small_account_scores,
    select_krx_small_account_satellite,
)


def _constituents(count: int = 25, snapshot: str = "2020-03-31") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": [snapshot] * count,
            "ticker": [f"{index:06d}" for index in range(1, count + 1)],
            "name": [f"stock-{index}" for index in range(1, count + 1)],
            "weight_pct": [3.0] * count,
        }
    )


def _fundamentals(
    count: int = 25,
    snapshot: str = "2020-03-31",
    available: str = "2020-04-01",
) -> pd.DataFrame:
    rows = []
    for index in range(1, count + 1):
        rows.append(
            {
                "snapshot_date": snapshot,
                "available_date": available,
                "ticker": f"{index:06d}",
                "market": "KOSPI",
                "name": f"stock-{index}",
                "close": 50_000 + index,
                "market_cap": 1_000_000_000_000 + index,
                "trading_value": 10_000_000_000 + index,
                "bps": 40_000 + index * 100,
                "eps": 4_000 + index * 20,
                "per": 4.0 + index,
                "pbr": 0.5 + index / 20,
                "dps": 500,
                "dividend_yield": 1.0,
                "source": "KRX Data Marketplace test fixture",
            }
        )
    return pd.DataFrame(rows)


def _prices(count: int = 25) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2019-01-01", "2020-04-01")
    result = {}
    for index in range(1, count + 1):
        close = np.linspace(30_000 + index, 50_000 + index * 100, len(dates))
        result[f"{index:06d}"] = pd.DataFrame({"close": close}, index=dates)
    return result


def test_scoring_requires_aligned_observable_snapshots():
    scores, coverage = compute_krx_small_account_scores(
        _constituents(snapshot="2019-12-30"),
        _fundamentals(),
        _prices(),
        "2020-04-01",
    )

    assert scores.empty
    assert coverage["snapshots_aligned"] is False
    assert select_krx_small_account_satellite(scores, coverage) == []


def test_scoring_selects_four_only_after_coverage_gate():
    scores, coverage = compute_krx_small_account_scores(
        _constituents(), _fundamentals(), _prices(), "2020-04-01"
    )

    selected = select_krx_small_account_satellite(scores, coverage)

    assert coverage["snapshots_aligned"] is True
    assert coverage["fundamental_members"] == 25
    assert coverage["index_weight_coverage_pct"] == 75.0
    assert len(selected) == 4


def test_expensive_share_is_ineligible_for_fixed_small_account_sleeve():
    prices = _prices()
    prices["000025"].loc[:, "close"] = np.linspace(
        200_000, 300_000, len(prices["000025"])
    )
    scores, _ = compute_krx_small_account_scores(
        _constituents(), _fundamentals(), prices, "2020-04-01"
    )

    expensive = scores.set_index("ticker").loc["000025"]

    assert expensive["close"] > 250_000
    assert not bool(expensive["eligible"])


def test_coverage_gate_rejects_22_members_even_with_high_weight():
    scores, coverage = compute_krx_small_account_scores(
        _constituents(count=22),
        _fundamentals(count=22),
        _prices(count=22),
        "2020-04-01",
    )

    assert coverage["index_weight_coverage_pct"] == 66.0
    assert select_krx_small_account_satellite(scores, coverage) == []
