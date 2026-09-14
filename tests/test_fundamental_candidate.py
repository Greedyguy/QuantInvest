import numpy as np
import pandas as pd

from fundamental_candidate import (
    compute_dart_quality_value_scores,
    select_dart_quality_value_satellite,
)


def _constituents(count=20):
    return pd.DataFrame(
        {
            "as_of_date": ["2020-03-30"] * count,
            "ticker": [f"{index:06d}" for index in range(1, count + 1)],
            "name": [f"stock-{index}" for index in range(1, count + 1)],
            "weight_pct": [3.0] * count,
        }
    )


def _fundamentals(count=20, future=False):
    rows = []
    for index in range(1, count + 1):
        scale = float(index)
        rows.append(
            {
                "ticker": f"{index:06d}",
                "period_end": "2019-12-31",
                "receipt_date": "2020-03-29" if not future else "2020-04-01",
                "available_date": "2020-03-30" if not future else "2020-04-02",
                "receipt_no": f"report-{index}",
                "assets": 1_000_000_000 * scale,
                "previous_assets": 900_000_000 * scale,
                "liabilities": 400_000_000 * scale,
                "previous_liabilities": 380_000_000 * scale,
                "equity": 600_000_000 * scale,
                "previous_equity": 520_000_000 * scale,
                "revenue": 800_000_000 * scale,
                "previous_revenue": 700_000_000 * scale,
                "operating_income": (60_000_000 + index * 2_000_000) * scale,
                "previous_operating_income": 50_000_000 * scale,
                "net_income": (40_000_000 + index * 1_000_000) * scale,
                "previous_net_income": 35_000_000 * scale,
                "cash_flow_from_operations": (45_000_000 + index * 1_500_000) * scale,
                "previous_cash_flow_from_operations": 40_000_000 * scale,
                "ordinary_issued_shares": 1_000_000 * scale,
            }
        )
    return pd.DataFrame(rows)


def _prices(count=20, periods=300):
    dates = pd.bdate_range("2019-02-01", periods=periods)
    prices = {}
    for index in range(1, count + 1):
        close = np.linspace(400 + index, 600 + index * 3, periods)
        prices[f"{index:06d}"] = pd.DataFrame(
            {"open": close, "close": close}, index=dates
        )
    return prices


def test_scoring_hides_filings_that_were_not_yet_available():
    scores, coverage = compute_dart_quality_value_scores(
        _constituents(), _fundamentals(future=True), _prices(), "2020-03-30"
    )

    assert scores.empty
    assert coverage["fundamental_members"] == 0


def test_scoring_is_prefix_invariant_and_selects_only_after_coverage_gate():
    prices = _prices()
    signal = pd.Timestamp("2020-03-30")
    prefix = {
        ticker: frame.loc[frame.index <= signal]
        for ticker, frame in prices.items()
    }
    prefix_scores, prefix_coverage = compute_dart_quality_value_scores(
        _constituents(), _fundamentals(), prefix, signal
    )
    full_scores, full_coverage = compute_dart_quality_value_scores(
        _constituents(), _fundamentals(), prices, signal
    )

    columns = ["ticker", "eligible", "composite_score"]
    pd.testing.assert_frame_equal(
        prefix_scores[columns], full_scores[columns], check_exact=False
    )
    selected = select_dart_quality_value_satellite(prefix_scores, prefix_coverage)
    assert len(selected) == 4

    failed_coverage = dict(full_coverage, fundamental_members=17)
    assert select_dart_quality_value_satellite(full_scores, failed_coverage) == []


def test_split_mismatch_value_outlier_is_ineligible():
    fundamentals = _fundamentals()
    fundamentals.loc[0, "ordinary_issued_shares"] = 1.0
    scores, _ = compute_dart_quality_value_scores(
        _constituents(), fundamentals, _prices(), "2020-03-30"
    )

    row = scores.set_index("ticker").loc["000001"]
    assert not bool(row["value_plausible"])
    assert not bool(row["eligible"])
