from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.report_k200_reentry_shadow import (
    build_shadow_payload,
    load_shadow_prices,
    write_immutable_shadow_record,
)


def _prices(end: str) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", end=end)
    close = 100.0 * np.cumprod(np.repeat(1.001, len(dates)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )


def _no_distributions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "record_date",
            "pay_date",
            "distribution_per_share",
            "taxable_per_share",
        ]
    )


def test_shadow_report_is_sanitized_and_never_plans_orders():
    prices = _prices("2026-09-10")

    payload = build_shadow_payload(
        prices,
        source_path=Path("069500_test.parquet"),
        distribution_events=_no_distributions(),
        generated_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )

    assert payload["execution_guard"] == "NO_ORDERS_SENT"
    assert set(payload["target_weights_effective_at_signal_open"]) == {
        "069500",
        "__CASH__",
    }
    assert "account" not in payload
    assert "orders" not in payload
    assert payload["source"]["signal"]["filename"] == "069500_test.parquet"
    assert payload["source"]["signal"]["price_basis"] == (
        "cash_distribution_adjusted"
    )
    assert payload["source"]["execution"]["price_basis"] == "actual_traded"
    assert payload["source"]["execution"]["provisional"] is True
    assert payload["paper_accounts"]["capital_authorized"] is False


def test_shadow_report_rejects_stale_input_from_eligible_evidence():
    payload = build_shadow_payload(
        _prices("2025-12-10"),
        source_path=Path("069500_stale.parquet"),
        distribution_events=_no_distributions(),
        generated_at=datetime(2027, 9, 15, tzinfo=timezone.utc),
    )

    assert payload["status"] == "stale_input_rejected"
    assert payload["data_freshness"]["fresh"] is False


def test_shadow_report_tracks_integer_share_candidate_and_benchmark_accounts():
    prices = _prices("2027-04-30")
    as_of = prices.index[-1]
    payload = build_shadow_payload(
        prices,
        source_path=Path("069500_paper.parquet"),
        distribution_events=_no_distributions(),
        generated_at=datetime(
            as_of.year, as_of.month, as_of.day, 7, tzinfo=timezone.utc
        ),
    )

    accounts = payload["paper_accounts"]
    assert payload["status"] == "eligible_observation"
    assert accounts["status"] == "collecting"
    assert isinstance(accounts["candidate"]["quantity"], int)
    assert isinstance(accounts["benchmark"]["quantity"], int)
    assert accounts["candidate"]["equity_krw"] == pytest.approx(
        accounts["candidate"]["cash_krw"]
        + accounts["candidate"]["quantity"] * float(prices["close"].iloc[-1])
    )
    assert accounts["comparison"]["subsequent_sessions"] > 0
    assert accounts["comparison"]["registered_gates"] is None
    assert accounts["capital_authorized"] is False


def test_shadow_report_uses_separate_actual_prices_for_integer_share_execution():
    signal_prices = _prices("2027-04-30")
    actual_prices = signal_prices * 2.0
    as_of = signal_prices.index[-1]

    payload = build_shadow_payload(
        signal_prices,
        source_path=Path("069500_adjusted.parquet"),
        execution_prices=actual_prices,
        execution_source_path=Path("069500_actual.parquet"),
        distribution_events=_no_distributions(),
        execution_price_authority="official_krx_actual_traded",
        generated_at=datetime(
            as_of.year, as_of.month, as_of.day, 7, tzinfo=timezone.utc
        ),
    )
    candidate = payload["paper_accounts"]["candidate"]

    assert payload["source"]["execution"]["provisional"] is False
    assert candidate["equity_krw"] == pytest.approx(
        candidate["cash_krw"]
        + candidate["quantity"] * float(actual_prices["close"].iloc[-1])
    )
    assert candidate["quantity"] < 2_100_000 / float(signal_prices["close"].iloc[-1])


def test_shadow_report_opens_registered_gates_only_after_252_subsequent_sessions():
    prices = _prices("2027-10-15")
    as_of = prices.index[-1]
    payload = build_shadow_payload(
        prices,
        source_path=Path("069500_one_year.parquet"),
        distribution_events=_no_distributions(),
        generated_at=datetime(
            as_of.year, as_of.month, as_of.day, 7, tzinfo=timezone.utc
        ),
    )

    comparison = payload["paper_accounts"]["comparison"]
    assert comparison["subsequent_sessions"] >= 252
    assert comparison["evidence_complete"] is True
    assert comparison["registered_gates"] is not None
    assert comparison["capital_authorized"] is False


def test_shadow_records_are_idempotent_but_conflicts_fail_closed(tmp_path):
    payload = {
        "signal_date": "2026-09-16",
        "generated_at_utc": "2026-09-16T07:10:00+00:00",
        "data_freshness": {"business_day_gap": 0},
        "evidence": "frozen",
    }
    path, created = write_immutable_shadow_record(payload, tmp_path)
    assert created is True

    rerun = dict(payload)
    rerun["generated_at_utc"] = "2026-09-16T08:10:00+00:00"
    rerun["data_freshness"] = {"business_day_gap": 1}
    same_path, created = write_immutable_shadow_record(rerun, tmp_path)
    assert same_path == path
    assert created is False

    conflict = dict(rerun)
    conflict["evidence"] = "revised"
    with pytest.raises(RuntimeError, match="immutable shadow record conflicts"):
        write_immutable_shadow_record(conflict, tmp_path)


def test_official_shadow_input_requires_strict_krx_actual_price_panel(tmp_path):
    path = tmp_path / "official.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "ticker": ["069500"],
            "open": [29_805],
            "high": [29_880],
            "low": [29_410],
            "close": [29_465],
            "source": ["KRX Data Marketplace screen 13103"],
            "price_basis": ["actual_traded"],
        }
    ).to_csv(path, index=False)

    result = load_shadow_prices(path, official_krx_input=True)

    assert result.index.tolist() == [pd.Timestamp("2020-01-02")]
    assert result.iloc[0]["close"] == 29_465
