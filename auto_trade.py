#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple automated trading pipeline that
1. Loads data & runs the multi-strategy allocator
2. Extracts the latest target weights (for next session)
3. Compares with previously stored targets
4. Emits order suggestions and updates state

NOTE: This is a reference pipeline for automation. Integration with a real broker
should replace the mock execution section.
"""

import json
import os
from datetime import datetime

from reports import load_data, filter_universe
from strategies import get_strategy

STATE_FILE = "data/auto_last_weights.json"
THRESHOLD = 0.01  # minimum weight change to trigger order


def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("weights", {}), data.get("timestamp")
    return {}, None


def save_state(weights):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "weights": weights,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compare_weights(old, new):
    tickers = set(old.keys()) | set(new.keys())
    deltas = {}
    for t in tickers:
        prev = old.get(t, 0.0)
        cur = new.get(t, 0.0)
        if abs(cur - prev) >= THRESHOLD:
            deltas[t] = cur - prev
    return deltas


def main():
    enriched, idx_map = load_data(use_cache=True)
    filtered = filter_universe(enriched)
    enriched = {ticker: enriched[ticker] for ticker in filtered if ticker in enriched}
    kosdaq = idx_map.get("KOSDAQ")

    strategy = get_strategy("multi_allocator")
    ec, _ = strategy.run_backtest(enriched, market_index=kosdaq, silent=True)

    target_df = strategy.get_latest_target_weights()
    if target_df is None or target_df.empty:
        print("No target weights available (fallback mode).")
        return

    latest_row = target_df.iloc[-1]
    weights = latest_row.dropna().to_dict()

    print(f"Target date: {latest_row.name}")
    print(f"Target weights (before thresholding): {weights}")

    prev_weights, prev_ts = load_previous_state()
    if prev_weights:
        print(f"Previous snapshot timestamp: {prev_ts}")

    deltas = compare_weights(prev_weights, weights)
    if not deltas:
        print("No significant changes. No orders required.")
    else:
        print("Order suggestions (weight delta):")
        for ticker, delta in deltas.items():
            action = "BUY" if delta > 0 else "SELL"
            print(f"  {ticker}: {action} weight change {delta:+.2%}")

    save_state(weights)
    print("State updated. Automation step complete.")


if __name__ == "__main__":
    main()
