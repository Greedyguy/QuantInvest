#!/usr/bin/env python3
"""Prepare offline backtest inputs and a private quality report. No API calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest_market_inputs import prepare_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--start', default='2025-01-01')
    p.add_argument('--end', default='2026-08-31')
    p.add_argument('--warmup-start', default='2024-01-01')
    p.add_argument('--master-date', default='2026-09-28')
    p.add_argument('--min-history', type=int, default=120)
    p.add_argument('--stocks-only', action='store_true')
    p.add_argument('--data-commit', required=True)
    p.add_argument('--code-commit', default='local')
    args = p.parse_args()
    report = prepare_inputs(args.store, args.output, start=args.start, end=args.end,
        warmup_start=args.warmup_start, master_date=args.master_date, min_history=args.min_history,
        include_etfs=not args.stocks_only, data_commit=args.data_commit, code_commit=args.code_commit)
    print({key: report[key] for key in ('status', 'target_tickers', 'verified_segments',
          'raw_rows', 'adjusted_rows', 'issue_counts', 'bundle_exported')})
    # Quality failures are reported, not disguised as infrastructure exceptions.
    # The workflow publishes this report before explicitly gating its conclusion.


if __name__ == '__main__':
    main()
