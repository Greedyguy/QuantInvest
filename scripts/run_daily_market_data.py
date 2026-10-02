#!/usr/bin/env python3
"""Daily read-only KIS collection and full-universe price input preparation."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from market_data_store import MarketStore
from kis_market_collection import KISPriceClient
from daily_market_data import run_daily


def main():
    store=MarketStore(os.environ['MARKET_STORE_PATH'])
    client=KISPriceClient(os.environ.get('KIS_APP_KEY'),os.environ.get('KIS_APP_SECRET'))
    result=run_daily(store,client,start=os.environ.get('DAILY_START') or '2026-09-01',
        max_requests=int(os.environ.get('MAX_REQUESTS') or '10000'),
        code_commit=os.environ.get('DAILY_CODE_SHA','local'))
    # Public Actions logs expose counts/status only, not private price rows.
    print(json.dumps({k:result.get(k) for k in ('status','decision_date','price_date',
        'target_tickers','requested','reused','stock_candidate_count','orders_enabled')}))


if __name__=='__main__':
    main()
