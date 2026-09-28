#!/usr/bin/env python3
"""Bounded Actions entry point, read-only KIS APIs; no trading imports."""
from datetime import datetime,timedelta
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from market_data_store import MarketStore,atomic_bytes,canonical,day
from kis_market_collection import KISPriceClient,collect_masters,collect_prices,read_master,select_kis


def main():
    store=MarketStore(os.environ['MARKET_STORE_PATH'])
    mode=os.environ.get('COLLECTION_MODE','incremental')
    today=datetime.now(ZoneInfo('Asia/Seoul')).date()
    observed=today.isoformat()
    if mode=='select':
        as_of=day(os.environ['SELECTION_DATE'])
        result=select_kis(store,as_of,as_of)
        atomic_bytes(store.root.parent/'signals'/f'{as_of}.json',canonical(result))
        return
    if mode not in ('master','backfill','incremental'):
        raise ValueError('Invalid collection mode')
    pinned_master=os.environ.get('MASTER_DATE')
    master_date=pinned_master or observed
    if not pinned_master:
        master=collect_masters(store,observed_date=observed)
    else:
        master=read_master(store,master_date)
    if mode=='master':
        print(f'Master rows: {len(master)}; historical universe not certified')
        return
    if mode=='backfill':
        start=day(os.environ.get('BACKFILL_START') or '2019-01-01')
        end=day(os.environ.get('BACKFILL_END') or '2026-08-31')
    else:
        end=(today-timedelta(days=1)).isoformat()
        start=(today-timedelta(days=8)).isoformat()
    if end>=observed:
        raise ValueError('Only completed prior calendar days may be collected')
    client=KISPriceClient(os.environ.get('KIS_APP_KEY'),os.environ.get('KIS_APP_SECRET'))
    result=collect_prices(store,client,master,master_date,start,end,
        max_requests=int(os.environ.get('MAX_REQUESTS') or '300'),refresh=mode=='incremental')
    result.update(start=start,end=end,master_date=master_date,mode=mode,orders_enabled=False)
    atomic_bytes(store.root/'last_kis_collection.json',canonical(result))
    print({k:v for k,v in result.items() if k!='candidates'})


if __name__=='__main__':
    main()
