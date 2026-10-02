#!/usr/bin/env python3
"""Collect resumable market-wide snapshots or read an offline candidate universe."""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_data_store import GROUPS, KRXClient, MarketStore, atomic_bytes, canonical, day


def collect(store, client, start, end, *, refresh=False, max_requests=300, delay=1.):
    import pandas as pd
    if day(start)>day(end):
        raise ValueError('Inverted range')
    requested, reused = 0, 0
    for stamp in pd.date_range(start,end):
        date = day(stamp.strftime('%Y-%m-%d'))
        if stamp.weekday() >= 5:
            continue
        for group in GROUPS:
            if store.has(group,date) and not refresh:
                reused += 1
                continue
            if requested >= max_requests:
                return dict(status='checkpoint_budget_exhausted', requested=requested, reused=reused,
                            next_group=group, next_date=date, complete=False)
            payload = client.fetch(group,date)
            store.ingest(group,date,payload)
            requested += 1
            print(json.dumps({'group':group,'date':date,'source_rows':len(payload['OutBlock_1'])}),flush=True)
            if delay:
                time.sleep(delay)
    empties = [k for k,r in store.manifest['snapshots'].items()
               if day(start)<=r['date']<=day(end) and r['coverage']=='empty_unconfirmed']
    return dict(status='range_queried', requested=requested, reused=reused,
                empty_unconfirmed_dates=empties, complete=False,
                note='Query completion is not calendar/corporate-action/full-universe certification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    fetch=sub.add_parser('collect')
    fetch.add_argument('--start',required=True)
    fetch.add_argument('--end',required=True)
    fetch.add_argument('--refresh',action='store_true')
    fetch.add_argument('--max-requests',type=int,default=300)
    daily=sub.add_parser('daily')
    daily.add_argument('--overlap-days',type=int,default=14)
    select=sub.add_parser('select')
    select.add_argument('--as-of',required=True)
    select.add_argument('--output',type=Path,required=True)
    select.add_argument('--min-value',type=float,default=1e9)
    select.add_argument('--max-price',type=float)
    select.add_argument('--slot-budget',type=float)
    select.add_argument('--include-unverified-etfs',action='store_true')
    select.add_argument('--include-inverse',action='store_true')
    select.add_argument('--limit',type=int,default=100)
    sub.add_parser('audit')
    master=sub.add_parser('kis-master')
    master.add_argument('--observed-date',default=datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat())
    kis=sub.add_parser('kis-collect')
    kis.add_argument('--master-date',required=True)
    kis.add_argument('--start',required=True)
    kis.add_argument('--end',required=True)
    kis.add_argument('--max-requests',type=int,default=300)
    kis.add_argument('--refresh',action='store_true')
    kis.add_argument('--env-file',type=Path)
    kis.add_argument('--demo',action='store_true')
    ks=sub.add_parser('kis-select')
    ks.add_argument('--master-date',required=True)
    ks.add_argument('--as-of',required=True)
    ks.add_argument('--output',type=Path,required=True)
    ks.add_argument('--allow-historical-master',action='store_true')
    ks.add_argument('--include-unverified-etfs',action='store_true')
    ks.add_argument('--min-value',type=float,default=1e9)
    ks.add_argument('--limit',type=int,default=100)
    args=parser.parse_args()
    store=MarketStore(args.store)
    if args.command.startswith('kis-'):
        from kis_market_collection import collect_masters,read_master,collect_prices,KISPriceClient,select_kis
        if args.command=='kis-master':
            # A downloaded current master cannot be relabelled as a past snapshot.
            today=datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()
            if day(args.observed_date)!=today:
                raise ValueError('Current master observed date must be today in KST')
            frame=collect_masters(store,observed_date=today)
            print(json.dumps({'master_date':today,'securities':len(frame),
                'eligible':int(frame.collection_eligible.sum()),'historical_universe_certified':False}))
        elif args.command=='kis-collect':
            if args.env_file:
                from dotenv import dotenv_values
                values=dotenv_values(args.env_file)
            else:
                values=os.environ
            client=KISPriceClient(values.get('KIS_APP_KEY'),values.get('KIS_APP_SECRET'),demo=args.demo)
            master=read_master(store,args.master_date)
            result=collect_prices(store,client,master,args.master_date,args.start,args.end,
                max_requests=args.max_requests,refresh=args.refresh)
            atomic_bytes(store.root/'last_kis_collection.json',canonical(result))
            print(json.dumps(result,ensure_ascii=False))
        else:
            if args.limit<1 or args.min_value<0:
                raise ValueError('Invalid selection bounds')
            result=select_kis(store,args.master_date,args.as_of,
                allow_historical_master=args.allow_historical_master,
                include_etfs=args.include_unverified_etfs,min_value=args.min_value,limit=args.limit)
            atomic_bytes(args.output,canonical(result))
            print(json.dumps({'as_of':result['as_of'],'candidates':len(result['candidates']),'orders_enabled':False}))
    elif args.command in ('collect','daily'):
        client=KRXClient(os.environ.get('KRX_OPENAPI_KEY'))
        if args.command=='daily':
            # Previous KST calendar day only: do not assume today's API EOD is ready.
            end=datetime.now(ZoneInfo('Asia/Seoul')).date()-timedelta(days=1)
            start=end-timedelta(days=args.overlap_days)
            result=collect(store,client,start.isoformat(),end.isoformat(),refresh=True,max_requests=100)
        else:
            if not 1 <= args.max_requests <= 1000:
                raise ValueError('max-requests must be 1..1000')
            result=collect(store,client,day(args.start),day(args.end),refresh=args.refresh,max_requests=args.max_requests)
        atomic_bytes(store.root/'last_collection.json',canonical(result))
        print(json.dumps(result,ensure_ascii=False))
    elif args.command=='select':
        if args.limit<1 or args.min_value<0 or (args.slot_budget is not None and args.slot_budget<=0):
            raise ValueError('Invalid selection bounds')
        result=store.select_universe(args.as_of,min_value=args.min_value,max_price=args.max_price,
            budget=args.slot_budget,include_unverified_etfs=args.include_unverified_etfs,
            include_inverse=args.include_inverse,limit=args.limit)
        atomic_bytes(args.output,canonical(result))
        print(json.dumps({'as_of':result['as_of'],'candidates':len(result['candidates']),'orders_enabled':False}))
    else:
        for namespace in ('snapshots','seeds','masters','kis_segments','calendars','daily_inputs','quarantines','kis_indices'):
            for record in store.manifest.get(namespace,{}).values():
                store.verify_record(record)
        print(json.dumps({'verified_snapshots':len(store.manifest['snapshots']),
            'seed_imports':len(store.manifest['seeds']),
            'kis_segments':len(store.manifest.get('kis_segments',{})),
            'masters':len(store.manifest.get('masters',{})),
            'full_universe_certified':False,'orders_enabled':False}))


if __name__=='__main__':
    main()
