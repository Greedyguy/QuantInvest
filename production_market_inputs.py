"""Offline, fail-closed bridge from private KIS data to existing KR strategies.

No provider requests, broker connection, price filling, ETF index proxies or ranking.
Current-master coverage is not historical survivorship certification.
"""
import re
import pandas as pd
from market_data_store import MarketStore,DataQualityError,day,digest
from daily_market_data import load_daily_selection_inputs
from backtest_market_inputs import _verified_frame,interval_gaps
from kis_index_store import load_indices
from signals import compute_indicators,add_rel_strength
from config import BLOCKED_TICKERS

INDEX_ETFS={'069500','102110','114800','122630','229200','091160','091180','152100'}


def legacy_name_allowed(row):
    # Exactly the existing get_universe name screen, using the stored master.
    return row.ticker in INDEX_ETFS or not any(k in row.name.upper() for k in ('우','리츠','스팩','SPAC'))


def load_production_inputs(store_path,*,start,data_commit,now=None):
    if not re.fullmatch('[a-f0-9]{40}',data_commit or ''):
        raise DataQualityError('Pin the private data repository commit')
    store=MarketStore(store_path)
    daily=load_daily_selection_inputs(store,now=now,include_unverified_etfs=True)
    start,end=day(start),daily.attrs['price_date']
    indices=load_indices(store,start,end)
    sessions=indices['KOSPI'].index
    if len(sessions)<125:
        raise DataQualityError('Insufficient index warmup')
    wanted=set(daily.ticker)
    records={}
    for r in store.manifest['kis_segments'].values():
        if r['ticker'] in wanted and r['end']>=start and r['start']<=end:
            records.setdefault((r['ticker'],r['price_basis']),[]).append(r)
    enriched,raw_refs,excluded={}, {}, []
    for row in daily.itertuples(index=False):
        if row.ticker in BLOCKED_TICKERS or not legacy_name_allowed(row):
            excluded.append(dict(ticker=row.ticker,reason='existing_universe_rule'))
            continue
        if not row.tradable:
            excluded.append(dict(ticker=row.ticker,reason=getattr(row,'price_status','')
                if getattr(row,'price_status','')=='confirmed_delisted' else 'confirmed_nontrading_on_price_date'))
            continue
        first=max(start,row.listed_date)
        expected=sessions[sessions>=pd.Timestamp(first)]
        if len(expected)<120:
            excluded.append(dict(ticker=row.ticker,reason='listing_history_below_120_sessions'))
            continue
        frames={}
        for basis in ('raw','adjusted'):
            items=sorted(records.get((row.ticker,basis),[]),key=lambda r:r['collected_at'])
            if interval_gaps([(r['start'],r['end']) for r in items],first,end):
                raise DataQualityError(f'Uncollected strategy history: {row.ticker}/{basis}')
            parts=[_verified_frame(store,r,row.isin) for r in items]
            f=pd.concat(parts).drop_duplicates('date',keep='last').sort_values('date')
            f=f.loc[f.date.between(first,end)].set_index('date')
            f.index=pd.to_datetime(f.index)
            if not f.index.equals(expected):
                raise DataQualityError(f'Missing or unexpected strategy sessions: {row.ticker}/{basis}')
            latest=f.iloc[-1]
            for field in ('open','high','low','close','volume','value'):
                if float(latest[field])!=float(getattr(row,basis+'_'+field)):
                    raise DataQualityError(f'Daily/history revision mismatch: {row.ticker}/{basis}')
            frames[basis]=f
        # Suspension observations remain in storage. Do not fabricate their OHLC.
        f=frames['adjusted']
        valid=f[['open','high','low','close']].gt(0).all(axis=1)
        if int(valid.sum())<120:
            excluded.append(dict(ticker=row.ticker,reason='trading_history_below_120_rows'))
            continue
        f=compute_indicators(f.loc[valid,['open','high','low','close','volume','value']])
        market=row.market.upper()
        if market not in indices: raise DataQualityError('Unknown master market')
        f=add_rel_strength(f,indices[market])
        enriched[row.ticker]=f
        raw_refs[row.ticker]=float(frames['raw'].iloc[-1]['close'])
    if '069500' not in enriched:
        raise DataQualityError('KODEX 200 benchmark missing after input validation')
    provenance=dict(source='private_kis_store',data_commit=data_commit,
        manifest_sha256=digest(store.manifest_path.read_bytes()),price_date=end,
        decision_date=daily.attrs['decision_date'],input_table_sha256=daily.attrs['input_table_sha256'],
        universe_membership_sha256=daily.attrs['universe_membership_sha256'],
        full_current_input_count=len(daily),strategy_input_count=len(enriched),
        order_blocked_tickers=sorted(daily.loc[daily.price_status.eq('confirmed_delisted'),'ticker'])
            if 'price_status' in daily else [],
        exclusion_counts=pd.Series([r['reason'] for r in excluded],dtype=str).value_counts().to_dict(),
        signal_price_basis='adjusted',execution_price_basis='raw',index_source='kis_actual_indices',
        market_cap_policy='unavailable_no_synthetic_fill_existing_price_style_fallback',
        historical_survivorship_certified=False,etf_multiplier_classification='name_screen_only')
    return enriched,indices,raw_refs,provenance
