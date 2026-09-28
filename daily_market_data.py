"""Whole-current-universe daily data supply, independent of trading decisions.

The price cut-off is strictly BEFORE the KST execution date. Today-observed
masters are labelled today, never backdated to match yesterday's prices.
"""
from datetime import datetime, timezone
import json
from zoneinfo import ZoneInfo

import pandas as pd

from market_data_store import MarketStore, DataQualityError, atomic_bytes, canonical, day, digest
from kis_market_collection import collect_masters, collect_prices, load_kis_panel, PRICE_FIELDS

KST = ZoneInfo('Asia/Seoul')


def _utc_now():
    return datetime.now(timezone.utc)


def normalize_calendar(payload):
    rows=payload.get('output')
    if isinstance(rows,dict):
        rows=[rows]
    if payload.get('rt_cd')!='0' or not isinstance(rows,list) or not rows:
        raise DataQualityError('Empty/invalid official calendar, not a holiday confirmation')
    output=[]
    for row in rows:
        date=day(row.get('bass_dt',''))
        opened=row.get('opnd_yn')
        if opened not in ('Y','N'):
            raise DataQualityError('Missing official market-open flag')
        output.append(dict(date=date,market_open=opened=='Y'))
    frame=pd.DataFrame(output).sort_values('date').reset_index(drop=True)
    if frame.date.duplicated().any():
        raise DataQualityError('Duplicate calendar date')
    return frame


def previous_open_day(store, client, observed_date):
    observed=day(observed_date)
    first=(pd.Timestamp(observed)-pd.Timedelta(days=14)).date().isoformat()
    last=(pd.Timestamp(observed)-pd.Timedelta(days=1)).date().isoformat()
    key=f'{observed}/kis'
    record=store.manifest.get('calendars',{}).get(key)
    if record:
        store.verify_record(record)
        calendar=pd.read_parquet(store.checked_path(record['table_path']))
    else:
        payload=client.fetch_calendar(first)
        calendar=normalize_calendar(payload)
        store.put_table('calendars',key,calendar,canonical(payload),dict(
            source='kis_chk_holiday',observed_date=observed,requested_base=first,
            coverage='returned_dates_only_not_full_holiday_database'))
        record=store.manifest['calendars'][key]
    required=pd.date_range(first,last).strftime('%Y-%m-%d')
    if not set(required)<=set(calendar.date):
        raise DataQualityError('Calendar does not cover all preceding 14 days; no guessed fallback')
    opened=calendar.loc[calendar.date.between(first,last)&calendar.market_open,'date']
    if opened.empty:
        raise DataQualityError('No confirmed previous open day in calendar response')
    return str(opened.max()),record


def _state(store, state):
    atomic_bytes(store.root/'daily_status'/'latest.json',canonical(state))
    atomic_bytes(store.root/'daily_status'/f'{state["decision_date"]}.json',canonical(state))


def run_daily(store, client, *, now=None, start='2026-09-01', max_requests=10000,
              master_loader=collect_masters, delay=.6, code_commit='local'):
    now=now or _utc_now()
    if now.tzinfo is None:
        raise ValueError('An aware decision time is required')
    observed=now.astimezone(KST).date().isoformat()
    start=day(start)
    if start>=observed:
        raise ValueError('Collection start must precede execution date')
    if not 1<=max_requests<=20000:
        raise ValueError('Invalid request budget')
    state=dict(schema_version=1,status='running',decision_date=observed,
        decision_started_at=now.astimezone(timezone.utc).isoformat(),collection_start=start,
        code_commit=code_commit,orders_enabled=False,strategy_approved=False,
        strategy_history_verified=False,full_historical_universe_certified=False,
        requested=0,reused=0,price_policy='latest_confirmed_open_day_strictly_before_execution_date')
    _state(store,state)
    try:
        target,calendar_record=previous_open_day(store,client,observed)
        state['price_date']=target
        if target<start:
            state['status']='no_open_day_in_collection_range'
            _state(store,state)
            return state
        # Fresh observation at each decision; neither today nor future IPOs are
        # retroactively inserted into the prior-day price universe.
        master=master_loader(store,observed_date=observed)
        master=master.loc[master.collection_eligible & master.listed_date.le(target)].copy()
        if master.empty or master.ticker.duplicated().any() or '069500' not in set(master.ticker):
            raise DataQualityError('Invalid current collection universe')
        state.update(target_tickers=len(master),master_date=observed,
            universe_membership_sha256=digest(canonical(sorted(
                [(r.ticker,r.isin,r.classification) for r in master.itertuples()]))))
        wanted=set(master.ticker)
        result=collect_prices(store,client,master,observed,start,target,
                              max_requests=max_requests,refresh=False,delay=delay)
        state.update(requested=result['requested'],reused=result['reused'])
        if not result['requested_universe_queried']:
            state.update(status='checkpoint_budget_exhausted',next_key=result.get('next_key'))
            _state(store,state)
            return state
        raw=load_kis_panel(store,wanted,target,target,basis='raw')
        adjusted=load_kis_panel(store,wanted,target,target,basis='adjusted')
        missing_raw=sorted(wanted-set(raw.ticker))
        missing_adjusted=sorted(wanted-set(adjusted.ticker))
        if missing_raw or missing_adjusted:
            state.update(status='blocked_missing_prices',missing_raw=missing_raw,missing_adjusted=missing_adjusted)
            _state(store,state)
            return state
        # Full universe input: no top-N truncation or strategy-specific filtering.
        # Nontrading securities remain explicitly flagged, not silently omitted.
        frame=master.merge(raw[['ticker',*PRICE_FIELDS]].rename(
            columns={k:'raw_'+k for k in PRICE_FIELDS}),on='ticker',validate='one_to_one')
        frame=frame.merge(adjusted[['ticker',*PRICE_FIELDS]].rename(
            columns={k:'adjusted_'+k for k in PRICE_FIELDS}),on='ticker',validate='one_to_one')
        frame['tradable']=frame[['raw_open','raw_high','raw_low','raw_close',
                               'adjusted_open','adjusted_high','adjusted_low','adjusted_close']].gt(0).all(axis=1) & frame.raw_volume.gt(0) & frame.raw_value.gt(0)
        benchmark=frame.loc[frame.ticker.eq('069500')]
        if not bool(benchmark.tradable.all()):
            raise DataQualityError('Benchmark price unavailable on confirmed open day')
        frame['price_date']=target
        frame['master_observed_date']=observed
        # Only known-at-decision metadata. History/strategy approval is separate.
        frame['stock_candidate_universe']=frame.tradable & frame.asset_type.eq('stock')
        source_versions=[]
        for record in store.manifest['kis_segments'].values():
            if record['ticker'] in wanted and record['start']<=target<=record['end']:
                source_versions.append({k:record[k] for k in ('ticker','isin','price_basis','start','end',
                    'raw_sha256','table_sha256','collected_at')})
        provenance=dict(price_date=target,decision_date=observed,
            calendar={k:calendar_record[k] for k in ('raw_sha256','table_sha256')},
            masters={m:{k:store.manifest['masters'][f'{observed}/{m}'][k] for k in ('raw_sha256','table_sha256')}
                     for m in ('kospi','kosdaq')},price_versions=source_versions)
        key=f'{observed}/{target}'
        store.put_table('daily_inputs',key,frame.sort_values('ticker'),canonical(provenance),dict(
            source='current_master_previous_day_price_inputs',price_date=target,decision_date=observed,
            master_observed_date=observed,orders_enabled=False,strategy_approved=False,
            strategy_history_verified=False,universe_membership_sha256=state['universe_membership_sha256']))
        record=store.manifest['daily_inputs'][key]
        state.update(status='daily_price_inputs_ready',input_key=key,input_table_sha256=record['table_sha256'],
            tradable_tickers=int(frame.tradable.sum()),stock_candidate_count=int(frame.stock_candidate_universe.sum()),
            completed_at=_utc_now().isoformat(),historical_catchup_queried=True)
        _state(store,state)
        return state
    except Exception:
        # No API body, credentials, or arbitrary provider exception in state/logs.
        state['status']='blocked_collection_or_quality_error'
        state['requested']=None  # A failed provider call may follow saved partial segments.
        state['reused']=None
        _state(store,state)
        raise


def load_daily_selection_inputs(store, *, now=None, include_unverified_etfs=False):
    """Fresh, full point-in-time price/universe input; NOT approved buy signals.

    A strategy must still load/validate its history, enforce its frozen rules and
    use the same prior-day lag in backtests. No ranking or orders occur here.
    """
    now=now or _utc_now()
    if now.tzinfo is None:
        raise ValueError('An aware decision time is required')
    current=now.astimezone(KST).date().isoformat()
    state=json.loads((store.root/'daily_status'/'latest.json').read_text())
    if state.get('status')!='daily_price_inputs_ready' or state.get('decision_date')!=current:
        raise DataQualityError('Daily input missing, incomplete or stale; no last-good fallback')
    if datetime.fromisoformat(state['completed_at'])>now:
        raise DataQualityError('Input was not yet available at this decision time')
    if not state['price_date']<current:
        raise DataQualityError('Expected strictly prior-day prices')
    record=store.manifest['daily_inputs'][state['input_key']]
    if (record['table_sha256']!=state['input_table_sha256'] or record['decision_date']!=current
            or record['price_date']!=state['price_date'] or record['master_observed_date']!=current
            or record['universe_membership_sha256']!=state['universe_membership_sha256']):
        raise DataQualityError('Daily input revision mismatch')
    store.verify_record(record)
    frame=pd.read_parquet(store.checked_path(record['table_path']))
    if (len(frame)!=state['target_tickers'] or frame.ticker.duplicated().any()
            or not frame.price_date.eq(state['price_date']).all()
            or not frame.master_observed_date.eq(current).all()):
        raise DataQualityError('Incomplete daily universe')
    if not include_unverified_etfs:
        frame=frame.loc[frame.asset_type.eq('stock') | frame.ticker.eq('069500')].copy()
    frame.attrs.update(state)
    return frame
