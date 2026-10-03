"""Date-bounded, identity-checked KIS lifecycle evidence, never price filling."""
import gzip
import json
import pandas as pd
from market_data_store import DataQualityError, canonical, day


def normalize_security_info(payload, ticker, isin, market):
    out = payload.get('output')
    if isinstance(out, list) and len(out) == 1:
        out = out[0]
    if payload.get('rt_cd') != '0' or not isinstance(out, dict):
        raise DataQualityError('Invalid security information')
    # Live CTPF1002R returns the internal product id 00000A + short code.
    # Only this observed, exact prefix is accepted; arbitrary suffix matches are not.
    if out.get('pdno') not in (ticker,'00000A'+ticker) or out.get('std_pdno') != isin:
        raise DataQualityError('Security lifecycle identity mismatch')
    field = {'kospi':'scts_mket_lstg_abol_dt', 'kosdaq':'kosdaq_mket_lstg_abol_dt'}.get(market)
    if not field:
        raise DataQualityError('Unknown lifecycle market')
    dates = set()
    for key in (field, 'lstg_abol_dt'):
        value = str(out.get(key, '')).strip()
        if value not in ('', '00000000'):
            dates.add(day(value))
    if len(dates) > 1:
        raise DataQualityError('Conflicting security delisting dates')
    return dict(ticker=ticker, isin=isin, market=market,
                delisted_date=next(iter(dates), ''),
                # A current suspension flag supplies no historical effective date.
                # It is recorded, but NEVER excuses missing prior-day prices.
                trading_halt=str(out.get('tr_stop_yn', '')).strip())


def read_security_info(store, key, row, observed):
    record = store.manifest.get('security_status', {}).get(key)
    if not record or record.get('observed_date') != observed:
        raise DataQualityError('Missing/stale lifecycle evidence')
    store.verify_record(record)
    payload = json.loads(gzip.decompress(store.checked_path(record['raw_path']).read_bytes()))
    info = normalize_security_info(payload, row.ticker, row.isin, row.market)
    actual = pd.read_parquet(store.checked_path(record['table_path']))
    try:
        pd.testing.assert_frame_equal(actual, pd.DataFrame([info]), check_exact=True)
    except AssertionError:
        raise DataQualityError('Lifecycle table differs from original response') from None
    return info, record


def collect_security_info(store, client, row, observed):
    key = f'{observed}/{row.ticker}'
    if key not in store.manifest.get('security_status', {}):
        payload = client.fetch_security_info(row.ticker)
        info = normalize_security_info(payload, row.ticker, row.isin, row.market)
        store.put_table('security_status', key, pd.DataFrame([info]), canonical(payload),
                        dict(source='kis_search_stock_info', observed_date=observed))
    info, record = read_security_info(store, key, row, observed)
    return info, key, record


def validate_daily_lifecycle(store, frame, observed, price_date):
    """Recheck every absent-price exception when consuming the daily input."""
    if 'price_status' not in frame:
        return
    if not frame.price_status.isin(['observed', 'observed_no_trades', 'confirmed_delisted']).all():
        raise DataQualityError('Unknown daily price status')
    no_trades=frame.loc[frame.price_status.eq('observed_no_trades')]
    if (not no_trades[['raw_volume','raw_value','adjusted_volume','adjusted_value']].eq(0).all().all()
            or no_trades.tradable.any() or no_trades.stock_candidate_universe.any()):
        raise DataQualityError('Invalid no-trade observation')
    for row in frame.loc[frame.price_status.eq('confirmed_delisted')].itertuples():
        info, record = read_security_info(store, row.lifecycle_key, row, observed)
        if (not info['delisted_date'] or info['delisted_date'] > price_date
                or record['raw_sha256'] != row.lifecycle_raw_sha256
                or record['table_sha256'] != row.lifecycle_table_sha256
                or row.tradable or row.stock_candidate_universe):
            raise DataQualityError('Invalid delisting exception')
        fields = [getattr(row, basis+'_'+field) for basis in ('raw','adjusted')
                  for field in ('open','high','low','close','volume','value')]
        if not all(pd.isna(value) for value in fields):
            raise DataQualityError('Delisted security must not have fabricated prices')
