import json

import pandas as pd
import pytest

from market_data_store import MarketStore, DataQualityError, canonical
from kis_market_collection import normalize_prices
from backtest_market_inputs import prepare_inputs, load_backtest_inputs, interval_gaps

DATES = ['2024-12-26', '2024-12-27', '2024-12-30', '2025-01-02', '2025-01-03']
IDENTITIES = {'069500': 'KR7069500007', '005930': 'KR7005930003'}


def put_prices(store, ticker, basis, dates=DATES, *, suffix='', adjusted_factor=None, start='2024-12-26'):
    rows = []
    for i, date in enumerate(dates):
        factor = adjusted_factor[i] if adjusted_factor else 1
        rows.append(dict(stck_bsop_date=date.replace('-', ''), stck_oprc=str(100*factor),
            stck_hgpr=str(110*factor), stck_lwpr=str(90*factor), stck_clpr=str(105*factor),
            acml_vol='1000', acml_tr_pbmn='104000'))
    payload = dict(rt_cd='0', output1=dict(stck_shrn_iscd=ticker), output2=rows)
    key = f'{ticker}/{basis}/{start}_2025-01-03{suffix}'
    frame = normalize_prices(payload, ticker, start, '2025-01-03', basis)
    store.put_table('kis_segments', key, frame, canonical(payload), dict(source='kis_period_price',
        ticker=ticker, isin=IDENTITIES[ticker], price_basis=basis, start=start, end='2025-01-03',
        master_date='2026-09-28', response_status='observed_rows' if dates else 'empty_unconfirmed'))
    return key


@pytest.fixture
def store(tmp_path):
    s = MarketStore(tmp_path/'store')
    for market, ticker, kind in [('kospi', '069500', 'etf'), ('kosdaq', '005930', 'stock')]:
        frame = pd.DataFrame([dict(ticker=ticker, isin=IDENTITIES[ticker], name=ticker,
            market=market, listed_date='2000-01-01', asset_type=kind, classification='stock',
            classification_verified=kind == 'stock', collection_eligible=True)])
        s.put_table('masters', '2026-09-28/'+market, frame, b'fixture master', {})
    for ticker in IDENTITIES:
        for basis in ('raw', 'adjusted'):
            put_prices(s, ticker, basis)
    return s


def run(store, tmp_path, **overrides):
    args = dict(start='2025-01-02', end='2025-01-03', warmup_start='2024-12-26',
                min_history=3, data_commit='a'*40)
    args.update(overrides)
    return prepare_inputs(store.root, tmp_path/'output', **args)


def codes(report):
    return {item['code'] for item in report['issues']}


def test_complete_offline_roundtrip(store, tmp_path, monkeypatch):
    import requests
    def forbidden(*args, **kwargs):
        raise AssertionError('Quality/adapter must never request network data')
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    report = run(store, tmp_path)
    assert report['status'] == 'ready_for_research_inputs'
    assert report['bundle_exported'] and not report['full_universe_certified']
    path = tmp_path/'output/bundle.json'
    with pytest.raises(DataQualityError):
        load_backtest_inputs(path)
    inputs = load_backtest_inputs(path, allow_provisional=True)
    assert len(inputs.calendar) == 5 and len(inputs.evaluation_dates) == 2
    assert isinstance(inputs.raw['005930'].index, pd.DatetimeIndex)
    assert inputs.raw['005930'].value.iloc[0] == 104000
    assert len(inputs.universe) == 2
    assert not inputs.eligibility.iloc[0].signal_eligible
    assert inputs.metadata['data_commit'] == 'a'*40


def test_request_gap_union():
    assert interval_gaps([('2025-01-01', '2025-01-03'), ('2025-01-03', '2025-01-07')],
                         '2025-01-01', '2025-01-07') == []
    assert interval_gaps([('2025-01-02', '2025-01-03')], '2025-01-01', '2025-01-05') == [
        ['2025-01-01', '2025-01-01'], ['2025-01-04', '2025-01-05']]


def test_warmup_coverage_and_sessions_block(store, tmp_path):
    report = run(store, tmp_path, warmup_start='2024-01-01', min_history=120)
    assert {'warmup_request_gaps', 'benchmark_warmup_sessions_insufficient'} <= codes(report)
    assert report['status'] == 'blocked' and not report['bundle_exported']
    assert not (tmp_path/'output/bundle.json').exists()


def test_missing_internal_session(store, tmp_path):
    put_prices(store, '005930', 'raw', dates=DATES[:3]+DATES[4:])
    report = run(store, tmp_path)
    assert {'missing_observed_sessions', 'raw_adjusted_dates_disagree'} <= codes(report)
    assert not report['bundle_exported']


def test_empty_response_is_not_holiday(store, tmp_path):
    put_prices(store, '005930', 'raw', dates=[])
    assert 'empty_response_unconfirmed' in codes(run(store, tmp_path))


def test_integrity_corruption_blocks(store, tmp_path):
    record = next(iter(store.manifest['kis_segments'].values()))
    store.checked_path(record['table_path']).write_bytes(b'corrupt')
    report = run(store, tmp_path)
    assert 'invalid_segment' in codes(report)
    assert not report['bundle_exported']


def test_normalized_table_must_match_original_response(store, tmp_path):
    key = '005930/raw/2024-12-26_2025-01-03'
    rec = store.manifest['kis_segments'][key]
    import gzip
    original = gzip.decompress(store.checked_path(rec['raw_path']).read_bytes())
    frame = pd.read_parquet(store.checked_path(rec['table_path']))
    frame.loc[0, 'close'] = 104
    metadata = {k: rec[k] for k in ('source', 'ticker', 'isin', 'price_basis', 'start', 'end')}
    store.put_table('kis_segments', key, frame, original, metadata)
    assert 'invalid_segment' in codes(run(store, tmp_path))


def test_conflicting_overlap_not_silently_latest(store, tmp_path):
    put_prices(store, '005930', 'raw', suffix='_other', adjusted_factor=[1, 1, 1, 1, 2])
    report = run(store, tmp_path)
    assert 'conflicting_overlap' in codes(report) and not report['bundle_exported']


def test_equal_overlap_can_be_reused(store, tmp_path):
    put_prices(store, '005930', 'raw', suffix='_same')
    assert run(store, tmp_path)['bundle_exported']


def test_adjustment_factor_review_gate(store, tmp_path):
    put_prices(store, '005930', 'adjusted', adjusted_factor=[.5, .5, .5, 1, 1])
    report = run(store, tmp_path)
    assert report['status'] == 'review_required' and report['bundle_exported']
    assert 'adjustment_factor_change_review' in codes(report)
    path = tmp_path/'output/bundle.json'
    with pytest.raises(DataQualityError):
        load_backtest_inputs(path, allow_provisional=True)
    assert load_backtest_inputs(path, allow_provisional=True, allow_action_review=True)


def test_prepared_bundle_tamper(store, tmp_path):
    run(store, tmp_path)
    (tmp_path/'output/raw-0000.parquet').write_bytes(b'corrupt')
    with pytest.raises(DataQualityError):
        load_backtest_inputs(tmp_path/'output/bundle.json', allow_provisional=True)


def test_missing_store_produces_report_not_empty_success(tmp_path):
    report = prepare_inputs(tmp_path/'missing', tmp_path/'output', data_commit='a'*40)
    assert report['status'] == 'blocked'
    assert (tmp_path/'output/report.json').exists()
    assert not (tmp_path/'missing').exists()


def test_existing_output_refused(store, tmp_path):
    run(store, tmp_path)
    with pytest.raises(FileExistsError):
        run(store, tmp_path)


def test_stocks_only_keeps_benchmark(store, tmp_path):
    report = run(store, tmp_path, include_etfs=False)
    assert report['target_tickers'] == 2
    assert 'etf_leverage_classification_name_only' not in codes(report)


def test_missing_benchmark_blocks(store, tmp_path):
    put_prices(store, '069500', 'raw', dates=[])
    report = run(store, tmp_path)
    assert 'no_benchmark_evaluation_calendar' in codes(report)


def test_data_commit_and_dates_required(store, tmp_path):
    with pytest.raises(ValueError):
        run(store, tmp_path, data_commit='main')
    with pytest.raises(ValueError):
        run(store, tmp_path, start='2026-01-01', end='2025-01-01')


def test_new_listing_stays_ineligible_until_history_ready(store, tmp_path):
    rec = store.manifest['masters']['2026-09-28/kosdaq']
    frame = pd.read_parquet(store.checked_path(rec['table_path']))
    frame['listed_date'] = '2025-01-02'
    store.put_table('masters', '2026-09-28/kosdaq', frame, b'new listing master', {})
    for basis in ('raw', 'adjusted'):
        put_prices(store, '005930', basis, dates=DATES[-2:])
    report = run(store, tmp_path)
    assert report['bundle_exported']
    bundle = load_backtest_inputs(tmp_path/'output/bundle.json', allow_provisional=True)
    assert not bundle.eligibility.loc[bundle.eligibility.ticker.eq('005930'), 'signal_eligible'].any()


def test_quality_workflow_is_separate_and_private():
    from pathlib import Path
    import yaml
    text = (Path(__file__).resolve().parents[1]/'.github/workflows/market-data-quality.yml').read_text()
    workflow = yaml.safe_load(text)
    assert set(workflow['on']) == {'workflow_call'}
    assert workflow['permissions'] == {'contents': 'read'}
    assert workflow['concurrency']['group'] != 'market-data-private-writer'
    assert set(workflow['on']['workflow_call']['secrets']) == {'MARKET_DATA_REPO_TOKEN'}
    assert 'KIS_APP' not in text and 'KIS_ACCOUNT' not in text
    assert 'upload-artifact' not in text and 'git reset' not in text and '--force' not in text
    assert 'git -C market-data-private push origin "$REPORT_BRANCH"' in text
    steps = workflow['jobs']['prepare']['steps']
    assert next(i for i, s in enumerate(steps) if s.get('id') == 'publish') > next(
        i for i, s in enumerate(steps) if s.get('id') == 'prepare')


def test_prepared_bundle_path_escape(store, tmp_path):
    run(store, tmp_path)
    path = tmp_path/'output/bundle.json'
    bundle = json.loads(path.read_text())
    bundle['files']['raw']['parts'][0]['path'] = '../outside.parquet'
    path.write_text(json.dumps(bundle))
    with pytest.raises(DataQualityError):
        load_backtest_inputs(path, allow_provisional=True)


def test_invalid_benchmark_trade_session_blocks(store, tmp_path):
    key = '069500/raw/2024-12-26_2025-01-03'
    rec = store.manifest['kis_segments'][key]
    import gzip
    payload = json.loads(gzip.decompress(store.checked_path(rec['raw_path']).read_bytes()))
    payload['output2'][0].update(acml_vol='0', acml_tr_pbmn='0')
    frame = normalize_prices(payload, '069500', '2024-12-26', '2025-01-03', 'raw')
    meta = {k: rec[k] for k in ('source', 'ticker', 'isin', 'price_basis', 'start', 'end')}
    store.put_table('kis_segments', key, frame, canonical(payload), meta)
    report = run(store, tmp_path)
    assert 'invalid_benchmark_sessions' in codes(report) and not report['bundle_exported']
