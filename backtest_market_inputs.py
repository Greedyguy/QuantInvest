"""Offline KIS input gate. No downloads, broker credentials, orders or returns.

Passing this gate only prepares research inputs. It does not certify historical
universe membership, corporate actions, distributions or strategy validity.
"""
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from market_data_store import MarketStore, DataQualityError, atomic_bytes, canonical, day, digest
from kis_market_collection import PRICE_FIELDS, normalize_prices, read_master

BENCHMARK = '069500'
PRICE_COLUMNS = ['date', 'ticker', 'price_basis', *PRICE_FIELDS]


def interval_gaps(intervals, start, end):
    """Missing calendar request ranges; never infer holidays from an empty API response."""
    cursor, finish = pd.Timestamp(start), pd.Timestamp(end)
    gaps = []
    for first, last in sorted(intervals):
        first, last = pd.Timestamp(first), pd.Timestamp(last)
        if last < cursor or first > finish:
            continue
        if first > cursor:
            gaps.append([cursor.date().isoformat(), (first-pd.Timedelta(days=1)).date().isoformat()])
        cursor = max(cursor, last+pd.Timedelta(days=1))
        if cursor > finish:
            break
    if cursor <= finish:
        gaps.append([cursor.date().isoformat(), finish.date().isoformat()])
    return gaps


def _verified_frame(store, record, identity):
    store.verify_record(record)
    if record.get('isin') != identity:
        raise DataQualityError('identity_mismatch')
    if record.get('source') != 'kis_period_price':
        raise DataQualityError('unexpected_source')
    first, last = day(record['start']), day(record['end'])
    basis, ticker = record['price_basis'], record['ticker']
    if basis not in ('raw', 'adjusted') or first > last:
        raise DataQualityError('invalid_price_metadata')
    expected = normalize_prices(json.loads(gzip.decompress(
        store.checked_path(record['raw_path']).read_bytes())), ticker, first, last, basis)
    frame = pd.read_parquet(store.checked_path(record['table_path']))
    if not set(PRICE_COLUMNS).issubset(frame.columns) or len(frame) != record['rows']:
        raise DataQualityError('invalid_table_schema_or_count')
    frame = frame[PRICE_COLUMNS].copy()
    frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.strftime('%Y-%m-%d')
    frame = frame.sort_values('date').reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(frame, expected, check_dtype=False, check_exact=True)
    except AssertionError:
        raise DataQualityError('raw_and_normalized_table_disagree') from None
    return frame


def prepare_inputs(store_path, output, *, start='2025-01-01', end='2026-08-31',
                   warmup_start='2024-01-01', master_date='2026-09-28',
                   min_history=120, include_etfs=True, data_commit, code_commit='local'):
    """Write private diagnostic reports; export no bundle when structural checks fail."""
    start, end, warmup_start, master_date = map(day, (start, end, warmup_start, master_date))
    if not warmup_start <= start <= end or not 1 <= min_history <= 1000:
        raise ValueError('Invalid period/history settings')
    if not re.fullmatch('[a-f0-9]{40}', data_commit):
        raise ValueError('Pin the private data commit with a full Git SHA')
    output = Path(output).resolve()
    # New output only: an old successful bundle must never survive a failed rerun.
    output.mkdir(parents=True, exist_ok=False)
    issues = []
    def issue(code, severity='blocker', **details):
        issues.append(dict(code=code, severity=severity, **details))
    report = dict(schema_version=1, start=start, end=end, warmup_start=warmup_start,
        master_date=master_date, min_history=min_history, include_etfs=include_etfs,
        data_commit=data_commit, code_commit=code_commit, benchmark=BENCHMARK,
        issues=issues, bundle_exported=False, full_universe_certified=False,
        corporate_actions_complete=False, distributions_complete=False,
        orders_enabled=False, calendar_source='KODEX 200 observed sessions, not independent exchange calendar',
        raw_rows=0, adjusted_rows=0, target_tickers=0, verified_segments=0)
    store = None
    if not (Path(store_path)/'manifest.json').is_file():
        issue('missing_store_manifest')
    else:
        try:
            store = MarketStore(store_path)
            report['source_manifest_sha256'] = digest(store.manifest_path.read_bytes())
            master = read_master(store, master_date)
        except (ValueError, OSError, KeyError, TypeError):
            issue('store_or_master_integrity_failure')
            store = None
    if store is None:
        return _finish(report, output)
    if master.ticker.duplicated().any() or BENCHMARK not in set(master.ticker):
        issue('invalid_or_missing_benchmark_master')
        return _finish(report, output)
    target = master.loc[master.collection_eligible & master.listed_date.le(end)].copy()
    if not include_etfs:
        target = target.loc[target.asset_type.eq('stock') | target.ticker.eq(BENCHMARK)]
    if BENCHMARK not in set(target.ticker):
        issue('benchmark_not_eligible')
        return _finish(report, output)
    report['target_tickers'] = len(target)
    issue('current_master_survivorship_bias', 'warning')
    issue('observed_benchmark_calendar_not_independently_certified', 'warning')
    issue('total_return_and_corporate_action_accounting_unverified', 'warning')
    if master_date > end:
        issue('master_observed_after_evaluation_period', 'warning')
    if include_etfs and target.asset_type.eq('etf').any():
        issue('etf_leverage_classification_name_only', 'warning')

    # Group once: do not scan every stored file once per security.
    records = {}
    wanted = set(target.ticker)
    for record in store.manifest['kis_segments'].values():
        if record.get('ticker') not in wanted:
            continue
        try:
            first, last = day(record['start']), day(record['end'])
        except (ValueError, KeyError):
            issue('invalid_segment_dates', ticker=record.get('ticker'))
            continue
        if last >= warmup_start and first <= end:
            records.setdefault((record['ticker'], record.get('price_basis')), []).append(record)
    raw, adjusted, coverage, versions = {}, {}, [], []
    for security in target.itertuples():
        first = max(warmup_start, security.listed_date)
        for basis, dest in (('raw', raw), ('adjusted', adjusted)):
            frames, intervals = [], []
            for record in records.get((security.ticker, basis), []):
                try:
                    frame = _verified_frame(store, record, security.isin)
                except (ValueError, OSError, KeyError, TypeError, EOFError):
                    issue('invalid_segment', ticker=security.ticker, basis=basis,
                          first=record['start'], last=record['end'])
                    continue
                report['verified_segments'] += 1
                intervals.append((record['start'], record['end']))
                versions.append({k: record[k] for k in ('ticker', 'isin', 'price_basis', 'start', 'end',
                    'raw_sha256', 'table_sha256', 'collected_at')})
                if frame.empty:
                    issue('empty_response_unconfirmed', ticker=security.ticker, basis=basis,
                          first=record['start'], last=record['end'])
                frames.append(frame.loc[frame.date.between(first, end)])
            gaps = interval_gaps(intervals, first, end)
            evaluation_gaps = interval_gaps(intervals, max(start, security.listed_date), end)
            if evaluation_gaps:
                issue('evaluation_request_gaps', ticker=security.ticker, basis=basis, ranges=evaluation_gaps)
            if gaps and not evaluation_gaps:
                issue('warmup_request_gaps', ticker=security.ticker, basis=basis, ranges=gaps)
            elif gaps and evaluation_gaps and first < start:
                warmup_gaps = interval_gaps(intervals, first, (pd.Timestamp(start)-pd.Timedelta(days=1)).date().isoformat())
                if warmup_gaps:
                    issue('warmup_request_gaps', ticker=security.ticker, basis=basis, ranges=warmup_gaps)
            panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
            for date, group in panel.loc[panel.date.duplicated(keep=False)].groupby('date'):
                if len(group[PRICE_COLUMNS].drop_duplicates()) > 1:
                    issue('conflicting_overlap', ticker=security.ticker, basis=basis, date=date)
            panel = panel.drop_duplicates('date').sort_values('date').reset_index(drop=True)
            dest[security.ticker] = panel
            coverage.append(dict(ticker=security.ticker, basis=basis, rows=len(panel),
                request_gaps=gaps, valid_segments=len(intervals)))
    report['raw_rows'] = sum(map(len, raw.values()))
    report['adjusted_rows'] = sum(map(len, adjusted.values()))
    report['coverage'] = coverage
    benchmark = raw[BENCHMARK]
    calendar = pd.Index(benchmark.date)
    if benchmark.empty or not calendar[(calendar >= start) & (calendar <= end)].size:
        issue('no_benchmark_evaluation_calendar')
    if len(calendar[calendar < start]) < min_history:
        issue('benchmark_warmup_sessions_insufficient', required=min_history,
              observed=int(len(calendar[calendar < start])))
    eligibility = []
    for security in target.itertuples():
        ticker = security.ticker
        r, a = raw[ticker], adjusted[ticker]
        expected = calendar[calendar >= max(warmup_start, security.listed_date)]
        for basis, frame in (('raw', r), ('adjusted', a)):
            missing = expected.difference(pd.Index(frame.date))
            extra = pd.Index(frame.date).difference(calendar)
            if len(missing):
                issue('missing_observed_sessions', ticker=ticker, basis=basis, count=len(missing), sample=missing[:5].tolist())
            if len(extra):
                issue('off_benchmark_calendar', ticker=ticker, basis=basis, count=len(extra), sample=extra[:5].tolist())
        if not r.date.equals(a.date):
            issue('raw_adjusted_dates_disagree', ticker=ticker)
            continue
        if r.empty:
            continue
        valid = r[['open', 'high', 'low', 'close']].gt(0).all(axis=1) & a[['open', 'high', 'low', 'close']].gt(0).all(axis=1)
        tradable = valid & r.volume.gt(0) & r.value.gt(0)
        if ticker == BENCHMARK and not tradable.all():
            issue('invalid_benchmark_sessions')
        elif not tradable.all():
            issue('nontradable_sessions_retained', 'warning', ticker=ticker, count=int((~tradable).sum()))
        ratio = a.close.where(valid)/r.close.where(valid)
        changes = ratio.pct_change(fill_method=None).abs().gt(.01)
        if changes.any():
            issue('adjustment_factor_change_review', 'review', ticker=ticker,
                  count=int(changes.sum()), sample=r.loc[changes, 'date'].head(5).tolist())
        jumps = a.close.where(valid).pct_change(fill_method=None).abs().gt(.35)
        if jumps.any():
            issue('adjusted_price_jump_review', 'review', ticker=ticker,
                  count=int(jumps.sum()), sample=r.loc[jumps, 'date'].head(5).tolist())
        # All eligibility uses information available by that close. New listings
        # remain in the universe but cannot bypass their own history requirement.
        history = valid.astype(int).rolling(min_history, min_periods=min_history).sum().eq(min_history)
        for date, ok, trade in zip(r.date, history & tradable, tradable):
            eligibility.append(dict(date=date, ticker=ticker, signal_eligible=bool(ok), tradable=bool(trade)))

    if not any(i['severity'] == 'blocker' for i in issues):
        files = {}
        for name, frame in [('raw', pd.concat(raw.values(), ignore_index=True)),
                            ('adjusted', pd.concat(adjusted.values(), ignore_index=True)),
                            ('universe', target), ('eligibility', pd.DataFrame(eligibility))]:
            parts = []
            # Bounded files keep a multi-year whole-current-universe bundle below
            # GitHub's per-file limit without requiring a raw-data public artifact.
            for number, offset in enumerate(range(0, len(frame), 100_000)):
                part = frame.iloc[offset:offset+100_000]
                path = output/f'{name}-{number:04d}.parquet'
                part.to_parquet(path, index=False)
                parts.append(dict(path=path.name, sha256=digest(path.read_bytes()), rows=len(part)))
            files[name] = dict(parts=parts, rows=len(frame))
        report['bundle_exported'] = True
        bundle = {k: report[k] for k in ('schema_version', 'start', 'end', 'warmup_start', 'master_date',
            'min_history', 'include_etfs', 'data_commit', 'code_commit', 'benchmark', 'source_manifest_sha256')}
        bundle.update(files=files, source_versions=versions, provisional_research_only=True,
                      review_required=any(i['severity'] == 'review' for i in issues), orders_enabled=False)
        atomic_bytes(output/'bundle.json', canonical(bundle))
    return _finish(report, output)


def _finish(report, output):
    counts = {level: sum(i['severity'] == level for i in report['issues']) for level in ('blocker', 'review', 'warning')}
    report['issue_counts'] = counts
    report['status'] = ('blocked' if counts['blocker'] else 'review_required' if counts['review'] else 'ready_for_research_inputs')
    atomic_bytes(output/'report.json', canonical(report))
    lines = ['# 백테스트 데이터 연결·품질 검사', '', f'- 상태: `{report["status"]}`',
        f'- 평가 기간: {report["start"]} ~ {report["end"]}',
        f'- 지표 준비 시작: {report["warmup_start"]} / 최소 {report["min_history"]}거래일',
        f'- 데이터 버전: `{report["data_commit"]}`',
        f'- 대상 종목: {report["target_tickers"]} / 검증 조각: {report["verified_segments"]}',
        f'- 원주가 {report["raw_rows"]}행 / 수정주가 {report["adjusted_rows"]}행',
        f'- 차단 {counts["blocker"]}건 / 검토 {counts["review"]}건 / 주의 {counts["warning"]}건',
        f'- 백테스트 입력 생성: {report["bundle_exported"]}', '',
        '## 해석', '', '수집 중 누락 또는 2024년 준비 자료 부족은 검사 오류가 아니라 백테스트 시작 차단 사유입니다.',
        '상장폐지 종목·과거 편입 종목·기업행동·ETF 분배금·세금의 검증 완료를 뜻하지 않습니다.',
        '준비된 입력도 연구 전용이며, 전략 실행/수익률 산출/매매 주문을 수행하지 않습니다.', '', '## 항목별 집계', '']
    for code in sorted({i['code'] for i in report['issues']}):
        lines.append(f'- `{code}`: {sum(i["code"] == code for i in report["issues"])}건')
    lines += ['', '종목별 누락 기간·이상 날짜는 같은 폴더의 report.json을 확인하세요.', '']
    atomic_bytes(output/'report.md', '\n'.join(lines).encode())
    return report


@dataclass
class BacktestInputs:
    raw: dict
    adjusted: dict
    calendar: pd.DatetimeIndex
    evaluation_dates: pd.DatetimeIndex
    universe: pd.DataFrame
    eligibility: pd.DataFrame
    metadata: dict


def load_backtest_inputs(bundle_path, *, allow_provisional=False, allow_action_review=False, include_etfs=False):
    """Price dictionaries have DatetimeIndex and OHLCV/value for existing engines.

    Warmup rows remain in calendar; simulate only evaluation_dates. The caller
    must honor eligibility and next-session execution. No forward-fill/seed fallback.
    """
    root = Path(bundle_path).resolve().parent
    bundle = json.loads(Path(bundle_path).read_text())
    if bundle.get('schema_version') != 1 or not allow_provisional:
        raise DataQualityError('Explicit provisional-research opt-in required')
    if bundle.get('review_required') and not allow_action_review:
        raise DataQualityError('Corporate-action/price-jump review is required')
    frames = {}
    for name in ('raw', 'adjusted', 'universe', 'eligibility'):
        record = bundle['files'][name]
        parts = []
        for part in record['parts']:
            path = (root/part['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or digest(path.read_bytes()) != part['sha256']:
                raise DataQualityError('Prepared bundle missing/corrupt or path escape')
            frame = pd.read_parquet(path)
            if len(frame) != part['rows']:
                raise DataQualityError('Prepared part row count differs')
            parts.append(frame)
        if not parts:
            raise DataQualityError('Prepared bundle is empty')
        frames[name] = pd.concat(parts, ignore_index=True)
        if len(frames[name]) != record['rows']:
            raise DataQualityError('Prepared bundle row count differs')
    prices = {}
    universe = frames['universe']
    if not include_etfs:
        universe = universe.loc[universe.asset_type.eq('stock') | universe.ticker.eq(bundle['benchmark'])]
    wanted = set(universe.ticker)
    for basis in ('raw', 'adjusted'):
        panel = frames[basis].loc[frames[basis].ticker.isin(wanted)].copy()
        panel['date'] = pd.to_datetime(panel.date)
        # Zero quotes are nontradable/missing, never a zero liquidation value.
        ohlc = ['open', 'high', 'low', 'close']
        panel[ohlc] = panel[ohlc].where(panel[ohlc].gt(0), np.nan)
        prices[basis] = {ticker: f.set_index('date')[list(PRICE_FIELDS)].sort_index()
                         for ticker, f in panel.groupby('ticker')}
    calendar = pd.DatetimeIndex(prices['raw'][bundle['benchmark']].index)
    evaluation = calendar[(calendar >= bundle['start']) & (calendar <= bundle['end'])]
    return BacktestInputs(prices['raw'], prices['adjusted'], calendar, evaluation,
                          universe, frames['eligibility'].loc[frames['eligibility'].ticker.isin(wanted)], bundle)
