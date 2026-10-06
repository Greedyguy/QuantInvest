"""Offline regressions for the October 6 stale decision-date failure."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.check_kr_signal_ready import check_latest, main
from signal_safety import SIGNAL_PATH_VERSION, validate_snapshot


def snapshot():
    return dict(signal_date='2026-10-02', strategy='multi_allocator_plus_safe_etf_kqm',
        targets={'069500': .3, '__CASH__': .7}, ref_prices={'069500': 60000.},
        meta=dict(signal_path_version=SIGNAL_PATH_VERSION, market='kr', allocation_policy='legacy',
            data_as_of=dict(primary_index='2026-10-02', secondary_index='2026-10-02',
                            target_securities={'069500': '2026-10-02'}),
            market_inputs=dict(source='private_kis_store', decision_date='2026-10-05',
                price_date='2026-10-02', data_commit='a'*40, manifest_sha256='b'*64,
                input_table_sha256='c'*64, universe_membership_sha256='d'*64)))


def test_actual_failure_explains_decision_date_and_preserves_rejection():
    payload = snapshot()
    original = copy.deepcopy(payload)
    validate_snapshot(payload, today='2026-10-05', require_private_inputs=True)
    with pytest.raises(ValueError) as error:
        validate_snapshot(payload, today='2026-10-06', require_private_inputs=True)
    message = str(error.value)
    assert "decision_date='2026-10-05', expected='2026-10-06'" in message
    assert 'Daily EOD Signal Prep' in message
    assert payload == original


def test_explicit_utc_clock_uses_kst_decision_date():
    payload = snapshot()
    payload['meta']['market_inputs']['decision_date'] = '2026-10-06'
    validate_snapshot(payload, today=pd.Timestamp('2026-10-05T21:20:00Z'), require_private_inputs=True)


@pytest.mark.parametrize('field', ['data_commit', 'manifest_sha256', 'input_table_sha256',
                                  'universe_membership_sha256'])
@pytest.mark.parametrize('value', [None, 123, '', 'not-a-hash'])
def test_bad_provenance_hash_has_field_specific_error(field, value):
    payload = snapshot()
    payload['meta']['market_inputs'][field] = value
    with pytest.raises(ValueError, match=field + '=missing/invalid'):
        validate_snapshot(payload, today='2026-10-05', require_private_inputs=True)


def test_latest_snapshot_only_never_falls_back_to_older_valid_file(tmp_path):
    first = tmp_path / 'signal_kr_2026-10-02.json'
    first.write_text(json.dumps(snapshot()))
    assert check_latest(tmp_path, today='2026-10-05')[0]
    newest = tmp_path / 'signal_kr_2026-10-03.json'
    newest.write_text('{broken')
    ready, path, reason = check_latest(tmp_path, today='2026-10-05')
    assert not ready and path == newest and reason


def test_missing_signal_and_stale_signal_are_not_ready(tmp_path):
    assert not check_latest(tmp_path, today='2026-10-06')[0]
    (tmp_path / 'signal_kr_2026-10-02.json').write_text(json.dumps(snapshot()))
    ready, _, reason = check_latest(tmp_path, today='2026-10-06')
    assert not ready and 'decision_date' in reason


@pytest.mark.parametrize('required', [False, True])
def test_cli_outputs_false_and_required_mode_exits_before_execution(tmp_path, monkeypatch, required):
    output = tmp_path / 'output'
    argv = ['check', '--directory', str(tmp_path), '--github-output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv + (['--require-ready'] if required else []))
    if required:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    else:
        main()
    assert output.read_text() == 'ready=false\n'


def test_preflight_imports_no_broker_krx_or_credentials():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c',
        "import sys; import scripts.check_kr_signal_ready; "
        "assert not any(n.startswith(('pykrx', 'kiwoom_api', 'dotenv', 'multi_allocator_plus_trader')) "
        "for n in sys.modules)"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_live_workflow_prepares_missing_signal_before_acquiring_broker_lock():
    import yaml
    root = Path(__file__).resolve().parents[1]
    live = yaml.safe_load((root / '.github/workflows/daily-open-exec-a.yml').read_text())
    prep = yaml.safe_load((root / '.github/workflows/daily-eod-signal.yml').read_text())
    jobs = live['jobs']
    assert 'concurrency' not in live  # Collector needs the same token lock.
    assert jobs['check-signal']['if'] == "github.ref == 'refs/heads/main'"
    assert 'secrets' not in str(jobs['check-signal'])
    assert jobs['prepare-signal']['needs'] == 'check-signal'
    assert jobs['prepare-signal']['if'] == "needs.check-signal.outputs.ready != 'true'"
    assert jobs['prepare-signal']['uses'] == './.github/workflows/daily-eod-signal.yml'
    assert jobs['prepare-signal']['with']['publish_signal'] is True
    assert 'KIS_ACCOUNT' not in jobs['prepare-signal']['secrets']
    run = jobs['run-live']
    assert run['needs'] == ['check-signal', 'prepare-signal']
    assert '!cancelled()' in run['if'] and "needs.check-signal.result == 'success'" in run['if']
    assert "needs.prepare-signal.outputs.signal_published == 'true'" in run['if']
    assert run['concurrency']['group'] == 'kis-real-token'
    steps = run['steps']
    assert steps[0]['with']['ref'] == 'main'
    check = next(i for i, s in enumerate(steps) if '--require-ready' in s.get('run', ''))
    trade = next(i for i, s in enumerate(steps) if '--real' in s.get('run', ''))
    assert check < trade
    assert 'env' not in steps[check] and 'continue-on-error' not in steps[check]
    assert '--require-private-inputs' in steps[trade]['run']
    # YAML 1.1 parses unquoted GitHub "on" as True.
    call = prep.get('on', prep.get(True))['workflow_call']
    assert call['inputs']['publish_signal']['default'] is False
    assert 'signal_published' in call['outputs']
    publish = next(s for s in prep['jobs']['prep-signal']['steps'] if s.get('id') == 'publish')
    assert publish['run'].index('git push origin HEAD:main') < publish['run'].index('signal_published=true')
