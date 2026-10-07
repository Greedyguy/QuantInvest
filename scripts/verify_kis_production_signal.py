"""Generate and reload the real strategy signal; synthetic plans, never orders."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from market_data_store import atomic_bytes,canonical
from multi_allocator_plus_trader import MultiAllocatorPlusTrader
from signal_safety import validate_snapshot,KR_BUY_BLOCKED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',required=True)
    p.add_argument('--data-commit',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--start',default='2026-01-01')
    args=p.parse_args()
    report=dict(status='blocked',orders_sent=0,account_read=False,data_commit=args.data_commit,
                synthetic_capital=1_000_000,min_trade_value=10_000)
    try:
        t=MultiAllocatorPlusTrader(start_date=args.start,dry_run=True,prepare_signal_only=True,
            market_store_path=args.store,market_data_commit=args.data_commit,signal_repair_mode='on',
            require_private_inputs=True,min_trade_value=report['min_trade_value'])
        assert t.kis is None
        t.load_market_data()
        date,targets=t.compute_target_weights()
        path=t.save_signal_snapshot(date,targets)
        payload=json.loads(path.read_text())
        validated=validate_snapshot(payload,require_private_inputs=True)
        # Read exactly this new snapshot, never an older file in reports/signals.
        t.signal_snapshot=str(path)
        loaded_date,loaded,refs=t.load_signal_snapshot()
        if loaded_date!=date or not validated.equals(loaded):
            raise RuntimeError('Signal producer/consumer mismatch')
        account={'total_value':1_000_000,'available_cash':1_000_000,'stock_value':0}
        plans=t.build_order_plan(loaded,account,{},price_cache_override=refs)
        report['synthetic_plan_count_before_recheck']=len(plans)
        plans,logs=t.apply_execution_recheck(plans,account)
        if any(plan.action=='BUY' and plan.symbol in KR_BUY_BLOCKED for plan in plans):
            raise RuntimeError('Buy block regression')
        report.update(status='signal_and_synthetic_plan_verified',signal_date=str(date.date()),
            targets=len(targets)-1,target_sum=float(targets.sum()),synthetic_plan_count=len(plans),
            signal_snapshot=path.name,market_inputs=t.market_input_provenance,
            note='Synthetic KRW 1m sizing only; not actual account or realized returns')
    except Exception as exc:
        # Controlled application errors only; the offline path has no credentials.
        report.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        atomic_bytes(Path(args.report),canonical(report))
        print(json.dumps({k:report[k] for k in ('status','orders_sent','account_read')}),flush=True)


if __name__=='__main__': main()
