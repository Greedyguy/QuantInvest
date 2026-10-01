# KR production signal repair

Scope: restore the existing seven-child allocation, not a new concentration strategy.
The KR trader now enables `child_signal_mode` with explicit market `KR`. US paths
and ordinary strategy `run_backtest` defaults remain unchanged.

## Behavior

- Signal generation does not liquidate merely because its input ends today.
- Short-small-cap real stop, profit and holding-period exits still run on the cutoff.
- KQM and K200 mean-reversion export actual cash/holding target histories directly.
- Missing child data, child exceptions, or missing weight histories fail closed;
  there is no silent partial-portfolio reallocation. All seven children need warmup.
- Existing entry filters, child allocations, target turnover caps, minimum orders,
  universe loading, and exposure rules remain unchanged. No 3/5/8-stock concentration.
- The final KR producer target is capped per security (existing 30%) and in aggregate
  (100%). Excess is reduced; an underinvested portfolio is never expanded to 100%.
- `305720` is KODEX secondary-battery industry, **not US Treasuries**. Per owner
  instruction, only additional BUYs are blocked, at planning, recheck and final send.
  Existing holdings may remain or be reduced/exited under the original target rules.
  Its unused budget stays cash; no replacement ETF is inferred or purchased.

## Versioned snapshot contract

EOD snapshots include `meta.signal_path_version=kr-signal-repair-v1`, `market=kr`,
`allocation_policy=legacy`, limits and the buy-block list. Open execution rejects:

- old/unversioned or policy/market/strategy-mismatched snapshots;
- negative/nonfinite weights, inconsistent cash, >100% budget or >30% security targets;
- missing/nonpositive reference prices or input dates different from the signal date;
- same-day/future EOD signals or signals more than three weekdays old (KST clock).

The weekday age rule is intentionally conservative and is **not an exchange holiday
calendar**. Long holidays can pause execution until a fresh snapshot is produced.
No stale-signal fallback is enabled to avoid missed trades. The producer validates
its input-date contract before saving. The consumer repeats validation before the
account/order stage. Live repaired mode requires the existing execution recheck and
cash-preserving policy; it cannot opt into `legacy_renorm`.
Real repaired KR execution also requires `eod_fixed`, so the direct live-computation
path cannot bypass the consumer's date checks. Additional buys require a finite,
positive current quote; an unavailable quote cannot fall back to yesterday's price.
Normal sells retain their original execution behavior.

## Rollout

1. Merge only after offline safety CI passes.
2. Run **Daily EOD Signal Prep** (prepare-only, no broker orders) or wait for its
   scheduled run. A pre-deployment snapshot deliberately cannot be consumed.
3. Inspect the new snapshot version/date, 100% sum, and per-security limits.
4. The next scheduled open execution consumes the new valid snapshot. The existing
   B shadow can read actual holdings without sending orders. Do not dispatch A live
   manually as a deployment test.

If signal preparation fails, fix the data/child failure; do not relax the contract
or switch to an old snapshot merely to make execution continue.

The frozen KODEX research report is independent of production validation. Its
failure does not discard an already validated production snapshot: the commit step
runs, then an explicit final failure marks the workflow red. Failed shadow evidence
is never accepted. At deployment review, its simulator's pinned hash was already
different from the unchanged main-branch file; that research contract is not reset
or silently re-approved by this deployment.

## Rollback

Rollback is explicit: revert the deployment or set `--signal-repair-mode off` in
both KR producer and consumers, then regenerate a matching legacy snapshot.
A repaired snapshot is rejected by a KR consumer with repair disabled. The
305720 buy-only block stays in place even in off mode unless separately reviewed.
Never use rollback flags to bypass the user's instrument restriction.

## Verification

`tests/test_signal_connection.py` checks cutoff holdings, true risk exits, repeat
calls and explicit KR market detection. `tests/test_signal_repair_live.py` checks
producer/consumer contracts, a synthetic holdings-to-order roundtrip without broker
calls, invalid budgets, child failures, US isolation and the three buy-only guards.
The CI workflow runs these plus existing execution, shadow, freshness and allocator
regression tests. Research evidence is in the task's signal-repair comparison report;
it is not proof of realized live-account performance.

Local full-suite check: 222 passed, 10 failed before the workflow-isolation test
was added. All 10 failures are the existing frozen-shadow simulator-hash mismatch
(`78a8d4d4...` on main vs pinned `b5ec0bb8...`), not changed strategy behavior.
