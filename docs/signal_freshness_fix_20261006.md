# 2026-10-06 KR signal freshness failure

## Evidence

- Failed live workflow: https://github.com/Greedyguy/QuantInvest/actions/runs/37391910283
- Code/snapshot checkout: `f7b827cc67ae366fd38306ee168e7813613f0943`.
- Latest KR snapshot: `reports/signals/signal_kr_2026-10-02.json`.
- Its private input source is `private_kis_store`, price date is `2026-10-02`,
  and decision date is `2026-10-05`. Execution on `2026-10-06` correctly rejects it.
- The October 5 preparation run published its validated production snapshot:
  https://github.com/Greedyguy/QuantInvest/actions/runs/37245704266
  Its separate frozen research evidence step failed after publication.
- At investigation time the returned workflow history contained no October 6
  preparation run. Independent schedules do not establish a dependency, and a
  checkout fixed at the event SHA can also miss a later signal publication.
- The KRX login message is not the exception that stopped this execution.
  The failure occurs in private-input provenance validation before account
  retrieval or orders. Adding KRX credentials does not refresh this snapshot.

## Change

The live workflow first checks the latest main snapshot without broker, KRX,
account, or messaging credentials. If it is not ready, it invokes the existing
whole-universe collection and no-order signal verification workflow, with
publication enabled. This automatic preparation-to-live connection was explicitly
approved by the user. No manual live workflow is dispatched during this fix.

The reusable workflow reports publication only after a successful push. A
separate research failure remains visible; it cannot invalidate already completed
production verification. The live job requires a ready preflight or successful
publication output, checks out current main, and revalidates the full snapshot
contract before broker credentials are exposed to the trading step. A missing,
invalid, stale, or uncommitted signal never unlocks orders.

The KIS token lock belongs to the live job, not its parent workflow, because the
collector needs the same lock. Existing collection writer locks remain in place.
Existing target limits, private hash checks, reference price checks, execution
rechecks, duplicate-execution handling, and the 305720 buy block remain active.

The validation error now identifies the mismatched date/source or invalid hash
field and the required preparation step. A timezone-aware validation clock is
converted to KST before its decision date is compared.

## Validation and limits

- 130 offline tests passed, including the exact stale-decision-date regression,
  KST date conversion, malformed hashes, no older-file fallback, credential-free
  preflight imports, workflow dependencies, and existing execution safety tests.
- Running the new required preflight against the failed checkout exits 1 and
  reports `decision_date='2026-10-05', expected='2026-10-06'`.
- This validation does not claim a new October 6 market-data collection, a new
  production signal, or successful real execution. Broker/account/order calls
  and manual live workflow dispatch are not part of these checks.
- Preparation can take substantial time if the daily schedule has not completed.
  Collection or validation failures still block trading. Calendar/data problems
  must be repaired at the source; never edit provenance dates to force readiness.
