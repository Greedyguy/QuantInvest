# Private KIS production signal bridge

## Scope and evidence

The three authorized steps are collection repair, stored-input integration and
no-order verification. Allocation parameters, entry/exit rules and the 305720
additional-buy block are unchanged. No manual live-order dispatch is part of this
rollout. Inputs can change when complete stored history replaces partial legacy
caches; unchanged strategy rules do **not** imply identical targets or returns.

The previous collection stopped at 0191M0/raw/2026-09-01_2026-10-01 with only a
generic quality error. Its rejected response was not retained, so the exact
original defect cannot be reconstructed. A scoped Actions re-query returned 21
valid rows for each basis. This does not prove every future response is valid.
Malformed price responses are now stored in a private `quarantines` namespace,
retried up to three times under identical validation, then fail closed. Wrong
security identity is never retried as an acceptable substitution. Quarantines
are hash-audited but never consumed as prices; valid checkpoints survive failures.

## Data contract

- The private repository commit and manifest hash are pinned in each signal.
- A ready, complete current-master input is required on the KST decision date;
  its price date must be the preceding officially confirmed open day.
- Every strategy-eligible security needs collected request coverage and exact
  observed index sessions, raw and adjusted histories, matching security identity,
  verified raw/table hashes and exact latest daily/history price agreement.
- The existing name screen and minimum 120 price-history rows are retained.
  Newly listed/confirmed nontrading securities are counted as exclusions rather
  than treated as missing API responses. Missing eligible history blocks the run.
- Signals use adjusted OHLC; execution references use the stored raw close.
  Nontrading zero OHLC is not fabricated or forward-filled.
- Market regimes use actual KOSPI 0001 and KOSDAQ 1001 index histories, not ETF
  proxies. Sources: [KIS index endpoint](https://github.com/koreainvestment/open-trading-api/tree/main/examples_llm/domestic_stock/inquire_daily_indexchartprice)
  and [official index codes](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/inquire_index_timeprice/chk_inquire_index_timeprice.py).
- Historical market cap is not invented from today's data. These KIS histories
  lack it; the existing allocator's missing-cap price-style fallback applies and
  is explicitly recorded. Current-master survivorship, complete corporate-action
  reconstruction and ETF multiplier classification are not certified by this gate.

## Workflow and verification

`Daily EOD Signal Prep` collects prior completed data first, then runs the offline
bridge and the actual seven-child strategy. It reloads the produced versioned
snapshot through the consumer contract and checks a synthetic KRW 1m order plan.
The preparation job receives no broker credentials and does not even initialize
the order connector. Real account reads and order submissions are absent.

The main preparation schedule is 06:20 KST Tuesday–Saturday (previous session's
close). This avoids adding an extra day of lag before the open execution job;
GitHub scheduling delays remain possible. Existing 18:40 prior-day collection is
retained. Collection success is a dependency, not an independent race.
The imported historical collector's separate 06:20 timer is removed to avoid a
duplicate backfill competing with preparation; manual historical runs still work.
Collection and live tasks serialize KIS token use.
Missing data, request budget exhaustion, indices or history blocks publication. Existing
open execution keeps rejecting old snapshots; no fallback is introduced.

Manual runs default to `publish_signal=false`. Only main-branch scheduled runs
or explicit main-branch `publish_signal=true` may commit a new production signal.
Validation reports are saved on a unique branch in the private data repository.
The separate frozen-shadow research failure remains visible and cannot veto
an independently validated production snapshot's save.

The imported historical/backtest pipeline remains separate from order approval.
No historical collection completion or retrospective backtest approves a live
signal by itself. Local safe tests include corrected/uncorrected responses,
identity/hash/date/history failures, true indices, raw references, no broker
construction and unchanged 305720 buy-only protection.
