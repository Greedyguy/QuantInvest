# 2026-10-07: selection succeeds but minimum sizing prevents buys

## Recorded evidence (KST)

- October 6 manual run `37398640888` submitted one 069500 SELL. Its summary
  `a440ffa31de7` has `completed_for_signal=true`.
- October 6 delayed scheduled run `37410770407` correctly skipped the same
  trading-day/signal pair at 12:50, citing that manual run. It was not a
  previous-day execution suppressing a new day's trading.
- October 7 manual run `37549830399` (started 09:02, trading step 09:14) and
  scheduled run `37566019884` (trading step 12:18) both planned zero orders.
  Neither was blocked as already executed; both summaries have
  `completed_for_signal=false`, zero failed orders, and 38 sizing exclusions.
- October 7 snapshot: decision date 2026-10-07, price date 2026-10-06,
  private data commit `8700936467f6067184ed9c828cb9f6449965d7fc`.
  The 3,866 current inputs contain 3,325 strategy-eligible securities after
  existing filters. Seven child strategies produce 38 positive security
  targets. The stress level 2 rule reduces exposure from 40% to 18%; cash is 82%.
- Every target was below the configured KRW 50,000 minimum or its one-share
  price at the recorded account size. This is an allocation/sizing constraint,
  not missing selection or duplicate-execution suppression.
- The separate frozen research failure remains unchanged. It does not mean
  the trading step was skipped; this diagnosis uses the execution summaries.

## Authorized correction

The user explicitly selected a KRW 10,000 minimum while preserving selected
securities and target weights. Live KR workflow and no-order verifier now both
use that amount. Strategy allocation, risk exposure, integer floor sizing,
same-day duplicate protection, current-price rechecks, cash checks, and the
305720 buy block remain unchanged. There is no automatic concentration or
redistribution of unspent budgets, and the shadow comparison's fixed historical
policies remain unchanged.

Reports and GitHub step summaries now distinguish selection/sizing exclusions,
already-at-target holdings, recheck exclusions, duplicate suppression, submitted
orders, and failures. Duplicate reports include the prior execution timestamp,
order count, and GitHub run ID when available. Exact balances and account numbers
remain excluded from published summaries. Sizing decisions separately record
whether the minimum amount or one-share constraint was unmet.

## No-order replay and tests

Replay uses the saved October 6 price-date signal with empty holdings and
explicitly synthetic capital, not a fresh account read or a promise of execution.

| Synthetic capital | 50,000 minimum | 10,000 minimum |
|---:|---|---|
| 500,000 | No orders | 446540 × 1, 036540 × 1 |
| 700,000 | No orders | 446540 × 1, 036540 × 1 |
| 1,000,000 | No orders | 446540 × 2, 036540 × 2 |

At reference prices the one-share pair totals KRW 23,240. The two-share pair
totals KRW 46,480. Each plan remains within its original target budget; most of
the selected securities still cannot fit a whole share. Live quotes, available
cash, and existing guards can further reduce or reject these plans.

141 offline tests passed, including original execution safety tests, minimum
configuration parity, no previous-day duplicate blocking, same-day duplicate
protection, sizing-only no-order outcomes, and sanitized GitHub summaries.
No broker/account calls or live workflow dispatch were made for this repair.
