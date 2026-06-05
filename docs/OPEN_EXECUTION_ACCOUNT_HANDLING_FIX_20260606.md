# Open Execution Account Handling Fix - 2026-06-06

## Summary

2026-06-05 장 시작 실행에서 전일 EOD 신호(`signal_kr_2026-06-04.json`) 기준 주문이 기대대로 나가지 않은 문제를 점검하고, 계좌 조회/현금 파싱/주문 재검증 로직을 보강했다.

이번 수정은 커밋 `8a9a5504` (`fix: harden open execution account handling`)로 `main`에 푸시했다.

수정 파일:

- `multi_allocator_plus_trader.py`
- `kiwoom_api/core/korea_investment_connector.py`

## Incident Context

2026-06-04 EOD 신호에는 다음 목표가 있었다.

- `034220`: 21.84%
- `357880`: 18.15%
- `242040`: 15.08%
- `018880`: 14.00%
- `229200`: 10.78% (`KODEX 코스닥150`)
- `043260`: 9.95%
- `__CASH__`: 10.19%

2026-06-05 오전 GitHub Actions 수동 실행은 전일 EOD 파일을 읽어 매수를 기대했지만 주문이 정상적으로 발생하지 않았다. 이후 지연된 스케줄 실행에서도 `229200` 매수가 누락된 것으로 보였다.

중요한 정정:

- `229200`은 `KODEX 코스닥150`이므로, 2026-06-04 EOD 신호에는 KODEX ETF가 포함되어 있었다.
- `069500` 포함 신호는 2026-06-05 장 종료 후 생성된 다음 거래일용 신호이므로, 2026-06-05 오전 주문 대상이 아니다.

## Root Cause Candidates Found

### 1. Duplicate Balance Lookup Could Corrupt Holdings

기존 `fetch_account_snapshot()`은 다음 순서로 KIS API를 호출했다.

1. `get_account_balance()`
2. `parse_account_balance_data()`
3. `get_account_stocks()`

문제는 `get_account_stocks()` 내부에서 다시 `get_account_balance()`를 호출한다는 점이다. 이 두 번째 잔고 조회가 KIS 초당 호출 제한(`EGW00201`)에 걸리면 보유 종목이 빈 리스트처럼 처리될 수 있다.

수정:

- 잔고 API 응답 한 번에서 계좌 요약과 보유 종목을 모두 파싱한다.
- 주식 평가금액이 있는데 보유 종목 파싱 결과가 비면 실거래를 중단한다.
- 코드가 인식한 보유 종목을 Actions 로그에 남긴다.

예상 로그:

```text
계좌 총자산: ...원 / 매수가능: ...원
보유 종목: 034220:53주, 018880:114주, ...
```

### 2. Wrong Available Cash Field Priority

기존 `parse_account_balance_data()`는 `available_cash` 산정 시 `ord_psbl_cash`보다 `prvs_rcdl_excc_amt`를 먼저 사용했다.

기존 우선순위:

```text
prvs_rcdl_excc_amt -> ord_psbl_cash -> dnca_tot_amt -> ...
```

`prvs_rcdl_excc_amt`는 전일정산/정산 관련 금액이라 실시간 주문가능현금보다 낮게 나올 수 있다. 이 경우 실제 계좌에는 현금이 충분해도 코드 내부에서는 현금 부족으로 판단할 수 있다.

수정 우선순위:

```text
ord_psbl_cash -> dnca_tot_amt -> prvs_rcdl_excc_amt -> nxdy_excc_amt -> dncl_amt
```

검증 예:

```text
prvs_rcdl_excc_amt = 483,783
ord_psbl_cash = 900,000
available_cash = 900,000
```

### 3. Balance API Rate Limit Was Not Applied

`get_account_balance()`는 GET 요청 직전에 `_wait_for_api_limit()`를 호출하지 않았다.

수정:

- 잔고 조회 GET 요청 전에도 `_wait_for_api_limit()` 적용
- `EGW00201`이면 두 번째 조회 시도까지 허용
- 잔고 조회 요청에 `timeout=10` 추가

### 4. Recheck Did Not Credit Planned Sell Proceeds

기존 주문 직전 재검증은 매도 계획을 먼저 보더라도 그 매도 예정 대금을 `remaining_cash`에 더하지 않았다. 그래서 매수 계획이 실제보다 작아지거나 0주로 스킵될 수 있었다.

수정:

- `SELL` 계획은 수수료/세금 차감 후 예상 현금을 `remaining_cash`에 더한다.
- `BUY` 계획은 수수료/슬리피지를 고려해 주문 가능 수량을 계산한다.
- 재검증 전후 현금, 최종 수량, 스킵 사유를 JSONL 및 Actions 로그에 남긴다.

### 5. Current Price Parsing Missed Nested KIS Response

KIS 현재가 응답은 보통 다음처럼 중첩된다.

```json
{
  "output": {
    "stck_prpr": "5210"
  }
}
```

기존 `_safe_get_current_price()`는 top-level 키만 봐서 현재가 재검증이 사실상 기준가 fallback에 의존할 수 있었다.

수정:

- top-level과 `output` 내부를 모두 확인
- `current_price`, `price`, `stck_prpr`, `prpr` 지원

## Logging Added

다음 상황이 Actions 로그에 직접 남는다.

주문 계획 생성:

```text
계획 생성: BUY 229200 16주 (보유 0주 -> 목표 16주, 목표 10.78%)
```

주문 계획 제외:

```text
계획 제외: 229200 보유 16주 = 목표 16주 (목표 10.78%, 기준가 18570원)
계획 제외: 229200 목표금액 300000원 < 최소매매 50000원
계획 제외: 229200 기준가 없음
```

재검증 제외:

```text
재검증 제외: BUY 229200 - 현금/주문가능 부족 (계획 16주, 현금 ...원 -> ...원)
재검증 제외: BUY 229200 - 가격 괴리 ...%
재검증 제외: BUY 229200 - 현재가 확인 실패
```

재검증 수량 조정:

```text
재검증 수량 조정: BUY 357880 96주 -> 92주 (현금 ...원 -> ...원)
```

## Telegram Note

Actions 로그에서 Telegram 전송은 `401 Unauthorized`가 확인되었다.

이 문제는 이번 코드 수정 범위가 아니라 GitHub Secrets 또는 봇/채팅 권한 문제다.

확인 대상:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- 봇이 해당 채팅에 접근 가능한지

## Verification Performed

문법 검사:

```bash
python -m py_compile multi_allocator_plus_trader.py kiwoom_api/core/korea_investment_connector.py
```

현금 필드 우선순위 검증:

```text
ord_psbl_cash가 prvs_rcdl_excc_amt보다 우선되어 available_cash로 사용됨
```

2026-06-04 EOD 숫자 기반 재현:

```text
229200 미보유 + 주문가능현금 충분
=> RAW: BUY 229200 16주
=> FINAL: BUY 229200 16주
```

## Operational Checklist For Next Live Run

다음 GitHub Actions 실행 로그에서 아래를 확인한다.

1. 신호 파일 로드

```text
신호 스냅샷 로드: ... signal_kr_YYYY-MM-DD.json
```

2. 계좌 현금 필드

```text
ord_psbl_cash: ...
매수가능금액 사용 필드: ord_psbl_cash = ...
```

3. 보유 종목

```text
보유 종목: ...
```

4. `229200` 주문 계획

```text
계획 생성: BUY 229200 ...
```

또는 제외된다면:

```text
계획 제외: 229200 ...
재검증 제외: BUY 229200 ...
```

5. 최종 주문 전송

```text
➡️ BUY 229200 x ...
```

## Files Changed

### `kiwoom_api/core/korea_investment_connector.py`

- 잔고 조회 전 rate limit 대기 추가
- 잔고 조회 timeout 추가
- `EGW00201` 재시도 처리 추가
- `available_cash` 필드 우선순위를 `ord_psbl_cash` 중심으로 변경

### `multi_allocator_plus_trader.py`

- 신호 snapshot의 strategy 이름을 실제 전략명으로 저장
- 실거래 계좌 조회 시 중복 잔고 API 호출 제거
- 잔고 응답 `output1`에서 보유 종목 직접 파싱
- 보유 종목 조회 실패 시 실거래 중단
- 보유 종목/주문 계획/제외 사유 로그 추가
- KIS 현재가 nested response 파싱 지원
- 주문 재검증 시 매도 예상 대금 반영
- 매수 수량 계산 시 수수료/슬리피지 buffer 반영
- 재검증 스킵/수량 조정 로그 및 Telegram 알림 추가
