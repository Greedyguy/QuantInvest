# GitHub Actions 기반 multi_allocator_plus 자동매매 설정 가이드

이 문서는 `multi_allocator_plus_trader.py`를 GitHub Actions에서 자동 실행해  
매일 종목을 선정하고 한국투자증권 API로 매매·리포트·텔레그램 알림까지 처리하는 방법을 정리합니다.

> ⚠️ 실제 매매 전에는 반드시 `--dry-run` 모드로 충분히 검증하세요.

---

## 1. 전체 구조 개요

### 1.1 일일 실행 플로우

1. **GitHub Actions 트리거**
   - 매 영업일 특정 시각(예: 한국 시간 08:30, 장 시작 전)에 워크플로우 실행.
2. **환경 준비**
   - Python 세팅, `requirements.txt` 설치.
3. **데이터 로드 & 인디케이터 계산**
   - `reports.load_data()` 호출
   - pykrx 기반 OHLCV + KOSPI/KOSDAQ 지수 다운로드
   - `signals.compute_indicators()`로 가격/거래량/RSI/이동평균 등 인디케이터 계산
   - `universe_filter.filter_universe()`로 유동성·가격·상장일 등 조건 필터링
4. **multi_allocator_plus 목표 비중 계산**
   - `multi_allocator_plus.compute_security_targets()` 호출
   - 자식 전략별 백테스트를 돌려 일별 equity·weight history 생성
   - 전략별 일일 수익률로 롤링 모멘텀·샤프 계산
   - KOSDAQ 지수 기반으로 레짐·익스포저·스트레스 레이어 적용
   - 최종적으로 **개별 티커별 목표 비중(포지션 weight)**을 만든 후 가장 최근 일자의 row만 사용
5. **한국투자증권 계좌 스냅샷 조회**
   - `KoreaInvestmentConnector.get_account_balance()`로 총자산·매수가능금액 조회
   - `KoreaInvestmentConnector.get_account_stocks()`로 현재 보유 종목/수량/평단·평가금액 조회
6. **주문 계획 생성**
   - 계좌 총자산 × 목표 비중 = 각 종목의 목표 금액
   - `target_qty = floor(target_value / 현재가)`
   - 기존 보유 수량과 비교하여 `delta = target_qty - current_qty` 계산
     - `delta > 0` → **BUY** (추가 매수/신규 매수)
     - `delta < 0` → **SELL** (부분·전량 매도)
   - 타깃에 없는 기존 보유 종목은 전량 매도 계획 생성
   - 너무 작은 주문(`est_value < --min-trade`)은 스킵
7. **주문 실행**
   - `--dry-run` 모드: 주문 로그만 출력, KIS API 호출 없음
   - 실거래 모드(`--real`):  
     `KoreaInvestmentConnector.send_order()` 로 시장가(quote_type="03") 주문 전송
     - SELL 주문들을 먼저 실행 후 BUY 주문 실행
8. **일일 리포트 생성**
   - `automation.daily_reporter.DailyReporter`로  
     `reports/daily/daily_report_YYYYMMDD.csv/json` 저장
   - CSV: 종목, 액션, 수량, 추정가격/가치, 타깃 비중, 보유/목표 수량
   - JSON: 계좌 요약, 보유 포지션, 주문 리스트, 총자산 등 포함
9. **텔레그램 알림**
   - `automation.telegram_notifier.TelegramNotifier`가  
     실행 날짜, 총자산, 주문 요약(상위 5개), 리포트 파일명 등을 텍스트로 발송

---

## 2. 데이터 수집 관련 Q&A

### 2.1 별도의 “일별 데이터 수집 스크립트”가 필요한가?

현재 구조에서는 **반드시 필요하지 않습니다.**

- `reports.load_data()` 내부에서:
  - `data_loader.load_panel()`을 통해 전 종목 OHLCV를 pykrx로부터 다운로드
  - `signals.compute_indicators()`를 통해 지표 계산
  - `cache_manager`를 이용해 로컬 캐시(`data/` 하위)에 저장/재사용
- GitHub Actions 러너는 매 실행마다 깨끗한 환경이기 때문에  
  매번 필요한 구간의 데이터를 새로 받게 됩니다(속도는 다소 느릴 수 있지만 기능적으로 문제 없음).

**추천 패턴**

- **개인 PC/서버에서**는 `update_data.py` 또는 `build_cache_v2.py`를 크론으로 돌려  
  장 종료 후 데이터를 미리 캐싱하고, 트레이더는 `use_cache=True`로 빠르게 실행.
- **GitHub Actions에서만 돌린다면**, 현재처럼 `load_data()` 한 번으로  
  “데이터 수집 + 지표 계산 + 캐싱”을 동시에 수행해도 문제 없습니다.

---

## 3. GitHub Actions 설정 절차

### 3.1 시크릿 등록

GitHub 저장소에서:  
`Settings → Secrets and variables → Actions → New repository secret` 으로 아래 키들을 등록합니다.

- `TELEGRAM_BOT_TOKEN` : 텔레그램 봇 토큰
- `TELEGRAM_CHAT_ID` : 알림을 받을 채팅방 ID
- `KIS_APPKEY` : 한국투자증권 앱키
- `KIS_APPSECRET` : 한국투자증권 앱시크릿
- `KIS_ACCOUNT` : 계좌번호(예: 12345678-01 형태에서 앞자리 8자리)

> 주의: 실제 계좌로 운용하기 전에는 **반드시 모의투자(virtual_account=True)**로 충분히 테스트하세요.

### 3.2 워크플로우 파일

이미 저장소에는 예시 워크플로우 `.github/workflows/daily-trade.yml` 이 추가되어 있습니다.  
내용 요약:

```yaml
name: Daily Multi Allocator Trade

on:
  schedule:
    - cron: "30 23 * * 0-4"  # UTC 기준 → KST 08:30
  workflow_dispatch: {}

jobs:
  run-trader:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run daily trader
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          KIS_APPKEY: ${{ secrets.KIS_APPKEY }}
          KIS_APPSECRET: ${{ secrets.KIS_APPSECRET }}
          KIS_ACCOUNT: ${{ secrets.KIS_ACCOUNT }}
        run: |
          python multi_allocator_plus_trader.py --start-date 2020-01-01 --dry-run
```

#### 3.2.1 실거래 전환

- `--dry-run`을 제거하고 `--real`을 추가하면 실거래 모드로 전환됩니다.

```yaml
run: |
  python multi_allocator_plus_trader.py \
    --start-date 2020-01-01 \
    --real
```

> 권장:  
> 1) 먼저 `--dry-run` + 모의계좌(virtual_account=True)로 충분히 검증  
> 2) 모의계좌 실주문(virtual_account=False, 모의계좌용 APPKEY)  
> 3) 최종적으로 실계좌용 키로 전환

---

## 4. multi_allocator_plus 일일 로직 상세

### 4.1 데이터 로드 & 인디케이터

1. `multi_allocator_plus_trader.py` → `load_market_data()`  
2. `reports.load_data(use_cache, start_date)` 호출
   - `get_universe()`로 KOSPI/KOSDAQ + 필요 ETF 유니버스 생성
   - `load_panel()`이 pykrx로 OHLCV를 받아 `panel[ticker]` DataFrame 구성
   - `get_index_close()`로 KOSPI/KOSDAQ 지수 시계열 다운
   - `compute_indicators()` + `add_rel_strength()` 로 각 종목에:
     - `returns`, `ma5/20/60`, `rsi`, `std20/60`, `atr60`, 거래대금 등 계산
   - 결과를 `enriched: Dict[ticker, DataFrame]`로 반환
3. `filter_universe(enriched)`는 유동성/가격/상장일 등 조건으로 종목을 필터링

### 4.2 multi_allocator_plus 시그널 결합

`strategy = get_strategy("multi_allocator_plus")`  
`targets = strategy.compute_security_targets(enriched, market_index=idx_kosdaq)`

내부 흐름:

1. 자식 전략 실행
   - `kqm_small_cap_v22_short`, `hybrid_portfolio_v2_4`, `kqm_small_cap_v22`, `etf_defensive`, `k200_mean_rev`
   - 각 전략은 자체적으로 `run_backtest(enriched, market_index)`를 수행하며
     - 일별 equity curve
     - `_record_weights()`로 포지션 weight history를 남김
2. 전략별 일일 수익률 계산
   - `_build_child_returns()`에서 각 equity curve → 일일 수익률 DataFrame(`ret_df`)
3. 시장 레짐/익스포저 계산
   - `_prepare_regime(market_index)`로 지수의 `ma60`, `mom20`, `mom5`, `vol20` 등을 계산
   - `_dynamic_exposure()` + `vol_target_map`으로 날짜별 총 익스포저 시리즈 생성
4. 전략 가중치 계산
   - `_dynamic_strategy_weights()`로 롤링 모멘텀/샤프 기반 기본 가중치 계산
   - `_performance_stress()`로 최근 누적수익/드로다운이 나쁜 전략/기간에서 익스포저를 줄이고 스트레스 레벨 산출
   - `_apply_momentum_exposure_boost()`로 단기 모멘텀에 따라 익스포저 가속/감속
   - `_apply_regime_bias()` + `regime_role_targets`로 offensive/defensive/short 역할 비중 강제
   - `_apply_performance_filter()`, `_apply_fast_momentum_boost()`, `_apply_recent_acceleration()`로  
     최근 성과/모멘텀을 반영해 전략별 가중치를 미세 조정
5. 티커별 최종 비중 생성
   - 자식 전략별 weight history(또는 trade log → `_convert_trades_to_weights`)를 가져와  
     `_combine_strategy_targets()`로 전략 가중치 × 익스포저 × 종목별 weight 을 결합
   - 마지막 날짜의 row에서 `__CASH__`를 제외한 티커 열만 추출 후 100% 합이 되도록 재정규화  
     → 이게 **실거래에서 목표로 삼을 포트폴리오 비중**

### 4.3 계좌 & 주문 로직

`multi_allocator_plus_trader.py` 흐름:

1. `fetch_account_snapshot()`
   - `get_account_balance()` → `parse_account_balance_data()`  
     → total_value, available_cash, stock_value 등 추출
   - `get_account_stocks()` → 현재 보유 종목/수량/평단/현재가
2. `build_order_plan(targets, account, holdings)`
   - 총자산 × 목표 비중으로 각 종목의 target_value 계산
   - `target_qty = int(target_value / 현재가)`
   - 보유 수량 대비 `delta = target_qty - current_qty`
     - `delta > 0` → BUY delta
     - `delta < 0` → SELL |delta|
   - 타깃에 없는 종목은 target_qty=0 → SELL 전량
   - 너무 작은 주문은 `--min-trade` 기준으로 제거
   - SELL을 먼저, 큰 주문을 앞에 두도록 정렬
3. `execute(plans, account, holdings, as_of)`
   - `dry_run=True` → 주문 전송 없이 로그/리포트/텔레그램만
   - `dry_run=False` & `virtual_account=False` → 실거래
   - 각 주문은 `send_order(..., quote_type="03")`로 시장가로 전송
4. `DailyReporter.save_report()`
   - `reports/daily/daily_report_YYYYMMDD.csv/json`에 결과 저장
5. `TelegramNotifier.send_message()`
   - 실행 날짜, 총자산, 상위 5개 주문 요약, 리포트 파일명을 발송

---

## 5. 정리

- **일별 데이터 수집**은 `load_data()`가 pykrx 호출 + 인디케이터 계산까지 수행하기 때문에  
  GitHub Actions 환경에서도 별도의 스크립트 없이 자동으로 처리됩니다.
- `multi_allocator_plus_trader.py`는  
  “데이터 로드 → multi_allocator_plus 목표비중 계산 → KIS 계좌 조회 → 주문 계획 → (선택) 실 주문 → 리포트/텔레그램”  
  의 풀 파이프라인을 제공하므로, GitHub Actions 또는 자체 크론에서 쉽게 자동화할 수 있습니다.
- 실전 투입 전에는 반드시:
  - 충분한 기간의 백테스트/워크포워드 테스트
  - 모의투자 계좌에서의 리허설
  - 주문 한도/슬리피지/수수료 등을 감안한 리스크 점검  
  을 선행한 후 실계좌 키를 연결하는 것을 권장합니다.

