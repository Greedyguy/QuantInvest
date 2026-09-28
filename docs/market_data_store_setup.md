# 재사용 시장 데이터 저장소 — KIS 우선, 주문 기능 없음

## 현재 상태

2026-09-28 구현. 기존 일별 신호/주문 워크플로와 운영 파일은 변경하지 않았다.
새 코드가 수집·보관·오프라인 후보 목록 생성만 수행한다. 종목 선정 결과는 거래대금 순의
연구용 후보 목록이지 매수 신호나 검증된 상대강도 전략이 아니다.

- 기존 연구 원본 183개 / 149,328행(중복 기간·원주가/수정주가를 포함한 적재 행 수)을
  2019-01-01~2026-08-31 범위로 로컬 저장소에 적재했다. 재실행 신규 적재 0개.
- 공식 KIS 마스터 2개를 실제 다운로드했다. 현재 스냅샷 3,942개 중 주식류 2,767개,
  ETF 1,175개다. 이름에서 레버리지/배수를 확인한 ETF 72개를 수집 후보에서 제외해
  3,870개가 잠정 수집 대상이다. 이 중 ETF 1,103개는 **정식 배율 검증 전**이다.
- 기존 원주가로 삼성전자·카카오의 2024-06-03~2026-08-31 1,092행을 네트워크 없이 읽었다.
- 로컬 KIS 키 인증은 거절됐지만 **GitHub Secrets의 키로 실제 수집·저장에 성공**했다.
  첫 연결 검증은 000020의 2019-01-01~2019-03-31 원주가·수정주가 각 59행이다.
  로컬 파일과 GitHub Secrets의 키가 같거나 모두 유효하다고 가정하지 않는다.
- `Greedyguy/stock_rawdata` 비공개 여부를 확인하고 기존 자료와 신규 KIS 자료를 적재했다.
  별도 비공개 Actions에서 파일 해시 및 2종목 1,092행 오프라인 재사용 검증도 통과했다.
- 전체 2020~2026년 8월 원주가·상장폐지·기업행동 검증 완료가 아니다.

## 데이터와 코드 저장소 분리

코드 저장소 QuantInvest는 공개다. 원시 자료는 공개 저장소, Release, 공개 Actions
artifact에 업로드하지 않는다. 별도 비공개 데이터 저장소의 `store/`에만 보관한다.
데이터 저장소는 README 등 첫 커밋과 `main` 브랜치가 있어야 한다.

코드 저장소 Actions 설정:

| 종류 | 이름 | 내용 |
|---|---|---|
| 기존 Secret | `KIS_APP_KEY`, `KIS_APP_SECRET` | 실전 환경의 시세 조회 가능한 키. 계좌번호 불필요 |
| 새 Secret | `MARKET_DATA_REPO_TOKEN` | **해당 비공개 데이터 저장소만** Contents read/write 가능한 fine-grained token |
| Variable | `MARKET_DATA_REPOSITORY` | `Greedyguy/stock_rawdata`（비공개 확인 완료） |
| Variable | `MARKET_DATA_BRANCH` | 기본 `main` |
| Variable | `MARKET_DATA_PIPELINE_ENABLED` | 일배치 활성화 시에만 `true` |
| Variable | `MARKET_DATA_DAILY_MAX_REQUESTS` | 기본 10000, 실행당 시세 요청 상한 |

KRX Open API는 선택적인 별도 경로다. 기존 KIS 키와 혼용하지 않는다. 사용하려면
`KRX_OPENAPI_KEY` 및 코스피·코스닥·ETF 서비스 승인이 따로 필요하다.
브라우저 KRX 로그인 정보를 추출하거나 세션을 GitHub에 복사하지 않는다.

비공개 저장소 연결 전에는 공개 저장소로 대체 저장하지 않고 작업을 중단한다.
키·토큰은 원본 자료/매니페스트에 넣지 않으며 임의 오류 응답도 로그로 출력하지 않는다.

## Actions 실행

`Market Data Store (no orders)`는 다음 모드를 제공한다.

1. `master`: 현재 KIS 코스피·코스닥 마스터 저장. 조회 시점 날짜를 과거 날짜로 위조할 수 없다.
2. `backfill`: 기본 2019-01-01~2026-08-31, 기본 실행당 300회. 동일 시작·끝·마스터 날짜로
   반복하면 완료 조각을 검증해 재사용하고 다음 조각으로 이어간다. 상한 도달은 완료가 아닌
   `checkpoint_budget_exhausted`다. 무제한 자동 재호출은 하지 않는다.
3. `incremental`: 전날까지 최근 8일을 다시 확인해 정정 자료를 새 버전으로 저장한다.
4. `select`: 지정 날짜의 저장 자료만 사용한다. 기본값은 검증되지 않은 ETF와 인버스를 제외한다.
   같은 날짜의 종목 마스터나 대상 종목 가격이 하나라도 없으면 결과 생성을 중단한다.

예약은 활성화 변수 및 기본 브랜치 반영 이후 화~토 한국시각 06:20이다.
현재는 별도 브랜치에서 수동 수집을 검증하는 단계이며 예약 수집은 아직 활성화하지 않았다.
수집/선정을 주문 워크플로에 자동 연결하지 않았다. 기존 장마감 16:10 작업이 사용할 데이터의
확정 시각·신선도·신호 규칙을 별도 검증한 뒤 읽기 어댑터를 연결해야 한다.
Actions 실행 중 키 발급/조회 제한·일부 종목 조회 오류가 있으면 숨기지 않고 실패한다.
그 전까지 검증된 조각은 비공개 저장소에 체크포인트로 커밋한다. 강제 푸시/리셋은 없다.
API 호출은 순차 처리하고 시세 호출 사이 0.6초를 둔다. 여러 작업의 동일 키 호출량을 합산해야
하므로 운영 시간과 겹칠 경우 제한/지연을 다시 조정해야 한다. 기본 10000회는 호출 보장이 아니다.

`Market data offline tests`는 비밀키 없이 별도 브랜치/PR에서 자동 검사한다.
`codex/market-data-store` 브랜치의 수집 워크플로 변경을 푸시하면 시세 요청을 **2회**로
제한한 연결 검증을 수행한다. 목적지는 동일한 비공개 저장소이며 주문 기능은 없다.
기본 브랜치 병합이나 일배치 활성화 없이 GitHub에 등록된 KIS 키로 연결을 검증하기 위한 경로다.
연결 토큰이 없거나 목적지의 비공개 여부를 확인할 수 없으면 시세 조회 전에 중단한다.

## 저장 구조와 재현성

- `masters/`: 조회 당시 종목 목록 원본 ZIP과 정규화 표. 날짜와 ISIN을 함께 보존.
- `kis_segments/`: 종목·원주가/수정주가·최대 90일 기간별 원본 응답과 표.
- `indexes/kis/`: 종목 접두사별 작은 인덱스. 거대한 단일 파일로 GitHub 파일 제한에 닿지 않게 분리.
- `snapshots/`: 선택적 KRX 날짜별 시장 전체 스냅샷.
- `seeds/`: 이전 연구에서 확보한 **일부 종목 자료**. 전체 시장 스냅샷으로 승격하지 않음.
- `manifest.json`: 버전·출처·검증 상태·파일 해시. 시세 파일은 내용 기반 이름으로 보존.

키와 계좌정보는 저장하지 않는다. 원본과 정규화 파일의 SHA256을 확인하며, 깨진 캐시를
정상으로 재사용하지 않는다. 정정/변환 변경은 새 파일이며 이전 파일을 덮어쓰지 않는다.
실험은 비공개 데이터 저장소의 **커밋 번호**를 고정해 재현한다. 최신 자료를 소급 적용한 결과와
당시 저장한 버전으로 재생한 결과는 구분한다. 대규모/장기 운영에서 Git 저장소 크기가 커지면
같은 매니페스트를 유지한 비공개 객체 저장소로 옮기는 것이 후속 과제다.

## 로컬 사용 예

```sh
python scripts/market_data_pipeline.py --store data/market_store kis-master
python scripts/market_data_pipeline.py --store data/market_store kis-collect \
  --master-date 2026-09-28 --start 2019-01-01 --end 2026-08-31 --max-requests 300
python scripts/market_data_pipeline.py --store data/market_store audit
```

키는 환경변수 또는 명시적 `--env-file`로만 전달한다. 모의 키는 로컬 `--demo` 옵션을 써야 한다.
Actions는 현재 기존 실전 키용이며 모의 키를 자동 추측하지 않는다.

오프라인 읽기:

```python
from market_data_store import MarketStore
from kis_market_collection import load_kis_panel, select_kis
store = MarketStore('data/market_store')
prices = load_kis_panel(store, {'005930'}, '2020-01-01', '2024-12-31', basis='raw')
# 명시적으로 같은 시점 종목 목록과 가격을 사용; 누락 시 실패.
universe = select_kis(store, '2026-09-28', '2026-09-28')
```

이전 연구 자료는 `store.load_seed_panel(..., price_basis='raw')`처럼 별도 조회한다.
그 자료가 없는 종목/기간은 네트워크 수집이나 수정주가로 몰래 대체하지 않는다.

## 아직 인증하지 않은 항목

KIS 현재 마스터에서 과거 가격을 받는 것은 **현재 종목들의 과거 이력**이다. 과거에만 존재했던
상장폐지 종목 전체나 코드 재사용 이력을 복원했다고 볼 수 없다. 마스터의 상장일을 이용한
호출 생략도 현재 식별자 기준이며, 과거 재상장·합병 이력의 완전한 해석은 아니다.
역사적 투자 가능 종목군은 별도 과거 마스터/공식 시장 전체 자료로 보완해야 한다.

ETF 상품명 배수 필터는 1차 분류다. 200배당커버드콜 같은 이름을 200배 상품으로 오인하지 않는
검사를 넣었지만, 공식 추종배율/유형 메타데이터 검증을 대신하지 않는다.
일반 ETF까지 포함한 연구 후보 목록은 명시적인 `--include-unverified-etfs`로만 허용하고
여전히 `orders_enabled=false`다. -1배 인버스는 수집하되 기본 후보에서는 제외한다.

수정주가 응답은 그 수집 시점의 조정 기준이다. 원주가와 분리해도 기업행동에 따른 실제 주식 수,
배당·ETF 분배금·과세를 자동 해결하지 않는다. 현재 저장소는 총수익 회계나 실전 전략 승인 시스템이 아니다.

## 확인한 공식 명세

- [KIS 기간별 일봉 조회 공식 예제](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/inquire_daily_itemchartprice/inquire_daily_itemchartprice.py):
  회당 최대 100건, `FID_ORG_ADJ_PRC=1` 원주가 / `0` 수정주가, KRX 시장 `J`.
- [KIS 코스피 마스터 공식 예제](https://github.com/koreainvestment/open-trading-api/blob/main/stocks_info/kis_kospi_code_mst.py),
  [코스닥 마스터 공식 예제](https://github.com/koreainvestment/open-trading-api/blob/main/stocks_info/kis_kosdaq_code_mst.py).
  본 구현은 예제에 있는 TLS 검증 해제나 ZIP 전체 추출/임시 파일 삭제를 사용하지 않는다.
- [KRX Open API 이용 절차](https://openapi.krx.co.kr/contents/OPP/INFO/OPPINFO003.jsp).
