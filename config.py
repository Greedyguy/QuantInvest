from datetime import date

START = "2020-01-01"  # 캐시 파일 실제 시작일
# START = "2024-06-01"  # 캐시 파일 실제 시작일
END   = None  # None이면 오늘 이전 거래일까지
MARKETS = ["KOSPI", "KOSDAQ"]

# Liquidity
MIN_AVG_TRD_AMT_20 = 5e8  # 5억원 (원화)

# Portfolio & risk
MAX_HOLDINGS = 5
MAX_WEIGHT_PER_NAME = 0.30
SECTOR_CAP = 3          # 섹터 제한 (1차 OFF)
USE_SECTOR_CAP = False  # 섹터 매핑 붙이면 True

# Execution
ENTER_NEXT_OPEN = True
FORBID_LIMIT_UP_ENTRY = True

# Stops / exits
STOP_LOSS = -0.08
TAKE_PROFIT = 0.15
MAX_HOLD_DAYS = 10

# Costs
FEE_PER_SIDE = 0.000140527     # 0.0140527% per side (KIS 뱅키스) 
VENUE_FEE_PER_SIDE = 0.000036396  # 유관기관 제비용(참고)
SLIPPAGE_ENTRY = 0.002   # +0.20%
SLIPPAGE_EXIT  = 0.002   # +0.20%

# Tax schedule (매도시에만)
TAX_RATE_SELL = 0.0015   # 기본 0.15% (시나리오B면 0.002 로 변경)
TAX_SCHEDULE = [  # (YYYYMMDD, tax_rate) 효과발생일 순서
    ("20250101", 0.0015),
    # 예: 개편 시행 확정시 ("20260101", 0.0020)
]

# Signal weights
W = dict(LC=2.0, VS=1.5, BO=1.0, RS=1.0, VCP=0.5, GG=0.5)

RANDOM_SEED = 42

# Data freshness (business days allowed between cached data end and request end)
DATA_STALE_TOLERANCE_BDAYS = 3
