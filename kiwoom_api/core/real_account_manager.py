"""
실제 계좌 정보 관리 시스템
- 실제 계좌 잔고 조회
- 기존 보유 포지션 반영
- 모의투자/실제매매 모드 일관성 보장
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import asyncio
from decimal import Decimal

from .korea_investment_connector import KoreaInvestmentConnector
from .pykiwoom_connector import PyKiwoomConnector


@dataclass
class RealPosition:
    """실제 보유 포지션 정보"""
    symbol: str
    name: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_rate: float
    purchase_date: Optional[str] = None
    sector: Optional[str] = None
    
    @property
    def total_cost(self) -> float:
        """총 매입 금액"""
        return self.avg_price * self.quantity
    
    @property
    def profit_loss(self) -> float:
        """손익 금액"""
        return self.market_value - self.total_cost


@dataclass
class RealAccountInfo:
    """실제 계좌 정보"""
    account_no: str
    total_cash: float  # 예수금 총액
    available_cash: float  # 매수 가능 금액
    total_value: float  # 총 평가 금액
    total_profit_loss: float  # 총 손익
    profit_loss_rate: float  # 손익률
    positions: List[RealPosition] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def stock_value(self) -> float:
        """보유 주식 평가 금액"""
        return sum(pos.market_value for pos in self.positions)
    
    @property
    def position_count(self) -> int:
        """보유 종목 수"""
        return len(self.positions)


class RealAccountManager:
    """실제 계좌 정보 관리자"""
    
    def __init__(self, use_korea_investment: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_korea_investment = use_korea_investment
        
        # API 커넥터 초기화
        self.kis_connector = None
        self.kiwoom_connector = None
        
        # 계좌 정보 캐시
        self.account_info: Optional[RealAccountInfo] = None
        self.last_update_time: Optional[datetime] = None
        self.update_interval = 30  # 30초마다 업데이트
        
        # 포지션 추적
        self.position_cache: Dict[str, RealPosition] = {}
        
    async def initialize(self) -> bool:
        """계좌 관리자 초기화"""
        try:
            self.logger.info("🏦 실제 계좌 관리자 초기화 중...")
            
            if self.use_korea_investment:
                # 한국투자증권 API 초기화
                if not await self._initialize_kis():
                    return False
                    
            else:
                # 키움 API 초기화 (향후 구현)
                self.logger.warning("키움 API는 현재 미구현")
                return False
            
            # 초기 계좌 정보 로드
            await self.refresh_account_info()
            
            if self.account_info:
                self.logger.info(f"✅ 계좌 초기화 완료")
                self.logger.info(f"   📊 계좌번호: {self.account_info.account_no}")
                self.logger.info(f"   💰 가용자금: {self.account_info.available_cash:,.0f}원")
                self.logger.info(f"   📈 보유종목: {self.account_info.position_count}개")
                
                # 기존 포지션 로그 출력
                for pos in self.account_info.positions:
                    self.logger.info(f"   📊 {pos.name}({pos.symbol}): {pos.quantity}주, "
                                   f"평단가 {pos.avg_price:,.0f}원")
                return True
            else:
                self.logger.error("❌ 계좌 정보 로드 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"계좌 관리자 초기화 실패: {e}")
            return False
    
    async def _initialize_kis(self) -> bool:
        """한국투자증권 API 초기화"""
        try:
            # 환경변수에서 설정 로드
            from config.secrets_manager import SecretsManager
            import os
            
            try:
                secrets_manager = SecretsManager()
                appkey = secrets_manager.retrieve_secret('korea_investment_appkey')
                appsecret = secrets_manager.retrieve_secret('korea_investment_appsecret')
                account = secrets_manager.retrieve_secret('korea_investment_account')
            except Exception as e:
                self.logger.warning(f"SecretsManager 로드 실패, 환경변수 사용: {e}")
                appkey = appsecret = account = None
            
            # 환경변수에서 fallback
            if not appkey:
                appkey = os.getenv('KOREA_INVESTMENT_APPKEY') or os.getenv('KIS_APP_KEY') or os.getenv('KIS_APPKEY')
            if not appsecret:
                appsecret = os.getenv('KOREA_INVESTMENT_APPSECRET') or os.getenv('KIS_APP_SECRET') or os.getenv('KIS_APPSECRET')
            if not account:
                account = os.getenv('KOREA_INVESTMENT_ACCOUNT') or os.getenv('KIS_ACCOUNT')
            
            # 실거래 모드로 KoreaInvestmentConnector 초기화
            self.kis_connector = KoreaInvestmentConnector(
                appkey=appkey or "",
                appsecret=appsecret or "",
                account=account or "",
                virtual_account=False  # 🔥 실거래 모드 명시적 설정
            )
            
            self.logger.info(f"[DEBUG] RealAccountManager KIS Connector 설정:")
            self.logger.info(f"[DEBUG]   virtual_account: {self.kis_connector.virtual_account}")
            self.logger.info(f"[DEBUG]   base_url: {self.kis_connector.BASE_URL}")
            
            # 접속 및 토큰 발급 (connect 메서드가 인증을 포함함)
            if not self.kis_connector.connect():
                self.logger.error("한국투자증권 API 접속 실패")
                return False
                
            self.logger.info("✅ 한국투자증권 API 초기화 완료 (실거래 모드)")
            return True
            
        except Exception as e:
            self.logger.error(f"한국투자증권 API 초기화 오류: {e}")
            return False
    
    async def refresh_account_info(self, force: bool = False) -> bool:
        """계좌 정보 새로고침"""
        try:
            # 캐시 확인 (강제 업데이트가 아니면)
            if not force and self.last_update_time:
                elapsed = (datetime.now() - self.last_update_time).total_seconds()
                if elapsed < self.update_interval:
                    return True
            
            self.logger.info("🔄 계좌 정보 업데이트 중...")
            
            if self.use_korea_investment and self.kis_connector:
                account_data = await self._fetch_kis_account_info()
                if account_data:
                    self.account_info = account_data
                    self.last_update_time = datetime.now()
                    self._update_position_cache()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"계좌 정보 업데이트 실패: {e}")
            return False
    
    async def _fetch_kis_account_info(self) -> Optional[RealAccountInfo]:
        """한국투자증권 API로 계좌 정보 조회"""
        try:
            # 계좌 잔고 조회
            balance_result = self.kis_connector.get_account_balance()
            if not balance_result:
                self.logger.error("계좌 잔고 조회 실패")
                return None
            
            # 잔고 데이터 파싱
            balance_info = self.kis_connector.parse_account_balance_data(balance_result)
            if not balance_info:
                self.logger.error("계좌 잔고 데이터 파싱 실패")
                return None
            
            # 보유 종목 조회
            positions_data = self.kis_connector.get_account_stocks()
            positions = []
            
            if positions_data:
                for stock_data in positions_data:
                    position = RealPosition(
                        symbol=stock_data.get('symbol', ''),
                        name=stock_data.get('name', ''),
                        quantity=int(stock_data.get('quantity', 0)),
                        avg_price=float(stock_data.get('avg_price', 0)),
                        current_price=float(stock_data.get('current_price', 0)),
                        market_value=float(stock_data.get('market_value', 0)),
                        unrealized_pnl=float(stock_data.get('unrealized_pnl', 0)),
                        unrealized_pnl_rate=float(stock_data.get('unrealized_pnl_rate', 0)),
                        purchase_date=stock_data.get('purchase_date'),
                        sector=stock_data.get('sector')
                    )
                    positions.append(position)
            
            # 계좌 정보 생성
            account_info = RealAccountInfo(
                account_no=balance_info.get('account_no', ''),
                total_cash=float(balance_info.get('total_cash', 0)),
                available_cash=float(balance_info.get('available_cash', 0)),
                total_value=float(balance_info.get('total_value', 0)),
                total_profit_loss=float(balance_info.get('total_profit_loss', 0)),
                profit_loss_rate=float(balance_info.get('profit_loss_rate', 0)),
                positions=positions
            )
            
            self.logger.info(f"💰 계좌 정보 업데이트:")
            self.logger.info(f"   총 자산: {account_info.total_value:,.0f}원")
            self.logger.info(f"   가용 현금: {account_info.available_cash:,.0f}원")
            self.logger.info(f"   보유 종목: {len(positions)}개")
            
            return account_info
            
        except Exception as e:
            self.logger.error(f"KIS 계좌 정보 조회 오류: {e}")
            return None
    
    def _update_position_cache(self):
        """포지션 캐시 업데이트"""
        self.position_cache.clear()
        if self.account_info:
            for position in self.account_info.positions:
                self.position_cache[position.symbol] = position
    
    def get_available_cash(self) -> float:
        """매수 가능 금액 반환"""
        if not self.account_info:
            return 0.0
        return self.account_info.available_cash
    
    def get_position(self, symbol: str) -> Optional[RealPosition]:
        """특정 종목의 포지션 정보 반환"""
        return self.position_cache.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """특정 종목 보유 여부 확인"""
        return symbol in self.position_cache
    
    def get_total_value(self) -> float:
        """총 계좌 가치 반환"""
        if not self.account_info:
            return 0.0
        return self.account_info.total_value
    
    def calculate_position_size(self, symbol: str, price: float, 
                              risk_percent: float = 2.0) -> int:
        """실제 가용 자금 기반 포지션 사이징"""
        try:
            if not self.account_info:
                return 0
            
            # 위험 금액 계산 (총 자산의 일정 비율)
            risk_amount = self.account_info.total_value * (risk_percent / 100)
            
            # 가용 현금과 위험 금액 중 작은 값 사용
            max_investment = min(self.account_info.available_cash, risk_amount)
            
            # 주식 수 계산
            quantity = int(max_investment / price)
            
            # 최소 주문 단위 적용 (1주)
            return max(0, quantity)
            
        except Exception as e:
            self.logger.error(f"포지션 사이징 계산 오류: {e}")
            return 0
    
    def get_account_summary(self) -> Dict[str, Any]:
        """계좌 요약 정보 반환"""
        if not self.account_info:
            return {}
        
        return {
            'account_no': self.account_info.account_no,
            'total_cash': self.account_info.total_cash,
            'available_cash': self.account_info.available_cash,
            'stock_value': self.account_info.stock_value,
            'total_value': self.account_info.total_value,
            'total_profit_loss': self.account_info.total_profit_loss,
            'profit_loss_rate': self.account_info.profit_loss_rate,
            'position_count': self.account_info.position_count,
            'last_updated': self.account_info.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """포지션 요약 정보 반환"""
        if not self.account_info:
            return []
        
        positions = []
        for pos in self.account_info.positions:
            positions.append({
                'symbol': pos.symbol,
                'name': pos.name,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_rate': pos.unrealized_pnl_rate,
                'sector': pos.sector
            })
        
        return positions
    
    async def start_monitoring(self):
        """실시간 계좌 모니터링 시작"""
        self.logger.info("📊 실시간 계좌 모니터링 시작")
        
        while True:
            try:
                await self.refresh_account_info()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"계좌 모니터링 오류: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기


# 하나마이크론 포지션 시뮬레이션을 위한 테스트 함수
async def simulate_existing_position():
    """기존 포지션(하나마이크론 2주) 시뮬레이션"""
    manager = RealAccountManager()
    
    # 실제 계좌 정보 기반 시뮬레이션 데이터
    hanamicron_position = RealPosition(
        symbol='067310',
        name='하나마이크론',
        quantity=2,
        avg_price=12250.0,
        current_price=12210.0,  # 실제 현재가
        market_value=24420.0,   # 실제 평가금액
        unrealized_pnl=-80.0,   # 실제 손익
        unrealized_pnl_rate=-0.326530612244898,  # 실제 손익률
        sector='기타'
    )
    
    test_account = RealAccountInfo(
        account_no='50141961',    # 실제 계좌번호
        total_cash=9975500.0,     # 실제 현금 잔고
        available_cash=9975500.0, # 실제 가용 현금
        total_value=9999920.0,    # 실제 총 자산
        total_profit_loss=-80.0,  # 실제 총 손익
        profit_loss_rate=-0.00080000,  # 실제 손익률
        positions=[hanamicron_position]
    )
    
    manager.account_info = test_account
    manager._update_position_cache()
    
    return manager


if __name__ == "__main__":
    # 테스트 실행
    async def test_run():
        manager = await simulate_existing_position()
        
        print("=== 계좌 요약 ===")
        summary = manager.get_account_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\n=== 보유 포지션 ===")
        positions = manager.get_positions_summary()
        for pos in positions:
            print(f"{pos['name']}({pos['symbol']}): {pos['quantity']}주")
            print(f"  평단가: {pos['avg_price']:,.0f}원")
            print(f"  현재가: {pos['current_price']:,.0f}원")
            print(f"  평가금액: {pos['market_value']:,.0f}원")
    
    asyncio.run(test_run()) 