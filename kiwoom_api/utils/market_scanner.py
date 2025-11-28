"""
동적 시장 스캐닝 시스템
실시간으로 전체 시장에서 조건에 맞는 종목을 발굴
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScanningCriteria:
    """스캐닝 조건 - Threshold 최적화 적용"""
    min_price: float = 1000          # 최소 주가 (유지)
    max_price: float = 100000        # 최대 주가 (500,000 → 100,000 중소형주 포함)
    min_volume: int = 50000          # 최소 거래량 (100,000 → 50,000 완화)
    min_market_cap: float = 10e9     # 최소 시가총액 (500억 → 100억 중소형주 포함)
    exclude_sectors: List[str] = None # 제외 섹터
    technical_filters: Dict = None   # 기술적 필터
    
    def __post_init__(self):
        if self.exclude_sectors is None:
            self.exclude_sectors = ['관리종목', '투자위험종목']
        if self.technical_filters is None:
            self.technical_filters = {
                'min_rsi': 30,
                'max_rsi': 70,
                'min_volume_ratio': 1.2  # 평균 대비 거래량
            }

class MarketScanner:
    """동적 시장 스캐너"""
    
    def __init__(self, kis_connector=None):
        self.kis_connector = kis_connector
        self._stock_universe = None
        self._last_update = None
        self._cache_expiry = 3600  # 1시간
        
    async def get_stock_universe(self) -> pd.DataFrame:
        """전체 종목 리스트 조회 (캐시 포함)"""
        if (self._stock_universe is None or 
            self._last_update is None or 
            (datetime.now() - self._last_update).seconds > self._cache_expiry):
            
            await self._update_stock_universe()
        
        return self._stock_universe
    
    async def _update_stock_universe(self):
        """종목 유니버스 업데이트"""
        try:
            if self.kis_connector:
                # 한국투자증권 API로 전체 종목 리스트 조회
                universe_data = await self._fetch_from_kis_api()
            else:
                # API 없으면 기본 종목 리스트 사용
                universe_data = self._get_default_universe()
            
            self._stock_universe = pd.DataFrame(universe_data)
            self._last_update = datetime.now()
            logger.info(f"종목 유니버스 업데이트 완료: {len(self._stock_universe)}개 종목")
            
        except Exception as e:
            logger.error(f"종목 유니버스 업데이트 실패: {e}")
            if self._stock_universe is None:
                self._stock_universe = pd.DataFrame(self._get_default_universe())
    
    async def _fetch_from_kis_api(self) -> List[Dict]:
        """한국투자증권 API에서 종목 정보 조회"""
        # 실제 구현 시 KIS API 호출
        # 현재는 모의 데이터
        return self._get_default_universe()
    
    def _get_default_universe(self) -> List[Dict]:
        """기본 종목 유니버스 (확장 가능)"""
        return [
            # 대형주
            {'code': '005930', 'name': '삼성전자', 'sector': 'IT', 'market_cap': 500e12},
            {'code': '000660', 'name': 'SK하이닉스', 'sector': 'IT', 'market_cap': 100e12},
            {'code': '035420', 'name': 'NAVER', 'sector': 'IT', 'market_cap': 50e12},
            {'code': '051910', 'name': 'LG화학', 'sector': '화학', 'market_cap': 40e12},
            {'code': '373220', 'name': 'LG에너지솔루션', 'sector': '배터리', 'market_cap': 80e12},
            
            # 중형주
            {'code': '067310', 'name': '하나마이크론', 'sector': '반도체', 'market_cap': 5e12},
            {'code': '035720', 'name': '카카오', 'sector': 'IT', 'market_cap': 30e12},
            {'code': '028260', 'name': '삼성물산', 'sector': '건설', 'market_cap': 25e12},
            {'code': '086790', 'name': '하나금융지주', 'sector': '금융', 'market_cap': 20e12},
            {'code': '207940', 'name': '삼성바이오로직스', 'sector': '바이오', 'market_cap': 60e12},
            
            # 추가 가능한 종목들 (실제로는 수천 개)
            {'code': '005380', 'name': '현대차', 'sector': '자동차', 'market_cap': 45e12},
            {'code': '012330', 'name': '현대모비스', 'sector': '자동차부품', 'market_cap': 20e12},
            {'code': '000270', 'name': '기아', 'sector': '자동차', 'market_cap': 35e12},
            {'code': '068270', 'name': '셀트리온', 'sector': '바이오', 'market_cap': 45e12},
            {'code': '326030', 'name': 'SK바이오팜', 'sector': '제약', 'market_cap': 8e12},
            {'code': '042700', 'name': '한미반도체', 'sector': '반도체', 'market_cap': 3e12},
            {'code': '357780', 'name': '솔브레인', 'sector': '화학', 'market_cap': 2e12},
            {'code': '042660', 'name': '대우조선해양', 'sector': '조선', 'market_cap': 15e12},
            {'code': '161390', 'name': '한국타이어앤테크놀로지', 'sector': '타이어', 'market_cap': 10e12},
            {'code': '034020', 'name': '두산에너빌리티', 'sector': '에너지', 'market_cap': 12e12},
        ]
    
    async def scan_for_candidates(self, criteria: ScanningCriteria, strategy_name: str = None) -> List[str]:
        """조건에 맞는 종목 스캔"""
        universe = await self.get_stock_universe()
        
        # 기본 필터링
        candidates = universe[
            (universe['market_cap'] >= criteria.min_market_cap) &
            (~universe['sector'].isin(criteria.exclude_sectors))
        ].copy()
        
        # 전략별 추가 필터링
        if strategy_name == 'minervini_orb':
            candidates = await self._apply_minervini_filters(candidates, criteria)
        elif strategy_name == 'high_volatility':
            candidates = await self._apply_volatility_filters(candidates, criteria)
        
        # 실시간 시장 데이터 기반 필터링
        final_candidates = await self._apply_realtime_filters(candidates.copy(), criteria)
        
        result_codes = final_candidates['code'].tolist()
        logger.info(f"스캔 결과: {len(result_codes)}개 종목 발견 (전략: {strategy_name})")
        
        return result_codes
    
    async def _apply_minervini_filters(self, candidates: pd.DataFrame, criteria: ScanningCriteria) -> pd.DataFrame:
        """미너비니 전략 필터"""
        # 실제 구현 시 기술적 분석 조건 추가
        # - 52주 최고가 대비 가격 위치
        # - 이동평균선 정렬
        # - 거래량 급증 여부
        return candidates
    
    async def _apply_volatility_filters(self, candidates: pd.DataFrame, criteria: ScanningCriteria) -> pd.DataFrame:
        """고변동성 전략 필터"""
        # 실제 구현 시 변동성 관련 조건 추가
        # - 최근 변동성 수준
        # - 가격 급변동 여부
        return candidates
    
    async def _apply_realtime_filters(self, candidates: pd.DataFrame, criteria: ScanningCriteria) -> pd.DataFrame:
        """실시간 시장 데이터 필터"""
        filtered_candidates = []
        
        for _, stock in candidates.iterrows():
            try:
                # 실시간 가격 및 거래량 정보 조회
                current_price = await self._get_current_price(stock['code'])
                current_volume = await self._get_current_volume(stock['code'])
                
                # 가격 범위 체크
                if criteria.min_price <= current_price <= criteria.max_price:
                    # 거래량 체크
                    if current_volume >= criteria.min_volume:
                        filtered_candidates.append(stock)
                        
            except Exception as e:
                logger.warning(f"종목 {stock['code']} 실시간 필터링 실패: {e}")
                continue
        
        return pd.DataFrame(filtered_candidates) if filtered_candidates else pd.DataFrame()
    
    async def _get_current_price(self, code: str) -> float:
        """실시간 현재가 조회"""
        if self.kis_connector:
            return await self.kis_connector.get_current_price(code)
        else:
            # 모의 데이터
            import random
            return random.uniform(10000, 100000)
    
    async def _get_current_volume(self, code: str) -> int:
        """실시간 거래량 조회"""
        if self.kis_connector:
            return await self.kis_connector.get_current_volume(code)
        else:
            # 모의 데이터
            import random
            return random.randint(100000, 1000000)
    
    def get_stock_name(self, code: str) -> str:
        """종목 코드로 종목명 조회 (동적)"""
        if self._stock_universe is not None:
            match = self._stock_universe[self._stock_universe['code'] == code]
            if not match.empty:
                return match.iloc[0]['name']
        
        # 캐시에 없으면 API 조회 또는 기본값
        return f"종목_{code}"
    
    async def update_stock_info(self, code: str) -> Optional[Dict]:
        """특정 종목 정보 업데이트/추가"""
        try:
            if self.kis_connector:
                info = await self.kis_connector.get_stock_info(code)
                if info and self._stock_universe is not None:
                    # 기존 정보 업데이트 또는 새로 추가
                    existing_idx = self._stock_universe[self._stock_universe['code'] == code].index
                    if not existing_idx.empty:
                        self._stock_universe.loc[existing_idx[0]] = info
                    else:
                        self._stock_universe = pd.concat([self._stock_universe, pd.DataFrame([info])], ignore_index=True)
                    return info
        except Exception as e:
            logger.error(f"종목 {code} 정보 업데이트 실패: {e}")
        
        return None

# 전역 스캐너 인스턴스
market_scanner = MarketScanner()

async def get_dynamic_candidates(strategy_name: str, max_count: int = 20) -> List[str]:
    """전략별 동적 후보 종목 조회 - Threshold 최적화 적용"""
    criteria = ScanningCriteria()
    
    # 전략별 조건 조정 (기존보다 완화)
    if strategy_name == 'minervini_orb':
        criteria.min_market_cap = 30e9   # 300억 이상 (1000억 → 300억 완화)
        criteria.min_volume = 100000     # 거래량 적정 (200,000 → 100,000)
        criteria.max_price = 150000      # 더 넓은 가격대 허용
    elif strategy_name == 'high_volatility':
        criteria.min_market_cap = 10e9   # 100억 이상 (500억 → 100억 완화)
        criteria.max_price = 80000       # 가격대 확대 (50,000 → 80,000)
        criteria.min_volume = 50000      # 거래량 완화
    
    candidates = await market_scanner.scan_for_candidates(criteria, strategy_name)
    
    # 최대 개수 제한
    return candidates[:max_count]

def get_stock_name_dynamic(code: str) -> str:
    """동적 종목명 조회"""
    return market_scanner.get_stock_name(code) 