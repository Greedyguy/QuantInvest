# -*- coding: utf-8 -*-
"""
한국투자증권 OpenAPI 연결 관리자
기존 키움 API와 호환되는 인터페이스 제공
"""

import requests
import time
import logging
import threading
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# 종목명 관리 시스템 import (상대 경로로 접근)
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.stock_name_manager import get_stock_name
    STOCK_NAME_AVAILABLE = True
except ImportError:
    STOCK_NAME_AVAILABLE = False
    def get_stock_name(symbol: str) -> str:
        return f"종목{symbol}"

ORDER_ERR_MARKET_OPERATION_DATE_MISMATCH = -1001
ORDER_ERR_ACCOUNT_NOT_ELIGIBLE = -1002


class KoreaInvestmentConnector:
    """한국투자증권 OpenAPI 기반 연결 관리자"""
    
    def __init__(self, appkey: str = "", appsecret: str = "", account: str = "", 
                 virtual_account: bool = True, base_url: str = ""):
        """
        연결 관리자 초기화
        
        Args:
            appkey: 앱키 (빈 문자열이면 환경변수에서 자동 로드)
            appsecret: 앱시크릿 (빈 문자열이면 환경변수에서 자동 로드)
            account: 계좌번호 (8자리, 빈 문자열이면 환경변수에서 자동 로드)
            virtual_account: 모의투자 여부 (환경변수에서 자동 로드)
            base_url: API 기본 URL
        """
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 환경변수에서 설정 로드
        self.appkey = appkey or os.getenv('KIS_APP_KEY')
        self.appsecret = appsecret or os.getenv('KIS_APP_SECRET')
        self.account = account or os.getenv('KIS_ACCOUNT')
        self.virtual_account = virtual_account
        
        # 🔧 새로운 토큰 매니저 사용
        from .kis_token_manager import get_token_manager
        self.token_manager = get_token_manager()
        
        # 기존 토큰 관련 속성은 토큰 매니저에서 관리
        self.access_token = None  # 이제 토큰 매니저에서 가져옴
        self.token_expiry = 0
        
        # API 설정
        if base_url:
            self.BASE_URL = base_url
        else:
            if virtual_account:
                self.BASE_URL = "https://openapivts.koreainvestment.com:29443"
            else:
                self.BASE_URL = "https://openapi.koreainvestment.com:9443"
        
        # 계좌 정보
        self.account_list = [self.account] if self.account else []
        self.account_product_code = "01"
        
        # 연결 상태
        self.is_connected = False
        self.is_logged_in = False
        self.login_attempts = 0
        
        # API 호출 제한
        self.last_api_call = 0
        self.api_calls = 0
        
        # 로그인 시간
        self.last_login_time = None
        
        self.logger.info(f"[INIT] KoreaInvestmentConnector 초기화 ({'모의투자' if virtual_account else '실거래'})")
        
        # 통계 정보
        self.api_calls = 0
        self.last_login_time = None
        
        # 로깅
        self._lock = threading.Lock()
        
        # API 호출 제한 관리
        self.api_call_delay = 0.5  # 500ms 딜레이 (더 안전하게)
        
        # 에러 코드 처리
        self.error_code = KoreaInvestmentErrorCode()
    
    def _get_headers(self, tr_id: str) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        # 🔧 토큰 매니저에서 토큰 가져오기
        token = self.token_manager.get_valid_token()
        if not token:
            raise Exception("유효한 토큰을 가져올 수 없습니다")
        
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.appkey,
            "appsecret": self.appsecret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        
        return headers
    
    def connect(self) -> bool:
        """연결 및 인증 (기존 키움 API의 connect()와 호환)"""
        login_success = self.authenticate()
        return login_success
    
    def authenticate(self) -> bool:
        """토큰 발급 및 인증 - 이제 토큰 매니저 사용"""
        print("[DEBUG] authenticate() 진입")
        try:
            self.login_attempts += 1
            
            # 🔧 토큰 매니저에서 토큰 가져오기
            token = self.token_manager.get_valid_token()
            
            if token:
                self.access_token = token  # 호환성을 위해 저장
                self.token_expiry = self.token_manager.token_expiry
                self.is_connected = True
                self.is_logged_in = True
                self.last_login_time = datetime.now()
                
                # 토큰 발급 통계 로그
                stats = self.token_manager.get_daily_stats()
                self.logger.info(f"[OK] 토큰 인증 성공 (일일 발급: {stats['daily_requests']}/{stats['daily_limit']})")
                return True
            else:
                self.logger.error("[FAIL] 토큰 매니저에서 유효한 토큰을 가져올 수 없음")
                return False
                
        except Exception as e:
            self.logger.error(f"[FAIL] 인증 오류: {e}")
            return False
    
    def ensure_token(self):
        """토큰 만료 시 자동 갱신 - 이제 토큰 매니저 사용"""
        try:
            # 🔧 토큰 매니저에서 자동으로 처리
            token = self.token_manager.get_valid_token()
            if token:
                self.access_token = token  # 호환성을 위해 업데이트
                self.token_expiry = self.token_manager.token_expiry
            else:
                self.logger.error("[FAIL] 토큰 갱신 실패")
                raise Exception("토큰 갱신 실패")
        except Exception as e:
            self.logger.error(f"[FAIL] 토큰 확인 오류: {e}")
            raise
    
    def _wait_for_api_limit(self):
        """API 호출 제한 대기 (개선된 버전)"""
        current_time = time.time()
        
        if hasattr(self, 'last_api_call'):
            time_since_last_call = current_time - self.last_api_call
            
            # API 호출 간격 조정 (더 안전하게 - 초당 거래건수 초과 방지)
            min_interval = 1.5  # 1.5초 최소 간격
            if time_since_last_call < min_interval:
                wait_time = min_interval - time_since_last_call
                self.logger.debug(f"API 호출 제한 대기: {wait_time:.2f}초")
                time.sleep(wait_time)
        
        self.last_api_call = time.time()
    
    def get_account_list(self) -> List[str]:
        """계좌 목록 반환 (키움 API 호환)"""
        return self.account_list.copy()
    
    def get_login_info(self, info_type: str) -> str:
        """로그인 정보 조회 (키움 API 호환)"""
        if info_type == "ACCNO":
            return ";".join(self.account_list) + ";"
        elif info_type == "ACCOUNT_CNT":
            return str(len(self.account_list))
        else:
            return ""
    
    def get_connect_state(self) -> int:
        """연결 상태 조회 (키움 API 호환)"""
        return 1 if self.is_connected else 0
    
    def send_order(self, request_name: str, screen_no: str, account_no: str,
                   order_type: int, stock_code: str, quantity: int, price: int,
                   quote_type: str, original_order_no: str = "") -> int:
        """
        주문 전송 (키움 API 호환) - 재시도 로직 포함
        
        Args:
            request_name: 사용자 구분명
            screen_no: 화면번호
            account_no: 계좌번호
            order_type: 주문유형 (1:신규매수, 2:신규매도)
            stock_code: 종목코드
            quantity: 주문수량
            price: 주문가격
            quote_type: 호가구분 ("00":지정가, "03":시장가 등)
            original_order_no: 원주문번호
        
        Returns:
            int: 0(성공) 또는 에러코드
        """
        max_retries = 3
        retry_delay = 2.0  # 2초 대기
        
        for attempt in range(max_retries):
            try:
                # API 호출 제한 대기
                self._wait_for_api_limit()
                
                # 토큰 확인 및 갱신
                self.ensure_token()
                self.api_calls += 1
                
                # 매수/매도 구분
                side = "buy" if order_type == 1 else "sell"
                
                # 🆕 호가단위 가격 조정
                adjusted_price = price
                if price > 0 and quote_type != "03":  # 시장가가 아닌 경우만 조정
                    adjusted_price = self._adjust_to_tick_size(price)
                    if adjusted_price != price:
                        self.logger.info(f"[ADJUST] 호가단위 조정: {price:,}원 → {adjusted_price:,}원")
                
                # 호가구분 변환
                order_dvsn = "00"  # 기본값: 지정가
                if quote_type == "03":
                    order_dvsn = "01"  # 시장가
                
                url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
                
                # 실거래와 모의투자, 매수/매도에 맞는 TR 코드 선택
                if self.virtual_account:
                    # 모의투자
                    tr_code = "VTTC0802U" if order_type == 1 else "VTTC0801U"  # 매수:0802, 매도:0801
                else:
                    # 실거래
                    tr_code = "TTTC0802U" if order_type == 1 else "TTTC0801U"  # 매수:0802, 매도:0801
                
                headers = self._get_headers(tr_code)
                
                self.logger.info(f"[DEBUG] 주문 TR 코드: {tr_code} (virtual: {self.virtual_account}, order_type: {order_type})")
                
                data = {
                    "CANO": account_no,
                    "ACNT_PRDT_CD": self.account_product_code,
                    "PDNO": stock_code,
                    "ORD_DVSN": order_dvsn,
                    "ORD_QTY": str(quantity),
                    "ORD_UNPR": str(adjusted_price) if adjusted_price > 0 else "0",
                }
                
                self.logger.info(f"[TRY] 주문 시도 {attempt + 1}/{max_retries}: {stock_code} {quantity}주 ({side})")
                
                response = requests.post(url, headers=headers, json=data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("rt_cd") == "0":  # 성공
                        self.logger.info(f"[OK] 주문 전송 성공: {stock_code} {quantity}주")
                        return 0
                    else:
                        error_msg = result.get("msg1", "주문 실패")
                        error_code = result.get("msg_cd", "")
                        
                        # 특정 오류는 재시도하지 않음
                        if "credentials_type" in error_msg or "EGW00205" in error_code:
                            self.logger.error(f"[FAIL] 인증 오류 (재시도 안함): {error_msg}")
                            return -1

                        if "장운영일자가 주문일과 상이합니다" in error_msg:
                            self.logger.error(f"[FAIL] 휴장/영업일 불일치 (재시도 안함): {error_msg}")
                            return ORDER_ERR_MARKET_OPERATION_DATE_MISMATCH

                        if "교육이수가 등록/승인된 계좌만" in error_msg:
                            self.logger.error(f"[FAIL] 계좌/종목 자격요건 미충족 (재시도 안함): {error_msg}")
                            return ORDER_ERR_ACCOUNT_NOT_ELIGIBLE
                        
                        if attempt < max_retries - 1:
                            self.logger.warning(f"[RETRY] 주문 실패, 재시도 예정: {error_msg}")
                            time.sleep(retry_delay)
                            continue
                        else:
                            self.logger.error(f"[FAIL] 주문 최종 실패: {error_msg}")
                            return -1
                else:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"[RETRY] HTTP 오류 (코드: {response.status_code}), 재시도 예정")
                        time.sleep(retry_delay)
                        continue
                    else:
                        self.logger.error(f"[FAIL] 주문 요청 최종 실패: {response.text}")
                        return -1
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"[RETRY] 주문 오류, 재시도 예정: {e}")
                    time.sleep(retry_delay)
                    continue
                else:
                    self.logger.error(f"[FAIL] 주문 전송 최종 오류: {e}")
                    return -999
        
        return -999  # 모든 재시도 실패
    
    def get_account_balance(self, account_no: str = "") -> Dict[str, Any]:
        """
        계좌 잔고 조회 (키움 API 호환)
        
        Args:
            account_no: 계좌번호 (빈 문자열이면 기본 계좌 사용)
        
        Returns:
            Dict: 잔고 정보
        """
        try:
            self.ensure_token()
            self.api_calls += 1
            
            acc_no = account_no or self.account
            if not acc_no:
                self.logger.error("[FAIL] 계좌번호가 없습니다")
                return {}
            
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
            # 실거래와 모의투자에 맞는 TR 코드 선택
            tr_code = "VTTC8434R" if self.virtual_account else "TTTC8434R"
            headers = self._get_headers(tr_code)
            
            self.logger.info(f"[DEBUG] 잔고 조회 TR 코드: {tr_code} (virtual: {self.virtual_account})")
            
            # INQR_DVSN=01 이 기본값이며(대출일별), 일부 계좌에서는 02(종목별) 사용 시
            # INPUT INVALID_CHECK_INQR_DVSN 오류가 발생한다. 기본 01로 보내고,
            # 필요한 경우 02로 한 번 더 재시도한다.
            base_params = {
                "CANO": acc_no,
                "ACNT_PRDT_CD": self.account_product_code,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "N",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": ""
            }

            def _request_balance(inqr_dvsn: str):
                params = dict(base_params)
                params["INQR_DVSN"] = inqr_dvsn
                return requests.get(url, headers=headers, params=params)

            # 1차: INQR_DVSN=01
            response = _request_balance("01")
            retry_with_02 = False
            
            if response.status_code == 200:
                result = response.json()
                if result.get("rt_cd") == "0":
                    self.logger.info("[OK] 잔고 조회 성공")
                    # API 응답 디버깅을 위한 로깅 추가
                    output2 = result.get("output2", [{}])[0] if result.get("output2") else {}
                    self.logger.info(f"[DEBUG] 잔고 API 응답 output2 필드들:")
                    for key, value in output2.items():
                        self.logger.info(f"[DEBUG]   {key}: {value}")
                    return result
                # INQR_DVSN 관련 오류 시 02로 재시도
                error_msg = result.get("msg1", "잔고 조회 실패")
                if "INVALID_CHECK_INQR_DVSN" in error_msg:
                    retry_with_02 = True
                else:
                    self.logger.error(f"[FAIL] 잔고 조회 실패: {error_msg}")
                    return {}
            else:
                self.logger.error(f"[FAIL] 잔고 조회 요청 실패: {response.text}")
                return {}

            if retry_with_02:
                self.logger.warning("[RETRY] INQR_DVSN=02로 잔고 조회 재시도")
                response = _request_balance("02")
                if response.status_code == 200:
                    result = response.json()
                    if result.get("rt_cd") == "0":
                        self.logger.info("[OK] 잔고 조회 성공 (INQR_DVSN=02)")
                        return result
                    error_msg = result.get("msg1", "잔고 조회 실패")
                    self.logger.error(f"[FAIL] 잔고 조회 실패(재시도): {error_msg}")
                    return {}
                else:
                    self.logger.error(f"[FAIL] 잔고 조회 요청 실패(재시도): {response.text}")
                    return {}
                
        except Exception as e:
            self.logger.error(f"[FAIL] 잔고 조회 오류: {e}")
            return {}
    
    def get_account_stocks(self, account_no: str = "") -> List[Dict[str, Any]]:
        """
        계좌 보유 종목 조회
        
        Args:
            account_no: 계좌번호 (빈 문자열이면 기본 계좌 사용)
        
        Returns:
            List[Dict]: 보유 종목 리스트
        """
        try:
            # 먼저 잔고 조회로 보유 종목 정보 가져오기
            balance_result = self.get_account_balance(account_no)
            if not balance_result:
                return []
            
            stocks = []
            output1 = balance_result.get("output1", [])  # 보유종목 리스트
            
            for stock_data in output1:
                # 보유 수량이 0보다 큰 종목만 처리
                quantity = self._safe_int(stock_data.get("hldg_qty", "0"))
                if quantity <= 0:
                    continue
                
                # 현재가 조회 (실시간 가격 업데이트)
                symbol = stock_data.get("pdno", "")
                current_price_data = self.get_current_price(symbol)
                current_price = self._safe_float(
                    current_price_data.get("output", {}).get("stck_prpr", "0")
                ) if current_price_data else self._safe_float(stock_data.get("prpr", "0"))
                
                avg_price = self._safe_float(stock_data.get("pchs_avg_pric", "0"))
                market_value = current_price * quantity
                unrealized_pnl = market_value - (avg_price * quantity)
                unrealized_pnl_rate = (unrealized_pnl / (avg_price * quantity)) * 100 if avg_price > 0 else 0
                
                stock_info = {
                    "symbol": symbol,
                    "name": stock_data.get("prdt_name", ""),
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_rate": unrealized_pnl_rate,
                    "purchase_date": stock_data.get("ord_dt", ""),
                    "sector": self._get_sector_from_symbol(symbol)
                }
                stocks.append(stock_info)
            
            self.logger.info(f"[OK] 보유 종목 조회 성공: {len(stocks)}개")
            return stocks
                
        except Exception as e:
            self.logger.error(f"[FAIL] 보유 종목 조회 오류: {e}")
            return []
    
    def parse_account_balance_data(self, balance_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        계좌 잔고 조회 결과를 파싱하여 표준 형태로 변환
        
        Args:
            balance_result: get_account_balance() 결과
            
        Returns:
            Dict: 파싱된 계좌 정보
        """
        try:
            if not balance_result:
                return {}
            
            output2 = balance_result.get("output2", [{}])[0]  # 계좌 요약 정보
            
            # 🔧 현금 잔고 계산 로직 개선 - 실제 현금 잔고 우선 사용
            total_cash = self._safe_float(output2.get("dnca_tot_amt", "0"))  # 예수금총액
            
            # 매수가능금액 - 실제 현금 잔고 필드 우선 사용
            available_cash = 0.0
            possible_cash_fields = [
                "prvs_rcdl_excc_amt",  # 🆕 전일정산금액 (실제 현금 잔고) - 최우선
                "ord_psbl_cash",       # 주문가능현금 (일반적으로 매수가능금액)
                "dnca_tot_amt",        # 예수금총액 (fallback)
                "nxdy_excc_amt",       # 익일정산금액 (실제 현금과 다를 수 있음)
                "dncl_amt"             # 예수금 (최종 fallback)
            ]
            
            # 🔧 모든 필드값 로깅 (디버깅용)
            self.logger.info(f"[DEBUG] 계좌 잔고 API 응답 분석:")
            for field in possible_cash_fields:
                value = self._safe_float(output2.get(field, "0"))
                self.logger.info(f"[DEBUG]   {field}: {value:,.0f}원")
            
            # 실제 사용할 필드 선택
            for field in possible_cash_fields:
                value = self._safe_float(output2.get(field, "0"))
                if value > 0:
                    available_cash = value
                    self.logger.info(f"[DEBUG] ✅ 매수가능금액 사용 필드: {field} = {value:,.0f}원")
                    break
            
            total_value = self._safe_float(output2.get("tot_evlu_amt", "0"))  # 총평가금액
            total_profit_loss = self._safe_float(output2.get("evlu_pfls_smtl_amt", "0"))  # 평가손익합계금액
            profit_loss_rate = self._safe_float(output2.get("tot_evlu_pfls_rt", "0"))  # 총평가손익률
            
            # 추가 정보 로깅
            self.logger.info(f"[DEBUG] 계좌 정보 파싱:")
            self.logger.info(f"[DEBUG]   예수금총액(dnca_tot_amt): {total_cash:,.0f}원")
            self.logger.info(f"[DEBUG]   매수가능금액: {available_cash:,.0f}원")
            self.logger.info(f"[DEBUG]   총평가금액(tot_evlu_amt): {total_value:,.0f}원")
            
            account_info = {
                "account_no": self.account,
                "total_cash": total_cash,
                "available_cash": available_cash,
                "total_value": total_value,
                "total_profit_loss": total_profit_loss,
                "profit_loss_rate": profit_loss_rate,
                "stock_value": total_value - total_cash  # 주식 평가금액
            }
            
            return account_info
            
        except Exception as e:
            self.logger.error(f"계좌 잔고 데이터 파싱 오류: {e}")
            return {}
    
    def _safe_int(self, value: str) -> int:
        """문자열을 안전하게 정수로 변환"""
        try:
            if not value or value == "":
                return 0
            # 음수, 숫자, 소수점만 남기고 제거
            cleaned = ''.join(c for c in str(value) if c.isdigit() or c in '-.')
            return int(float(cleaned)) if cleaned and cleaned not in ['.', '-', '-.'] else 0
        except (ValueError, TypeError):
            return 0
    
    def _safe_float(self, value: str) -> float:
        """문자열을 안전하게 실수로 변환"""
        try:
            if not value or value == "":
                return 0.0
            # 음수, 숫자, 소수점만 남기고 제거
            cleaned = ''.join(c for c in str(value) if c.isdigit() or c in '-.')
            return float(cleaned) if cleaned and cleaned not in ['.', '-', '-.'] else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _get_sector_from_symbol(self, symbol: str) -> str:
        """종목코드로부터 섹터 정보 추출 (간단한 매핑)"""
        # 실제로는 별도 API나 DB에서 조회해야 함
        sector_mapping = {
            "005930": "전자/반도체",  # 삼성전자
            "000660": "반도체",       # SK하이닉스
            "035420": "인터넷/게임",  # NAVER
            "042700": "반도체",       # 한미반도체
            "051910": "화학",         # LG화학
            "035720": "인터넷/게임",  # 카카오
        }
        return sector_mapping.get(symbol, "기타")
    
    def _adjust_to_tick_size(self, price: float) -> int:
        """
        한국 주식 호가단위에 맞게 가격 조정
        
        Args:
            price: 조정할 가격
            
        Returns:
            int: 호가단위에 맞게 조정된 가격
        """
        price = int(price)  # 정수로 변환
        
        if price >= 100000:
            # 10만원 이상: 500원 단위
            return (price // 500) * 500
        elif price >= 50000:
            # 5만원 이상 10만원 미만: 100원 단위
            return (price // 100) * 100
        elif price >= 10000:
            # 1만원 이상 5만원 미만: 50원 단위
            return (price // 50) * 50
        elif price >= 5000:
            # 5천원 이상 1만원 미만: 10원 단위
            return (price // 10) * 10
        elif price >= 1000:
            # 1천원 이상 5천원 미만: 5원 단위
            return (price // 5) * 5
        else:
            # 1천원 미만: 1원 단위
            return price
    
    def get_current_price(self, stock_code: str) -> Dict[str, Any]:
        """
        주식 현재가 조회
        
        Args:
            stock_code: 종목코드 (6자리)
        
        Returns:
            Dict: 현재가 정보
        """
        try:
            # API 호출 제한 대기
            self._wait_for_api_limit()
            
            self.ensure_token()
            self.api_calls += 1
            
            url = f"{self.BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
            # 실거래와 모의투자에 맞는 TR 코드 선택 (현재가는 보통 동일하지만 안전하게 분기)
            tr_code = "FHKST01010100"  # 현재가 조회는 실거래/모의투자 공통
            headers = self._get_headers(tr_code)
            
            self.logger.debug(f"[DEBUG] 현재가 조회 TR 코드: {tr_code} (virtual: {self.virtual_account})")
            
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("rt_cd") == "0":
                    stock_name = get_stock_name(stock_code) if STOCK_NAME_AVAILABLE else f"종목{stock_code}"
                    self.logger.info(f"[OK] 현재가 조회 성공: {stock_code} ({stock_name})")
                    return result
                else:
                    error_msg = result.get("msg1", "현재가 조회 실패")
                    self.logger.error(f"[FAIL] 현재가 조회 실패: {error_msg}")
                    return {}
            else:
                self.logger.error(f"[FAIL] 현재가 조회 요청 실패: {response.text}")
                return {}
                
        except Exception as e:
            self.logger.error(f"[FAIL] 현재가 조회 오류: {e}")
            return {}
    
    def block_request(self, tr_code: str, **kwargs) -> Dict[str, Any]:
        """
        TR 요청 (키움 API 호환)
        주로 잔고조회에서 사용됨
        """
        if tr_code == "opw00018":  # 계좌평가잔고내역요청
            return self.get_account_balance(kwargs.get("계좌번호", ""))
        else:
            self.logger.warning(f"[WARN] 지원하지 않는 TR 코드: {tr_code}")
            return {}
    
    def get_status_summary(self) -> Dict[str, Any]:
        """상태 요약 정보 반환"""
        return {
            'is_connected': self.is_connected,
            'is_logged_in': self.is_logged_in,
            'account_count': len(self.account_list),
            'login_attempts': self.login_attempts,
            'api_calls': self.api_calls,
            'last_login_time': self.last_login_time,
            'virtual_account': self.virtual_account
        }
    
    def disconnect(self):
        """연결 해제"""
        with self._lock:
            self.is_connected = False
            self.is_logged_in = False
            self.access_token = None
            self.token_expiry = 0
            self.logger.info("[INFO] 한국투자증권 API 연결 해제됨")


class KoreaInvestmentErrorCode:
    """한국투자증권 API 에러 코드 처리"""
    
    def __init__(self):
        self.error_messages = {
            "0": "정상처리",
            "-1": "실패",
            "40310000": "잘못된 계좌번호",
            "40320000": "잘못된 종목코드",
            # 추가 에러 코드들...
        }
    
    def get_error_message(self, error_code: str) -> str:
        """에러 코드에 대한 메시지 반환"""
        return self.error_messages.get(error_code, f"알 수 없는 오류: {error_code}")
    
    def is_success(self, error_code: str) -> bool:
        """성공 여부 확인"""
        return error_code == "0" 
