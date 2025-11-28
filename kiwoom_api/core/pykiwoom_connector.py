# -*- coding: utf-8 -*-
"""
한국투자증권 OpenAPI 기반 연결 관리자 (PyKiwoom 호환)
"""

import sys
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import pandas as pd
from .korea_investment_connector import KoreaInvestmentConnector, KoreaInvestmentErrorCode
from kiwoom_api.utils.config_loader import load_kis_config

config = load_kis_config()
if not config:
    print("❌ 설정 로드 실패. .env 파일을 확인하세요.")
    exit(1)

class PyKiwoomConnector:
    """한국투자증권 OpenAPI 기반 연결 관리자 (PyKiwoom 호환)"""
    
    def __init__(self, appkey: str = "", appsecret: str = "", account: str = "", virtual_account: bool = True, base_url: str = ""):
        """연결 관리자 초기화"""
        # 한국투자증권 API 커넥터 초기화
        self.korea_investment = KoreaInvestmentConnector(
            appkey=config.appkey, 
            appsecret=config.appsecret, 
            account=config.account, 
            virtual_account=config.virtual,
            base_url=config.base_url
        )
        
        # 기존 PyKiwoom과 호환성을 위한 속성들
        self.kiwoom = self.korea_investment  # 호환성을 위해 kiwoom으로 참조
        self.app = None  # Qt 애플리케이션 (한국투자증권 API에서는 불필요)
        self.is_connected = False
        self.is_logged_in = False
        self.account_list = []
        self.selected_account = account
        self.initialization_error = None
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # 통계 정보
        self.connection_attempts = 0
        self.login_attempts = 0
        self.last_login_time = None
        
        # 시장 상태 (기본값)
        self.market_open = True
        
        # 사용자 정보
        self.user_info = {}
        
        # 실시간 데이터 핸들러 (한국투자증권 API에서는 WebSocket 사용)
        self.real_handlers = {}
        
    def initialize(self) -> bool:
        """한국투자증권 API 초기화"""
        try:
            print("[DEBUG] with self._lock 진입 시도")
            with self._lock:
                print("[DEBUG] with self._lock 진입 성공")
                self.logger.info("[INFO] 한국투자증권 API 초기화 시도...")
                
                # 한국투자증권 API는 별도 초기화가 필요 없음
                self.is_connected = True
                self.logger.info("[OK] 한국투자증권 API 초기화 성공")
                return True
                    
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"[FAIL] 한국투자증권 API 초기화 실패: {e}")
            return False
    
    def _load_account_info(self):
        """계좌 정보 로드"""
        try:
            if self.korea_investment:
                self.account_list = self.korea_investment.get_account_list()
                if self.account_list and not self.selected_account:
                    self.selected_account = self.account_list[0]
                
                # 사용자 정보 설정 (기본값)
                self.user_info = {
                    'user_id': 'korea_investment_user',
                    'user_name': '한국투자증권 사용자',
                    'account_cnt': str(len(self.account_list))
                }
                
                self.logger.info(f"[OK] 계좌 정보 로드: {len(self.account_list)}개 계좌")
                
        except Exception as e:
            self.logger.warning(f"[WARN] 계좌 정보 로드 실패: {e}")
    
    def _on_receive_tr_data(self, screen_no, rqname, trcode, record_name, prev_next):
        """TR 데이터 수신 이벤트 핸들러 (호환성용)"""
        pass
    
    def _on_receive_real_data(self, code, real_type, real_data):
        """실시간 데이터 수신 이벤트 핸들러 (호환성용)"""
        pass
    
    def _on_receive_chejan_data(self, gubun, item_cnt, fid_list):
        """체결 데이터 수신 이벤트 핸들러 (호환성용)"""
        pass
    
    def connect(self) -> bool:
        print("[DEBUG] connect() 진입")
        """연결 및 로그인"""
        
        try:
            print("[DEBUG] with self._lock 진입 시도")
            with self._lock:
                print("[DEBUG] with self._lock 진입 성공")
                self.connection_attempts += 1
                self.login_attempts += 1
                
                if not self.is_connected:
                    if not self.initialize():
                        return False
                print("[DEBUG] 인증 요청 전")
                print("[DEBUG] self.korea_investment:", self)
                print("[DEBUG] type(self.korea_investment):", type(self))
                print("[DEBUG] hasattr(authenticate):", hasattr(self, "authenticate"))
                print("[DEBUG] self.korea_investment.authenticate:", self.korea_investment.authenticate)
                print("[DEBUG] self._lock:", self._lock)
                print("[DEBUG] type(self._lock):", type(self._lock))
                # 한국투자증권 API 인증
                login_success = self.korea_investment.authenticate()
                print("[DEBUG] 인증 요청 후, 응답:")
                if login_success:
                    self.is_logged_in = True
                    self.last_login_time = datetime.now()
                    self._load_account_info()
                    
                    self.logger.info("[OK] 한국투자증권 API 로그인 성공")
                    return True
                else:
                    self.logger.error("[FAIL] 한국투자증권 API 로그인 실패")
                    return False
                    
        except Exception as e:
            self.logger.error(f"[FAIL] 연결 중 오류: {e}")
            return False
    
    def disconnect(self):
        """연결 해제"""
        try:
            with self._lock:
                if self.korea_investment:
                    self.korea_investment.disconnect()
                
                self.is_connected = False
                self.is_logged_in = False
                self.account_list.clear()
                self.selected_account = ""
                
                self.logger.info("[INFO] 한국투자증권 API 연결 해제됨")
                
        except Exception as e:
            self.logger.error(f"[FAIL] 연결 해제 중 오류: {e}")
    
    def get_connection_state(self) -> int:
        """연결 상태 조회"""
        if self.korea_investment:
            return self.korea_investment.get_connect_state()
        return 0
    
    def get_login_info(self, tag: str) -> str:
        """로그인 정보 조회"""
        if not self.is_logged_in or not self.korea_investment:
            return ""
        
        return self.korea_investment.get_login_info(tag)
    
    def get_account_list(self) -> List[str]:
        """계좌 목록 반환"""
        return self.account_list.copy()
    
    def get_user_info(self) -> Dict[str, str]:
        """사용자 정보 반환"""
        return self.user_info.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = {
            'is_connected': self.is_connected,
            'is_logged_in': self.is_logged_in,
            'connection_attempts': self.connection_attempts,
            'login_attempts': self.login_attempts,
            'last_login_time': self.last_login_time,
            'account_count': len(self.account_list),
            'selected_account': self.selected_account,
            'market_open': self.market_open,
            'initialization_error': self.initialization_error
        }
        
        # 한국투자증권 API 통계 추가
        if self.korea_investment:
            ki_stats = self.korea_investment.get_status_summary()
            stats.update({
                'api_calls': ki_stats.get('api_calls', 0),
                'virtual_account': ki_stats.get('virtual_account', True)
            })
        
        return stats
    
    def get_code_list_by_market(self, market: str) -> List[str]:
        """시장별 종목 코드 리스트 (한국투자증권 API에서는 별도 구현 필요)"""
        # 한국투자증권 API에서는 별도의 종목 리스트 API를 사용해야 함
        # 현재는 기본 샘플 코드만 반환
        self.logger.warning("[WARN] get_code_list_by_market은 한국투자증권 API에서 별도 구현 필요")
        
        # 샘플 코드 (실제로는 API 호출 필요)
        if market == "0":  # 코스피
            return ["005930", "000660", "035420"]  # 삼성전자, 하이닉스, 네이버
        elif market == "10":  # 코스닥
            return ["091990", "066570", "035720"]  # 셀트리온헬스케어, LG전자, 카카오
        else:
            return []
    
    def get_master_code_name(self, code: str) -> str:
        """종목명 조회 (한국투자증권 API에서는 별도 구현 필요)"""
        # 한국투자증권 API에서는 별도의 종목 정보 API를 사용해야 함
        self.logger.warning("[WARN] get_master_code_name은 한국투자증권 API에서 별도 구현 필요")
        
        # 샘플 데이터 (실제로는 API 호출 필요)
        sample_names = {
            "005930": "삼성전자",
            "000660": "SK하이닉스", 
            "035420": "NAVER",
            "091990": "셀트리온헬스케어",
            "066570": "LG전자",
            "035720": "카카오"
        }
        return sample_names.get(code, "")
    
    def get_master_listed_stock_cnt(self, code: str) -> int:
        """상장주식수 조회 (한국투자증권 API에서는 별도 구현 필요)"""
        self.logger.warning("[WARN] get_master_listed_stock_cnt는 한국투자증권 API에서 별도 구현 필요")
        return 0
    
    def get_master_last_price(self, code: str) -> int:
        """현재가 조회 (한국투자증권 API에서는 별도 구현 필요)"""
        self.logger.warning("[WARN] get_master_last_price는 한국투자증권 API에서 별도 구현 필요")
        return 0
    
    def register_real_handler(self, real_type: str, handler: Callable):
        """실시간 데이터 핸들러 등록"""
        self.real_handlers[real_type] = handler
        self.logger.info(f"[OK] 실시간 핸들러 등록: {real_type}")
    
    def set_real_reg(self, screen_no: str, code_list: str, fid_list: str, opt_type: str):
        """실시간 등록 (한국투자증권 API에서는 WebSocket 사용)"""
        try:
            # 한국투자증권 API에서는 WebSocket을 통한 실시간 데이터 수신
            # 현재는 로그만 출력
            self.logger.info(f"[INFO] 실시간 등록 요청: 화면번호={screen_no}, 종목={code_list}")
            
            # 실제 구현 시에는 WebSocket 연결 및 구독 처리 필요
            # TODO: 한국투자증권 WebSocket API 구현
            
        except Exception as e:
            self.logger.error(f"[FAIL] 실시간 등록 실패: {e}")
    
    def send_order(self, rqname: str, screen_no: str, acc_no: str, order_type: int,
                   code: str, quantity: int, price: int, hoga_gubun: str, 
                   order_no: str = "") -> str:
        """주문 전송"""
        if not self.is_logged_in or not self.korea_investment:
            self.logger.error("[FAIL] 로그인되지 않아 주문 전송 불가")
            return "-1"
        
        try:
            # 한국투자증권 API를 통한 주문 전송
            result = self.korea_investment.send_order(
                rqname, screen_no, acc_no, order_type,
                code, quantity, price, hoga_gubun, order_no
            )
            
            if result == 0:
                self.logger.info(f"[OK] 주문 전송 성공: {code} {quantity}주")
                return "0"
            else:
                self.logger.error(f"[FAIL] 주문 전송 실패: 에러코드 {result}")
                return str(result)
                
        except Exception as e:
            self.logger.error(f"[FAIL] 주문 전송 중 오류: {e}")
            return "-999"
    
    def block_request(self, tr_code: str, **kwargs) -> Dict[str, Any]:
        """TR 요청 (호환성용)"""
        if tr_code == "opw00018":  # 계좌평가잔고내역요청
            return self.korea_investment.get_account_balance(kwargs.get("계좌번호", ""))
        else:
            self.logger.warning(f"[WARN] 지원하지 않는 TR 코드: {tr_code}")
            return {}
    
    # 추가 호환성 메서드들
    def GetConnectState(self) -> int:
        """연결 상태 조회 (대문자 메서드명 호환)"""
        return self.get_connection_state()
    
    def GetLoginInfo(self, tag: str) -> str:
        """로그인 정보 조회 (대문자 메서드명 호환)"""
        return self.get_login_info(tag)
    
    def GetAccountList(self) -> str:
        """계좌 목록 반환 (대문자 메서드명 호환)"""
        return ";".join(self.account_list) + ";"
    
    def SendOrder(self, rqname: str, screen_no: str, acc_no: str, order_type: int,
                  code: str, quantity: int, price: int, hoga_gubun: str, 
                  order_no: str = "") -> int:
        """주문 전송 (대문자 메서드명 호환)"""
        result = self.send_order(rqname, screen_no, acc_no, order_type, 
                               code, quantity, price, hoga_gubun, order_no)
        return int(result)
    
    def get_balance(self):
        # 실제로 KoreaInvestmentConnector의 잔고 조회 API를 호출
        return self.korea_investment.get_balance() 