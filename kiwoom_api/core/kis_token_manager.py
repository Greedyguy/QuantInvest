# -*- coding: utf-8 -*-
"""
KIS API Token Manager - 싱글톤 패턴으로 토큰 중복 발급 방지
"""

import os
import json
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class KISTokenManager:
    """
    한국투자증권 API 토큰 관리자 (싱글톤)
    - 일 1회 발급 원칙 준수
    - 토큰 캐시 및 공유
    - 중복 발급 방지
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.access_token = None
        self.token_expiry = 0
        self.last_token_request = 0
        self.token_request_count = 0  # 일일 토큰 발급 횟수 추적
        self.daily_reset_time = 0     # 일일 리셋 시간
        
        # 캐시 디렉토리
        self.cache_dir = Path(".kis_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "shared_token.json"
        
        # 환경 설정
        self.appkey = os.getenv('KIS_APP_KEY')
        self.appsecret = os.getenv('KIS_APP_SECRET')
        self.virtual_account = os.getenv('KIS_VIRTUAL_ACCOUNT', 'true').lower() == 'true'
        
        if self.virtual_account:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"
            
        # 토큰 로드 시도
        self._load_cached_token()
        self._initialized = True
        
        logger.info(f"[INIT] KIS 토큰 관리자 초기화 완료 ({'모의투자' if self.virtual_account else '실거래'})")
    
    def _reset_daily_counter_if_needed(self):
        """자정이 지나면 일일 카운터 리셋"""
        import datetime
        current_date = datetime.datetime.now().date()
        
        if self.daily_reset_time == 0:
            self.daily_reset_time = time.mktime(current_date.timetuple())
        
        cache_date = datetime.datetime.fromtimestamp(self.daily_reset_time).date()
        
        if current_date > cache_date:
            logger.info("[RESET] 일일 토큰 발급 카운터 리셋")
            self.token_request_count = 0
            self.daily_reset_time = time.mktime(current_date.timetuple())
            self._save_token_cache()
    
    def _load_cached_token(self):
        """캐시된 토큰 로드"""
        try:
            if not self.cache_file.exists():
                return False
                
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            # 토큰 유효성 검사
            if data.get('expires_at', 0) > time.time() + 300:  # 5분 여유
                self.access_token = data.get('access_token')
                self.token_expiry = data.get('expires_at', 0)
                self.token_request_count = data.get('daily_count', 0)
                self.daily_reset_time = data.get('daily_reset', 0)
                
                logger.info(f"[LOAD] 캐시된 토큰 로드 (일일 발급: {self.token_request_count}회)")
                return True
            else:
                logger.info("[LOAD] 캐시된 토큰 만료됨")
                
        except Exception as e:
            logger.warning(f"[LOAD] 토큰 캐시 로드 실패: {e}")
            
        return False
    
    def _save_token_cache(self):
        """토큰 캐시 저장"""
        try:
            data = {
                'access_token': self.access_token,
                'expires_at': self.token_expiry,
                'daily_count': self.token_request_count,
                'daily_reset': self.daily_reset_time,
                'environment': 'virtual' if self.virtual_account else 'real',
                'last_update': time.time()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"[SAVE] 토큰 캐시 저장 실패: {e}")
    
    def get_valid_token(self) -> Optional[str]:
        """
        유효한 토큰 반환
        - 일 1회 발급 원칙 준수
        - 캐시된 토큰 우선 사용
        """
        with self._lock:
            self._reset_daily_counter_if_needed()
            
            # 토큰이 유효하면 기존 토큰 반환
            if self.access_token and time.time() < self.token_expiry - 300:
                return self.access_token
            
            # 일일 발급 제한 체크 (안전을 위해 3회로 제한)
            if self.token_request_count >= 3:
                logger.error(f"[LIMIT] 일일 토큰 발급 한도 초과! ({self.token_request_count}/3)")
                if self.access_token:  # 만료된 토큰이라도 반환 (긴급 상황 대비)
                    logger.warning("[FALLBACK] 만료된 토큰으로 시도합니다")
                    return self.access_token
                return None
            
            # 새 토큰 발급
            return self._request_new_token()
    
    def _request_new_token(self) -> Optional[str]:
        """새 토큰 발급"""
        try:
            import requests
            
            # 이전 요청으로부터 최소 65초 대기
            if self.last_token_request > 0:
                elapsed = time.time() - self.last_token_request
                if elapsed < 65:
                    wait_time = 65 - elapsed
                    logger.info(f"[WAIT] 토큰 발급 제한으로 {wait_time:.1f}초 대기")
                    time.sleep(wait_time)
            
            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json; charset=utf-8"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.appkey,
                "appsecret": self.appsecret
            }
            
            logger.info(f"[REQ] 새 토큰 발급 요청 (일일 {self.token_request_count + 1}회차)")
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                self.access_token = result["access_token"]
                self.token_expiry = time.time() + int(result["expires_in"])
                self.last_token_request = time.time()
                self.token_request_count += 1
                
                self._save_token_cache()
                
                logger.info(f"✅ 토큰 발급 성공! (일일 {self.token_request_count}/3)")
                return self.access_token
            else:
                logger.error(f"[FAIL] 토큰 발급 실패: {response.text}")
                
        except Exception as e:
            logger.error(f"[ERROR] 토큰 발급 오류: {e}")
        
        return None
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """일일 토큰 발급 통계"""
        self._reset_daily_counter_if_needed()
        
        return {
            'daily_requests': self.token_request_count,
            'daily_limit': 3,
            'remaining': max(0, 3 - self.token_request_count),
            'last_request': self.last_token_request,
            'token_valid': self.access_token is not None and time.time() < self.token_expiry - 300,
            'environment': 'virtual' if self.virtual_account else 'real'
        }
    
    def force_refresh(self) -> bool:
        """강제 토큰 갱신 (긴급시에만 사용)"""
        logger.warning("[FORCE] 강제 토큰 갱신 시도")
        
        with self._lock:
            self.access_token = None
            self.token_expiry = 0
            
            token = self._request_new_token()
            return token is not None


# 전역 토큰 매니저 인스턴스
_token_manager = None

def get_token_manager() -> KISTokenManager:
    """토큰 매니저 인스턴스 반환"""
    global _token_manager
    if _token_manager is None:
        _token_manager = KISTokenManager()
    return _token_manager

def get_shared_token() -> Optional[str]:
    """공유 토큰 반환 (편의 함수)"""
    return get_token_manager().get_valid_token()

def get_token_stats() -> Dict[str, Any]:
    """토큰 발급 통계 반환"""
    return get_token_manager().get_daily_stats() 