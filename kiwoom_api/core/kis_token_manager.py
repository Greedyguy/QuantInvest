# -*- coding: utf-8 -*-
"""
KIS API Token Manager - 거래 환경별 공유로 토큰 중복 발급 방지
"""

import os
import json
import time
import threading
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

TOKEN_RATE_LIMIT_RETRY_SECONDS = 65
TOKEN_RATE_LIMIT_ERROR_CODE = "EGW00133"

class KISTokenManager:
    """
    한국투자증권 API 토큰 관리자
    - 일 1회 발급 원칙 준수
    - 토큰 캐시 및 공유
    - 중복 발급 방지
    """
    _lock = threading.Lock()

    def __init__(
        self,
        appkey: Optional[str] = None,
        appsecret: Optional[str] = None,
        virtual_account: Optional[bool] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
            
        self.access_token = None
        self.token_expiry = 0
        self.last_token_request = 0
        self.token_request_count = 0  # 일일 토큰 발급 횟수 추적
        self.daily_reset_time = 0     # 일일 리셋 시간
        
        # 캐시 디렉토리
        self.cache_dir = Path(cache_dir or ".kis_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # 환경 설정
        self.appkey = appkey or os.getenv('KIS_APP_KEY')
        self.appsecret = appsecret or os.getenv('KIS_APP_SECRET')
        if virtual_account is None:
            self.virtual_account = os.getenv(
                'KIS_VIRTUAL_ACCOUNT', 'true'
            ).lower() == 'true'
        else:
            self.virtual_account = bool(virtual_account)

        default_base_url = (
            "https://openapivts.koreainvestment.com:29443"
            if self.virtual_account
            else "https://openapi.koreainvestment.com:9443"
        )
        self.base_url = (base_url or default_base_url).rstrip("/")
        self.environment = 'virtual' if self.virtual_account else 'real'
        self.appkey_fingerprint = hashlib.sha256(
            (self.appkey or "").encode("utf-8")
        ).hexdigest()[:12]
        self.cache_file = self.cache_dir / (
            f"shared_token_{self.environment}_{self.appkey_fingerprint}.json"
        )
            
        # 토큰 로드 시도
        self._load_cached_token()
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

            if (
                data.get('environment') != self.environment
                or data.get('appkey_fingerprint') != self.appkey_fingerprint
            ):
                logger.warning("[LOAD] 현재 거래 환경과 다른 토큰 캐시를 무시합니다")
                return False
            
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
                'environment': self.environment,
                'appkey_fingerprint': self.appkey_fingerprint,
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
        """새 토큰 발급. 분당 발급 제한은 한 번 대기 후 재시도한다."""
        try:
            import requests

            url = f"{self.base_url}/oauth2/tokenP"
            headers = {"content-type": "application/json; charset=utf-8"}
            data = {
                "grant_type": "client_credentials",
                "appkey": self.appkey,
                "appsecret": self.appsecret
            }

            for attempt in range(2):
                if attempt > 0:
                    logger.warning(
                        "[RETRY] KIS 접근토큰 1분당 발급 제한으로 %d초 후 재시도합니다.",
                        TOKEN_RATE_LIMIT_RETRY_SECONDS,
                    )
                    time.sleep(TOKEN_RATE_LIMIT_RETRY_SECONDS)
                elif self.last_token_request > 0:
                    elapsed = time.time() - self.last_token_request
                    if elapsed < TOKEN_RATE_LIMIT_RETRY_SECONDS:
                        wait_time = TOKEN_RATE_LIMIT_RETRY_SECONDS - elapsed
                        logger.info(f"[WAIT] 토큰 발급 제한으로 {wait_time:.1f}초 대기")
                        time.sleep(wait_time)

                logger.info(
                    f"[REQ] 새 토큰 발급 요청 (일일 {self.token_request_count + 1}회차)"
                )
                response = requests.post(url, headers=headers, json=data, timeout=10)
                self.last_token_request = time.time()

                if response.status_code == 200:
                    result = response.json()

                    self.access_token = result["access_token"]
                    self.token_expiry = time.time() + int(result["expires_in"])
                    self.token_request_count += 1

                    self._save_token_cache()

                    logger.info(f"✅ 토큰 발급 성공! (일일 {self.token_request_count}/3)")
                    return self.access_token

                if attempt == 0 and self._is_minute_rate_limit(response):
                    continue

                logger.error(f"[FAIL] 토큰 발급 실패: {response.text}")
                break

        except Exception as e:
            logger.error(f"[ERROR] 토큰 발급 오류: {e}")

        return None

    @staticmethod
    def _is_minute_rate_limit(response) -> bool:
        """KIS의 접근토큰 1분당 1회 제한 응답인지 판별한다."""
        response_text = response.text or ""
        error_code = ""
        error_description = ""
        try:
            payload = response.json()
            error_code = str(payload.get("error_code") or payload.get("msg_cd") or "")
            error_description = str(
                payload.get("error_description") or payload.get("msg1") or ""
            )
        except (TypeError, ValueError):
            pass

        combined = f"{response_text} {error_description}"
        return (
            error_code == TOKEN_RATE_LIMIT_ERROR_CODE
            or "1분당 1회" in combined
            or "접근토큰 발급 잠시 후 다시 시도" in combined
        )
    
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


# 거래 환경별 토큰 매니저 인스턴스. 실전/모의 토큰은 서로 호환되지 않는다.
_token_managers: Dict[tuple, KISTokenManager] = {}
_token_managers_lock = threading.Lock()


def get_token_manager(
    appkey: Optional[str] = None,
    appsecret: Optional[str] = None,
    virtual_account: Optional[bool] = None,
    base_url: Optional[str] = None,
) -> KISTokenManager:
    """API 키와 거래 환경이 같은 호출끼리만 토큰 매니저를 공유한다."""
    resolved_appkey = appkey or os.getenv('KIS_APP_KEY')
    if virtual_account is None:
        resolved_virtual = os.getenv(
            'KIS_VIRTUAL_ACCOUNT', 'true'
        ).lower() == 'true'
    else:
        resolved_virtual = bool(virtual_account)
    resolved_base_url = (
        base_url
        or (
            "https://openapivts.koreainvestment.com:29443"
            if resolved_virtual
            else "https://openapi.koreainvestment.com:9443"
        )
    ).rstrip("/")
    manager_key = (resolved_appkey or "", resolved_virtual, resolved_base_url)

    with _token_managers_lock:
        manager = _token_managers.get(manager_key)
        if manager is None:
            manager = KISTokenManager(
                appkey=resolved_appkey,
                appsecret=appsecret,
                virtual_account=resolved_virtual,
                base_url=resolved_base_url,
            )
            _token_managers[manager_key] = manager
        return manager

def get_shared_token() -> Optional[str]:
    """공유 토큰 반환 (편의 함수)"""
    return get_token_manager().get_valid_token()

def get_token_stats() -> Dict[str, Any]:
    """토큰 발급 통계 반환"""
    return get_token_manager().get_daily_stats()
