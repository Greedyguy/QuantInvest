# -*- coding: utf-8 -*-
"""
한국투자증권 OpenAPI 설정 로더
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv


class KoreaInvestmentConfig:
    """한국투자증권 API 설정 클래스"""
    
    def __init__(self, env_file: str = ".env"):
        """설정 초기화"""
        # .env 파일 로드
        load_dotenv(env_file)
        
        # 다양한 환경 변수 패턴 지원
        self.appkey = (os.getenv("KIS_APP_KEY", "") or 
                      os.getenv("KIS_APPKEY", "") or 
                      os.getenv("KOREA_INVESTMENT_APPKEY", ""))
        
        self.appsecret = (os.getenv("KIS_APP_SECRET", "") or 
                         os.getenv("KIS_APPSECRET", "") or 
                         os.getenv("KOREA_INVESTMENT_APPSECRET", ""))
        
        self.account = (os.getenv("KIS_ACCOUNT", "") or 
                       os.getenv("KOREA_INVESTMENT_ACCOUNT", ""))
        
        virtual_env = (os.getenv("KIS_VIRTUAL", "") or 
                      os.getenv("KIS_VIRTUAL_ACCOUNT", "") or 
                      os.getenv("KOREA_INVESTMENT_VIRTUAL", "True"))
        self.virtual = virtual_env.lower() == "true"
        
        self.base_url = (os.getenv("KIS_BASE_URL", "") or 
                        os.getenv("KOREA_INVESTMENT_BASE_URL", ""))
        
        # 기본 URL 설정 (virtual에 따라)
        if not self.base_url:
            if self.virtual:
                self.base_url = "https://openapivts.koreainvestment.com:29443"
            else:
                self.base_url = "https://openapi.koreainvestment.com:9443"
        
        print("[DEBUG] appkey:", self.appkey[:10] + "***" if self.appkey else "비어있음")
        print("[DEBUG] appsecret:", "설정됨" if self.appsecret else "비어있음") 
        print("[DEBUG] account:", self.account)
        print("[DEBUG] virtual:", self.virtual)
        print("[DEBUG] base_url:", self.base_url)
    
    def is_valid(self) -> bool:
        """설정 유효성 검사"""
        return bool(self.appkey and self.appsecret and self.account)
    
    def get_config_dict(self) -> Dict[str, str]:
        """설정을 딕셔너리로 반환"""
        return {
            "appkey": self.appkey,
            "appsecret": self.appsecret, 
            "account": self.account,
            "virtual_account": self.virtual,
            "base_url": self.base_url
        }
    
    def __str__(self) -> str:
        """설정 정보 출력 (민감 정보 마스킹)"""
        return f"""
한국투자증권 API 설정:
- 앱키: {self.appkey[:8]}***
- 앱시크릿: {self.appsecret[:8]}***
- 계좌번호: {self.account[:4]}****
- 모의투자: {self.virtual}
- 기본 URL: {self.base_url}
"""


def load_kis_config(env_file: str = ".env") -> Optional[KoreaInvestmentConfig]:
    """한국투자증권 API 설정 로드"""
    try:
        config = KoreaInvestmentConfig(env_file)
        if config.is_valid():
            return config
        else:
            print("❌ API 설정이 유효하지 않습니다. .env 파일을 확인하세요.")
            return None
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return None 