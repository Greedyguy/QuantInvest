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
        
        # 자격증명과 계좌번호는 로그에 일부라도 출력하지 않는다. GitHub Actions
        # 마스킹은 정확히 일치하는 secret만 보장하므로 접두사 출력도 피한다.
    
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
        """민감 정보 없이 설정 상태만 출력"""
        return f"""
한국투자증권 API 설정:
- 자격증명: {'설정됨' if self.is_valid() else '미설정'}
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
            print("❌ API 설정이 유효하지 않습니다. KIS 환경 변수를 확인하세요.")
            return None
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return None
