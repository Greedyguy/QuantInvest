from .korea_investment_connector import KoreaInvestmentConnector
from .kis_token_manager import KISTokenManager

__all__ = [
    'KoreaInvestmentConnector',
    'KISTokenManager', 
    'RealAccountManager',
    'PyKiwoomConnector'
]


def __getattr__(name):
    """키움 전용 모듈은 실제로 요청될 때만 불러온다.

    PyKiwoomConnector는 플랫폼/자격증명 검사를 import 시점에 수행하므로 KIS만
    사용하는 테스트와 GitHub Actions까지 종료시키지 않도록 지연 로딩한다.
    """
    if name == "RealAccountManager":
        from .real_account_manager import RealAccountManager
        return RealAccountManager
    if name == "PyKiwoomConnector":
        from .pykiwoom_connector import PyKiwoomConnector
        return PyKiwoomConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
