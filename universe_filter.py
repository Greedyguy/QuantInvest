import pandas as pd
import numpy as np

def filter_universe(enriched: dict,
                    min_tvalue=2_000_000_000,   # 20억
                    min_price=2000,
                    max_price=40000,
                    min_active_days=15):
    """
    현실적인 소액 계좌용 Universe 필터링 v1.0
    Args:
        enriched: {ticker: df} OHLCV 데이터셋
        min_tvalue: 20일 평균 거래대금 최소값
        min_price: 최소 주가
        max_price: 최대 주가
        min_active_days: 최근 20일 중 거래된 날 수 최소값
    Returns:
        list – 필터 통과 종목 리스트
    """

    valid = []

    for ticker, df in enriched.items():
        if df is None or len(df) < 20:
            continue

        # 최근 20일
        recent = df.tail(20)

        # 1) 가격 필터
        last_price = recent["close"].iloc[-1]
        if not (min_price <= last_price <= max_price):
            continue

        # 2) 거래대금 필터
        recent_tvalue = (recent["close"] * recent["volume"]).mean()
        if recent_tvalue < min_tvalue:
            continue

        # 3) 연속 거래 필터
        active_days = (recent["volume"] > 0).sum()
        if active_days < min_active_days:
            continue

        valid.append(ticker)

    return valid