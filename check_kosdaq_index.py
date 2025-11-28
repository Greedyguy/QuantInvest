#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
코스닥 지수 데이터 수집 및 사용 가능 여부 점검 (수정 버전)
"""

from data_loader import get_index_close
from cache_manager import load_index, save_index
from datetime import date, timedelta
import pandas as pd

print('=' * 80)
print('🔍 코스닥 지수 데이터 수집 및 사용 가능 여부 점검 (수정 버전)')
print('=' * 80)

# 날짜 범위 설정 (과거 날짜로 확실히)
start = '2024-01-01'
end = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')  # 어제까지

print(f'\n📅 테스트 기간: {start} ~ {end}')
print(f'   (현재 날짜: {date.today()})')

# 1. pykrx 지수 티커 조회
print('\n1️⃣ pykrx 지수 티커 조회')
try:
    from pykrx import stock
    
    # 사용 가능한 지수 목록 조회
    print('\n   지수 티커 목록 조회:')
    try:
        # KOSPI 관련 지수
        kospi_tickers = stock.get_index_ticker_list("20240101", market="KOSPI")
        print(f'   - KOSPI 지수 티커: {kospi_tickers[:10]}... (총 {len(kospi_tickers)}개)')
        
        # KOSDAQ 관련 지수
        kosdaq_tickers = stock.get_index_ticker_list("20240101", market="KOSDAQ")
        print(f'   - KOSDAQ 지수 티커: {kosdaq_tickers[:10]}... (총 {len(kosdaq_tickers)}개)')
        
        # 종합지수 찾기
        print('\n   종합지수 찾기:')
        for ticker in kospi_tickers:
            try:
                name = stock.get_index_ticker_name(ticker)
                if '종합' in name or 'KOSPI' in name:
                    print(f'   - KOSPI: {ticker} = {name}')
            except:
                pass
        
        for ticker in kosdaq_tickers:
            try:
                name = stock.get_index_ticker_name(ticker)
                if '종합' in name or 'KOSDAQ' in name:
                    print(f'   - KOSDAQ: {ticker} = {name}')
            except:
                pass
                
    except Exception as e:
        print(f'   ❌ 지수 티커 조회 오류: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f'   ❌ pykrx import 오류: {type(e).__name__}: {e}')

# 2. 올바른 티커로 지수 데이터 가져오기
print('\n2️⃣ 올바른 티커로 지수 데이터 가져오기')
try:
    from pykrx import stock
    
    # 일반적으로 사용되는 지수 티커
    # KOSPI: "1001" 또는 "코스피" 또는 실제 티커
    # KOSDAQ: "2001" 또는 "코스닥" 또는 실제 티커
    
    # 방법 1: 티커 코드 직접 사용
    print('\n   방법 1: 티커 코드 직접 사용')
    for idx_code, name in [("1001", "KOSPI"), ("2001", "KOSDAQ")]:
        try:
            print(f'\n   {name} (코드: {idx_code}):')
            idx_data = stock.get_index_ohlcv_by_date(start, end, idx_code)
            if idx_data is not None and not idx_data.empty:
                print(f'   ✅ 성공! shape={idx_data.shape}')
                print(f'   - 컬럼: {list(idx_data.columns)}')
                print(f'   - 샘플:\n{idx_data.head(3)}')
            else:
                print(f'   ❌ 빈 데이터')
        except Exception as e:
            print(f'   ❌ 오류: {type(e).__name__}: {e}')
            # 상세 오류 출력
            import traceback
            traceback.print_exc()
    
    # 방법 2: 지수명으로 조회
    print('\n   방법 2: 지수명으로 조회')
    for idx_name in ["코스피", "코스닥", "KOSPI", "KOSDAQ"]:
        try:
            print(f'\n   {idx_name}:')
            idx_data = stock.get_index_ohlcv_by_date(start, end, idx_name)
            if idx_data is not None and not idx_data.empty:
                print(f'   ✅ 성공! shape={idx_data.shape}')
                print(f'   - 컬럼: {list(idx_data.columns)}')
            else:
                print(f'   ❌ 빈 데이터')
        except Exception as e:
            print(f'   ❌ 오류: {type(e).__name__}: {e}')
    
except Exception as e:
    print(f'   ❌ 오류: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

# 3. 저수준 API 테스트
print('\n3️⃣ 저수준 API 테스트 (get_index_ohlcv_by_date)')
try:
    from pykrx.website.krx.market import get_index_ohlcv_by_date
    
    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")
    
    print(f'\n   날짜 포맷: {start_str} ~ {end_str}')
    
    for idx_code, name in [("1001", "KOSPI"), ("2001", "KOSDAQ")]:
        try:
            print(f'\n   {name} (코드: {idx_code}):')
            idx_data = get_index_ohlcv_by_date(start_str, end_str, idx_code)
            if idx_data is not None and not idx_data.empty:
                print(f'   ✅ 성공! shape={idx_data.shape}')
                print(f'   - 컬럼: {list(idx_data.columns)}')
                print(f'   - 샘플:\n{idx_data.head(3)}')
            else:
                print(f'   ❌ 빈 데이터 반환')
        except Exception as e:
            print(f'   ❌ 오류: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f'   ❌ import 오류: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 80)
print('💡 결론 및 권장 사항')
print('=' * 80)
print('''
1. 날짜 범위 확인: 미래 날짜가 포함되지 않았는지 확인
2. pykrx 버전 확인: pip show pykrx 로 버전 확인
3. 지수 티커 확인: get_index_ticker_list()로 올바른 티커 확인
4. 대안: yfinance나 다른 데이터 소스 고려
''')