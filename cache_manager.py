#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캐시 관리 모듈

enriched 데이터, 인덱스 데이터, 백테스트 결과를 캐싱하여 성능 향상
"""

import os
import pickle
import hashlib
import json
import pandas as pd
from pathlib import Path


# 캐시 디렉토리 설정
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
ENRICHED_DIR = os.path.join(os.path.dirname(__file__), "data", "enriched")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "reports", "cache")

# 디렉토리 생성
for d in [CACHE_DIR, ENRICHED_DIR, INDEX_DIR, BACKTEST_DIR]:
    os.makedirs(d, exist_ok=True)


def get_config_hash(config_dict: dict) -> str:
    """
    설정 딕셔너리의 해시값 생성
    
    Args:
        config_dict: 설정 딕셔너리
        
    Returns:
        MD5 해시 문자열
    """
    # 딕셔너리를 정렬된 JSON 문자열로 변환
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ============================================================
# Enriched 데이터 캐싱
# ============================================================

def save_enriched(ticker: str, df: pd.DataFrame):
    """
    종목의 enriched 데이터를 파일로 저장
    
    Args:
        ticker: 종목 코드
        df: enriched DataFrame
    """
    if df is None or df.empty:
        return
    
    # 날짜 범위 추출
    start_date = df.index.min().strftime("%Y%m%d")
    end_date = df.index.max().strftime("%Y%m%d")
    filepath = os.path.join(ENRICHED_DIR, f"{ticker}_{start_date}_{end_date}.parquet")
    
    try:
        df.to_parquet(filepath, compression='gzip')
    except Exception as e:
        print(f"⚠️  {ticker} enriched 저장 실패: {e}")


def load_enriched(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    종목의 enriched 데이터를 파일에서 로드
    
    Args:
        ticker: 종목 코드
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        
    Returns:
        enriched DataFrame 또는 None
    """
    import glob
    
    # 해당 종목의 모든 캐시 파일 찾기
    pattern = os.path.join(ENRICHED_DIR, f"{ticker}_*.parquet")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return None
    
    # 가장 최근 파일 사용
    filepath = max(matching_files, key=os.path.getmtime)
    
    try:
        df = pd.read_parquet(filepath)
        
        if len(df) > 0:
            req_start = pd.to_datetime(start_date)
            req_end = pd.to_datetime(end_date)
            df_start = df.index.min()
            df_end = df.index.max()
            
            # ✅ FIX: df_end가 None 또는 NaT인지 확인
            if df_end is None or pd.isna(df_end) or df_start is None or pd.isna(df_start):
                # 인덱스가 유효하지 않으면 캐시 무효
                return None
            
            # 요청 범위가 캐시 범위 내에 있는지 확인
            if df_start <= req_start and df_end >= req_end:
                # ✅ 시작 날짜와 종료 날짜 모두 필터링
                df = df[(df.index >= req_start) & (df.index <= req_end)]
                return df
            elif df_start <= req_end and df_end >= req_start:
                # 부분적으로 겹치는 경우: 겹치는 부분만 반환
                df = df[(df.index >= req_start) & (df.index <= req_end)]
                if len(df) > 100:  # 최소 데이터 개수 확인
                    return df
            
            # 데이터 품질 검증: 마지막 날짜가 요청 종료일보다 30일 이상 이전이면 무효
            try:
                days_diff = (req_end - df_end).days
            except (TypeError, AttributeError):
                # 날짜 연산 실패 시 캐시 무효
                return None
            
            if days_diff > 30:
                # 오래된 캐시는 사용하지 않음
                return None
        
        return None
        
    except Exception as e:
        print(f"⚠️  {ticker} enriched 로드 실패: {e}")
        return None

# ============================================================
# 인덱스 데이터 캐싱
# ============================================================

def save_index(index_name: str, df: pd.DataFrame):
    """
    인덱스 데이터를 파일로 저장
    
    Args:
        index_name: 인덱스 이름 (예: KOSPI, KOSDAQ)
        df: 인덱스 DataFrame
    """
    if df is None or df.empty:
        return
    
    filepath = os.path.join(INDEX_DIR, f"{index_name}.parquet")
    
    try:
        df.to_parquet(filepath, compression='gzip')
    except Exception as e:
        print(f"⚠️  {index_name} 인덱스 저장 실패: {e}")


def load_index(index_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    인덱스 데이터를 파일에서 로드
    
    Args:
        index_name: 인덱스 이름
        start_date: 시작일
        end_date: 종료일
        
    Returns:
        인덱스 DataFrame 또는 None
    """
    filepath = os.path.join(INDEX_DIR, f"{index_name}.parquet")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_parquet(filepath)
        
        if len(df) > 0:
            df_start = df.index.min()
            df_end = df.index.max()
            req_start = pd.to_datetime(start_date)
            req_end = pd.to_datetime(end_date)
            
            # 요청 범위가 캐시 범위 내에 있는지 확인
            if df_start <= req_start and df_end >= req_end:
                df = df[(df.index >= req_start) & (df.index <= req_end)]
                return df
        
        return None
        
    except Exception as e:
        print(f"⚠️  {index_name} 인덱스 로드 실패: {e}")
        return None


# ============================================================
# 마지막 계산 날짜 저장
# ============================================================

def save_last_calc_date(date: str):
    """
    마지막 계산 날짜 저장
    
    Args:
        date: 날짜 문자열 (YYYY-MM-DD)
    """
    filepath = os.path.join(CACHE_DIR, "last_calc_date.txt")
    
    try:
        with open(filepath, 'w') as f:
            f.write(date)
    except Exception as e:
        print(f"⚠️  마지막 계산 날짜 저장 실패: {e}")


def get_last_calc_date() -> str:
    """
    마지막 계산 날짜 로드
    
    Returns:
        날짜 문자열 또는 None
    """
    filepath = os.path.join(CACHE_DIR, "last_calc_date.txt")
    
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️  마지막 계산 날짜 로드 실패: {e}")
        return None


# ============================================================
# 백테스트 결과 캐싱
# ============================================================

def save_backtest_result(strategy_name: str, config_hash: str, equity_curve: pd.DataFrame, trade_log: list):
    """
    백테스트 결과 저장
    
    Args:
        strategy_name: 전략 이름
        config_hash: 설정 해시값
        equity_curve: equity curve DataFrame
        trade_log: 거래 로그 리스트
    """
    filename = f"{strategy_name}_{config_hash}.pkl"
    filepath = os.path.join(BACKTEST_DIR, filename)
    
    try:
        result = {
            'equity_curve': equity_curve,
            'trade_log': trade_log,
            'strategy_name': strategy_name,
            'config_hash': config_hash,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
            
    except Exception as e:
        print(f"⚠️  백테스트 결과 저장 실패: {e}")


def load_backtest_result(strategy_name: str, config_hash: str) -> tuple:
    """
    백테스트 결과 로드
    
    Args:
        strategy_name: 전략 이름
        config_hash: 설정 해시값
        
    Returns:
        (equity_curve, trade_log) 튜플 또는 (None, None)
    """
    filename = f"{strategy_name}_{config_hash}.pkl"
    filepath = os.path.join(BACKTEST_DIR, filename)
    
    if not os.path.exists(filepath):
        return None, None
    
    try:
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        
        return result.get('equity_curve'), result.get('trade_log')
        
    except Exception as e:
        print(f"⚠️  백테스트 결과 로드 실패: {e}")
        return None, None


# ============================================================
# 캐시 정리
# ============================================================

def clear_cache(cache_type: str = "all"):
    """
    캐시 정리
    
    Args:
        cache_type: 'all', 'enriched', 'index', 'backtest' 중 하나
    """
    import shutil
    
    if cache_type in ["all", "enriched"]:
        if os.path.exists(ENRICHED_DIR):
            shutil.rmtree(ENRICHED_DIR)
            os.makedirs(ENRICHED_DIR, exist_ok=True)
            print("✅ Enriched 캐시 삭제")
    
    if cache_type in ["all", "index"]:
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            os.makedirs(INDEX_DIR, exist_ok=True)
            print("✅ Index 캐시 삭제")
    
    if cache_type in ["all", "backtest"]:
        if os.path.exists(BACKTEST_DIR):
            shutil.rmtree(BACKTEST_DIR)
            os.makedirs(BACKTEST_DIR, exist_ok=True)
            print("✅ Backtest 캐시 삭제")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        cache_type = sys.argv[2] if len(sys.argv) > 2 else "all"
        clear_cache(cache_type)
        print(f"✅ 캐시 정리 완료: {cache_type}")
    else:
        print("Usage: python cache_manager.py clear [all|enriched|index|backtest]")
