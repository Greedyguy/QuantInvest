#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 업데이트 스크립트
- 캐시를 무시하고 최신 데이터 다운로드
- 시가총액 데이터 포함
"""

import os
import sys
from datetime import date

# 캐시 디렉토리 정리
print("🔄 데이터 업데이트 시작...")
print("=" * 80)

# 1. 오래된 enriched 캐시 파일 정리
enriched_dir = "data/enriched"
if os.path.exists(enriched_dir):
    import glob
    files = glob.glob(f"{enriched_dir}/*_*.parquet")
    print(f"\n1️⃣ Enriched 캐시 정리: {len(files):,}개 파일")
    
    # 종목별로 최신 파일만 남기기
    from collections import defaultdict
    ticker_files = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        ticker = basename.split("_")[0]
        ticker_files[ticker].append(f)
    
    removed = 0
    for ticker, file_list in ticker_files.items():
        if len(file_list) > 1:
            # 가장 최근 파일만 보관
            latest = max(file_list, key=os.path.getmtime)
            for f in file_list:
                if f != latest:
                    try:
                        os.remove(f)
                        removed += 1
                    except:
                        pass
    
    print(f"   ✅ {removed:,}개 중복 파일 제거")

# 2. last_calc_date 초기화
cache_dir = "data/cache"
last_calc_file = os.path.join(cache_dir, "last_calc_date.txt")
if os.path.exists(last_calc_file):
    os.remove(last_calc_file)
    print(f"\n2️⃣ 증분 업데이트 캐시 초기화: ✅")

# 3. reports.py 실행 (캐시 무시 모드)
print(f"\n3️⃣ 최신 데이터 다운로드 중...")
print("   이 작업은 5~10분 소요될 수 있습니다.")
print("-" * 80)

# 캐시를 무시하고 데이터 로드
import subprocess
result = subprocess.run(
    [sys.executable, "reports.py", "--strategy", "baseline", "--no-cache"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("✅ 데이터 업데이트 완료!")
    print("=" * 80)
    print("\n📊 데이터 품질 재점검:")
    print("   python data_quality_check.py")
else:
    print("\n⚠️  데이터 업데이트 중 오류 발생")
    print("   수동 실행: python reports.py --strategy baseline --no-cache")

