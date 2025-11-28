#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 텔레그램 sendMessage 테스트 스크립트

사용법:
    python test_telegram_send.py "보낼 메시지 내용"

사전 준비:
    .env 파일 또는 환경변수에 아래 값 설정
      TELEGRAM_BOT_TOKEN=...
      TELEGRAM_CHAT_ID=...
"""

import os
import sys
import json
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "텔레그램 sendMessage 테스트입니다."

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[ERROR] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 가 설정되어 있지 않습니다.")
        print(" .env 또는 환경변수를 확인하세요.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    print(f"[INFO] URL: {url}")
    print(f"[INFO] CHAT_ID: {chat_id}")

    try:
        resp = requests.post(url, json=payload, timeout=15)
        print(f"[INFO] HTTP status: {resp.status_code}")
        try:
            data = resp.json()
            print("[INFO] Response JSON:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
        except Exception:
            print("[INFO] Raw response text:")
            print(resp.text[:1000])
    except Exception as exc:
        print(f"[ERROR] 요청 중 예외 발생: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

