#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 텔레그램 알림 유틸리티
환경 변수 또는 .env 에서 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 를 읽어 메시지를 전송한다.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import requests


logger = logging.getLogger(__name__)


class TelegramNotifier:
    """텔레그램 봇을 사용해 간단한 텍스트 알림을 전송한다."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        if not self.bot_token or not self.chat_id:
            logger.warning("텔레그램 토큰/채팅 ID가 설정되지 않아 알림 기능이 비활성화됩니다.")

    def can_send(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _build_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text: str, parse_mode: Optional[str] = None) -> bool:
        """메시지를 전송한다. 실패해도 예외를 던지지 않고 False를 반환."""
        if not self.can_send():
            logger.debug("텔레그램 토큰이 없어 메시지를 전송하지 않습니다.")
            return False
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            resp = requests.post(self._build_url(), json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            logger.error("텔레그램 전송 실패: %s - %s", resp.status_code, resp.text)
        except Exception as exc:  # pragma: no cover
            logger.error("텔레그램 전송 중 오류: %s", exc)
        return False


def format_alert(title: str, lines: list[str]) -> str:
    """알림 메시지를 보기 좋은 텍스트 형태로 만든다."""
    joined = "\n".join(lines)
    return f"[{title}]\\n{joined}"
