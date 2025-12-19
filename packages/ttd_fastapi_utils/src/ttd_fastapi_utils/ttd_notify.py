
# --- Simple notifier API (wraps requests) ---
from datetime import datetime
import json
import logging
import os
import socket

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional
    requests = None

hostname = socket.gethostname()
logger = logging.getLogger(__name__)


class TTDNotify:
    def __init__(self, webhook_url: str):
        self.muted = False
        self.webhook_url = webhook_url
        if os.getenv("ENVIRONMENT", "") == "development":
            logger.info("Development environment, muted notifications")
            self.muted = True

    def send_message(self, content: str, level: str = "Info", message_type: str = "text"):
        if self.muted or not self.webhook_url or requests is None:
            return
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"[{level}][{current_time}][{hostname}]\n{content}"
        headers = {"Content-Type": "application/json"}
        data = {
            "msgtype": message_type,
            "text": {
                "content": content,
                "mentioned_list": ["@all"],
            },
        }
        try:
            resp = requests.post(self.webhook_url, headers=headers, data=json.dumps(data))
            resp.raise_for_status()
            result = resp.json()
            if result.get("errcode") != 0:
                logger.error(f"Failed to send message: {result}")
        except Exception:
            logger.exception("Request error while sending notification")

    def send_error_message(self, error_message: str):
        self.send_message(error_message, "Error")
        logger.error(error_message)

    def send_info_message(self, info_message: str):
        self.send_message(info_message, "Info")
        logger.info(info_message)

    def send_warning_message(self, warning_message: str):
        self.send_message(warning_message, "Warning")
        logger.warning(warning_message)


# Default notifier (solidified in package, override with TTD_WEBHOOK_URL)
_DEFAULT_WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4e5f3845-cd5f-43a0-b004-231e710c68e4"
_notifier_url = os.getenv("TTD_WEBHOOK_URL", _DEFAULT_WEBHOOK_URL)
notifier = TTDNotify(_notifier_url)
