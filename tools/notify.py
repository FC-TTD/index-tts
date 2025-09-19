import logging
import os
from datetime import datetime
import socket

import requests
import json

logger = logging.getLogger(__name__)

hostname = socket.gethostname()

class Notify:
    def __init__(self, webhook_url):
        """
        初始化通知类
        :param webhook_url: 企业微信机器人 webhook 地址
        """
        self.muted = False
        self.webhook_url = webhook_url
        if os.getenv("ENVIRONMENT","") == "development":
            logger.info("Development environment, muted notifications")
            self.muted = True

    def send_message(self, content, level="Info", message_type="text"):
        """
        发送消息到企业微信机器人
        :param content: 消息内容
        :param message_type: 消息类型，默认是 text
        """
        if self.muted:
            return
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content = f"[{level}][{current_time}][{hostname}]\n{content}"

        headers = {"Content-Type": "application/json"}
        data = {
            "msgtype": message_type,
            "text": {
                "content": content,
                "mentioned_list": ["@all"],
            }
        }
        try:
            response = requests.post(self.webhook_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            if result.get("errcode") != 0:
                logger.error(f"Failed to send message: {result}")
        except requests.exceptions.RequestException as e:
            logger.exception(f"Request error: {e}")

    def send_error_message(self, error_message):
        """
        发送报错消息
        :param error_message: 报错消息内容
        """
        self.send_message(error_message, "Error")
        logger.error(error_message)

    def send_info_message(self, info_message):
        """
        发送报错消息
        :param info_message: 报错消息内容
        """
        self.send_message(info_message, "Info")
        logger.info(info_message)

    def send_warning_message(self, warning_message):
        """
        发送警告消息
        :param warning_message: 警告消息内容
        """
        self.send_message(warning_message, "Warning")
        logger.warning(warning_message)


webhook = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a72c830d-7d5b-4d88-93e5-326d888600ce"
notifier = Notify(webhook)
