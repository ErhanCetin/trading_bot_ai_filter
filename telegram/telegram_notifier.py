import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import load_environment

load_environment()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_token")
BOT_CHAT_ID = os.getenv("TELEGRAM_BOT_CHAT_ID", "your_chat_id")
GROUP_CHAT_ID = os.getenv("TELEGRAM_GROUP_CHAT_ID", "your_chat_id")


def send_telegram_message(message: str):
    payload = get_telegram_message(BOT_CHAT_ID, message)
    send_telegram_message_to_user(payload)
    payload_group = get_telegram_message(GROUP_CHAT_ID, message)
    send_telegram_message_to_user(payload_group)


  

def send_telegram_message_to_user(payload: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print("Telegram gönderim hatası:", response.text)

def get_telegram_message(chat_id: str,message: str):
    return {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }


# https://api.telegram.org/bot7747655121:AAHbBR56ZAebAgyFXI1IaEs8l_3kgmwJPFU/getUpdates
# group bilgilerini almak için yukarıdaki linki kullanabilirsin.
