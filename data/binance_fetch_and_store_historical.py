import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_environment
load_environment()

from data.binance_fetch_historical import (
    fetch_kline,
    fetch_funding_rate,
    fetch_open_interest,
    fetch_long_short_ratio,
)
from db.writer import (
    insert_kline,
    insert_funding_rate,
    insert_open_interest,
    insert_long_short_ratio,
)

SYMBOL = "ETHFIUSDT"
INTERVAL = "5m"
DAYS = 7

def fetch_and_store_all():
    print(f"📥 {SYMBOL} için {DAYS} günlük 5m veriler çekiliyor...")

    try:
        df_kline = fetch_kline(SYMBOL, INTERVAL, days=DAYS)
        insert_kline(df_kline)
        print(f"✅ Kline verisi: {len(df_kline)} satır")

        df_fr = fetch_funding_rate(SYMBOL, days=DAYS)
        insert_funding_rate(df_fr)
        print(f"✅ Funding rate: {len(df_fr)} satır")

        df_oi = fetch_open_interest(SYMBOL, interval=INTERVAL, days=DAYS)
        insert_open_interest(df_oi)
        print(f"✅ Open Interest: {len(df_oi)} satır")

        df_lsr = fetch_long_short_ratio(SYMBOL, interval=INTERVAL, days=DAYS)
        insert_long_short_ratio(df_lsr)
        print(f"✅ Long/Short Ratio: {len(df_lsr)} satır")

        print("🎉 Tüm veriler başarıyla veritabanına kaydedildi.")
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    fetch_and_store_all()
