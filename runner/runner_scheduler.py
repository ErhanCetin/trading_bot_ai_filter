import os
import sys
import asyncio
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from utils.data_freshness import is_data_fresh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_environment, get_config
from db.postgresql import engine
from signal_engine.indicators_original import add_indicators
from signal_engine.signal_generator import generate_signals
from signal_engine.signal_strength import apply_signal_strength
from telegram.telegram_notifier import send_telegram_message
from live_trade.binance_executor import open_position
from risk.position_sizer import calculate_position_size
from fetcher_strategy import get_fetcher
from utils.price_utils import get_current_market_price,is_entry_price_valid



load_environment()
config = get_config()

SYMBOL = "ETHUSDT" #config["SYMBOL"]
INTERVAL = config["INTERVAL"]
ACCOUNT_BALANCE = config["ACCOUNT_BALANCE"]
RISK_PER_TRADE = config["RISK_PER_TRADE"]
SL_MULTIPLIER = config["SL_MULTIPLIER"]
TP_MULTIPLIER = config["TP_MULTIPLIER"]

last_open_time = None
ENV = os.getenv("ENV", "local")


async def check_new_candle():
    global last_open_time

    # TODO : veri cekme işlemini async hale getir.fethch_scheduler.py kullan.
    # TODO: README.md dosyasindaki TOODO'da ayrinti var. Asagidaki kod kalacak local icin.
    fetcher = get_fetcher()
    fetcher.fetch(SYMBOL, INTERVAL)

    query = f"""
        SELECT * FROM kline_data
        WHERE symbol = %(symbol)s AND interval = %(interval)s
        ORDER BY open_time DESC
        LIMIT 100
    """
    df = pd.read_sql(query, engine, params={"symbol": SYMBOL, "interval": INTERVAL})
    df = df.sort_values("open_time")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    
    last_close_time = df["close_time"].iloc[-1]
    if not is_data_fresh(last_close_time, INTERVAL):
        print("⛔️ Veri güncel değil, sinyal üretimi atlandı.")
        return
    latest_open_time = df["open_time"].iloc[-1]
    if last_open_time == latest_open_time:
        print("⏳ Yeni mum oluşmamış, bekleniyor...")
        return
    last_open_time = latest_open_time

    print("⏳ Indikator hesaplmasi basladi ...")

    df = add_indicators(df)
    df = generate_signals(df)
    df = apply_signal_strength(df)
    
    df["open_time_fmt"] = pd.to_datetime(df["open_time"], unit="ms")

    print("📊 Son 3 mum:")
    print(df[["open_time_fmt", "close", "rsi", "macd", "atr", "long_signal", "short_signal", "signal_strength"]].tail(3))

    recent = df.tail(3)
    for _, row in recent.iterrows():
        if row["signal_strength"] >= 3 and row["long_signal"]:
            
            try:
                # islem acmak isten önce canlı fiyatı kontrol et. Eger fiyat sinyal fiyatından %1'den fazla sapma varsa işlem açma.
                current_price = get_current_market_price(SYMBOL)
            except Exception as e:
                print("❌ Retry sonrası da canlı fiyat alınamadı:", e)
                continue
            
            if not is_entry_price_valid(current_price, row['close']):
                print(f"❌ Canlı fiyat ({current_price}) ile sinyal fiyatı ({row['close']}) uyumsuz. İşlem iptal.")
                continue
            
            print(f"✅ Giriş fiyatı geçerli: {current_price} == {row['close']}")

            entry_price = row["close"]
            sl_price = entry_price - (row["atr"] * SL_MULTIPLIER)
            tp_price = entry_price + (row["atr"] * TP_MULTIPLIER)
            position_size = calculate_position_size(ACCOUNT_BALANCE, RISK_PER_TRADE, row["atr"], entry_price, sl_price)

            msg = f"""*🚨 LONG Sinyali*
                *{SYMBOL}* 🔥 `LONG`
                `Entry:` {entry_price:.2f}
                `SL:` {sl_price:.2f}
                `TP:` {tp_price:.2f}
                *Lot:* {position_size}
                Güç Skoru: {row["signal_strength"]} / 4"""

            send_telegram_message(msg)
            open_position(SYMBOL, "BUY", position_size, entry_price, sl_price, tp_price)
            break

        elif row["signal_strength"] >= 3 and row["short_signal"]:
            entry_price = row["close"]
            sl_price = entry_price + (row["atr"] * SL_MULTIPLIER)
            tp_price = entry_price - (row["atr"] * TP_MULTIPLIER)
            position_size = calculate_position_size(ACCOUNT_BALANCE, RISK_PER_TRADE, row["atr"], entry_price, sl_price)

            msg = f"""*🚨 SHORT Sinyali*
                *{SYMBOL}* 🔥 `SHORT`
                `Entry:` {entry_price:.2f}
                `SL:` {sl_price:.2f}
                `TP:` {tp_price:.2f}
                *Lot:* {position_size}
                Güç Skoru: {row["signal_strength"]} / 4"""

            send_telegram_message(msg)
            open_position(SYMBOL, "SELL", position_size, entry_price, sl_price, tp_price)
            break

async def main():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(check_new_candle, "interval", seconds=10)
    scheduler.start()
    print("📡 Otomatik sinyal motoru başlatıldı.")
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
