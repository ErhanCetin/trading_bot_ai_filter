import pandas as pd
import os
import sys
from db.postgresql import engine
from signal_engine.indicators_original import add_indicators
from signal_engine.signal_generator import generate_signals
from signal_engine.signal_strength import apply_signal_strength
from telegram.telegram_notifier import send_telegram_message
from live_trade.binance_executor import open_position
from risk.position_sizer import calculate_position_size
from runner_data_sync import fetch_and_store
from config import SYMBOL, INTERVAL, SL_MULTIPLIER, TP_MULTIPLIER
from fetcher_strategy import get_fetcher
from utils.data_freshness import is_data_fresh
from utils.price_utils import get_current_market_price, is_entry_price_valid



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_environment, get_config

load_environment()
config = get_config()

# Config
SYMBOL = "BTCUSDT"#"AIOTUSDT" #config["SYMBOL"]
INTERVAL = "1m"#config["INTERVAL"]
ACCOUNT_BALANCE = config["ACCOUNT_BALANCE"]
RISK_PER_TRADE = config["RISK_PER_TRADE"]
LIMIT = config["LIMIT"]
SL_MULTIPLIER = config["SL_MULTIPLIER"]
TP_MULTIPLIER = config["TP_MULTIPLIER"]
LEVERAGE = config["LEVERAGE"]

#datayi binance'dan çek
# TODO : veri cekme işlemini async hale getir.fethch_scheduler.py kullan.
# TODO: README.md dosyasindaki TOODO'da ayrinti var. Asagidaki kod kalacak local icin.
fetcher = get_fetcher()
fetcher.fetch(SYMBOL, INTERVAL)


# Veri çek
query = f"""
    SELECT * FROM kline_data
    WHERE symbol = %(symbol)s AND interval = %(interval)s
    ORDER BY open_time DESC
    LIMIT {LIMIT}
"""

df = pd.read_sql(query, engine, params={"symbol": SYMBOL, "interval": INTERVAL})


df = df.sort_values("open_time")
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

# İndikatör ve sinyal hesapla
df = add_indicators(df)
df = generate_signals(df)
df = apply_signal_strength(df)
df["open_time_fmt"] = pd.to_datetime(df["open_time"], unit="ms")

print("📊 Son 5 mum:")
print(df[["open_time_fmt", "close", "rsi", "macd", "atr", "long_signal", "short_signal", "signal_strength"]].tail(5))




last = df.iloc[-1]
if not is_data_fresh(last["close_time"], INTERVAL):
    raise ValueError(f"⛔️ Veri güncel değil, sinyal üretimi atlandı.")

print("📍 Son mum verisi:")
print(last[["open_time_fmt", "close", "rsi", "macd", "atr", "long_signal", "short_signal", "signal_strength"]])
    # Sinyal motorları genelde en son oluşmuş muma göre karar verir çünkü:
    # Canlı işlem yapacaksan, şimdi oluşan muma göre pozisyona girmen gerekir
    # Backtest'te bir döngüyle tüm mumlara bakarsın, ama canlıda sadece "şu an" önemlidir
    # iloc[-1] → son satır (en yeni mum) → o anda sistem ne diyor?
    # Yani: "Bu an itibariyle pozisyon açmalı mıyım?" sorusunu cevaplamak için sadece last kullanılır.
entry_price = last["close"]
sl_price = entry_price - (last["atr"] * SL_MULTIPLIER)
tp_price = entry_price + (last["atr"] * TP_MULTIPLIER)
    # ATR → volatilite ölçer, yani "piyasa ne kadar oynak?".
    # Bu nedenle SL (stop-loss) ve TP (take-profit) seviyeleri genelde ATR’ye göre ayarlanır.
    # 1.5 x ATR	SL → daha geniş, küçük oynaklıkta stop olmasın
    # 2.5 x ATR	TP → daha uzak kâr hedefi = Risk:Ödül ≈ 1:1.6
    # Yani: SL çok dar olursa "noise" da bile stop olursun,
    # TP çok uzak olursa hiç hedefe varamazsın. Bu oranlar yaygın pratiklerdir.
    # Daha agresif: SL: ATR × 1   TP: ATR × 1.5
    # Daha temkinli: --> SL: ATR × 2  TP: ATR × 3


position_size = calculate_position_size(ACCOUNT_BALANCE, RISK_PER_TRADE, last["atr"], entry_price, sl_price)
print(last["long_signal"])
print(last["signal_strength"])


if last["long_signal"] and last["signal_strength"] >= 3:
      
    try:
        # islem acmak isten önce canlı fiyatı kontrol et. Eger fiyat sinyal fiyatından %1'den fazla sapma varsa işlem açma.
        current_price = get_current_market_price(SYMBOL)
    except Exception as e:
        print("❌ Retry sonrası da canlı fiyat alınamadı:", e)
        exit
        
    if not is_entry_price_valid(current_price, row['close']):
          print(f"❌ Canlı fiyat ({current_price}) ile sinyal fiyatı ({row['close']}) uyumsuz. İşlem iptal.")
    else: 
        msg = f"""*🚨 LONG Sinyali*
            *{SYMBOL}* 🔥 `LONG`
            `Entry:` {entry_price:.2f}
            `SL:` {sl_price:.2f}
            `TP:` {tp_price:.2f}
            *Lot:* {position_size}
            Güç Skoru: {last['signal_strength']} / 4"""
        
        send_telegram_message(msg)
        open_position(SYMBOL, "BUY", position_size, entry_price, sl_price, tp_price)

elif last["short_signal"] and last["signal_strength"] >= 3:
     # islem acmak isten önce canlı fiyatı kontrol et. Eger fiyat sinyal fiyatından %1'den fazla sapma varsa işlem açma
    try:
        # islem acmak isten önce canlı fiyatı kontrol et. Eger fiyat sinyal fiyatından %1'den fazla sapma varsa işlem açma.
        current_price = get_current_market_price(SYMBOL)
    except Exception as e:
        print("❌ Retry sonrası da canlı fiyat alınamadı:", e)
        exit
        
    if not is_entry_price_valid(current_price, row['close']):
          print(f"❌ Canlı fiyat ({current_price}) ile sinyal fiyatı ({row['close']}) uyumsuz. İşlem iptal.")
    else:
        msg = f"""*🚨 SHORT Sinyali*
            *{SYMBOL}* 🔥 `SHORT`
            `Entry:` {entry_price:.2f}
            `SL:` {tp_price:.2f}
            `TP:` {sl_price:.2f}
            *Lot:* {position_size}
            Güç Skoru: {last['signal_strength']} / 4"""

        send_telegram_message(msg)
        open_position(SYMBOL, "SELL", position_size, entry_price, tp_price, sl_price)

else:
    print("🔍 Sinyal yok ya da zayıf.")



   
