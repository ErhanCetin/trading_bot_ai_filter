import os

# Genel işlem parametreleri
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "5m")
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", 10000))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))

# ATR çarpanları
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", 1.5))
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", 2.5))

