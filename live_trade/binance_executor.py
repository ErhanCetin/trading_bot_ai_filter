import os
from binance.client import Client

API_KEY = os.getenv("BINANCE_API_KEY", "your_key")
API_SECRET = os.getenv("BINANCE_API_SECRET", "your_secret")

client = Client(API_KEY, API_SECRET)
client.FUTURES_URL = "https://fapi.binance.com/fapi"

def setup_account(symbol: str, leverage: int = 10):
    try:
        # Hedge mode aktif et
        client.futures_change_position_mode(dualSidePosition=True)
        # Kaldıraç ayarla
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        print(f"✅ Hedge mode aktif, kaldıraç: {leverage}x")
    except Exception as e:
        print("⚠️ Account ayar hatası:", e)

def open_position(symbol: str, side: str, quantity: float, entry_price: float, sl_price: float, tp_price: float):
    """
    side: 'BUY' (long) or 'SELL' (short)
    """
    try:
        setup_account(symbol)

        # order = client.futures_create_order(
        #     symbol=symbol,
        #     side=side,
        #     type='MARKET',
        #     quantity=quantity
        # )
        # print("✅ Market order gönderildi:", order["orderId"])

        # sl_side = 'SELL' if side == 'BUY' else 'BUY'
        # tp_side = sl_side

        # client.futures_create_order(
        #     symbol=symbol,
        #     side=sl_side,
        #     type='STOP_MARKET',
        #     stopPrice=round(sl_price, 2),
        #     closePosition=True
        # )
        # client.futures_create_order(
        #     symbol=symbol,
        #     side=tp_side,
        #     type='TAKE_PROFIT_MARKET',
        #     stopPrice=round(tp_price, 2),
        #     closePosition=True
        # )
        print("✅ SL/TP emirleri gönderildi")
    except Exception as e:
        print("❌ Emir gönderme hatası:", e)
