import time
import pandas as pd
import requests
from datetime import datetime, timedelta



# Tablo	API	Süre sınırı	Notlar
# kline_data	/api/v3/klines	1500 bar/req	✅ hazır
# funding_rate	/fapi/v1/fundingRate	1000 kayıt/req	8 saatte bir kayıt
# open_interest	/futures/data/openInterestHist	30 gün geriye	5 dakikalık destekler
# long_short_ratio	/futures/data/globalLongShortAccountRatio	30 gün geriye	genelde 15m/1h


#df = get_historical_klines("BTCUSDT", "5m", 7)
#print(df.tail())

BASE_URL = "https://fapi.binance.com"

def get_unix_ms(dt):
    return int(dt.timestamp() * 1000)

def fetch_kline(symbol: str, interval: str, days: int = 7) -> pd.DataFrame:
    end_time = int(time.time() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []

    while start_time < end_time:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1500
        }
        res = requests.get(url, params=params)
        data = res.json()

        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "_"
    ])
    df["symbol"] = symbol
    df["interval"] = interval
    df = df.drop(columns=["_"])
    return df

def fetch_funding_rate(symbol: str, days: int = 7) -> pd.DataFrame:
    end_time = int(time.time() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []

    while start_time < end_time:
        url = f"{BASE_URL}/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        res = requests.get(url, params=params)
        data = res.json()

        if not data:
            break
        all_data.extend(data)
        start_time = data[-1]["fundingTime"] + 1
        time.sleep(0.2)

    return pd.DataFrame(all_data)

def fetch_open_interest(symbol: str, interval: str = "5m", days: int = 7) -> pd.DataFrame:
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    all_data = []

    while start_time < end_time:
        url = f"{BASE_URL}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": interval,
            "limit": 500,
            "startTime": get_unix_ms(start_time),
            "endTime": get_unix_ms(end_time)
        }
        res = requests.get(url, params=params)
        data = res.json()

        if not data or isinstance(data, dict) and data.get("code"):
            break
        all_data.extend(data)
        start_time = datetime.fromtimestamp(data[-1]["timestamp"] / 1000.0) + timedelta(minutes=5)
        time.sleep(0.5)

    return pd.DataFrame(all_data)

def fetch_long_short_ratio(symbol: str, interval: str = "5m", days: int = 7) -> pd.DataFrame:
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    all_data = []

    while start_time < end_time:
        url = f"{BASE_URL}/futures/data/globalLongShortAccountRatio"
        params = {
            "symbol": symbol,
            "period": interval,
            "limit": 500,
            "startTime": get_unix_ms(start_time),
            "endTime": get_unix_ms(end_time)
        }
        res = requests.get(url, params=params)
        data = res.json()

        if not data or isinstance(data, dict) and data.get("code"):
            break
        all_data.extend(data)
        start_time = datetime.fromtimestamp(data[-1]["timestamp"] / 1000.0) + timedelta(minutes=5)
        time.sleep(0.5)

    return pd.DataFrame(all_data)
