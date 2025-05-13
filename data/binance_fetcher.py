import requests
import pandas as pd

BASE_URL = "https://fapi.binance.com"

def fetch_kline(symbol, interval, limit=1500):
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "_"
    ])
    df["symbol"] = symbol
    df["interval"] = interval
    df = df.drop(columns=["_"])
    return df

def fetch_funding_rate(symbol, limit=1000):
    url = f"{BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data)
    df["symbol"] = symbol
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["symbol", "fundingTime", "fundingRate"]]

def fetch_open_interest(symbol):
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": "5m", "limit": 500}
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data)
    df["symbol"] = symbol
    df["openInterest"] = df["sumOpenInterest"].astype(float)
    return df[["symbol", "timestamp", "openInterest"]]

def fetch_long_short_ratio(symbol):
    url = f"{BASE_URL}/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": "5m", "limit": 500}
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data)
    df["symbol"] = symbol
    df["longAccount"] = df["longAccount"]
    df["shortAccount"] = df["shortAccount"]
    df["ratio"] = df["longShortRatio"].astype(float)
    return df[["symbol", "timestamp", "longAccount", "shortAccount", "ratio"]]
