import os
import sys

def fetch_and_store(symbol: str, interval: str = "5m"):
    from data.binance_fetcher import fetch_kline, fetch_funding_rate, fetch_open_interest, fetch_long_short_ratio
    from db.writer import insert_kline, insert_funding_rate, insert_open_interest, insert_long_short_ratio

    df_kline = fetch_kline(symbol, interval)
    insert_kline(df_kline)

    df_fr = fetch_funding_rate(symbol)
    insert_funding_rate(df_fr)

    df_oi = fetch_open_interest(symbol)
    insert_open_interest(df_oi)

    df_lsr = fetch_long_short_ratio(symbol)
    insert_long_short_ratio(df_lsr)
