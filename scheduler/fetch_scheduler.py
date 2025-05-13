import asyncio
from data.binance_fetcher import (
    fetch_kline, fetch_funding_rate,
    fetch_open_interest, fetch_long_short_ratio
)
from db.writer_sonra_sil import (
    insert_kline, insert_funding_rate,
    insert_open_interest, insert_long_short_ratio
)

SYMBOL = "BTCUSDT"
INTERVAL = "1m"

async def task_kline():
    try:
        df = fetch_kline(SYMBOL, INTERVAL, limit=100)
        insert_kline(df)
        print("✅ Kline güncellendi")
    except Exception as e:
        print("Kline HATA:", e)

async def task_funding_rate():
    try:
        df = fetch_funding_rate(SYMBOL)
        insert_funding_rate(df)
        print("✅ Funding Rate güncellendi")
    except Exception as e:
        print("Funding HATA:", e)

async def task_open_interest():
    try:
        df = fetch_open_interest(SYMBOL)
        insert_open_interest(df)
        print("✅ Open Interest güncellendi")
    except Exception as e:
        print("OI HATA:", e)

async def task_long_short_ratio():
    try:
        df = fetch_long_short_ratio(SYMBOL)
        insert_long_short_ratio(df)
        print("✅ Long/Short Ratio güncellendi")
    except Exception as e:
        print("LSR HATA:", e)

async def main_loop():
    while True:
        await asyncio.gather(
            task_kline(),
            task_funding_rate(),
            task_open_interest(),
            task_long_short_ratio()
        )
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main_loop())
