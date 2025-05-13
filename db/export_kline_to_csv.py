import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing in environment.")

# DB bağlantısı
engine = create_engine(DATABASE_URL)

def export_kline_data_to_csv(symbol: str, interval: str, output_dir: str = ".") -> str:
    """
    Export kline data for given symbol and interval to CSV.

    Args:
        symbol (str): e.g., "ETHUSDT"
        interval (str): e.g., "5m"
        output_dir (str): path to write the CSV

    Returns:
        str: path of the written CSV file
    """
    query = """
        SELECT 
            symbol, interval, open_time, open, high, low, close, volume,
            close_time, quote_asset_volume, number_of_trades,
            taker_buy_base_volume, taker_buy_quote_volume
        FROM kline_data
        WHERE symbol = %s AND interval = %s
        ORDER BY open_time
    """

    with engine.connect() as conn:
        #df = pd.read_sql(query, conn, params={"symbol": symbol, "interval": interval})
        df = pd.read_sql(query, conn, params=(symbol, interval))

    filename = f"{symbol}_{interval}_kline_export.csv"
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Exported {len(df)} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    # 🔧 Buraya parametreleri gir
    export_kline_data_to_csv(symbol="ETHFIUSDT", interval="5m")