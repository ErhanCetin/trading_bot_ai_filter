# check_db_data.py - Hızlı kontrol için
import pandas as pd
from backtest.utils.config_loader import load_env_config
from backtest.utils.data_loader import load_price_data

def quick_db_check():
    """Hızlı veritabanı veri kontrolü"""
    env_config = load_env_config()
    db_url = env_config.get("db_url")
    
    # ✅ TEST EDİLECEK SYMBOL/INTERVAL ÇIFTLERI
    test_pairs = [
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "4h"), 
        ("ETHUSDT", "1h"),
        ("ADAUSDT", "1h"),
        ("SOLUSDT", "4h"),
        ("BTCUSDT", "15m")
    ]
    
    print("🔍 DATABASE DATA CHECK:")
    print("=" * 50)
    
    for symbol, interval in test_pairs:
        try:
            df = load_price_data(symbol, interval, db_url)
            if not df.empty:
                start_date = pd.to_datetime(df['open_time'].min(), unit='ms').strftime('%Y-%m-%d')
                end_date = pd.to_datetime(df['open_time'].max(), unit='ms').strftime('%Y-%m-%d')
                print(f"✅ {symbol} {interval}: {len(df)} records ({start_date} to {end_date})")
            else:
                print(f"❌ {symbol} {interval}: NO DATA")
        except Exception as e:
            print(f"❌ {symbol} {interval}: ERROR - {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    quick_db_check()