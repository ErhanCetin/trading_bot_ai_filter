# debug_database.py - Veritabanında ne var kontrol et
import pandas as pd
from sqlalchemy import create_engine, text

def check_database_content(db_url: str):
    """Veritabanındaki mevcut verileri kontrol et"""
    engine = create_engine(db_url)
    
    # ✅ MEVCUT SEMBOLLER
    symbols_query = "SELECT DISTINCT symbol FROM kline_data ORDER BY symbol"
    symbols_df = pd.read_sql(text(symbols_query), engine)
    print("🔍 AVAILABLE SYMBOLS:")
    print(symbols_df['symbol'].tolist())
    
    # ✅ MEVCUT INTERVALS
    intervals_query = "SELECT DISTINCT interval FROM kline_data ORDER BY interval"
    intervals_df = pd.read_sql(text(intervals_query), engine)
    print("\n🔍 AVAILABLE INTERVALS:")
    print(intervals_df['interval'].tolist())
    
    # ✅ SEMBOL/INTERVAL KOMBİNASYONLARI
    combo_query = """
    SELECT symbol, interval, COUNT(*) as record_count, 
           MIN(open_time) as first_record, MAX(open_time) as last_record
    FROM kline_data 
    GROUP BY symbol, interval 
    ORDER BY symbol, interval
    """
    combo_df = pd.read_sql(text(combo_query), engine)
    print("\n🔍 SYMBOL/INTERVAL COMBINATIONS:")
    print(combo_df.to_string())
    
    engine.dispose()
    return combo_df

# Ana kodda kullan
from backtest.utils.config_loader import load_env_config
env_config = load_env_config()
available_data = check_database_content(env_config.get("db_url"))