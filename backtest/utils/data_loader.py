"""
Veri yükleme ve hazırlama işlevleri
"""
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import text, create_engine

def load_price_data(symbol: str, interval: str, db_url: str) -> pd.DataFrame:
    """
    Veritabanından fiyat verilerini yükler
    
    Args:
        symbol: İşlem sembolü (örn. "BTCUSDT")
        interval: Zaman aralığı (örn. "1m", "5m", "1h")
        db_url: Veritabanı bağlantı URL'si
        
    Returns:
        Fiyat verilerini içeren DataFrame
    """
    print(f"📊 Loading price data for {symbol} at {interval} interval from {db_url}")
    
    try:
        engine = create_engine(db_url)
        
        query = f"""
        SELECT * FROM kline_data
        WHERE symbol = '{symbol}' AND interval = '{interval}'
        ORDER BY open_time
        """
        
        print(f"🔍 DATA DEBUG: SQL Query = {query}")
        
        df = pd.read_sql(text(query), engine)
        
        print(f"🔍 DATA DEBUG: Raw query result:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Empty: {df.empty}")
        print(f"  - Columns: {df.columns.tolist()}")
        
        if df.empty:
            print("❌ DATA DEBUG: Query returned empty result!")
            print("❌ Checking if table exists and has data...")
            
            # Tablo var mı kontrol et
            check_query = "SELECT COUNT(*) as count FROM kline_data"
            count_df = pd.read_sql(text(check_query), engine)
            print(f"❌ Total records in kline_data: {count_df.iloc[0]['count']}")
            
            # Symbol ve interval değerlerini kontrol et
            symbol_query = f"SELECT DISTINCT symbol FROM kline_data WHERE symbol LIKE '%{symbol[:4]}%'"
            symbol_df = pd.read_sql(text(symbol_query), engine)
            print(f"❌ Similar symbols in database: {symbol_df['symbol'].tolist()}")
            
            interval_query = "SELECT DISTINCT interval FROM kline_data"
            interval_df = pd.read_sql(text(interval_query), engine)
            print(f"❌ Available intervals: {interval_df['interval'].tolist()}")
            
            engine.dispose()
            return pd.DataFrame()  # Boş DataFrame döndür
        
        print(f"🔍 DATA DEBUG: Data loaded successfully:")
        print(f"  - First row: {df.iloc[0].to_dict()}")
        print(f"  - Last row: {df.iloc[-1].to_dict()}")
        
        # Sayısal sütunları dönüştür
        numeric_columns = ["open", "high", "low", "close", "volume"]
        missing_numeric = [col for col in numeric_columns if col not in df.columns]
        if missing_numeric:
            print(f"⚠️ Missing numeric columns: {missing_numeric}")
            # Eksik sütunları 0 ile doldur
            for col in missing_numeric:
                df[col] = 0.0
        
        # Mevcut numeric sütunları dönüştür
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        df[existing_numeric] = df[existing_numeric].astype(float)
        
        # Zaman sütununu standartlaştır
        if "open_time" in df.columns:
            df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms")
        else:
            print("⚠️ open_time column missing!")
        
        print(f"🔍 DATA DEBUG: Final DataFrame:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        print(f"  - Index: {df.index}")
        
        engine.dispose()
        return df
        
    except Exception as e:
        print(f"❌ DATA LOADER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Hata durumunda boş DataFrame

def parse_indicators_config(config_str: str) -> Dict[str, Any]:
    """
    İndikatör konfigürasyonlarını JSON formatından Python sözlüğüne dönüştürür
    
    Args:
        config_str: JSON formatında indikatör konfigürasyonu
        
    Returns:
        İndikatör konfigürasyonu sözlüğü
    """
    import json
    
    if not config_str:
        return {}
    
    try:
        return json.loads(config_str)
    except json.JSONDecodeError:
        print(f"⚠️ Error parsing indicators config: {config_str}")
        return {}

def load_config_combinations(csv_path: str) -> pd.DataFrame:
    """
    Konfigürasyon kombinasyonlarını CSV dosyasından yükler
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        Konfigürasyon kombinasyonlarını içeren DataFrame
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Error loading config combinations: {e}")
        return pd.DataFrame()

def transform_config_row(row: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Konfigürasyon satırını Signal Engine formatına dönüştürür
    
    Args:
        row: Konfigürasyon satırı
        
    Returns:
        Signal Engine formatında konfigürasyon
    """
    config = {
        "indicators": {},
        "strategies": {},
        "strength": {},
        "filters": {}
    }
    
    # İndikatörleri ekle
    indicators = {}
    
    # EMA indikatörleri
    if not pd.isna(row.get("EMA_FAST")) and not pd.isna(row.get("EMA_SLOW")):
        indicators["ema"] = {
            "fast_period": int(row["EMA_FAST"]), 
            "slow_period": int(row["EMA_SLOW"])
        }
    
    # RSI indikatörü
    if not pd.isna(row.get("RSI")):
        indicators["rsi"] = {"period": int(row["RSI"])}
    
    # MACD indikatörü
    if not pd.isna(row.get("MACD")) and bool(row["MACD"]):
        indicators["macd"] = {}
    
    # ATR indikatörü
    if not pd.isna(row.get("ATR")):
        indicators["atr"] = {"period": int(row["ATR"])}
    
    # OBV indikatörü
    if not pd.isna(row.get("OBV")) and bool(row["OBV"]):
        indicators["obv"] = {}
    
    # ADX indikatörü
    if not pd.isna(row.get("ADX")):
        indicators["adx"] = {"period": int(row["ADX"])}
    
    # CCI indikatörü
    if not pd.isna(row.get("CCI")):
        indicators["cci"] = {"period": int(row["CCI"])}
    
    # SuperTrend indikatörü
    if not pd.isna(row.get("SUPER_TREND_period")) and not pd.isna(row.get("SUPER_TREND_multiplier")):
        indicators["supertrend"] = {
            "period": int(row["SUPER_TREND_period"]),
            "multiplier": float(row["SUPER_TREND_multiplier"])
        }
    
    # Bollinger Bands indikatörü
    if not pd.isna(row.get("BOLLINGER_length")) and not pd.isna(row.get("BOLLINGER_stddev")):
        indicators["bollinger"] = {
            "period": int(row["BOLLINGER_length"]),
            "std_dev": float(row["BOLLINGER_stddev"])
        }
    
    # Donchian Channel indikatörü
    if not pd.isna(row.get("DONCHIAN_period")):
        indicators["donchian"] = {
            "period": int(row["DONCHIAN_period"])
        }
    
    # Z-Score indikatörü
    if not pd.isna(row.get("Z_SCORE_length")):
        indicators["zscore"] = {
            "period": int(row["Z_SCORE_length"])
        }
    
    config["indicators"] = indicators
    
    # Standart stratejileri ekle
    config["strategies"] = {
        "trend_following": {},
        "oscillator_signals": {},
        "volatility_breakout": {}
    }
    
    # Standart strength hesaplayıcıları ekle
    config["strength"] = {
        "trend_indicators": {},
        "oscillator_levels": {},
        "volatility_measures": {}
    }
    
    # Standart filtreleri ekle
    config["filters"] = {
        "rsi_threshold": {},
        "macd_confirmation": {},
        "atr_volatility": {},
        "min_checks": 2,
        "min_strength": 3
    }
    
    return config