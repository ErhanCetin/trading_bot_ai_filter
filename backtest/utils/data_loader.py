"""
Veri yükleme ve hazırlama işlevleri
ENHANCED: Added comprehensive data validation and preprocessing
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import text, create_engine
import logging

logger = logging.getLogger(__name__)

def load_price_data(symbol: str, interval: str, db_url: str, validate_data: bool = True, min_rows: int = 100) -> pd.DataFrame:
    """
    ENHANCED: Veritabanından fiyat verilerini yükler with comprehensive validation
    
    Args:
        symbol: İşlem sembolü (örn. "BTCUSDT")
        interval: Zaman aralığı (örn. "1m", "5m", "1h")
        db_url: Veritabanı bağlantı URL'si
        validate_data: Whether to validate data quality
        min_rows: Minimum required rows
        
    Returns:
        Fiyat verilerini içeren DataFrame
    """
    print(f"📊 Loading price data for {symbol} at {interval} interval from {db_url}")
    
    try:
        engine = create_engine(db_url)
        
        # ENHANCED: Query with data quality checks
        query = f"""
        SELECT * FROM kline_data
        WHERE symbol = '{symbol}' 
        AND interval = '{interval}'
        AND open > 0 AND high > 0 AND low > 0 AND close > 0
        AND high >= low AND high >= open AND high >= close
        AND low <= open AND low <= close
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
            engine.dispose()
            return _suggest_alternatives(symbol, interval, db_url)
        
        print(f"🔍 DATA DEBUG: Data loaded successfully:")
        print(f"  - First row: {df.iloc[0].to_dict()}")
        print(f"  - Last row: {df.iloc[-1].to_dict()}")
        
        # ENHANCED: Data validation and preprocessing
        if validate_data:
            df = _validate_and_clean_data(df, symbol, interval, min_rows)
            if df.empty:
                engine.dispose()
                return df
        
        # ENHANCED: Preprocessing
        df = _preprocess_price_data(df)
        
        print(f"✅ Loaded {len(df)} validated records")
        engine.dispose()
        return df
        
    except Exception as e:
        print(f"❌ DATA LOADER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Hata durumunda boş DataFrame

def _suggest_alternatives(symbol: str, interval: str, db_url: str) -> pd.DataFrame:
    """
    ENHANCED: Suggest alternative symbols or intervals when data not found
    """
    try:
        engine = create_engine(db_url)
        
        # Find similar symbols
        similar_query = f"""
        SELECT DISTINCT symbol FROM kline_data 
        WHERE symbol LIKE '%{symbol[:4]}%' 
        LIMIT 10
        """
        similar_df = pd.read_sql(text(similar_query), engine)
        
        # Find available intervals
        interval_query = "SELECT DISTINCT interval FROM kline_data ORDER BY interval"
        interval_df = pd.read_sql(text(interval_query), engine)
        
        print(f"💡 SUGGESTIONS:")
        print(f"   Similar symbols: {similar_df['symbol'].tolist()}")
        print(f"   Available intervals: {interval_df['interval'].tolist()}")
        
        # Check total records in database
        total_query = "SELECT COUNT(*) as count FROM kline_data"
        total_df = pd.read_sql(text(total_query), engine)
        print(f"   Total records in database: {total_df.iloc[0]['count']}")
        
        engine.dispose()
        return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Error finding alternatives: {e}")
        return pd.DataFrame()

def _validate_and_clean_data(df: pd.DataFrame, symbol: str, interval: str, min_rows: int) -> pd.DataFrame:
    """
    ENHANCED: Validate and clean price data
    
    Args:
        df: Raw DataFrame
        symbol: Symbol name
        interval: Time interval
        min_rows: Minimum required rows
        
    Returns:
        Cleaned DataFrame
    """
    print(f"🔍 Validating data quality...")
    
    initial_rows = len(df)
    
    # 1. Remove rows with invalid prices
    df = df[
        (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) &
        (df['high'] >= df['low']) & 
        (df['high'] >= df['open']) & (df['high'] >= df['close']) &
        (df['low'] <= df['open']) & (df['low'] <= df['close'])
    ]
    
    # 2. Remove extreme outliers (price spikes)
    for col in ['open', 'high', 'low', 'close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # More conservative than 1.5
        upper_bound = Q3 + 3 * IQR
        
        outliers_before = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        outliers_removed = outliers_before - len(df)
        
        if outliers_removed > 0:
            print(f"   Removed {outliers_removed} outliers from {col}")
    
    # 3. Check for gaps in time series
    if 'open_time' in df.columns:
        df['open_time_dt'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.sort_values('open_time_dt')
        
        # Detect time gaps
        time_diffs = df['open_time_dt'].diff()
        expected_interval = _get_expected_interval(interval)
        
        if expected_interval:
            large_gaps = time_diffs[time_diffs > expected_interval * 2]
            if len(large_gaps) > 0:
                print(f"   ⚠️ Found {len(large_gaps)} time gaps larger than expected")
    
    # 4. Remove duplicate timestamps
    if 'open_time' in df.columns:
        duplicates = df.duplicated(subset=['open_time'])
        if duplicates.sum() > 0:
            print(f"   Removed {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
    
    # 5. Volume validation
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > len(df) * 0.1:  # More than 10% zero volume
            print(f"   ⚠️ Warning: {zero_volume} rows with zero volume ({zero_volume/len(df)*100:.1f}%)")
    
    # 6. Check minimum data requirement
    if len(df) < min_rows:
        print(f"   ❌ Insufficient data: {len(df)} rows (required: {min_rows})")
        return pd.DataFrame()
    
    cleaned_rows = len(df)
    removed_rows = initial_rows - cleaned_rows
    
    if removed_rows > 0:
        print(f"   📋 Data cleaning: {removed_rows} rows removed ({removed_rows/initial_rows*100:.1f}%)")
    
    print(f"   ✅ Data validation complete: {cleaned_rows} clean rows")
    return df

def _get_expected_interval(interval: str) -> pd.Timedelta:
    """
    ENHANCED: Get expected time interval between candles
    """
    interval_map = {
        '1m': pd.Timedelta(minutes=1),
        '5m': pd.Timedelta(minutes=5),
        '15m': pd.Timedelta(minutes=15),
        '30m': pd.Timedelta(minutes=30),
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        '1d': pd.Timedelta(days=1),
        '1w': pd.Timedelta(weeks=1)
    }
    return interval_map.get(interval)

def _preprocess_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENHANCED: preprocessing of price data
    """
    print(f"🔧 Preprocessing data...")
    
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
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"   Handling {missing_before} missing values...")
        
        # Forward fill price data
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Fill remaining missing values
        df = df.fillna(0)
    
    # Zaman sütununu standartlaştır
    if "open_time" in df.columns:
        df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms")
    else:
        print("⚠️ open_time column missing!")
    
    # ENHANCED: Add basic derived columns
    if all(col in df.columns for col in ["high", "low", "close"]):
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Price range
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close'] * 100
        
        # Body and wick sizes
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Add time-based features
    if 'open_time_dt' in df.columns:
        df['hour'] = df['open_time_dt'].dt.hour
        df['day_of_week'] = df['open_time_dt'].dt.dayofweek
        df['month'] = df['open_time_dt'].dt.month
    
    # Sort by time
    if 'open_time' in df.columns:
        df = df.sort_values('open_time').reset_index(drop=True)
    
    print(f"🔍 DATA DEBUG: Final DataFrame:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Index: {df.index}")
    
    print(f"   ✅ Preprocessing complete")
    return df

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    ENHANCED: Comprehensive data quality assessment
    
    Args:
        df: Price data DataFrame
        
    Returns:
        Data quality report
    """
    if df.empty:
        return {"status": "error", "message": "Empty DataFrame"}
    
    quality_report = {
        "status": "success",
        "total_rows": len(df),
        "date_range": {},
        "price_analysis": {},
        "volume_analysis": {},
        "issues": []
    }
    
    # Date range analysis
    if 'open_time_dt' in df.columns:
        quality_report["date_range"] = {
            "start": df['open_time_dt'].min().strftime('%Y-%m-%d %H:%M'),
            "end": df['open_time_dt'].max().strftime('%Y-%m-%d %H:%M'),
            "duration_days": (df['open_time_dt'].max() - df['open_time_dt'].min()).days
        }
    
    # Price analysis
    price_cols = ['open', 'high', 'low', 'close']
    available_price_cols = [col for col in price_cols if col in df.columns]
    
    if available_price_cols:
        price_stats = df[available_price_cols].describe()
        quality_report["price_analysis"] = {
            "mean_price": float(df['close'].mean()) if 'close' in df.columns else 0,
            "price_volatility": float(df['close'].std() / df['close'].mean() * 100) if 'close' in df.columns else 0,
            "price_range": {
                "min": float(df[available_price_cols].min().min()),
                "max": float(df[available_price_cols].max().max())
            }
        }
    
    # Volume analysis
    if 'volume' in df.columns:
        quality_report["volume_analysis"] = {
            "avg_volume": float(df['volume'].mean()),
            "zero_volume_pct": float((df['volume'] == 0).sum() / len(df) * 100),
            "volume_consistency": float(1 - df['volume'].std() / df['volume'].mean()) if df['volume'].mean() > 0 else 0
        }
    
    # Data quality issues
    issues = []
    
    # Check for gaps
    if 'open_time_dt' in df.columns and len(df) > 1:
        time_diffs = df['open_time_dt'].diff().dropna()
        median_diff = time_diffs.median()
        large_gaps = (time_diffs > median_diff * 2).sum()
        
        if large_gaps > 0:
            issues.append(f"Found {large_gaps} time gaps in data")
    
    # Check for extreme values
    for col in available_price_cols:
        col_data = df[col]
        q99 = col_data.quantile(0.99)
        q01 = col_data.quantile(0.01)
        extreme_values = ((col_data > q99 * 2) | (col_data < q01 * 0.5)).sum()
        
        if extreme_values > 0:
            issues.append(f"Found {extreme_values} extreme values in {col}")
    
    quality_report["issues"] = issues
    quality_report["quality_score"] = _calculate_quality_score(quality_report)
    
    return quality_report

def _calculate_quality_score(quality_report: Dict[str, Any]) -> float:
    """
    ENHANCED: Calculate data quality score (0-10)
    """
    score = 10.0
    
    # Penalize for issues
    score -= len(quality_report.get("issues", [])) * 0.5
    
    # Volume quality
    volume_analysis = quality_report.get("volume_analysis", {})
    zero_volume_pct = volume_analysis.get("zero_volume_pct", 0)
    if zero_volume_pct > 10:
        score -= 2
    elif zero_volume_pct > 5:
        score -= 1
    
    # Price volatility check
    price_analysis = quality_report.get("price_analysis", {})
    volatility = price_analysis.get("price_volatility", 0)
    if volatility > 50:  # Very high volatility might indicate data issues
        score -= 1
    
    # Data completeness
    total_rows = quality_report.get("total_rows", 0)
    if total_rows < 100:
        score -= 3
    elif total_rows < 500:
        score -= 1
    
    return max(0, min(10, score))

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