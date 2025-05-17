"""
Konfigürasyon yükleme modülü
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import sys

# Root dizinini Python yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mevcut env_loader'ı içe aktar
from env_loader import load_environment, get_config, get_indicator_config, get_position_direction

# Logger ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_env_config() -> Dict[str, Any]:
    """
    Çevre değişkenlerinden konfigürasyon yükler
    
    Returns:
        Konfigürasyon sözlüğü
    """
    # Önce mevcut çevre değişkenlerini yükle
    load_environment()
    
    # Global konfigürasyonu al
    global_config = get_config()

    #print(f" ✅ ✅ ✅  Global config: {global_config}")
    
    # Backtest parametreleri
    config = {
        "symbol": global_config.get("SYMBOL", "BTCUSDTXXX"),
        "interval": global_config.get("INTERVAL", "100m"),
        "initial_balance": float(global_config.get("ACCOUNT_BALANCE", 0.0)),
        "risk_per_trade": float(global_config.get("RISK_PER_TRADE", 0.99)),
        "sl_multiplier": float(global_config.get("SL_MULTIPLIER", 999.5)),
        "tp_multiplier": float(global_config.get("TP_MULTIPLIER", 999.0)),
        "leverage": float(global_config.get("LEVERAGE", 0.0)),
        "commission_rate": float(global_config.get("COMMISSION_RATE", 0.999)),
        "db_url": global_config.get("DB_URL", "postgresql://localhost/crypto"),
        "results_dir": global_config.get("RESULTS_DIR", "backtest/results")
    }
    
    # Pozisyon yönü yapılandırması
    try:
        config["position_direction"] = get_position_direction()
    except ValueError:
        logger.warning("POSITION_DIRECTION tanımlı değil, varsayılan değerler kullanılıyor.")
        config["position_direction"] = {"Long": True, "Short": True}
    
    # İndikatör yapılandırması
    try:
        config["indicators"] = {
            "long": get_indicator_config("Long"),
            "short": get_indicator_config("Short")
        }
    except ValueError as e:
        logger.warning(f"İndikatör yapılandırması yüklenemedi: {e}")
        config["indicators"] = {"long": {}, "short": {}}
    
    logger.info(f"✅ Konfigürasyon başarıyla yüklendi: {config['symbol']} {config['interval']}")
    return config


def load_config_csv(csv_path: str) -> pd.DataFrame:
    """
    CSV dosyasından konfigürasyon yükler
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        Konfigürasyon DataFrame'i
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ {len(df)} konfigürasyon {csv_path} dosyasından yüklendi")
        return df
    except Exception as e:
        logger.error(f"❌ Konfigürasyon CSV dosyası yüklenirken hata: {csv_path}: {e}")
        return pd.DataFrame()


def transform_config_row(row: pd.Series) -> Dict[str, Any]:
    """
    CSV konfigürasyon satırını Signal Engine formatına dönüştürür
    
    Args:
        row: CSV'den okunan konfigürasyon satırı
        
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


def convert_csv_row_to_config(row: pd.Series) -> Dict[str, Any]:
    """
    CSV satırını konfigürasyon sözlüğüne dönüştürür
    
    Args:
        row: CSV satırı
        
    Returns:
        Konfigürasyon sözlüğü
    """
    # Mevcut çevre değişkenlerini yükle
    load_environment()
    
    # Pozisyon yönünü al
    try:
        position_direction = get_position_direction()
    except ValueError:
        position_direction = {"Long": True, "Short": True}
    
    config = {"indicators": {"long": {}, "short": {}}}
    
    # İndikatör parametreleri
    if not pd.isna(row.get("EMA_FAST")) and not pd.isna(row.get("EMA_SLOW")):
        config["indicators"]["long"]["ema"] = {
            "fast_period": int(row["EMA_FAST"]),
            "slow_period": int(row["EMA_SLOW"])
        }
        config["indicators"]["short"]["ema"] = {
            "fast_period": int(row["EMA_FAST"]),
            "slow_period": int(row["EMA_SLOW"])
        }
    
    # RSI
    if not pd.isna(row.get("RSI")):
        config["indicators"]["long"]["rsi"] = {"period": int(row["RSI"])}
        config["indicators"]["short"]["rsi"] = {"period": int(row["RSI"])}
    
    # MACD
    if not pd.isna(row.get("MACD")) and row["MACD"]:
        config["indicators"]["long"]["macd"] = {}
        config["indicators"]["short"]["macd"] = {}
    
    # ATR
    if not pd.isna(row.get("ATR")):
        config["indicators"]["long"]["atr"] = {"period": int(row["ATR"])}
        config["indicators"]["short"]["atr"] = {"period": int(row["ATR"])}
    
    # OBV
    if not pd.isna(row.get("OBV")) and row["OBV"]:
        config["indicators"]["long"]["obv"] = {}
        config["indicators"]["short"]["obv"] = {}
    
    # ADX
    if not pd.isna(row.get("ADX")):
        config["indicators"]["long"]["adx"] = {"period": int(row["ADX"])}
        config["indicators"]["short"]["adx"] = {"period": int(row["ADX"])}
    
    # CCI
    if not pd.isna(row.get("CCI")):
        config["indicators"]["long"]["cci"] = {"period": int(row["CCI"])}
        config["indicators"]["short"]["cci"] = {"period": int(row["CCI"])}
    
    # SuperTrend
    if not pd.isna(row.get("SUPER_TREND_period")) and not pd.isna(row.get("SUPER_TREND_multiplier")):
        config["indicators"]["long"]["supertrend"] = {
            "period": int(row["SUPER_TREND_period"]),
            "multiplier": float(row["SUPER_TREND_multiplier"])
        }
        config["indicators"]["short"]["supertrend"] = {
            "period": int(row["SUPER_TREND_period"]),
            "multiplier": float(row["SUPER_TREND_multiplier"])
        }
    
    # Bollinger Bands
    if not pd.isna(row.get("BOLLINGER_length")) and not pd.isna(row.get("BOLLINGER_stddev")):
        config["indicators"]["long"]["bollinger"] = {
            "period": int(row["BOLLINGER_length"]),
            "std_dev": float(row["BOLLINGER_stddev"])
        }
        config["indicators"]["short"]["bollinger"] = {
            "period": int(row["BOLLINGER_length"]),
            "std_dev": float(row["BOLLINGER_stddev"])
        }
    
    # Donchian Channel
    if not pd.isna(row.get("DONCHIAN_period")):
        config["indicators"]["long"]["donchian"] = {
            "period": int(row["DONCHIAN_period"])
        }
        config["indicators"]["short"]["donchian"] = {
            "period": int(row["DONCHIAN_period"])
        }
    
    # Z-Score
    if not pd.isna(row.get("Z_SCORE_length")):
        config["indicators"]["long"]["zscore"] = {
            "period": int(row["Z_SCORE_length"])
        }
        config["indicators"]["short"]["zscore"] = {
            "period": int(row["Z_SCORE_length"])
        }
    
    # Pozisyon yönünü ekle
    config["position_direction"] = position_direction
    
    return config


if __name__ == "__main__":
    # Test
    config = load_env_config()
    print(json.dumps(config, indent=2))