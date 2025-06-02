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
from env_loader import (
    load_environment, 
    get_config, 
    get_indicator_config, 
    get_position_direction,
    get_strategies_config,
    get_filter_config,
    get_strength_config
)

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
        "results_dir": global_config.get("RESULTS_DIR", "backtest/results"),
        "max_holding_bars": int(global_config.get("MAX_HOLDING_BARS", 500))  # YENİ
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
    
    # Strateji yapılandırması
    try:
        config["strategies"] = get_strategies_config()
    except ValueError as e:
        logger.warning(f"Strateji yapılandırması yüklenemedi: {e}")
        config["strategies"] = {}
    
    # Filtre yapılandırması
    try:
        config["filters"] = get_filter_config()
    except ValueError as e:
        logger.warning(f"Filtre yapılandırması yüklenemedi: {e}")
        config["filters"] = {}
    
    # Güç hesaplayıcı yapılandırması
    try:
        config["strength"] = get_strength_config()
    except ValueError as e:
        logger.warning(f"Güç hesaplayıcı yapılandırması yüklenemedi: {e}")
        config["strength"] = {}
    
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
    CSV konfigürasyon satırını Signal Engine formatına dönüştürür.
    Detaylı parametre kombinasyonlarını destekler.
    
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
    
    # Temel İndikatörler
    
    # EMA indikatörleri
    if not pd.isna(row.get("EMA_ENABLE")) and bool(row["EMA_ENABLE"]):
        periods = []
        
        # Dinamik olarak EMA periyotlarını ekle
        for i in range(1, 6):  # En fazla 5 periyot destekleyelim
            period_key = f"EMA_PERIOD_{i}"
            if not pd.isna(row.get(period_key)):
                periods.append(int(row[period_key]))
        
        # Eski formatı da destekle (EMA_FAST, EMA_SLOW)
        if not periods and not pd.isna(row.get("EMA_FAST")) and not pd.isna(row.get("EMA_SLOW")):
            periods = [int(row["EMA_FAST"]), int(row["EMA_SLOW"])]
        
        # En az bir periyod varsa ekle
        if periods:
            indicators["ema"] = {"periods": periods}
    
    # SMA indikatörü
    if not pd.isna(row.get("SMA_ENABLE")) and bool(row["SMA_ENABLE"]):
        periods = []
        
        # Dinamik olarak SMA periyotlarını ekle
        for i in range(1, 6):  # En fazla 5 periyot destekleyelim
            period_key = f"SMA_PERIOD_{i}"
            if not pd.isna(row.get(period_key)):
                periods.append(int(row[period_key]))
        
        # Eski formatı da destekle (SMA_PERIOD)
        if not periods and not pd.isna(row.get("SMA_PERIOD")):
            periods = [int(row["SMA_PERIOD"])]
        
        # En az bir periyod varsa ekle
        if periods:
            indicators["sma"] = {"periods": periods}
    
    # RSI indikatörü
    if not pd.isna(row.get("RSI_ENABLE")) and bool(row["RSI_ENABLE"]):
        periods = []
        
        # Dinamik olarak RSI periyotlarını ekle
        for i in range(1, 4):  # En fazla 3 periyot destekleyelim
            period_key = f"RSI_PERIOD_{i}"
            if not pd.isna(row.get(period_key)):
                periods.append(int(row[period_key]))
        
        # Eski formatı da destekle (RSI)
        if not periods and not pd.isna(row.get("RSI")):
            periods = [int(row["RSI"])]
        
        # En az bir periyod varsa ekle
        if periods:
            indicators["rsi"] = {"periods": periods}
    
    # MACD indikatörü
    if not pd.isna(row.get("MACD_ENABLE")) and bool(row["MACD_ENABLE"]):
        macd_params = {}
        
        # Dinamik MACD parametreleri
        if not pd.isna(row.get("MACD_FAST_PERIOD")):
            macd_params["fast_period"] = int(row["MACD_FAST_PERIOD"])
        
        if not pd.isna(row.get("MACD_SLOW_PERIOD")):
            macd_params["slow_period"] = int(row["MACD_SLOW_PERIOD"])
        
        if not pd.isna(row.get("MACD_SIGNAL_PERIOD")):
            macd_params["signal_period"] = int(row["MACD_SIGNAL_PERIOD"])
        
        # Boş ise varsayılan değerleri kullan, aksi halde tüm parametreleri içeren sözlük kullan
        indicators["macd"] = macd_params if macd_params else {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
    
    # ATR indikatörü
    if not pd.isna(row.get("ATR_ENABLE")) and bool(row["ATR_ENABLE"]):
        atr_params = {}
        
        if not pd.isna(row.get("ATR_PERIOD")):
            atr_params["window"] = int(row["ATR_PERIOD"])
        
        # Eski formatı da destekle (ATR)
        elif not pd.isna(row.get("ATR")):
            atr_params["window"] = int(row["ATR"])
        
        indicators["atr"] = atr_params if atr_params else {"window": 14}
    
    # Bollinger Bands indikatörü
    if not pd.isna(row.get("BOLLINGER_ENABLE")) and bool(row["BOLLINGER_ENABLE"]):
        bb_params = {}
        
        if not pd.isna(row.get("BOLLINGER_PERIOD")):
            bb_params["window"] = int(row["BOLLINGER_PERIOD"])
        elif not pd.isna(row.get("BOLLINGER_length")):
            bb_params["window"] = int(row["BOLLINGER_length"])
        
        if not pd.isna(row.get("BOLLINGER_STDDEV")):
            bb_params["window_dev"] = float(row["BOLLINGER_STDDEV"])
        elif not pd.isna(row.get("BOLLINGER_stddev")):
            bb_params["window_dev"] = float(row["BOLLINGER_stddev"])
        
        indicators["bollinger"] = bb_params if bb_params else {
            "window": 20, 
            "window_dev": 2.0
        }
    
    # Stochastic indikatörü
    if not pd.isna(row.get("STOCHASTIC_ENABLE")) and bool(row["STOCHASTIC_ENABLE"]):
        stoch_params = {}
        
        if not pd.isna(row.get("STOCHASTIC_K")):
            stoch_params["window"] = int(row["STOCHASTIC_K"])
        
        if not pd.isna(row.get("STOCHASTIC_D")):
            stoch_params["d_window"] = int(row["STOCHASTIC_D"])
        
        if not pd.isna(row.get("STOCHASTIC_SMOOTH")):
            stoch_params["smooth_window"] = int(row["STOCHASTIC_SMOOTH"])
        
        indicators["stochastic"] = stoch_params if stoch_params else {
            "window": 14,
            "smooth_window": 3,
            "d_window": 3
        }
    
    # Gelişmiş İndikatörler
    
    # SuperTrend indikatörü
    if not pd.isna(row.get("SUPERTREND_ENABLE")) and bool(row["SUPERTREND_ENABLE"]):
        st_params = {}
        
        if not pd.isna(row.get("SUPERTREND_PERIOD")):
            st_params["atr_period"] = int(row["SUPERTREND_PERIOD"])
        elif not pd.isna(row.get("SUPER_TREND_period")):
            st_params["atr_period"] = int(row["SUPER_TREND_period"])
        
        if not pd.isna(row.get("SUPERTREND_MULTIPLIER")):
            st_params["atr_multiplier"] = float(row["SUPERTREND_MULTIPLIER"])
        elif not pd.isna(row.get("SUPER_TREND_multiplier")):
            st_params["atr_multiplier"] = float(row["SUPER_TREND_multiplier"])
        
        indicators["supertrend"] = st_params if st_params else {
            "atr_period": 10,
            "atr_multiplier": 3.0
        }
    
    # Ichimoku indikatörü
    if not pd.isna(row.get("ICHIMOKU_ENABLE")) and bool(row["ICHIMOKU_ENABLE"]):
        ichi_params = {}
        
        if not pd.isna(row.get("ICHIMOKU_TENKAN")):
            ichi_params["tenkan_period"] = int(row["ICHIMOKU_TENKAN"])
        
        if not pd.isna(row.get("ICHIMOKU_KIJUN")):
            ichi_params["kijun_period"] = int(row["ICHIMOKU_KIJUN"])
        
        if not pd.isna(row.get("ICHIMOKU_SENKOU_B")):
            ichi_params["senkou_b_period"] = int(row["ICHIMOKU_SENKOU_B"])
        
        if not pd.isna(row.get("ICHIMOKU_DISPLACEMENT")):
            ichi_params["displacement"] = int(row["ICHIMOKU_DISPLACEMENT"])
        
        indicators["ichimoku"] = ichi_params if ichi_params else {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "displacement": 26
        }
    
    # Adaptive RSI indikatörü
    if not pd.isna(row.get("ADAPTIVE_RSI_ENABLE")) and bool(row["ADAPTIVE_RSI_ENABLE"]):
        adaptive_rsi_params = {}
        
        if not pd.isna(row.get("ADAPTIVE_RSI_BASE_PERIOD")):
            adaptive_rsi_params["base_period"] = int(row["ADAPTIVE_RSI_BASE_PERIOD"])
        
        if not pd.isna(row.get("ADAPTIVE_RSI_VOLATILITY_WINDOW")):
            adaptive_rsi_params["volatility_window"] = int(row["ADAPTIVE_RSI_VOLATILITY_WINDOW"])
        
        if not pd.isna(row.get("ADAPTIVE_RSI_MIN_PERIOD")):
            adaptive_rsi_params["min_period"] = int(row["ADAPTIVE_RSI_MIN_PERIOD"])
        
        if not pd.isna(row.get("ADAPTIVE_RSI_MAX_PERIOD")):
            adaptive_rsi_params["max_period"] = int(row["ADAPTIVE_RSI_MAX_PERIOD"])
        
        indicators["adaptive_rsi"] = adaptive_rsi_params if adaptive_rsi_params else {
            "base_period": 14,
            "volatility_window": 100,
            "min_period": 5,
            "max_period": 30
        }
    
    # Z-Score indikatörü
    if not pd.isna(row.get("ZSCORE_ENABLE")) and bool(row["ZSCORE_ENABLE"]):
        zscore_params = {}
        
        if not pd.isna(row.get("ZSCORE_WINDOW")):
            zscore_params["window"] = int(row["ZSCORE_WINDOW"])
        elif not pd.isna(row.get("Z_SCORE_length")):
            zscore_params["window"] = int(row["Z_SCORE_length"])
        
        # Apply_to kolonları
        apply_to = []
        for i in range(1, 6):  # En fazla 5 kolon destekleyelim
            apply_key = f"ZSCORE_APPLY_TO_{i}"
            if not pd.isna(row.get(apply_key)):
                apply_to.append(row[apply_key])
        
        if apply_to:
            zscore_params["apply_to"] = apply_to
        
        indicators["zscore"] = zscore_params if zscore_params else {
            "window": 100,
            "apply_to": ["close", "rsi_14", "macd_line"]
        }
    
    # Rejim indikatörleri
    
    # Market Regime indikatörü
    if not pd.isna(row.get("MARKET_REGIME_ENABLE")) and bool(row["MARKET_REGIME_ENABLE"]):
        mr_params = {}
        
        if not pd.isna(row.get("MARKET_REGIME_LOOKBACK")):
            mr_params["lookback_window"] = int(row["MARKET_REGIME_LOOKBACK"])
        
        if not pd.isna(row.get("MARKET_REGIME_ADX_THRESHOLD")):
            mr_params["adx_threshold"] = int(row["MARKET_REGIME_ADX_THRESHOLD"])
        
        indicators["market_regime"] = mr_params if mr_params else {
            "lookback_window": 50,
            "adx_threshold": 25
        }
    
    # Diğer indikatörler için benzer yaklaşım uygulanabilir...
    
    config["indicators"] = indicators
    
    # Stratejileri ekle
    strategies = {}
    
    # Trend Following stratejisi
    if not pd.isna(row.get("TREND_FOLLOWING_ENABLE")) and bool(row["TREND_FOLLOWING_ENABLE"]):
        tf_params = {}
        
        if not pd.isna(row.get("TREND_FOLLOWING_ADX_THRESHOLD")):
            tf_params["adx_threshold"] = int(row["TREND_FOLLOWING_ADX_THRESHOLD"])
        
        if not pd.isna(row.get("TREND_FOLLOWING_RSI_THRESHOLD")):
            tf_params["rsi_threshold"] = int(row["TREND_FOLLOWING_RSI_THRESHOLD"])
        
        if not pd.isna(row.get("TREND_FOLLOWING_CONFIRMATION_COUNT")):
            tf_params["confirmation_count"] = int(row["TREND_FOLLOWING_CONFIRMATION_COUNT"])
        
        strategies["trend_following"] = tf_params if tf_params else {
            "adx_threshold": 25,
            "rsi_threshold": 50,
            "macd_threshold": 0,
            "confirmation_count": 3
        }
    
    # MTF Trend stratejisi
    if not pd.isna(row.get("MTF_TREND_ENABLE")) and bool(row["MTF_TREND_ENABLE"]):
        mtf_params = {}
        
        if not pd.isna(row.get("MTF_TREND_ALIGNMENT")):
            mtf_params["alignment_required"] = float(row["MTF_TREND_ALIGNMENT"])
        
        strategies["mtf_trend"] = mtf_params if mtf_params else {
            "alignment_required": 0.8
        }
    
    # Adaptive Trend stratejisi
    if not pd.isna(row.get("ADAPTIVE_TREND_ENABLE")) and bool(row["ADAPTIVE_TREND_ENABLE"]):
        at_params = {}
        
        if not pd.isna(row.get("ADAPTIVE_TREND_MAX_THRESHOLD")):
            at_params["adx_max_threshold"] = int(row["ADAPTIVE_TREND_MAX_THRESHOLD"])
        
        if not pd.isna(row.get("ADAPTIVE_TREND_MIN_THRESHOLD")):
            at_params["adx_min_threshold"] = int(row["ADAPTIVE_TREND_MIN_THRESHOLD"])
        
        strategies["adaptive_trend"] = at_params if at_params else {
            "adx_max_threshold": 40,
            "adx_min_threshold": 15
        }
    
    # Diğer stratejiler için benzer yaklaşım...
    
    if strategies:
        config["strategies"] = strategies
    else:
        # Varsayılan stratejiler
        config["strategies"] = {
            "trend_following": {},
            "mtf_trend": {},
            "adaptive_trend": {}
        }
    
    # Filtreler
    filters = {}
    
    # Market Regime filtresi
    if not pd.isna(row.get("MARKET_REGIME_FILTER_ENABLE")) and bool(row["MARKET_REGIME_FILTER_ENABLE"]):
        filters["market_regime"] = {}
    
    # Volatility Regime filtresi
    if not pd.isna(row.get("VOLATILITY_REGIME_FILTER_ENABLE")) and bool(row["VOLATILITY_REGIME_FILTER_ENABLE"]):
        filters["volatility_regime"] = {}
    
    # Dynamic Threshold filtresi
    if not pd.isna(row.get("DYNAMIC_THRESHOLD_FILTER_ENABLE")) and bool(row["DYNAMIC_THRESHOLD_FILTER_ENABLE"]):
        dt_params = {}
        
        if not pd.isna(row.get("DYNAMIC_THRESHOLD_BASE")):
            dt_params["base_threshold"] = float(row["DYNAMIC_THRESHOLD_BASE"])
        
        if not pd.isna(row.get("DYNAMIC_THRESHOLD_VOL_IMPACT")):
            dt_params["volatility_impact"] = float(row["DYNAMIC_THRESHOLD_VOL_IMPACT"])
        
        if not pd.isna(row.get("DYNAMIC_THRESHOLD_TREND_IMPACT")):
            dt_params["trend_impact"] = float(row["DYNAMIC_THRESHOLD_TREND_IMPACT"])
        
        filters["dynamic_threshold_filter"] = dt_params if dt_params else {
            "base_threshold": 0.6,
            "volatility_impact": 0.2,
            "trend_impact": 0.2
        }
    
    # FilterManager parametreleri
    if not pd.isna(row.get("MIN_CHECKS")):
        filters["min_checks"] = int(row["MIN_CHECKS"])
    else:
        filters["min_checks"] = 2
    
    if not pd.isna(row.get("MIN_STRENGTH")):
        filters["min_strength"] = int(row["MIN_STRENGTH"])
    else:
        filters["min_strength"] = 3
    
    if filters:
        config["filters"] = filters
    else:
        # Varsayılan filtreler
        config["filters"] = {
            "market_regime": {},
            "dynamic_threshold_filter": {},
            "min_checks": 2,
            "min_strength": 3
        }
    
    # Güç hesaplayıcılar
    strength = {}
    
    # Market Context Strength
    if not pd.isna(row.get("MARKET_CONTEXT_STRENGTH_ENABLE")) and bool(row["MARKET_CONTEXT_STRENGTH_ENABLE"]):
        strength["market_context_strength"] = {
            "volatility_adjustment": True,
            "trend_health_adjustment": True
        }
    
    # Risk Reward Strength
    if not pd.isna(row.get("RISK_REWARD_STRENGTH_ENABLE")) and bool(row["RISK_REWARD_STRENGTH_ENABLE"]):
        rr_params = {}
        
        if not pd.isna(row.get("RISK_REWARD_RATIO")):
            rr_params["min_reward_risk_ratio"] = float(row["RISK_REWARD_RATIO"])
        
        strength["risk_reward_strength"] = rr_params if rr_params else {
            "risk_factor": 1.0,
            "reward_factor": 1.0,
            "min_reward_risk_ratio": 1.5
        }
    
    if strength:
        config["strength"] = strength
    else:
        # Varsayılan güç hesaplayıcılar
        config["strength"] = {
            "market_context_strength": {},
            "risk_reward_strength": {}
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
    # transform_config_row fonksiyonunu kullanarak önce Signal Engine formatına dönüştür
    config = transform_config_row(row)
    
    # Pozisyon yönünü ekle
    try:
        config["position_direction"] = get_position_direction()
    except ValueError:
        config["position_direction"] = {"Long": True, "Short": True}
    
    return config


if __name__ == "__main__":
    # Test
    config = load_env_config()
    print(json.dumps(config, indent=2))