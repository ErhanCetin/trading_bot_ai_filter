# backtest/utils/indicator_helper.py dosyası (düzeltilmiş)

import json
import logging
from typing import Dict, Any, List

def check_available_indicators():
    """
    Signal Engine'de mevcut indikatörleri kontrol eder
    """
    from signal_engine.indicators import registry
    
    available_indicators = {}
    
    print("\nAVAILABLE INDICATORS IN SIGNAL ENGINE:")
    print("=" * 80)
    
    # IndicatorRegistry sınıfının get_all_indicators() metodunu kullan
    indicators_dict = registry.get_all_indicators()
    
    for name, indicator_class in indicators_dict.items():
        print(f"- {name}: {getattr(indicator_class, 'description', 'No description')}")
        available_indicators[name] = indicator_class
    
    return available_indicators

def create_indicators_config(indicators_list: List[str], with_params: bool = True) -> str:
    """
    Verilen indikatör listesi için yapılandırma JSON string'i oluşturur
    
    Args:
        indicators_list: Kullanılacak indikatör isimleri listesi
        with_params: İndikatörler için varsayılan parametreleri ekle
        
    Returns:
        JSON formatında indikatör yapılandırması
    """
    from signal_engine.indicators import registry
    
    config = {}
    indicators_dict = registry.get_all_indicators()
    
    for name in indicators_list:
        if name in indicators_dict:
            if with_params:
                # İndikatör sınıfından varsayılan parametreleri al
                indicator_class = indicators_dict[name]
                try:
                    # Parametreleri doğrudan sınıftan al veya bir instance oluşturarak al
                    if hasattr(indicator_class, 'default_params'):
                        params = indicator_class.default_params
                    else:
                        # Varsayılan parametreler belirtilmemişse boş bir instance oluştur
                        params = indicator_class().params
                except Exception as e:
                    logging.debug(f"Could not get params for {name}: {e}")
                    # Hata durumunda boş parametre
                    params = {}
                
                config[name] = params
            else:
                # Parametresiz indikatör
                config[name] = {}
        else:
            logging.warning(f"Indicator '{name}' not found in registry, skipping.")
    
    return json.dumps(config)

def get_recommended_config():
    """
    Desteklenen indikatörlere dayalı tavsiye edilen yapılandırmayı döndürür
    """
    # Kapsamlı indikatör listesi
    recommended = [
        # Trend indikatörleri
        "ema", "sma", "wma", "dema", "tema", "trix", "vwap", "mtf_ema", "kama", "ichimoku",
        "supertrend", "ttm_trend", "hma", "alma", "vidya", "zlema", "arnaud_legoux_ma",
        
        # Momentum indikatörleri
        "rsi", "stoch", "stoch_rsi", "macd", "ppo", "cci", "adx", "dmi", "adaptive_rsi",
        "awesome_oscillator", "mfi", "williams_r", "roc", "tsi", "ultimate_oscillator", "cmo",
        
        # Volatilite indikatörleri
        "atr", "bollinger", "keltner", "donchian", "zscore", "volatility_regime", "atr_percent",
        "chandelier_exit", "price_channel", "true_range", "atr_bands", "stdev", "parabolic_sar",
        
        # Hacim indikatörleri
        "obv", "volume_profile", "cmf", "vwma", "mfi", "adl", "volume_oscillator",
        "vwap", "pvt", "ease_of_movement", "force_index", "vpt", "klinger",
        
        # Diğer/Özel indikatörler
        "market_regime", "trend_strength", "cycle_finder", "fibonacci_retracement",
        "divergence_detector", "pivot_points", "support_resistance", "renko", "waves",
        "heiken_ashi", "gann_angles", "fractals", "elder_ray", "demarker", "camarilla",
        
        # Ek indikatörler (Error mesajında belirtilenler)
        "mtf_ema", "adaptive_rsi", "market_regime", "volatility_regime", "trend_strength"
    ]
    
    # Registry'de var olan indikatörleri kontrol et
    from signal_engine.indicators import registry
    indicators_dict = registry.get_all_indicators()
    available = [name for name in recommended if name in indicators_dict]
    
    # İndikatör bulunamazsa
    if not available:
        # Mevcut tüm indikatörleri kullan
        available = list(indicators_dict.keys())
        logging.warning(f"No recommended indicators found in registry. Using all {len(available)} available indicators.")
    
    # Yapılandırma oluştur
    config = create_indicators_config(available)
    
    return {
        "long": json.loads(config),
        "short": json.loads(config),
        "recommended_env": f'INDICATORS_LONG=\'{config}\'\nINDICATORS_SHORT=\'{config}\''
    }