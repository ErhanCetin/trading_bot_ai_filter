"""
Ana çalıştırıcı modül - VSCode'dan doğrudan çalıştırılabilir
"""
import os
import sys
import json
from typing import Dict, List, Any, Optional
import logging

# Modül yolunu ayarla - mevcut dizinin bir üst dizinini Python yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backtest modüllerini içe aktar
from backtest.utils.config_loader import load_env_config
from backtest.runners.single_backtest import run_single_backtest
from backtest.runners.batch_backtest import run_batch_backtest
from backtest.utils.config_viewer import print_config_details
from backtest.utils.signal_engine_components import check_signal_engine_components


def print_registered_indicators():
    """
    Kayıtlı indikatörleri gösterir
    """
    from backtest.utils.indicator_helper import check_available_indicators, get_recommended_config

    # Kullanılabilir indikatörleri göster
    available = check_available_indicators()
    print(f"Found {len(available)} indicators in registry.\n")

    # Tavsiye edilen yapılandırmayı al
    recommended = get_recommended_config()

    print("\nRECOMMENDED CONFIGURATION:")
    print("=" * 80)
    print(recommended["recommended_env"])
    print("=" * 80)
    print("\nThis configuration includes only the indicators available in the Signal Engine registry.")


def run_backtest(mode: str = "single", config_id: str = "default", custom_config: Dict[str, Any] = None):
    """
    Backtest çalıştırır - VSCode'dan direkt çağrılabilir
    
    Args:
        mode: Çalıştırma modu ("single" veya "batch")
        config_id: Konfigürasyon ID'si (single mod için)
        custom_config: Özel konfigürasyon sözlüğü (varsayılanları ezmek için)
    """
    # Çevre değişkenlerinden konfigürasyon yükle
    env_config = load_env_config()
    
    # Özel konfigürasyonu entegre et (varsa)
    if custom_config:
        for key, value in custom_config.items():
            env_config[key] = value
    
    # Çıktı dizini oluştur
    output_dir = env_config.get("results_dir", "backtest/results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Gerekli parametreleri kontrol et
    symbol = env_config.get("symbol")
    interval = env_config.get("interval")
    
    if not symbol or not interval:
        logger.error("Symbol ve interval parametreleri gerekli. Lütfen ENV dosyasını kontrol edin.")
        return
    
    if mode == "single":
        logger.info(f"🚀 {symbol} {interval} için tek backtest çalıştırılıyor (Config ID: {config_id})")
        
        # Backtest parametrelerini hazırla
        backtest_params = {
            "initial_balance": float(env_config.get("initial_balance", 10000.0)),
            "risk_per_trade": float(env_config.get("risk_per_trade", 0.01)),
            "sl_multiplier": float(env_config.get("sl_multiplier", 1.5)),
            "tp_multiplier": float(env_config.get("tp_multiplier", 3.0)),
            "leverage": float(env_config.get("leverage", 1.0)),
            "position_direction": env_config.get("position_direction", {"Long": True, "Short": True}),
            "commission_rate": float(env_config.get("commission_rate", 0.001)),
        }
        
        # Tek backtest çalıştır
        result = run_single_backtest(
            symbol=symbol,
            interval=interval,
            db_url=env_config.get("db_url"),
            output_dir=os.path.join(output_dir, "single"),
            backtest_params=backtest_params,
            # Tüm signal engine yapılandırmalarını (indikatörler, stratejiler, filtreler, güç hesaplayıcılar) aktarmaya gerek yok
            # çünkü güncellenmiş single_backtest.py zaten bunları çevre değişkenlerinden alıyor
            config_id=config_id
        )
        
        if result.get("status") == "success":
            logger.info(f"✅ Backtest başarıyla tamamlandı. Sonuçlar: {output_dir}/single klasörüne kaydedildi.")
            
            # Kısa özet yazdır
            metrics = result.get("result", {}).get("metrics", {})
            if metrics:
                logger.info(f"📊 Özet Sonuçlar:")
                logger.info(f"   - Toplam İşlem: {result['result']['total_trades']}")
                logger.info(f"   - Kazanç Oranı: {metrics.get('win_rate', 0):.2f}%")
                logger.info(f"   - ROI: {result['result']['roi_pct']:.2f}%")
                logger.info(f"   - Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                
                # Ek metrikler
                if 'sharpe_ratio' in metrics:
                    logger.info(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                
                if 'profit_factor' in metrics:
                    logger.info(f"   - Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        else:
            logger.error(f"❌ Backtest sırasında hata oluştu: {result.get('message')}")
    
    elif mode == "batch":
        # Config CSV yolunu belirle
        config_csv = os.path.join("backtest", "config", "config_combinations.csv")
        
        if not os.path.exists(config_csv):
            logger.error(f"❌ Konfigürasyon CSV dosyası bulunamadı: {config_csv}")
            return
        
        logger.info(f"🚀 CSV konfigürasyonları ile toplu backtest çalıştırılıyor: {config_csv}")
        
        # Maksimum işlemci sayısını belirle
        max_workers = os.cpu_count() - 1  # Bir CPU boşta bırak
        
        # Backtest parametrelerini hazırla
        backtest_params = {
            "initial_balance": float(env_config.get("initial_balance", 10000.0)),
            "risk_per_trade": float(env_config.get("risk_per_trade", 0.01)),
            "sl_multiplier": float(env_config.get("sl_multiplier", 1.5)),
            "tp_multiplier": float(env_config.get("tp_multiplier", 3.0)),
            "leverage": float(env_config.get("leverage", 1.0)),
            "position_direction": env_config.get("position_direction", {"Long": True, "Short": True}),
            "commission_rate": float(env_config.get("commission_rate", 0.001)),
        }
        
        # Toplu backtest çalıştır
        result = run_batch_backtest(
            symbol=symbol,
            interval=interval,
            config_csv_path=config_csv,
            db_url=env_config.get("db_url"),
            output_dir=os.path.join(output_dir, "batch"),
            backtest_params=backtest_params,
            max_workers=max_workers
        )
        
        if result.get("status") == "success":
            logger.info(f"✅ Toplu backtest başarıyla tamamlandı. Sonuçlar: {output_dir}/batch klasörüne kaydedildi.")
            logger.info(f"   - Toplam Konfigürasyon: {result.get('total_configs')}")
            logger.info(f"   - Tamamlanan: {result.get('completed_configs')}")
            logger.info(f"   - En İyi ROI: {result.get('best_roi_pct'):.2f}% (Config: {result.get('best_roi_config')})")
            logger.info(f"   - En İyi Kazanç Oranı: {result.get('best_winrate_pct'):.2f}% (Config: {result.get('best_winrate_config')})")
        else:
            logger.error(f"❌ Toplu backtest sırasında hata oluştu: {result.get('message')}")
    
    else:
        logger.error(f"❌ Bilinmeyen çalıştırma modu: {mode}. 'single' veya 'batch' kullanın.")


def load_test_config() -> Dict[str, Any]:
    """
    Test için yapılandırma yükle
    
    Returns:
        Test yapılandırma sözlüğü
    """
    # Signal engine'deki tüm bileşenleri içeren yapılandırma
    
    # Bu, signal engine'in tüm bileşenlerini etkinleştiren örnek bir yapılandırmadır
    # Gerçek uygulama için .env dosyasındaki yapılandırmayı kullanmanız önerilir
    
    # Tüm indikatörleri içeren yapılandırma
    INDICATORS_LONG = {
        "ema": {"periods": [9, 21, 50, 200]},
        "rsi": {"periods": [14]},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger": {"window": 20, "window_dev": 2.0},
        "atr": {"window": 14},
        "supertrend": {"atr_period": 10, "atr_multiplier": 3.0},
        "market_regime": {},
        "trend_strength": {}
    }
    
    # Tüm stratejileri içeren yapılandırma
    STRATEGIES_CONFIG = {
        "trend_following": {},
        "mtf_trend": {},
        "adaptive_trend": {},
        "overextended_reversal": {},
        "pattern_reversal": {},
        "divergence_reversal": {},
        "volatility_breakout": {},
        "range_breakout": {},
        "sr_breakout": {}
    }
    
    # Tüm filtreleri içeren yapılandırma
    FILTER_CONFIG = {
        "market_regime": {},
        "volatility_regime": {},
        "trend_strength": {},
        "zscore_extreme_filter": {},
        "dynamic_threshold_filter": {},
        "min_checks": 1,
        "min_strength": 1
    }
    
    # Tüm güç hesaplayıcıları içeren yapılandırma
    STRENGTH_CONFIG = {
        "probabilistic_strength": {},
        "risk_reward_strength": {},
        "market_context_strength": {}
    }
    
    return {
        "indicators": {"long": INDICATORS_LONG, "short": INDICATORS_LONG},
        "strategies": STRATEGIES_CONFIG,
        "filters": FILTER_CONFIG,
        "strength": STRENGTH_CONFIG
    }


if __name__ == "__main__":
    # VSCode'dan doğrudan çalıştırmak için yapılandırma
    # Burada mod, config_id ve özel parametreler ayarlanabilir
    
    # Çalıştırma modu: "single" veya "batch"
    RUN_MODE = "single"
    
    # Tek backtest için konfigürasyon ID'si
    CONFIG_ID = "default"
    
    # Kapsamlı bir test için signal engine bileşenlerini içeren yapılandırmayı yükleme seçeneği
    # Varsayılan olarak kapalı, test etmek için aktifleştirebilirsiniz
    USE_TEST_CONFIG = False
    
    # Özel konfigürasyon (çevre değişkenlerini ezmek için)
    CUSTOM_CONFIG = {
        # "symbol": "ETHUSDT",        # Varsayılan: ENV dosyasından
        # "interval": "1h",           # Varsayılan: ENV dosyasından
        # "initial_balance": 5000,    # Varsayılan: 10000
        # "risk_per_trade": 0.02,     # Varsayılan: 0.01
        # "sl_multiplier": 2.0,       # Varsayılan: 1.5
        # "tp_multiplier": 4.0,       # Varsayılan: 3.0
        # "leverage": 2.0,            # Varsayılan: 1.0
        # "position_direction": {"Long": True, "Short": False}  # Varsayılan: Her iki yön
    }
    
    # Test yapılandırmasını etkinleştir
    if USE_TEST_CONFIG:
        test_config = load_test_config()
        CUSTOM_CONFIG.update(test_config)
    
    # Backtest çalıştır
    run_backtest(mode=RUN_MODE, config_id=CONFIG_ID, custom_config=CUSTOM_CONFIG)