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
    # Bu kodu bir yerde çalıştırın (örneğin main.py dosyasına ekleyin)

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
    #print_config_details(env_config, "BACKTEST CONFIGURATION")

    #print_registered_indicators()
    #check_signal_engine_components()

    
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
        
        # Tek backtest çalıştır
        result = run_single_backtest(
            symbol=symbol,
            interval=interval,
            db_url=env_config.get("db_url"),
            output_dir=os.path.join(output_dir, "single"),
            backtest_params={
                "initial_balance": env_config.get("initial_balance"),
                "risk_per_trade": env_config.get("risk_per_trade"),
                "sl_multiplier": env_config.get("sl_multiplier"),
                "tp_multiplier": env_config.get("tp_multiplier"),
                "leverage": env_config.get("leverage"),
                "position_direction": env_config.get("position_direction"),
                "commission_rate": env_config.get("commission_rate"),
            },
            indicators_config=env_config.get("indicators"),
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
        
        # Toplu backtest çalıştır
        result = run_batch_backtest(
            symbol=symbol,
            interval=interval,
            config_csv_path=config_csv,
            db_url=env_config.get("db_url"),
            output_dir=os.path.join(output_dir, "batch"),
            backtest_params={
                "initial_balance": env_config.get("initial_balance"),
                "risk_per_trade": env_config.get("risk_per_trade"),
                "sl_multiplier": env_config.get("sl_multiplier"),
                "tp_multiplier": env_config.get("tp_multiplier"),
                "leverage": env_config.get("leverage"),
                "position_direction": env_config.get("position_direction"),
                "commission_rate": env_config.get("commission_rate"),
            },
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


if __name__ == "__main__":
    # VSCode'dan doğrudan çalıştırmak için yapılandırma
    # Burada mod, config_id ve özel parametreler ayarlanabilir
    
    # Çalıştırma modu: "single" veya "batch"
    RUN_MODE = "single"
    
    # Tek backtest için konfigürasyon ID'si
    CONFIG_ID = "default"
    
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
    
    # Backtest çalıştır
    run_backtest(mode=RUN_MODE, config_id=CONFIG_ID, custom_config=CUSTOM_CONFIG)