# backtest/utils/config_viewer.py dosyasına eklenecek kod

import json
from typing import Dict, Any

def print_config_details(config: Dict[str, Any], title: str = "CONFIG DETAILS") -> None:
    """
    Konfigürasyon nesnesini detaylı bir şekilde yazdırır
    
    Args:
        config: Konfigürasyon nesnesi
        title: Çıktı başlığı
    """
    print(f"\n{title}")
    print("=" * 80)
    print(json.dumps(config, indent=2, sort_keys=True))
    print("=" * 80)

    # Önemli değerleri ayrı ayrı yazdır
    print("\nKEY CONFIG VALUES:")
    print(f"Symbol: {config.get('symbol')}")
    print(f"Interval: {config.get('interval')}")
    print(f"Initial Balance: {config.get('initial_balance')}")
    print(f"Risk Per Trade: {config.get('risk_per_trade')}")
    print(f"Position Direction: {config.get('position_direction')}")

    # İndikatör detaylarını derinlemesine yazdır
    print("\nINDICATOR DETAILS:")
    if 'indicators' in config:
        long_indicators = config['indicators'].get('long', {})
        short_indicators = config['indicators'].get('short', {})
        
        print("\nLONG INDICATORS:")
        for name, params in long_indicators.items():
            print(f"  - {name}: {json.dumps(params)}")
        
        print("\nSHORT INDICATORS:")
        for name, params in short_indicators.items():
            print(f"  - {name}: {json.dumps(params)}")
    else:
        print("No indicators configured.")