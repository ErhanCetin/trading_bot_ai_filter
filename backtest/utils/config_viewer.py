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


def print_enhanced_config_summary(env_config: Dict[str, Any]) -> None:
    """
    ENHANCED: Print enhanced configuration summary
    """
    print("\n📋 BACKTEST CONFIGURATION SUMMARY:")
    print("="*50)
    print(f"Symbol: {env_config.get('symbol', 'Not set')}")
    print(f"Interval: {env_config.get('interval', 'Not set')}")
    print(f"Initial Balance: ${env_config.get('initial_balance', 0):,.2f}")
    print(f"Risk per Trade: {float(env_config.get('risk_per_trade', 0))*100:.2f}%")
    print(f"Leverage: {env_config.get('leverage', 1)}x")
    print(f"SL Multiplier: {env_config.get('sl_multiplier', 1.5)}")
    print(f"TP Multiplier: {env_config.get('tp_multiplier', 3.0)}")
    print(f"Risk/Reward Ratio: {float(env_config.get('tp_multiplier', 3.0))/float(env_config.get('sl_multiplier', 1.5)):.2f}")
    print(f"Commission Rate: {float(env_config.get('commission_rate', 0.001))*100:.3f}%")
    
    pos_dir = env_config.get('position_direction', {})
    directions = []
    if pos_dir.get('Long', True):
        directions.append('Long')
    if pos_dir.get('Short', True):
        directions.append('Short')
    print(f"Allowed Directions: {', '.join(directions) if directions else 'None'}")
    
    # Enhanced configuration details
    indicators = env_config.get('indicators', {})
    if indicators:
        long_indicators = indicators.get('long', {})
        short_indicators = indicators.get('short', {})
        total_indicators = len(long_indicators) + len(short_indicators)
        print(f"Indicators: {total_indicators} configured ({len(long_indicators)} long, {len(short_indicators)} short)")
    
    strategies = env_config.get('strategies', {})
    if strategies:
        print(f"Strategies: {len(strategies)} configured")
    
    filters = env_config.get('filters', {})
    if filters:
        filter_count = len([k for k in filters.keys() if k not in ['min_checks', 'min_strength']])
        print(f"Filters: {filter_count} rules configured")
        print(f"Min Checks Required: {filters.get('min_checks', 'Not set')}")
        print(f"Min Strength Required: {filters.get('min_strength', 'Not set')}")
        