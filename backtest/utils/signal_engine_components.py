import logging
from typing import Dict, Any, List

def check_signal_engine_components():
    """
    Signal Engine'deki mevcut tüm bileşenleri kontrol eder
    """
    try:
        from signal_engine.indicators import registry as indicator_registry
        from signal_engine.filters import registry as filter_registry
        from signal_engine.strength import registry as strength_registry
        from signal_engine.strategies import registry as strategy_registry
        
        print("\n=== AVAILABLE INDICATORS ===")
        for name in indicator_registry.get_all_indicators():
            print(f"- {name}")
        
        print("\n=== AVAILABLE FILTER RULES ===")
        for name in filter_registry.get_all_filters():
            print(f"- {name}")
        
        print("\n=== AVAILABLE STRENGTH CALCULATORS ===")
        for name in strength_registry.get_all_calculators():
            print(f"- {name}")
        
        print("\n=== AVAILABLE STRATEGIES ===")
        for name in strategy_registry.get_all_strategies():
            print(f"- {name}")
            
        return {
            "indicators": list(indicator_registry.get_all_indicators().keys()),
            "filter_rules": list(filter_registry.get_all_filters().keys()),
            "strength_calculators": list(strength_registry.get_all_calculators().keys()),
            "strategies": list(strategy_registry.get_all_strategies().keys())
        }
    except Exception as e:
        logging.error(f"Error checking Signal Engine components: {e}")
        return {}
    

#check_signal_engine_components()   