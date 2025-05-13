"""
Trading system filter rules package.
Import all filter rules for automatic registration.
"""

# Temel bileşenleri import et
from signal_engine.signal_indicator_plugin_system import BaseIndicator, IndicatorRegistry
from signal_engine.signal_strategy_system import BaseStrategy, StrategyRegistry
from signal_engine.signal_filter_system import BaseFilter, FilterRuleRegistry

import logging

# Modüllerin global registry'leri
indicator_registry = IndicatorRegistry()
strategy_registry = StrategyRegistry()
registry = FilterRuleRegistry()

# Versiyon bilgisi
__version__ = '0.1.0'

# Tüm filtreleri kaydetmek için yardımcı fonksiyon
def register_filter(registry, filter_class):
    """Filtreyi registry'e kaydet, hata durumunda alternatif yöntemi dene."""
    try:
        # Standart register yöntemi
        registry.register(filter_class)
    except (TypeError, AttributeError) as e:
        # Alternatif yöntem: doğrudan sözlüğe ekle
        if hasattr(filter_class, 'name'):
            registry._filters[filter_class.name] = filter_class
        else:
            # Name özelliği yoksa, sınıf adını kullan
            registry._filters[filter_class.__name__.lower()] = filter_class
        logging.info(f"Registered {filter_class.__name__} using alternative method.")

# Filtre modüllerini import et ve registry'e kaydet
try:
    # Rejim filtreleri
    from .regime_filters import (
        MarketRegimeFilter, 
        VolatilityRegimeFilter, 
        TrendStrengthFilter
    )
    
    register_filter(registry, MarketRegimeFilter)
    register_filter(registry, VolatilityRegimeFilter)
    register_filter(registry, TrendStrengthFilter)
except ImportError as e:
    logging.warning(f"Error importing regime filters: {e}")

try:
    # İstatistiksel filtreler
    from .statistical_filters import (
        ZScoreExtremeFilter,
        OutlierDetectionFilter,
        HistoricalVolatilityFilter
    )
    
    register_filter(registry, ZScoreExtremeFilter)
    register_filter(registry, OutlierDetectionFilter)
    register_filter(registry, HistoricalVolatilityFilter)
except ImportError as e:
    logging.warning(f"Error importing statistical filters: {e}")

try:
    # ML filtreleri
    from .ml_filters import (
        ProbabilisticSignalFilter,
        PatternRecognitionFilter,
        PerformanceClassifierFilter
    )
    
    register_filter(registry, ProbabilisticSignalFilter)
    register_filter(registry, PatternRecognitionFilter)
    register_filter(registry, PerformanceClassifierFilter)
except ImportError as e:
    logging.warning(f"Error importing ML filters: {e}")

try:
    # Adaptif filtreler
    from .adaptive_filters import (
        DynamicThresholdFilter,
        ContextAwareFilter,
        MarketCycleFilter
    )
    
    register_filter(registry, DynamicThresholdFilter)
    register_filter(registry, ContextAwareFilter)
    register_filter(registry, MarketCycleFilter)
except ImportError as e:
    logging.warning(f"Error importing adaptive filters: {e}")

try:
    # Ensemble filtreler
    from .ensemble_filters import (
        VotingEnsembleFilter,
        SequentialFilterChain,
        WeightedMetaFilter
    )
    
    register_filter(registry, VotingEnsembleFilter)
    register_filter(registry, SequentialFilterChain)
    register_filter(registry, WeightedMetaFilter)
except ImportError as e:
    logging.warning(f"Error importing ensemble filters: {e}")

# Expose registry for import
__all__ = ['registry']