"""
Trading system filter rules package.
Import all filter rules for automatic registration.
"""

# Temel bileşenleri import et
from signal_engine.signal_indicator_plugin_system import BaseIndicator, IndicatorRegistry
from signal_engine.signal_strategy_system import BaseStrategy, StrategyRegistry
from signal_engine.signal_filter_system import BaseFilter, FilterRuleRegistry


# Modüllerin global registry'leri
indicator_registry = IndicatorRegistry()
strategy_registry = StrategyRegistry()
filter_registry = FilterRuleRegistry()

# Versiyon bilgisi
__version__ = '0.1.0'

# Import all filter rule modules to register them
from .regime_filters import (
    MarketRegimeFilter,
    VolatilityRegimeFilter,
    TrendStrengthFilter
)

from .statistical_filters import (
    ZScoreExtremeFilter,
    OutlierDetectionFilter,
    HistoricalVolatilityFilter
)

from .ml_filters import (
    ProbabilisticSignalFilter,
    PatternRecognitionFilter,
    PerformanceClassifierFilter
)

from .adaptive_filters import (
    DynamicThresholdFilter,
    ContextAwareFilter,
    MarketCycleFilter
)

from .ensemble_filters import (
    VotingEnsembleFilter,
    SequentialFilterChain,
    WeightedMetaFilter
)

# Register all filter rules
registry.register(MarketRegimeFilter)
registry.register(VolatilityRegimeFilter)
registry.register(TrendStrengthFilter)
registry.register(ZScoreExtremeFilter)
registry.register(OutlierDetectionFilter)
registry.register(HistoricalVolatilityFilter)
registry.register(ProbabilisticSignalFilter)
registry.register(PatternRecognitionFilter)
registry.register(PerformanceClassifierFilter)
registry.register(DynamicThresholdFilter)
registry.register(ContextAwareFilter)
registry.register(MarketCycleFilter)
registry.register(VotingEnsembleFilter)
registry.register(SequentialFilterChain)
registry.register(WeightedMetaFilter)

# Expose registry for import
__all__ = ['registry']