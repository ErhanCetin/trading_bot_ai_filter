"""
Trading system indicators package.
Import all indicators for automatic registration with Smart Dependencies.
"""
from signal_engine.signal_indicator_plugin_system import IndicatorRegistry
from .common_calculations import ADXCalculator

# Create registry
registry = IndicatorRegistry()

# Register common utilities first
registry.register(ADXCalculator)

# Import all indicator modules to register them
from .base_indicators import (
    EMAIndicator,
    SMAIndicator,
    RSIIndicator,
    MACDIndicator,
    BollingerBandsIndicator,
    ATRIndicator,
    StochasticIndicator
)

from .advanced_indicators import (
    AdaptiveRSIIndicator,
    MultitimeframeEMAIndicator,
    HeikinAshiIndicator,
    SupertrendIndicator,
    IchimokuIndicator
)

from .feature_indicators import (
    PriceActionIndicator,
    VolumePriceIndicator,
    MomentumFeatureIndicator,
    SupportResistanceIndicator
)

from .regime_indicators import (
    MarketRegimeIndicator,
    VolatilityRegimeIndicator,
    TrendStrengthIndicator
)

from .statistical_indicators import (
    ZScoreIndicator,
    KeltnerChannelIndicator,
    StandardDeviationIndicator,
    LinearRegressionIndicator
)

# Register all indicators - FIXED: No duplicate ADXCalculator registration
all_indicators = [
    # Base indicators
    EMAIndicator, SMAIndicator, RSIIndicator, MACDIndicator, 
    BollingerBandsIndicator, ATRIndicator, StochasticIndicator,
    
    # Advanced indicators
    AdaptiveRSIIndicator, MultitimeframeEMAIndicator, HeikinAshiIndicator,
    SupertrendIndicator, IchimokuIndicator,
    
    # Feature indicators
    PriceActionIndicator, VolumePriceIndicator, MomentumFeatureIndicator,
    SupportResistanceIndicator,
    
    # Regime indicators (now with Smart Dependencies!)
    MarketRegimeIndicator, VolatilityRegimeIndicator, TrendStrengthIndicator,
    
    # Statistical indicators
    ZScoreIndicator, KeltnerChannelIndicator, StandardDeviationIndicator,
    LinearRegressionIndicator
]

# Register all indicators
for indicator_class in all_indicators:
    registry.register(indicator_class)

# Log registration summary
import logging
logger = logging.getLogger(__name__)
logger.info(f"Successfully registered {len(all_indicators) + 1} indicators with Smart Dependencies")
logger.debug(f"Available indicators: {list(registry.get_all_indicators().keys())}")

# Expose registry for import
__all__ = ['registry']