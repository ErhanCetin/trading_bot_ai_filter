"""
Trading system indicators package.
Import all indicators for automatic registration.
"""
from signal_engine.signal_indicator_plugin_system import IndicatorRegistry
   
from .common_calculations import ADXCalculator

# Create registry
registry = IndicatorRegistry()

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

# Register all indicators
for indicator_class in [
    # Base indicators
    EMAIndicator, SMAIndicator, RSIIndicator, MACDIndicator, 
    BollingerBandsIndicator, ATRIndicator, StochasticIndicator,
    
    # Advanced indicators
    AdaptiveRSIIndicator, MultitimeframeEMAIndicator, HeikinAshiIndicator,
    SupertrendIndicator, IchimokuIndicator,
    
    # Feature indicators
    PriceActionIndicator, VolumePriceIndicator, MomentumFeatureIndicator,
    SupportResistanceIndicator,
    
    # Regime indicators
    MarketRegimeIndicator, VolatilityRegimeIndicator, TrendStrengthIndicator,
    
    # Statistical indicators
    ZScoreIndicator, KeltnerChannelIndicator, StandardDeviationIndicator,
    LinearRegressionIndicator  

]:
    registry.register(indicator_class)
    registry.register(ADXCalculator)

# Expose registry for import
__all__ = ['registry']