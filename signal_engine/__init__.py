"""
Updated signal engine module for the trading bot.
Includes all plugin systems with modular architecture.
"""
# Indicator system imports
from signal_engine.signal_indicator_plugin_system import BaseIndicator, IndicatorRegistry, IndicatorManager

# Signal strategy system imports
from signal_engine.signal_strategy_system import BaseStrategy, StrategyRegistry, StrategyManager

# Signal strength system imports
from signal_engine.signal_strength_system import BaseStrengthCalculator, StrengthCalculatorRegistry, StrengthManager

# Signal filter system imports
from signal_engine.signal_filter_system import BaseFilter, FilterRuleRegistry, FilterManager

# ML system imports (optional)
try:
    from signal_engine.signal_ml_system import MLManager
    from signal_engine.ml import ModelTrainer, FeatureSelector, SignalPredictor, StrengthPredictor
    __has_ml__ = True
except ImportError:
    __has_ml__ = False

# Main manager import
from signal_engine.signal_manager import SignalManager

# Versioning
__version__ = '0.1.0'

# Sub-modules direct access
from . import indicators
from . import strategies
from . import filters
from . import strength

# ML sub-module (optional)
if __has_ml__:
    from . import ml

# For backward compatibility, you may want to rename manager classes
# to match your existing naming conventions
SignalGeneratorManager = StrategyManager
SignalFilterManager = FilterManager
SignalStrengthManager = StrengthManager
BaseFilterRule = BaseFilter

__all__ = [
    # Indicator system
    'BaseIndicator',
    'IndicatorRegistry',
    'IndicatorManager',
    
    # Signal strategy system
    'BaseStrategy',
    'StrategyRegistry',
    'StrategyManager',
    'SignalGeneratorManager',  # Alias for backward compatibility
    
    # Signal strength system
    'BaseStrengthCalculator',
    'StrengthCalculatorRegistry',
    'StrengthManager',
    'SignalStrengthManager',  # Alias for backward compatibility
    
    # Signal filter system
    'BaseFilter',
    'BaseFilterRule',  # Alias for backward compatibility
    'FilterRuleRegistry',
    'FilterManager',
    'SignalFilterManager',  # Alias for backward compatibility
    
    # Main manager
    'SignalManager',
    
    # Version
    '__version__'
]

# Add ML components to __all__ if available
if __has_ml__:
    __all__.extend([
        'MLManager',
        'ModelTrainer',
        'FeatureSelector',
        'SignalPredictor',
        'StrengthPredictor',
        '__has_ml__'
    ])