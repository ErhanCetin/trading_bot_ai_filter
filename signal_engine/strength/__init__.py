"""
Trading system signal strength calculators package.
Import all strength calculators for automatic registration.
"""
from signal_engine.signal_strength_system import StrengthCalculatorRegistry

# Create registry
registry = StrengthCalculatorRegistry()

# Import all strength calculator modules to register them
from .base_strength import BaseStrengthCalculator
from .predictive_strength import (
    ProbabilisticStrengthCalculator,
    RiskRewardStrengthCalculator,
    MLPredictiveStrengthCalculator
)

from .context_strength import (
    MarketContextStrengthCalculator,
    IndicatorConfirmationStrengthCalculator,
    MultiTimeframeStrengthCalculator
)

# Register all strength calculators
registry.register(ProbabilisticStrengthCalculator)
registry.register(RiskRewardStrengthCalculator)
registry.register(MLPredictiveStrengthCalculator)
registry.register(MarketContextStrengthCalculator)
registry.register(IndicatorConfirmationStrengthCalculator)
registry.register(MultiTimeframeStrengthCalculator)

# Expose registry for import
__all__ = ['registry']