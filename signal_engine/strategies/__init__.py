"""
Trading system strategies package.
Import all strategies for automatic registration.
"""
from signal_engine.signal_strategy_system import StrategyRegistry

# Create registry
registry = StrategyRegistry()

# Import all strategy modules to register them
from .trend_strategy import (
    TrendFollowingStrategy,
    MultiTimeframeTrendStrategy,
    AdaptiveTrendStrategy
)

from .reversal_strategy import (
    OverextendedReversalStrategy,
    PatternReversalStrategy,
    DivergenceReversalStrategy
)

from .breakout_strategy import (
    VolatilityBreakoutStrategy,
    RangeBreakoutStrategy,
    SupportResistanceBreakoutStrategy
)

from .ensemble_strategy import (
    RegimeBasedEnsembleStrategy,
    WeightedVotingEnsembleStrategy,
    AdaptiveEnsembleStrategy
)

# Register all strategies
registry.register(TrendFollowingStrategy)
registry.register(MultiTimeframeTrendStrategy)
registry.register(AdaptiveTrendStrategy)
registry.register(OverextendedReversalStrategy)
registry.register(PatternReversalStrategy)
registry.register(DivergenceReversalStrategy)
registry.register(VolatilityBreakoutStrategy)
registry.register(RangeBreakoutStrategy)
registry.register(SupportResistanceBreakoutStrategy)
registry.register(RegimeBasedEnsembleStrategy)
registry.register(WeightedVotingEnsembleStrategy)
registry.register(AdaptiveEnsembleStrategy)

# Expose registry for import
__all__ = ['registry']