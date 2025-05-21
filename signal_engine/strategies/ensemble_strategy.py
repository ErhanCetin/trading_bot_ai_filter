"""
Ensemble strategies for the trading system.
These strategies combine multiple sub-strategies for more robust signals.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from signal_engine.signal_strategy_system import BaseStrategy, StrategyRegistry

# Get strategy registry
try:
    from signal_engine.strategies import registry
except ImportError:
    registry = StrategyRegistry()

logger = logging.getLogger(__name__)


class RegimeBasedEnsembleStrategy(BaseStrategy):
    """Strategy that selects appropriate sub-strategies based on market regime."""
    
    name = "regime_ensemble"
    display_name = "Regime-Based Ensemble Strategy"
    description = "Selects appropriate sub-strategies based on market regime"
    category = "ensemble"
    
    default_params = {
        "regime_weights": {
            "strong_uptrend": {"trend": 0.7, "reversal": 0.1, "breakout": 0.2},
            "weak_uptrend": {"trend": 0.5, "reversal": 0.3, "breakout": 0.2},
            "strong_downtrend": {"trend": 0.7, "reversal": 0.1, "breakout": 0.2},
            "weak_downtrend": {"trend": 0.5, "reversal": 0.3, "breakout": 0.2},
            "ranging": {"trend": 0.2, "reversal": 0.3, "breakout": 0.5},
            "volatile": {"trend": 0.3, "reversal": 0.2, "breakout": 0.5},
            "overbought": {"trend": 0.1, "reversal": 0.7, "breakout": 0.2},
            "oversold": {"trend": 0.1, "reversal": 0.7, "breakout": 0.2},
            "unknown": {"trend": 0.33, "reversal": 0.33, "breakout": 0.34}
        },
        "vote_threshold": 0.6,  # Threshold for weighted voting
        "strategy_mapping": {
            "trend": ["trend_following", "mtf_trend", "adaptive_trend"],
            "reversal": ["overextended_reversal", "pattern_reversal", "divergence_reversal"],
            "breakout": ["volatility_breakout", "range_breakout", "sr_breakout"]
        }
    }
    
    required_indicators = ["close"]
    optional_indicators = ["market_regime"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with strategy instances.
        
        Args:
            params: Optional parameters to override defaults
        """
        super().__init__(params)
        
        # Initialize strategies
        self.strategies = {}
        
        try:
            # Initialize all available strategies from registry
            for strategy_name, strategy_class in registry.get_all_strategies().items():
                # Avoid recursive initialization of ensemble strategies
                if strategy_name != self.name and "ensemble" not in strategy_class.category:
                    try:
                        self.strategies[strategy_name] = strategy_class()
                    except Exception as e:
                        logger.warning(f"Failed to initialize strategy {strategy_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize strategies from registry: {e}")
        
        # Map strategies to categories with improved error handling
        self.category_map = {}
        strategy_mapping = self.params.get("strategy_mapping", self.default_params["strategy_mapping"])
        
        # Validate strategy_mapping structure
        if not isinstance(strategy_mapping, dict):
            logger.warning(f"Invalid strategy_mapping format: {strategy_mapping}, using default")
            strategy_mapping = self.default_params["strategy_mapping"]
        
        for category, strategy_list in strategy_mapping.items():
            if not isinstance(strategy_list, list):
                logger.warning(f"Invalid strategy list for category {category}: {strategy_list}, skipping")
                continue
                
            for strategy_name in strategy_list:
                if strategy_name in self.strategies:
                    self.category_map[strategy_name] = category
                else:
                    logger.debug(f"Strategy {strategy_name} defined in mapping but not available")
        
        # Ensure all strategies have a category assigned
        for strategy_name in self.strategies:
            if strategy_name not in self.category_map:
                # Assign a default category
                self.category_map[strategy_name] = "unknown"
                logger.debug(f"Strategy {strategy_name} has no category defined, assigning 'unknown'")

    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate ensemble signal conditions based on market regime.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get market regime with improved error handling
        regime = "unknown"
        
        try:
            # First check if market_regime exists and is not None/NaN
            if "market_regime" in row and row["market_regime"] is not None and not pd.isna(row["market_regime"]):
                regime_value = row["market_regime"]
                
                # Convert to string for consistency
                regime = str(regime_value).lower().strip()
                
                # Log for debugging
                logger.debug(f"Found market_regime: '{regime}', type: {type(regime_value)}")
            else:
                logger.debug(f"No valid market_regime at index {i}, using default: 'unknown'")
        except Exception as e:
            logger.warning(f"Error accessing market_regime at index {i}: {e}")
            regime = "unknown"
        
        # Get regime weights with safety checks
        regime_weights = self.params.get("regime_weights", self.default_params["regime_weights"])
        
        # Check if we have weights for this regime
        if regime not in regime_weights:
            logger.debug(f"Unknown market regime: '{regime}', falling back to 'unknown'")
            regime = "unknown"
        
        # If unknown is also not in weights (which shouldn't happen), use default weights
        weights = regime_weights.get(regime, {"trend": 0.33, "reversal": 0.33, "breakout": 0.34})
        
        # Track weighted votes for long and short
        long_votes = 0
        short_votes = 0
        total_weight = 0
        
        # Generate signals from all strategies
        for name, strategy in self.strategies.items():
            # Safely get category with default
            category = self.category_map.get(name, "unknown")
            # Safely get weight with default (using explicit get with default value)
            weight = weights.get(category, 0.33)
            
            try:
                if strategy.validate_dataframe(df):
                    try:
                        # Try to generate conditions with explicit error handling
                        conditions = strategy.generate_conditions(df, row, i)
                        
                        # Ensure conditions are valid and contain necessary keys
                        if not isinstance(conditions, dict) or 'long' not in conditions or 'short' not in conditions:
                            logger.debug(f"Strategy {name} returned invalid conditions format")
                            continue
                        
                        # Safely calculate votes based on conditions
                        long_conditions = conditions.get('long', [])
                        short_conditions = conditions.get('short', [])
                        
                        # Filter out None values and false values
                        long_conditions = [c for c in long_conditions if c is not None]
                        short_conditions = [c for c in short_conditions if c is not None]
                        
                        # Calculate votes only if there are conditions
                        if long_conditions:
                            long_count = sum(1 for x in long_conditions if x)
                            long_vote = long_count / len(long_conditions) if len(long_conditions) > 0 else 0
                            long_votes += long_vote * weight
                        
                        if short_conditions:
                            short_count = sum(1 for x in short_conditions if x)
                            short_vote = short_count / len(short_conditions) if len(short_conditions) > 0 else 0
                            short_votes += short_vote * weight
                        
                        # Only add weight if there were conditions
                        if long_conditions or short_conditions:
                            total_weight += weight
                            
                    except Exception as inner_e:
                        logger.debug(f"Error processing conditions for strategy {name}: {inner_e}")
                else:
                    logger.debug(f"Strategy {name} validation failed for dataframe")
            except Exception as e:
                logger.debug(f"Error generating conditions for strategy {name}: {e}")
        
        # Normalize votes (safely handle zero division)
        if total_weight > 0:
            long_votes /= total_weight
            short_votes /= total_weight
        else:
            # If no valid strategies, defaults to no signals
            logger.debug("No valid strategy weights found, using default vote values of 0")
        
        # Get vote threshold with safety check
        threshold = self.params.get("vote_threshold", self.default_params["vote_threshold"])
        
        # Convert to conditions
        long_conditions = [long_votes >= threshold] if long_votes > 0 else []
        short_conditions = [short_votes >= threshold] if short_votes > 0 else []
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }

class WeightedVotingEnsembleStrategy(BaseStrategy):
    """Strategy that combines signals from multiple sub-strategies using weighted voting."""
    
    name = "weighted_voting"
    display_name = "Weighted Voting Ensemble Strategy"
    description = "Combines signals using weighted voting across all strategies"
    category = "ensemble"
    
    default_params = {
        "strategy_weights": {
            "trend_following": 1.0,
            "mtf_trend": 1.0,
            "adaptive_trend": 1.0,
            "overextended_reversal": 1.0,
            "pattern_reversal": 1.0,
            "divergence_reversal": 1.0,
            "volatility_breakout": 1.0,
            "range_breakout": 1.0,
            "sr_breakout": 1.0
        },
        "vote_threshold": 0.6  # Threshold for weighted voting
    }
    
    required_indicators = ["close"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with strategy instances.
        
        Args:
            params: Optional parameters to override defaults
        """
        super().__init__(params)
        
        # Initialize strategies
        self.strategies = {}
        
        try:
            # Initialize all available strategies from registry
            for strategy_name, strategy_class in registry.get_all_strategies().items():
                # Avoid recursive initialization of ensemble strategies
                if strategy_name != self.name and "ensemble" not in strategy_class.category:
                    try:
                        self.strategies[strategy_name] = strategy_class()
                    except Exception as e:
                        logger.warning(f"Failed to initialize strategy {strategy_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize strategies from registry: {e}")
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate ensemble signal conditions using weighted voting.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get strategy weights
        strategy_weights = self.params.get("strategy_weights", self.default_params["strategy_weights"])
        
        # Track weighted votes for long and short
        long_votes = 0
        short_votes = 0
        total_weight = 0
        
        # Generate signals from all strategies
        for name, strategy in self.strategies.items():
            weight = strategy_weights.get(name, 1.0)
            
            try:
                if strategy.validate_dataframe(df):
                    # Try to generate conditions with explicit error handling
                    try:
                        conditions = strategy.generate_conditions(df, row, i)
                        
                        # Ensure conditions are valid and contain necessary keys
                        if not isinstance(conditions, dict):
                            logger.warning(f"Strategy {name} returned invalid conditions type: {type(conditions)}")
                            continue
                        
                        if "long" not in conditions or "short" not in conditions:
                            logger.warning(f"Strategy {name} returned incomplete conditions: {conditions.keys()}")
                            continue
                        
                        # Safely calculate votes based on conditions
                        long_conditions = conditions.get("long", [])
                        short_conditions = conditions.get("short", [])
                        
                        # Filter out None values from conditions
                        long_conditions = [c for c in long_conditions if c is not None]
                        short_conditions = [c for c in short_conditions if c is not None]
                        
                        # Calculate votes based on valid conditions
                        long_count = sum(long_conditions)
                        short_count = sum(short_conditions)
                        
                        # Normalize to 0-1 range if there are conditions
                        total_conditions_long = max(1, len(long_conditions))
                        total_conditions_short = max(1, len(short_conditions))
                        
                        long_vote = long_count / total_conditions_long if long_count > 0 else 0
                        short_vote = short_count / total_conditions_short if short_count > 0 else 0
                        
                        # Apply weight
                        long_votes += long_vote * weight
                        short_votes += short_vote * weight
                        total_weight += weight
                        
                    except Exception as inner_e:
                        logger.warning(f"Error processing conditions for strategy {name}: {inner_e}")
            except Exception as e:
                logger.warning(f"Error generating conditions for strategy {name}: {e}")
        
        # Normalize votes
        if total_weight > 0:
            long_votes /= total_weight
            short_votes /= total_weight
        
        # Get vote threshold
        threshold = self.params.get("vote_threshold", self.default_params["vote_threshold"])
        
        # Convert to conditions
        long_conditions = [long_votes >= threshold]
        short_conditions = [short_votes >= threshold]
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class AdaptiveEnsembleStrategy(BaseStrategy):
    """Strategy that adapts weights based on recent performance of sub-strategies."""
    
    name = "adaptive_ensemble"
    display_name = "Adaptive Ensemble Strategy"
    description = "Adapts weights based on recent performance of sub-strategies"
    category = "ensemble"
    
    default_params = {
        "lookback_window": 50,  # Window to evaluate strategy performance
        "performance_decay": 0.95,  # Decay factor for older performance
        "initial_weight": 1.0,  # Initial weight for all strategies
        "min_weight": 0.1,  # Minimum weight for any strategy
        "vote_threshold": 0.6  # Threshold for weighted voting
    }
    
    required_indicators = ["close"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with strategy instances and performance tracking.
        
        Args:
            params: Optional parameters to override defaults
        """
        super().__init__(params)
        
        # Initialize strategies
        self.strategies = {}
        
        try:
            # Initialize all available strategies from registry
            for strategy_name, strategy_class in registry.get_all_strategies().items():
                # Avoid recursive initialization of ensemble strategies
                if strategy_name != self.name and "ensemble" not in strategy_class.category:
                    try:
                        self.strategies[strategy_name] = strategy_class()
                    except Exception as e:
                        logger.warning(f"Failed to initialize strategy {strategy_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize strategies from registry: {e}")
        
        # Initialize performance tracking
        self.strategy_weights = {name: self.params.get("initial_weight", self.default_params["initial_weight"]) 
                               for name in self.strategies}
        
        # Track historical signals for performance evaluation
        self.historical_signals = {name: [] for name in self.strategies}
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate adaptive ensemble signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Lookback window for performance evaluation
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        
        # Update weights based on historical performance
        if i >= lookback + 1:
            self._update_weights(df, i)
        
        # Track weighted votes for long and short
        long_votes = 0
        short_votes = 0
        total_weight = 0
        
        # Generate signals and store for future performance evaluation
        for name, strategy in self.strategies.items():
            weight = self.strategy_weights.get(name, self.params.get("initial_weight", self.default_params["initial_weight"]))
            
            try:
                if strategy.validate_dataframe(df):
                    # Try to generate conditions with explicit error handling
                    try:
                        conditions = strategy.generate_conditions(df, row, i)
                        
                        # Ensure conditions are valid and contain necessary keys
                        if not isinstance(conditions, dict):
                            logger.warning(f"Strategy {name} returned invalid conditions type: {type(conditions)}")
                            continue
                        
                        if "long" not in conditions or "short" not in conditions:
                            logger.warning(f"Strategy {name} returned incomplete conditions: {conditions.keys()}")
                            continue
                        
                        # Safely calculate votes based on conditions
                        long_conditions = conditions.get("long", [])
                        short_conditions = conditions.get("short", [])
                        
                        # Filter out None values from conditions
                        long_conditions = [c for c in long_conditions if c is not None]
                        short_conditions = [c for c in short_conditions if c is not None]
                        
                        # Calculate votes based on valid conditions
                        long_count = sum(long_conditions)
                        short_count = sum(short_conditions)
                        
                        # Normalize to 0-1 range if there are conditions
                        total_conditions_long = max(1, len(long_conditions))
                        total_conditions_short = max(1, len(short_conditions))
                        
                        long_vote = long_count / total_conditions_long if long_count > 0 else 0
                        short_vote = short_count / total_conditions_short if short_count > 0 else 0
                        
                        # Store signals for performance tracking
                        self.historical_signals[name].append({
                            "index": i,
                            "long": long_vote > 0,
                            "short": short_vote > 0
                        })
                        
                        # Keep historical signals limited to lookback window
                        if len(self.historical_signals[name]) > lookback:
                            self.historical_signals[name] = self.historical_signals[name][-lookback:]
                        
                        # Apply weight
                        long_votes += long_vote * weight
                        short_votes += short_vote * weight
                        total_weight += weight
                        
                    except Exception as inner_e:
                        logger.warning(f"Error processing conditions for strategy {name}: {inner_e}")
            except Exception as e:
                logger.warning(f"Error generating conditions for strategy {name}: {e}")
        
        # Normalize votes
        if total_weight > 0:
            long_votes /= total_weight
            short_votes /= total_weight
        
        # Get vote threshold
        threshold = self.params.get("vote_threshold", self.default_params["vote_threshold"])
        
        # Convert to conditions
        long_conditions = [long_votes >= threshold]
        short_conditions = [short_votes >= threshold]
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }
    
    def _update_weights(self, df: pd.DataFrame, current_index: int) -> None:
        """
        Update strategy weights based on historical performance.
        
        Args:
            df: DataFrame with price data
            current_index: Current index in the dataframe
        """
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        decay = self.params.get("performance_decay", self.default_params["performance_decay"])
        min_weight = self.params.get("min_weight", self.default_params["min_weight"])
        
        # Calculate performance for each strategy
        performance_scores = {}
        
        for name, signals in self.historical_signals.items():
            # Skip if not enough signals
            if len(signals) < 5:
                continue
                
            # Calculate profit/loss for each signal
            score = 0
            signal_count = 0
            
            for j, signal in enumerate(signals[:-1]):  # Skip the most recent signal
                idx = signal["index"]
                
                # Skip if we don't have enough future data
                if idx >= current_index - 1:
                    continue
                    
                # Calculate profit/loss
                if signal["long"]:
                    # For long signals: profit if price went up
                    try:
                        next_price = df["close"].iloc[idx + 1]
                        current_price = df["close"].iloc[idx]
                        pnl = (next_price / current_price - 1) * 100
                        score += pnl
                        signal_count += 1
                    except Exception as e:
                        logger.warning(f"Error calculating long PnL for strategy {name}: {e}")
                    
                elif signal["short"]:
                    # For short signals: profit if price went down
                    try:
                        next_price = df["close"].iloc[idx + 1]
                        current_price = df["close"].iloc[idx]
                        pnl = (1 - next_price / current_price) * 100
                        score += pnl
                        signal_count += 1
                    except Exception as e:
                        logger.warning(f"Error calculating short PnL for strategy {name}: {e}")
            
            # Calculate average score
            if signal_count > 0:
                performance_scores[name] = score / signal_count
            else:
                performance_scores[name] = 0
        
        # Update weights based on performance scores
        if performance_scores:
            # Normalize scores
            min_score = min(performance_scores.values())
            max_score = max(performance_scores.values())
            
            score_range = max_score - min_score
            for name, score in performance_scores.items():
                if score_range > 0:
                    # Linear mapping from score to weight
                    normalized_score = (score - min_score) / score_range
                    
                    # Apply decay to current weight and add new weight component
                    current_weight = self.strategy_weights.get(name, self.params.get("initial_weight", self.default_params["initial_weight"]))
                    new_weight = (current_weight * decay) + (normalized_score * (1 - decay))
                    
                    # Ensure minimum weight
                    self.strategy_weights[name] = max(min_weight, new_weight)