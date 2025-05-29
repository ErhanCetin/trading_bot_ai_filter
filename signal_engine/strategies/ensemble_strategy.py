"""
Ensemble strategies for the trading system.
These strategies combine multiple sub-strategies for more robust signals.
FIXED VERSION - Corrected strategy dependencies and fallback logic
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Type
import logging

from signal_engine.signal_strategy_system import BaseStrategy, StrategyRegistry

logger = logging.getLogger(__name__)


class EnsembleStrategyBase(BaseStrategy):
    """Base class for all ensemble strategies with improved registry management"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        
        # Safe registry import with fallback
        self.registry = self._get_registry()
        
        # Lazy loading - strategies loaded only when needed
        self.strategy_instances = {}
        self.available_strategies = self._get_available_strategies()
        
        # Performance tracking
        self.strategy_performance = {}
        
    def _get_registry(self) -> StrategyRegistry:
        """Safely get strategy registry with proper fallback"""
        try:
            # Try to import the global registry
            from signal_engine.strategies import registry as global_registry
            if global_registry and len(global_registry.get_all_strategies()) > 0:
                logger.debug("Using global strategy registry")
                return global_registry
            else:
                logger.warning("Global registry is empty, creating fallback registry")
                return self._create_fallback_registry()
        except ImportError:
            logger.warning("Cannot import global registry, creating fallback")
            return self._create_fallback_registry()
    
    def _create_fallback_registry(self) -> StrategyRegistry:
        """Create fallback registry with essential strategies"""
        fallback_registry = StrategyRegistry()
        
        try:
            # Import and register essential strategies manually
            from signal_engine.strategies.trend_strategy import (
                TrendFollowingStrategy, AdaptiveTrendStrategy
            )
            from signal_engine.strategies.reversal_strategy import (
                OverextendedReversalStrategy, PatternReversalStrategy
            )
            from signal_engine.strategies.breakout_strategy import (
                VolatilityBreakoutStrategy, RangeBreakoutStrategy
            )
            
            fallback_registry.register(TrendFollowingStrategy)
            fallback_registry.register(AdaptiveTrendStrategy)
            fallback_registry.register(OverextendedReversalStrategy)
            fallback_registry.register(PatternReversalStrategy)
            fallback_registry.register(VolatilityBreakoutStrategy)
            fallback_registry.register(RangeBreakoutStrategy)
            
            logger.info("Fallback registry created with essential strategies")
        except ImportError as e:
            logger.error(f"Failed to create fallback registry: {e}")
        
        return fallback_registry
    
    def _get_available_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """Get available strategies excluding other ensemble strategies"""
        available = {}
        
        for name, strategy_class in self.registry.get_all_strategies().items():
            # Exclude self and other ensemble strategies to prevent recursion
            if (name != self.name and 
                hasattr(strategy_class, 'category') and 
                strategy_class.category != "ensemble"):
                available[name] = strategy_class
        
        return available
    
    def _get_strategy_instance(self, strategy_name: str, 
                             params: Optional[Dict] = None) -> Optional[BaseStrategy]:
        """Lazy load strategy instance when needed"""
        cache_key = f"{strategy_name}_{hash(str(params)) if params else 'default'}"
        
        if cache_key not in self.strategy_instances:
            if strategy_name in self.available_strategies:
                try:
                    strategy_class = self.available_strategies[strategy_name]
                    self.strategy_instances[cache_key] = strategy_class(params)
                    logger.debug(f"Loaded strategy: {strategy_name}")
                except Exception as e:
                    logger.error(f"Failed to load strategy {strategy_name}: {e}")
                    return None
            else:
                logger.warning(f"Strategy {strategy_name} not available")
                return None
        
        return self.strategy_instances.get(cache_key)
    
    def _calculate_vote_strength(self, conditions: List[bool]) -> float:
        """Calculate vote strength from conditions list"""
        if not conditions:
            return 0.0
        
        # Filter out None values
        valid_conditions = [c for c in conditions if c is not None]
        
        if not valid_conditions:
            return 0.0
        
        true_count = sum(1 for c in valid_conditions if c)
        total_count = len(valid_conditions)
        
        # Vote strength is the confidence level
        confidence = true_count / total_count
        
        # Apply minimum threshold to avoid weak signals
        min_confidence = 0.5
        if confidence < min_confidence:
            return 0.0
        
        return confidence
    
    def _validate_conditions(self, conditions: Dict) -> bool:
        """Validate that conditions dict has proper structure"""
        return (isinstance(conditions, dict) and 
                "long" in conditions and 
                "short" in conditions and
                isinstance(conditions["long"], list) and
                isinstance(conditions["short"], list))


class WeightedVotingEnsembleStrategy(EnsembleStrategyBase):
    """Strategy that combines signals using proper weighted voting"""
    
    name = "weighted_voting"
    display_name = "Weighted Voting Ensemble Strategy"
    description = "Combines signals using mathematically correct weighted voting"
    category = "ensemble"
    
    default_params = {
        "strategy_weights": {
            "trend_following": 1.0,
            "adaptive_trend": 0.8,
            "overextended_reversal": 1.0,
            "pattern_reversal": 0.7,
            "volatility_breakout": 0.9,
            "range_breakout": 0.8
        },
        "vote_threshold": 0.6,           # Vote threshold for signal generation
        "min_strategies": 2,             # Minimum strategies that must provide signals
        "conflict_resolution": "strength" # How to resolve long/short conflicts: "strength", "threshold", "difference"
    }
    
    required_indicators = ["close"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate ensemble conditions using proper weighted voting"""
        
        strategy_weights = self.params.get("strategy_weights", self.default_params["strategy_weights"])
        vote_threshold = self.params.get("vote_threshold", self.default_params["vote_threshold"])
        min_strategies = self.params.get("min_strategies", self.default_params["min_strategies"])
        
        # Track individual strategy votes
        strategy_votes = {
            "long": {},   # {strategy_name: vote_strength}
            "short": {}   # {strategy_name: vote_strength}
        }
        
        total_available_weight = 0
        active_strategies = 0
        
        # Collect votes from each strategy
        for strategy_name, weight in strategy_weights.items():
            strategy = self._get_strategy_instance(strategy_name)
            
            if not strategy or not strategy.validate_dataframe(df):
                continue
            
            try:
                # Get strategy conditions
                conditions = strategy.generate_conditions(df, row, i)
                
                if not self._validate_conditions(conditions):
                    continue
                
                # Calculate vote strength for this strategy
                long_strength = self._calculate_vote_strength(conditions.get('long', []))
                short_strength = self._calculate_vote_strength(conditions.get('short', []))
                
                # Store votes if strategy has opinion
                if long_strength > 0:
                    strategy_votes["long"][strategy_name] = long_strength
                if short_strength > 0:
                    strategy_votes["short"][strategy_name] = short_strength
                
                # Track active strategies
                if long_strength > 0 or short_strength > 0:
                    total_available_weight += weight
                    active_strategies += 1
                    
            except Exception as e:
                logger.debug(f"Error getting votes from strategy {strategy_name}: {e}")
                continue
        
        # Check if we have enough active strategies
        if active_strategies < min_strategies:
            return {"long": [], "short": []}
        
        # Calculate weighted ensemble votes
        ensemble_votes = self._calculate_ensemble_votes(
            strategy_votes, strategy_weights, total_available_weight
        )
        
        # Apply vote threshold and conflict resolution
        final_conditions = self._resolve_voting_conflicts(
            ensemble_votes, vote_threshold
        )
        
        return final_conditions
    
    def _calculate_ensemble_votes(self, strategy_votes: Dict, 
                                strategy_weights: Dict, 
                                total_weight: float) -> Dict[str, float]:
        """Calculate proper weighted ensemble votes"""
        ensemble_votes = {"long": 0.0, "short": 0.0}
        
        if total_weight == 0:
            return ensemble_votes
        
        # Calculate weighted long votes
        for strategy_name, vote_strength in strategy_votes["long"].items():
            weight = strategy_weights.get(strategy_name, 1.0)
            ensemble_votes["long"] += vote_strength * weight
        
        # Calculate weighted short votes  
        for strategy_name, vote_strength in strategy_votes["short"].items():
            weight = strategy_weights.get(strategy_name, 1.0)
            ensemble_votes["short"] += vote_strength * weight
        
        # Normalize by total available weight
        ensemble_votes["long"] /= total_weight
        ensemble_votes["short"] /= total_weight
        
        return ensemble_votes
    
    def _resolve_voting_conflicts(self, ensemble_votes: Dict[str, float], 
                                threshold: float) -> Dict[str, List[bool]]:
        """Resolve conflicts between long and short votes"""
        conflict_resolution = self.params.get("conflict_resolution", "strength")
        
        long_vote = ensemble_votes["long"]
        short_vote = ensemble_votes["short"]
        
        long_conditions = []
        short_conditions = []
        
        if conflict_resolution == "strength":
            # Choose the stronger signal if it exceeds threshold
            if long_vote > threshold and long_vote > short_vote:
                long_conditions.append(True)
            elif short_vote > threshold and short_vote > long_vote:
                short_conditions.append(True)
                
        elif conflict_resolution == "threshold":
            # Both signals must independently exceed threshold
            if long_vote > threshold:
                long_conditions.append(True)
            if short_vote > threshold:
                short_conditions.append(True)
                
        elif conflict_resolution == "difference":
            # Require significant difference between signals
            min_difference = 0.2
            if (long_vote > threshold and 
                long_vote - short_vote > min_difference):
                long_conditions.append(True)
            elif (short_vote > threshold and 
                  short_vote - long_vote > min_difference):
                short_conditions.append(True)
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class RegimeBasedEnsembleStrategy(EnsembleStrategyBase):
    """Strategy that selects appropriate sub-strategies based on market regime"""
    
    name = "regime_ensemble"
    display_name = "Regime-Based Ensemble Strategy"
    description = "Dynamically weights strategies based on market regime"
    category = "ensemble"
    
    default_params = {
        "regime_weights": {
            "strong_uptrend": {"trend": 0.8, "reversal": 0.1, "breakout": 0.1},
            "weak_uptrend": {"trend": 0.5, "reversal": 0.3, "breakout": 0.2},
            "strong_downtrend": {"trend": 0.8, "reversal": 0.1, "breakout": 0.1},
            "weak_downtrend": {"trend": 0.5, "reversal": 0.3, "breakout": 0.2},
            "ranging": {"trend": 0.2, "reversal": 0.4, "breakout": 0.4},
            "volatile": {"trend": 0.3, "reversal": 0.3, "breakout": 0.4},
            "overbought": {"trend": 0.1, "reversal": 0.7, "breakout": 0.2},
            "oversold": {"trend": 0.1, "reversal": 0.7, "breakout": 0.2},
            "unknown": {"trend": 0.33, "reversal": 0.33, "breakout": 0.34}
        },
        "strategy_category_mapping": {
            "trend": ["trend_following", "adaptive_trend"],
            "reversal": ["overextended_reversal", "pattern_reversal"],
            "breakout": ["volatility_breakout", "range_breakout"]
        },
        "vote_threshold": 0.6
    }
    
    required_indicators = ["close"]
    optional_indicators = ["market_regime"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate conditions based on current market regime"""
        
        # Determine current market regime
        regime = self._determine_market_regime(row)
        logger.debug(f"Detected market regime: {regime}")
        
        # Get regime-specific weights
        regime_weights = self.params.get("regime_weights", 
                                       self.default_params["regime_weights"])
        category_weights = regime_weights.get(regime, regime_weights["unknown"])
        
        # Get strategy mapping
        strategy_mapping = self.params.get("strategy_category_mapping",
                                         self.default_params["strategy_category_mapping"])
        
        # Calculate votes with regime-based weighting
        ensemble_votes = {"long": 0.0, "short": 0.0}
        total_weight = 0.0
        
        for category, weight in category_weights.items():
            if category not in strategy_mapping:
                continue
                
            # Get strategies for this category
            category_strategies = strategy_mapping[category]
            category_vote_long = 0.0
            category_vote_short = 0.0
            active_strategies = 0
            
            for strategy_name in category_strategies:
                strategy = self._get_strategy_instance(strategy_name)
                
                if not strategy or not strategy.validate_dataframe(df):
                    continue
                
                try:
                    conditions = strategy.generate_conditions(df, row, i)
                    
                    if self._validate_conditions(conditions):
                        long_strength = self._calculate_vote_strength(conditions.get('long', []))
                        short_strength = self._calculate_vote_strength(conditions.get('short', []))
                        
                        if long_strength > 0 or short_strength > 0:
                            category_vote_long += long_strength
                            category_vote_short += short_strength
                            active_strategies += 1
                            
                except Exception as e:
                    logger.debug(f"Error processing strategy {strategy_name}: {e}")
                    continue
            
            # Average category votes and apply regime weight
            if active_strategies > 0:
                category_vote_long /= active_strategies
                category_vote_short /= active_strategies
                
                ensemble_votes["long"] += category_vote_long * weight
                ensemble_votes["short"] += category_vote_short * weight
                total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            ensemble_votes["long"] /= total_weight
            ensemble_votes["short"] /= total_weight
        
        # Apply threshold
        vote_threshold = self.params.get("vote_threshold", self.default_params["vote_threshold"])
        
        return {
            "long": [ensemble_votes["long"] > vote_threshold],
            "short": [ensemble_votes["short"] > vote_threshold]
        }
    
    def _determine_market_regime(self, row: pd.Series) -> str:
        """Determine current market regime with fallback logic"""
        
        # Try direct market regime indicator
        if "market_regime" in row and not pd.isna(row["market_regime"]):
            regime_value = str(row["market_regime"]).lower().strip()
            regime_weights = self.params.get("regime_weights", 
                                           self.default_params["regime_weights"])
            if regime_value in regime_weights:
                return regime_value
        
        # Fallback: analyze available indicators
        return self._analyze_market_regime(row)
    
    def _analyze_market_regime(self, row: pd.Series) -> str:
        """Analyze market regime from available indicators"""
        
        # Simple regime detection logic
        regime_indicators = {
            "trend_strength": 0,  # -1 to 1
            "volatility": 0,      # 0 to 1  
            "momentum": 0         # -1 to 1
        }
        
        # Analyze trend strength
        if "adx" in row and not pd.isna(row["adx"]):
            if row["adx"] > 30:
                regime_indicators["trend_strength"] = 1
            elif row["adx"] < 20:
                regime_indicators["trend_strength"] = -1
        
        # Analyze volatility
        if "atr_percent" in row and not pd.isna(row["atr_percent"]):
            if row["atr_percent"] > 2.0:
                regime_indicators["volatility"] = 1
            elif row["atr_percent"] < 0.5:
                regime_indicators["volatility"] = -1
        
        # Analyze momentum
        if "rsi_14" in row and not pd.isna(row["rsi_14"]):
            if row["rsi_14"] > 70:
                regime_indicators["momentum"] = 1
            elif row["rsi_14"] < 30:
                regime_indicators["momentum"] = -1
        
        # Determine regime based on indicators
        if regime_indicators["trend_strength"] > 0:
            if regime_indicators["momentum"] > 0:
                return "strong_uptrend"
            elif regime_indicators["momentum"] < 0:
                return "strong_downtrend"
            else:
                return "weak_uptrend" if regime_indicators["momentum"] >= 0 else "weak_downtrend"
        elif regime_indicators["volatility"] > 0:
            return "volatile"
        else:
            return "ranging"


class AdaptiveEnsembleStrategy(EnsembleStrategyBase):
    """Strategy that adapts weights based on recent performance of sub-strategies"""
    
    name = "adaptive_ensemble"
    display_name = "Adaptive Ensemble Strategy"
    description = "Adapts weights based on recent performance of sub-strategies"
    category = "ensemble"
    
    default_params = {
        "lookback_window": 50,     # Window to evaluate strategy performance
        "performance_decay": 0.95, # Decay factor for older performance
        "initial_weight": 1.0,     # Initial weight for all strategies
        "min_weight": 0.1,         # Minimum weight for any strategy
        "max_weight": 3.0,         # Maximum weight for any strategy
        "vote_threshold": 0.6,     # Threshold for weighted voting
        "adaptation_frequency": 10  # How often to update weights (in bars)
    }
    
    required_indicators = ["close"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with performance tracking"""
        super().__init__(params)
        
        # Initialize performance tracking
        initial_weight = self.params.get("initial_weight", self.default_params["initial_weight"])
        self.strategy_weights = {name: initial_weight for name in self.available_strategies}
        
        # Track historical signals for performance evaluation (memory-efficient)
        max_history = self.params.get("lookback_window", 50) * 2
        self.historical_signals = {}
        self.max_history = max_history
        
        # Track when weights were last updated
        self.last_weight_update = -1
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate adaptive ensemble signal conditions"""
        
        # Update weights periodically based on performance
        adaptation_freq = self.params.get("adaptation_frequency", self.default_params["adaptation_frequency"])
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        
        if (i >= lookback + 1 and 
            i - self.last_weight_update >= adaptation_freq):
            self._update_weights(df, i)
            self.last_weight_update = i
        
        # Track weighted votes for long and short
        long_votes = 0.0
        short_votes = 0.0
        total_weight = 0.0
        
        # Generate signals and store for future performance evaluation
        for strategy_name in self.available_strategies:
            weight = self.strategy_weights.get(strategy_name, 
                                             self.params.get("initial_weight", 
                                                           self.default_params["initial_weight"]))
            
            strategy = self._get_strategy_instance(strategy_name)
            
            if not strategy or not strategy.validate_dataframe(df):
                continue
            
            try:
                conditions = strategy.generate_conditions(df, row, i)
                
                if not self._validate_conditions(conditions):
                    continue
                
                # Calculate vote strengths
                long_strength = self._calculate_vote_strength(conditions.get('long', []))
                short_strength = self._calculate_vote_strength(conditions.get('short', []))
                
                # Store signals for performance tracking
                self._store_signal(strategy_name, {
                    "index": i,
                    "long": long_strength > 0,
                    "short": short_strength > 0,
                    "long_strength": long_strength,
                    "short_strength": short_strength
                })
                
                # Apply weight to votes
                long_votes += long_strength * weight
                short_votes += short_strength * weight
                total_weight += weight
                
            except Exception as e:
                logger.debug(f"Error processing strategy {strategy_name}: {e}")
                continue
        
        # Normalize votes
        if total_weight > 0:
            long_votes /= total_weight
            short_votes /= total_weight
        
        # Apply vote threshold
        vote_threshold = self.params.get("vote_threshold", self.default_params["vote_threshold"])
        
        return {
            "long": [long_votes > vote_threshold],
            "short": [short_votes > vote_threshold]
        }
    
    def _store_signal(self, strategy_name: str, signal_data: Dict) -> None:
        """Store signal with automatic memory cleanup"""
        if strategy_name not in self.historical_signals:
            self.historical_signals[strategy_name] = []
        
        self.historical_signals[strategy_name].append(signal_data)
        
        # Limit memory usage
        if len(self.historical_signals[strategy_name]) > self.max_history:
            self.historical_signals[strategy_name] = \
                self.historical_signals[strategy_name][-self.max_history:]
    
    def _update_weights(self, df: pd.DataFrame, current_index: int) -> None:
        """Update strategy weights based on historical performance"""
        
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        decay = self.params.get("performance_decay", self.default_params["performance_decay"])
        min_weight = self.params.get("min_weight", self.default_params["min_weight"])
        max_weight = self.params.get("max_weight", self.default_params["max_weight"])
        
        # Calculate performance for each strategy
        performance_scores = {}
        
        for strategy_name, signals in self.historical_signals.items():
            if len(signals) < 5:  # Need minimum signals for evaluation
                continue
            
            score = self._calculate_strategy_performance(signals, df, current_index, lookback)
            performance_scores[strategy_name] = score
        
        # Update weights based on performance scores
        if performance_scores:
            self._apply_performance_weights(performance_scores, decay, min_weight, max_weight)
        
        logger.debug(f"Updated strategy weights: {self.strategy_weights}")
    
    def _calculate_strategy_performance(self, signals: List[Dict], 
                                      df: pd.DataFrame, 
                                      current_index: int, 
                                      lookback: int) -> float:
        """Calculate performance score for a strategy based on historical signals"""
        
        total_score = 0.0
        signal_count = 0
        
        # Evaluate signals within lookback window
        start_idx = max(0, current_index - lookback)
        
        for signal in signals:
            signal_idx = signal["index"]
            
            # Skip signals outside lookback window or too recent to evaluate
            if signal_idx < start_idx or signal_idx >= current_index - 1:
                continue
            
            try:
                # Calculate forward return
                current_price = df["close"].iloc[signal_idx]
                future_price = df["close"].iloc[signal_idx + 1]
                
                if pd.isna(current_price) or pd.isna(future_price):
                    continue
                
                forward_return = (future_price / current_price - 1) * 100
                
                # Score based on signal direction and actual return
                if signal["long"] and forward_return > 0:
                    total_score += forward_return * signal["long_strength"]
                    signal_count += 1
                elif signal["short"] and forward_return < 0:
                    total_score += abs(forward_return) * signal["short_strength"]
                    signal_count += 1
                elif signal["long"] and forward_return < 0:
                    total_score += forward_return * signal["long_strength"]  # Negative score
                    signal_count += 1
                elif signal["short"] and forward_return > 0:
                    total_score -= forward_return * signal["short_strength"]  # Negative score
                    signal_count += 1
                    
            except (IndexError, KeyError, TypeError):
                continue
        
        # Return average score
        return total_score / signal_count if signal_count > 0 else 0.0
    
    def _apply_performance_weights(self, performance_scores: Dict[str, float], 
                                 decay: float, min_weight: float, max_weight: float) -> None:
        """Apply performance-based weight adjustments"""
        
        if not performance_scores:
            return
        
        # Normalize scores to avoid extreme weights
        scores = list(performance_scores.values())
        if len(scores) > 1:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Avoid extreme adjustments
            if std_score > 0:
                normalized_scores = {
                    name: (score - mean_score) / std_score
                    for name, score in performance_scores.items()
                }
            else:
                normalized_scores = {name: 0.0 for name in performance_scores}
        else:
            normalized_scores = {name: 0.0 for name in performance_scores}
        
        # Update weights
        for strategy_name, normalized_score in normalized_scores.items():
            current_weight = self.strategy_weights.get(strategy_name, 1.0)
            
            # Calculate adjustment factor (sigmoid-like function)
            adjustment = 1.0 + (normalized_score * 0.3)  # Max 30% adjustment
            
            # Apply decay and adjustment
            new_weight = (current_weight * decay) + (adjustment * (1 - decay))
            
            # Clamp to min/max bounds
            self.strategy_weights[strategy_name] = max(min_weight, min(max_weight, new_weight))