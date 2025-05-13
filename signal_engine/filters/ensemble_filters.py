"""
Ensemble filters for the trading system.
These filters combine multiple filtering techniques for more robust results.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from signal_engine.signal_filter_system import BaseFilter, FilterRuleRegistry

logger = logging.getLogger(__name__)


class VotingEnsembleFilter(BaseFilter):
    """Filter that applies multiple filters and uses voting to make final decisions."""
    
    name = "voting_ensemble_filter"
    display_name = "Voting Ensemble Filter"
    description = "Combines multiple filters using a voting mechanism"
    category = "ensemble"
    
    default_params = {
        "filters": [
            {"name": "market_regime_filter", "params": {}},
            {"name": "volatility_regime_filter", "params": {}},
            {"name": "trend_strength_filter", "params": {}}
        ],
        "voting_method": "majority",  # 'majority', 'unanimous', or 'weighted'
        "weights": {},  # Custom weights for 'weighted' method
        "threshold": 0.5  # Threshold for weighted voting (if method is 'weighted')
    }
    
    required_indicators = ["close"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with filter instances.
        
        Args:
            params: Optional parameters to override defaults
        """
        super().__init__(params)
        
        # Initialize filters
        self.filters = []
        
        try:
            # Get filter registry
            from signal_engine.filters import registry
            
            # Initialize filters from configuration
            filter_configs = self.params.get("filters", self.default_params["filters"])
            
            for config in filter_configs:
                filter_name = config.get("name")
                filter_params = config.get("params", {})
                
                # Create filter instance
                filter_instance = registry.create_filter(filter_name, filter_params)
                
                if filter_instance:
                    self.filters.append(filter_instance)
                else:
                    logger.warning(f"Filter {filter_name} not found in registry")
                    
        except Exception as e:
            logger.error(f"Error initializing ensemble filters: {e}")
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply ensemble of filters to signals using voting.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # If no filters available, return original signals
        if not self.filters:
            logger.warning("No filters available for ensemble")
            return signals
        
        # Get parameters
        voting_method = self.params.get("voting_method", self.default_params["voting_method"])
        weights = self.params.get("weights", self.default_params["weights"])
        threshold = self.params.get("threshold", self.default_params["threshold"])
        
        # Create a copy of the signals to hold filtered results
        filtered_signals = signals.copy()
        
        # Apply each filter and collect results
        filter_results = []
        for filter_instance in self.filters:
            try:
                result = filter_instance.apply(df, signals)
                filter_results.append(result)
            except Exception as e:
                logger.error(f"Error applying filter {filter_instance.name}: {e}")
        
        # Apply voting logic
        if voting_method == "unanimous":
            # Signal passes only if all filters pass it
            for i in range(len(signals)):
                if signals.iloc[i] != 0:  # Only check non-zero signals
                    # Check if any filter removed the signal
                    for result in filter_results:
                        if result.iloc[i] == 0:
                            filtered_signals.iloc[i] = 0
                            break
                            
        elif voting_method == "majority":
            # Signal passes if majority of filters pass it
            for i in range(len(signals)):
                if signals.iloc[i] != 0:  # Only check non-zero signals
                    # Count how many filters kept the signal
                    pass_count = sum(1 for result in filter_results if result.iloc[i] != 0)
                    
                    # If less than half passed, filter out
                    if pass_count < len(filter_results) / 2:
                        filtered_signals.iloc[i] = 0
                        
        elif voting_method == "weighted":
            # Signal passes based on weighted votes
            for i in range(len(signals)):
                if signals.iloc[i] != 0:  # Only check non-zero signals
                    # Calculate weighted votes
                    total_weight = 0
                    pass_weight = 0
                    
                    for j, result in enumerate(filter_results):
                        # Get filter name
                        filter_name = self.filters[j].name
                        
                        # Get weight for this filter (default to 1.0)
                        weight = weights.get(filter_name, 1.0)
                        total_weight += weight
                        
                        # Add to pass weight if filter passed the signal
                        if result.iloc[i] != 0:
                            pass_weight += weight
                    
                    # Calculate pass ratio
                    pass_ratio = pass_weight / total_weight if total_weight > 0 else 0
                    
                    # Filter out if below threshold
                    if pass_ratio < threshold:
                        filtered_signals.iloc[i] = 0
        
        return filtered_signals


class SequentialFilterChain(BaseFilter):
    """Filter that applies multiple filters in sequence."""
    
    name = "sequential_filter_chain"
    display_name = "Sequential Filter Chain"
    description = "Applies multiple filters in a sequential chain"
    category = "ensemble"
    
    default_params = {
        "filters": [
            {"name": "market_regime_filter", "params": {}},
            {"name": "volatility_regime_filter", "params": {}},
            {"name": "trend_strength_filter", "params": {}}
        ],
        "early_stopping": True  # Stop if a filter removes all signals
    }
    
    required_indicators = ["close"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize with filter instances.
        
        Args:
            params: Optional parameters to override defaults
        """
        super().__init__(params)
        
        # Initialize filters
        self.filters = []
        
        try:
            # Get filter registry
            from signal_engine.filters import registry
            
            # Initialize filters from configuration
            filter_configs = self.params.get("filters", self.default_params["filters"])
            
            for config in filter_configs:
                filter_name = config.get("name")
                filter_params = config.get("params", {})
                
                # Create filter instance
                filter_instance = registry.create_filter(filter_name, filter_params)
                
                if filter_instance:
                    self.filters.append(filter_instance)
                else:
                    logger.warning(f"Filter {filter_name} not found in registry")
                    
        except Exception as e:
            logger.error(f"Error initializing sequential filters: {e}")
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply filters in sequence to the signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # If no filters available, return original signals
        if not self.filters:
            logger.warning("No filters available for chain")
            return signals
        
        # Get parameters
        early_stopping = self.params.get("early_stopping", self.default_params["early_stopping"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Apply filters in sequence
        for filter_instance in self.filters:
            try:
                filtered_signals = filter_instance.apply(df, filtered_signals)
                
                # Check if all signals have been filtered out
                if early_stopping and filtered_signals.sum() == 0:
                    logger.info(f"Early stopping after filter {filter_instance.name} removed all signals")
                    break
                    
            except Exception as e:
                logger.error(f"Error applying filter {filter_instance.name}: {e}")
        
        return filtered_signals


class WeightedMetaFilter(BaseFilter):
    """Meta-filter that combines signals from multiple strategies with adaptive weights."""
    
    name = "weighted_meta_filter"
    display_name = "Weighted Meta-Filter"
    description = "Combines and filters signals from multiple strategies with adaptive weights"
    category = "ensemble"
    
    default_params = {
        "strategy_weights": {
            "trend_following": 1.0,
            "mtf_trend": 1.0,
            "adaptive_trend": 1.0,
            "overextended_reversal": 0.8,
            "pattern_reversal": 0.8,
            "divergence_reversal": 0.8,
            "volatility_breakout": 0.9,
            "range_breakout": 0.9,
            "sr_breakout": 0.9
        },
        "regime_weight_adjustments": {
            "strong_uptrend": {"trend": 1.2, "reversal": 0.7, "breakout": 0.9},
            "weak_uptrend": {"trend": 1.1, "reversal": 0.8, "breakout": 0.9},
            "ranging": {"trend": 0.7, "reversal": 1.0, "breakout": 1.2},
            "weak_downtrend": {"trend": 1.1, "reversal": 0.8, "breakout": 0.9},
            "strong_downtrend": {"trend": 1.2, "reversal": 0.7, "breakout": 0.9},
            "volatile": {"trend": 0.8, "reversal": 0.8, "breakout": 1.2},
            "overbought": {"trend": 0.7, "reversal": 1.2, "breakout": 0.9},
            "oversold": {"trend": 0.7, "reversal": 1.2, "breakout": 0.9}
        },
        "strategy_categories": {
            "trend": ["trend_following", "mtf_trend", "adaptive_trend"],
            "reversal": ["overextended_reversal", "pattern_reversal", "divergence_reversal"],
            "breakout": ["volatility_breakout", "range_breakout", "sr_breakout"]
        },
        "threshold": 0.6  # Minimum weight threshold for a signal to pass
    }
    
    required_indicators = ["close"]
    optional_indicators = ["market_regime", "strategy_name", "signal_strength"]
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply weighted meta-filtering to signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Validate dataframe
        if not self.validate_dataframe(df):
            return signals
        
        # Get parameters
        strategy_weights = self.params.get("strategy_weights", self.default_params["strategy_weights"])
        regime_adjustments = self.params.get("regime_weight_adjustments", self.default_params["regime_weight_adjustments"])
        strategy_categories = self.params.get("strategy_categories", self.default_params["strategy_categories"])
        threshold = self.params.get("threshold", self.default_params["threshold"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Check if we have strategy information
        has_strategy_info = "strategy_name" in df.columns
        
        # Apply weighted filtering
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Get strategy name
            strategy_name = None
            if has_strategy_info:
                strategy_name = df["strategy_name"].iloc[i]
            
            # If no strategy name, skip
            if not strategy_name:
                continue
            
            # Get base weight for this strategy
            weight = strategy_weights.get(strategy_name, 1.0)
            
            # Get strategy category
            strategy_category = "unknown"
            for category, strategies in strategy_categories.items():
                if strategy_name in strategies:
                    strategy_category = category
                    break
            
            # Adjust weight based on market regime
            if "market_regime" in df.columns:
                regime = df["market_regime"].iloc[i]
                
                if regime in regime_adjustments:
                    regime_adjustment = regime_adjustments[regime]
                    # Apply adjustment based on strategy category
                    category_adjustment = regime_adjustment.get(strategy_category, 1.0)
                    weight *= category_adjustment
            
            # Apply signal strength if available
            if "signal_strength" in df.columns:
                signal_strength = df["signal_strength"].iloc[i] / 100  # 0-1 scale
                
                # Combine weight with signal strength
                combined_weight = weight * signal_strength
                
                # Filter out if below threshold
                if combined_weight < threshold:
                    filtered_signals.iloc[i] = 0
            else:
                # No signal strength available, use weight directly
                if weight < threshold:
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals