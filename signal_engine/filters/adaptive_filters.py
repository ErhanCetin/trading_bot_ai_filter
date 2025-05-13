"""
Adaptive filters for the trading system.
These filters dynamically adjust their behavior based on market conditions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from signal_engine.signal_filter_system import BaseFilter

logger = logging.getLogger(__name__)


class DynamicThresholdFilter(BaseFilter):
    """Filter that dynamically adjusts thresholds based on market conditions."""
    
    name = "dynamic_threshold_filter"
    display_name = "Dynamic Threshold Filter"
    description = "Adjusts filtering thresholds based on current market conditions"
    category = "adaptive"
    
    default_params = {
        "base_threshold": 0.6,  # Base threshold value
        "volatility_impact": 0.2,  # Impact of volatility on threshold
        "trend_impact": 0.2,  # Impact of trend strength on threshold
        "min_threshold": 0.4,  # Minimum threshold value
        "max_threshold": 0.8   # Maximum threshold value
    }
    
    required_indicators = ["close"]
    optional_indicators = ["volatility_percentile", "trend_strength", "market_regime"]
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply dynamic threshold filter to signals.
        
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
        base_threshold = self.params.get("base_threshold", self.default_params["base_threshold"])
        vol_impact = self.params.get("volatility_impact", self.default_params["volatility_impact"])
        trend_impact = self.params.get("trend_impact", self.default_params["trend_impact"])
        min_threshold = self.params.get("min_threshold", self.default_params["min_threshold"])
        max_threshold = self.params.get("max_threshold", self.default_params["max_threshold"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Calculate dynamic thresholds for each row
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Start with base threshold
            threshold = base_threshold
            
            # Adjust based on volatility
            if "volatility_percentile" in df.columns:
                vol_percentile = df["volatility_percentile"].iloc[i] / 100  # 0-1 scale
                # Higher volatility = higher threshold
                threshold += (vol_percentile - 0.5) * vol_impact
            
            # Adjust based on trend strength
            if "trend_strength" in df.columns:
                trend_strength = df["trend_strength"].iloc[i] / 100  # 0-1 scale
                
                # Strong trend = lower threshold for trend-following
                signal_direction = signals.iloc[i]
                trend_direction = 1
                if "trend_direction" in df.columns:
                    trend_direction = df["trend_direction"].iloc[i]
                
                # If signal follows trend, reduce threshold
                if (signal_direction > 0 and trend_direction > 0) or \
                   (signal_direction < 0 and trend_direction < 0):
                    threshold -= trend_strength * trend_impact
                else:
                    # If signal against trend, increase threshold
                    threshold += trend_strength * trend_impact
            
            # Adjust based on market regime
            if "market_regime" in df.columns:
                regime = df["market_regime"].iloc[i]
                signal_direction = signals.iloc[i]
                
                # In strong trend regimes, lower threshold for trend-following
                if regime in ["strong_uptrend", "strong_downtrend"]:
                    # Check if signal aligns with trend
                    if (regime == "strong_uptrend" and signal_direction > 0) or \
                       (regime == "strong_downtrend" and signal_direction < 0):
                        threshold -= 0.1
                
                # In ranging regimes, increase threshold for all signals
                elif regime == "ranging":
                    threshold += 0.1
                
                # In volatile regimes, increase threshold
                elif regime == "volatile":
                    threshold += 0.15
            
            # Ensure threshold is within bounds
            threshold = max(min_threshold, min(max_threshold, threshold))
            
            # Apply threshold to signal
            # This is a placeholder for actual threshold application
            # In a real system, you would check signal strength or other metrics
            # For example:
            if "signal_strength" in df.columns:
                # If signal strength below threshold, filter out
                if df["signal_strength"].iloc[i] / 100 < threshold:
                    filtered_signals.iloc[i] = 0
            
            # If no strength measure available, use a random value for demonstration
            else:
                # Simulate signal quality (0-1)
                signal_quality = np.random.random()
                if signal_quality < threshold:
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals


class ContextAwareFilter(BaseFilter):
    """Filter that adapts its behavior based on the broader market context."""
    
    name = "context_aware_filter"
    display_name = "Context-Aware Filter"
    description = "Adapts filtering rules based on the broader market context"
    category = "adaptive"
    
    default_params = {
        "context_rules": {
            "trending": {
                "trend_follow_threshold": 0.4,
                "mean_reversion_threshold": 0.7
            },
            "ranging": {
                "trend_follow_threshold": 0.7,
                "mean_reversion_threshold": 0.4
            },
            "volatile": {
                "trend_follow_threshold": 0.6,
                "mean_reversion_threshold": 0.6
            }
        },
        "strategy_categories": {
            "trend_following": ["trend_following", "mtf_trend", "adaptive_trend"],
            "mean_reversion": ["overextended_reversal", "pattern_reversal", "divergence_reversal"],
            "breakout": ["volatility_breakout", "range_breakout", "sr_breakout"]
        }
    }
    
    required_indicators = ["market_regime"]
    optional_indicators = ["signal_strength", "strategy_name"]
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply context-aware filter to signals.
        
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
        context_rules = self.params.get("context_rules", self.default_params["context_rules"])
        strategy_categories = self.params.get("strategy_categories", self.default_params["strategy_categories"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Apply context-aware filtering
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Get market regime (context)
            market_regime = df["market_regime"].iloc[i]
            
            # Map regime to context
            context = "trending"
            if market_regime in ["strong_uptrend", "weak_uptrend", "strong_downtrend", "weak_downtrend"]:
                context = "trending"
            elif market_regime in ["ranging"]:
                context = "ranging"
            elif market_regime in ["volatile", "overbought", "oversold"]:
                context = "volatile"
            
            # Get strategy category
            strategy_category = "unknown"
            if "strategy_name" in df.columns:
                strategy_name = df["strategy_name"].iloc[i]
                for category, strategies in strategy_categories.items():
                    if strategy_name in strategies:
                        strategy_category = category
                        break
            
            # Get appropriate threshold based on context and strategy
            threshold = 0.5  # Default threshold
            
            if context in context_rules:
                context_rule = context_rules[context]
                
                if strategy_category == "trend_following":
                    threshold = context_rule.get("trend_follow_threshold", 0.5)
                elif strategy_category == "mean_reversion":
                    threshold = context_rule.get("mean_reversion_threshold", 0.5)
                elif strategy_category == "breakout":
                    # For breakout strategies, use different thresholds based on regime
                    if market_regime == "volatile":
                        threshold = 0.3  # Lower threshold in volatile markets
                    elif market_regime == "ranging":
                        threshold = 0.4  # Lower threshold in ranging markets
                    else:
                        threshold = 0.6  # Higher threshold in trending markets
            
            # Apply threshold to signal
            if "signal_strength" in df.columns:
                # If signal strength below threshold, filter out
                if df["signal_strength"].iloc[i] / 100 < threshold:
                    filtered_signals.iloc[i] = 0
            else:
                # No strength measure available, use a random value for demonstration
                signal_quality = np.random.random()
                if signal_quality < threshold:
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals


class MarketCycleFilter(BaseFilter):
    """Filter that adapts based on identified market cycles."""
    
    name = "market_cycle_filter"
    display_name = "Market Cycle Filter"
    description = "Adapts filtering rules based on identified market cycles"
    category = "adaptive"
    
    default_params = {
        "cycle_rules": {
            "accumulation": {
                "long_threshold": 0.4,
                "short_threshold": 0.7
            },
            "markup": {
                "long_threshold": 0.3,
                "short_threshold": 0.8
            },
            "distribution": {
                "long_threshold": 0.7,
                "short_threshold": 0.4
            },
            "markdown": {
                "long_threshold": 0.8,
                "short_threshold": 0.3
            }
        },
        "default_thresholds": {
            "long_threshold": 0.5,
            "short_threshold": 0.5
        }
    }
    
    required_indicators = ["close"]
    optional_indicators = ["market_cycle", "signal_strength", "volume_trend"]
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply market cycle filter to signals.
        
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
        cycle_rules = self.params.get("cycle_rules", self.default_params["cycle_rules"])
        default_thresholds = self.params.get("default_thresholds", self.default_params["default_thresholds"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Check if market cycle indicator is available
        has_market_cycle = "market_cycle" in df.columns
        
        # Apply market cycle filtering
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Determine signal direction
            is_long = signals.iloc[i] > 0
            
            # Get market cycle if available
            market_cycle = None
            if has_market_cycle:
                market_cycle = df["market_cycle"].iloc[i]
            
            # If market cycle not available, try to infer from other indicators
            if market_cycle is None:
                # Example inference using price and volume
                if "volume_trend" in df.columns:
                    volume_trend = df["volume_trend"].iloc[i]
                    
                    # Check recent price action (simple trend detection)
                    price_increasing = False
                    if i >= 10:
                        price_increasing = df["close"].iloc[i] > df["close"].iloc[i-10]
                    
                    # Infer market cycle based on price and volume
                    if price_increasing and volume_trend > 0:
                        market_cycle = "markup"  # Rising price, rising volume
                    elif price_increasing and volume_trend <= 0:
                        market_cycle = "distribution"  # Rising price, falling volume
                    elif not price_increasing and volume_trend > 0:
                        market_cycle = "accumulation"  # Falling price, rising volume
                    else:
                        market_cycle = "markdown"  # Falling price, falling volume
                
                # If still no cycle determined, use market regime as a fallback
                if market_cycle is None and "market_regime" in df.columns:
                    regime = df["market_regime"].iloc[i]
                    
                    if regime in ["strong_uptrend", "weak_uptrend"]:
                        market_cycle = "markup"
                    elif regime in ["strong_downtrend", "weak_downtrend"]:
                        market_cycle = "markdown"
                    elif regime == "overbought":
                        market_cycle = "distribution"
                    elif regime == "oversold":
                        market_cycle = "accumulation"
                    else:
                        market_cycle = None  # No clear cycle
            
            # Get appropriate threshold based on market cycle and signal direction
            if is_long:
                threshold = default_thresholds["long_threshold"]
                if market_cycle in cycle_rules:
                    threshold = cycle_rules[market_cycle]["long_threshold"]
            else:
                threshold = default_thresholds["short_threshold"]
                if market_cycle in cycle_rules:
                    threshold = cycle_rules[market_cycle]["short_threshold"]
            
            # Apply threshold to signal
            if "signal_strength" in df.columns:
                # If signal strength below threshold, filter out
                if df["signal_strength"].iloc[i] / 100 < threshold:
                    filtered_signals.iloc[i] = 0
            else:
                # No strength measure available, use a random value for demonstration
                signal_quality = np.random.random()
                if signal_quality < threshold:
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals