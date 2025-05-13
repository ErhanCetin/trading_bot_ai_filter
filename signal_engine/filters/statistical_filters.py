"""
Statistical filters for the trading system.
These filters use statistical methods to evaluate signal quality.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from scipy import stats

from signal_engine.signal_filter_system import BaseFilter

logger = logging.getLogger(__name__)


class ZScoreExtremeFilter(BaseFilter):
    """Filter that removes signals with extreme Z-scores."""
    
    name = "zscore_extreme_filter"
    display_name = "Z-Score Extreme Filter"
    description = "Filters out signals with extreme statistical deviations"
    category = "statistical"
    
    default_params = {
        "indicators": {
            "rsi_14_zscore": {"min": -3.0, "max": 3.0},
            "macd_line_zscore": {"min": -3.0, "max": 3.0},
            "close_zscore": {"min": -3.0, "max": 3.0}
        },
        "min_conditions_met": 1  # Minimum number of conditions that must be met
    }
    
    required_indicators = []  # Will be populated from params
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with dynamic required indicators."""
        super().__init__(params)
        
        # Set required indicators based on the parameters
        self.required_indicators = list(self.params.get(
            "indicators", self.default_params["indicators"]
        ).keys())
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply Z-score filter to signals.
        
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
        indicators = self.params.get("indicators", self.default_params["indicators"])
        min_conditions = self.params.get("min_conditions_met", self.default_params["min_conditions_met"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Apply Z-score filtering
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Count conditions met
            conditions_met = 0
            
            for indicator, thresholds in indicators.items():
                if indicator in df.columns:
                    value = df[indicator].iloc[i]
                    
                    # Check if the value is within acceptable range
                    if not pd.isna(value) and thresholds["min"] <= value <= thresholds["max"]:
                        conditions_met += 1
            
            # Filter out signal if not enough conditions are met
            if conditions_met < min_conditions:
                filtered_signals.iloc[i] = 0
        
        return filtered_signals


class OutlierDetectionFilter(BaseFilter):
    """Filter that removes statistical outliers in signal generation."""
    
    name = "outlier_detection_filter"
    display_name = "Outlier Detection Filter"
    description = "Filters out statistical outliers in signal patterns"
    category = "statistical"
    
    default_params = {
        "lookback_window": 100,
        "percentile_threshold": 95,  # Filter out top 5% of outliers
        "price_metrics": ["close", "high", "low"],
        "indicator_metrics": ["rsi_14", "macd_line", "bollinger_width"],
        "min_metrics_required": 2
    }
    
    required_indicators = ["close"]  # Will be extended based on params
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with dynamic required indicators."""
        super().__init__(params)
        
        # Set required indicators based on the parameters
        price_metrics = self.params.get("price_metrics", self.default_params["price_metrics"])
        indicator_metrics = self.params.get("indicator_metrics", self.default_params["indicator_metrics"])
        
        self.required_indicators = list(set(["close"] + price_metrics + indicator_metrics))
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply outlier filter to signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Check for minimum required data
        available_indicators = [col for col in self.required_indicators if col in df.columns]
        if len(available_indicators) < self.params.get("min_metrics_required", self.default_params["min_metrics_required"]):
            return signals
        
        # Get parameters
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        percentile = self.params.get("percentile_threshold", self.default_params["percentile_threshold"])
        price_metrics = [m for m in self.params.get("price_metrics", self.default_params["price_metrics"]) if m in df.columns]
        indicator_metrics = [m for m in self.params.get("indicator_metrics", self.default_params["indicator_metrics"]) if m in df.columns]
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Need enough history for statistics
        if len(df) <= lookback:
            return signals
        
        # Apply outlier filtering
        for i in range(lookback, len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
                
            # Track outlier metrics
            outlier_count = 0
            metrics_checked = 0
            
            # Check price metrics for outliers
            for metric in price_metrics:
                if metric not in df.columns:
                    continue
                    
                metrics_checked += 1
                window = df[metric].iloc[i-lookback:i]
                current_value = df[metric].iloc[i]
                
                # Calculate percentile of current value within window
                percentile_rank = stats.percentileofscore(window, current_value)
                
                # Check if it's an outlier (above threshold percentile)
                if percentile_rank > percentile or percentile_rank < (100 - percentile):
                    outlier_count += 1
            
            # Check indicator metrics for outliers
            for metric in indicator_metrics:
                if metric not in df.columns:
                    continue
                    
                metrics_checked += 1
                window = df[metric].iloc[i-lookback:i]
                current_value = df[metric].iloc[i]
                
                # Calculate percentile of current value within window
                percentile_rank = stats.percentileofscore(window, current_value)
                
                # Check if it's an outlier (above threshold percentile)
                if percentile_rank > percentile or percentile_rank < (100 - percentile):
                    outlier_count += 1
            
            # If more than half of the metrics are outliers, filter the signal
            if metrics_checked > 0 and outlier_count / metrics_checked > 0.5:
                filtered_signals.iloc[i] = 0
        
        return filtered_signals


class HistoricalVolatilityFilter(BaseFilter):
    """Filter signals based on historical volatility levels."""
    
    name = "historical_volatility_filter"
    display_name = "Historical Volatility Filter"
    description = "Filters signals based on historical volatility analysis"
    category = "statistical"
    
    default_params = {
        "lookback_window": 50,
        "volatility_threshold": 80,  # Percentile threshold for high volatility
        "atr_period": 14,
        "high_volatility_rules": {
            "trend_following": False,  # Filter out trend signals in high volatility
            "reversal": True,          # Allow reversal signals in high volatility
            "breakout": True           # Allow breakout signals in high volatility
        },
        "strategy_categories": {
            "trend_following": ["trend_following", "mtf_trend", "adaptive_trend"],
            "reversal": ["overextended_reversal", "pattern_reversal", "divergence_reversal"],
            "breakout": ["volatility_breakout", "range_breakout", "sr_breakout"]
        }
    }
    
    required_indicators = ["high", "low", "close"]
    optional_indicators = ["atr", "volatility_percentile", "strategy_name"]
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply historical volatility filter to signals.
        
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
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        vol_threshold = self.params.get("volatility_threshold", self.default_params["volatility_threshold"])
        atr_period = self.params.get("atr_period", self.default_params["atr_period"])
        high_vol_rules = self.params.get("high_volatility_rules", self.default_params["high_volatility_rules"])
        strategy_categories = self.params.get("strategy_categories", self.default_params["strategy_categories"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Calculate ATR if not available
        if "atr" not in df.columns:
            from ta.volatility import AverageTrueRange
            atr_indicator = AverageTrueRange(
                high=df["high"], 
                low=df["low"], 
                close=df["close"],
                window=atr_period
            )
            df["atr"] = atr_indicator.average_true_range()
            
            # Calculate ATR as percentage of price
            df["atr_percent"] = df["atr"] / df["close"] * 100
        
        # Calculate volatility percentile if not available
        if "volatility_percentile" not in df.columns and "atr" in df.columns:
            # Need enough history
            if len(df) > lookback:
                df["volatility_percentile"] = 0.0
                
                for i in range(lookback, len(df)):
                    # Get ATR window
                    atr_window = df["atr"].iloc[i-lookback:i]
                    current_atr = df["atr"].iloc[i]
                    
                    # Calculate percentile
                    percentile = stats.percentileofscore(atr_window, current_atr)
                    df.loc[df.index[i], "volatility_percentile"] = percentile
        
        # Apply volatility filtering
        for i in range(lookback, len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
                
            # Check if we're in high volatility
            high_volatility = False
            
            if "volatility_percentile" in df.columns:
                vol_percentile = df["volatility_percentile"].iloc[i]
                high_volatility = vol_percentile >= vol_threshold
            elif "atr_percent" in df.columns:
                # Calculate percentile manually
                atr_pct = df["atr_percent"].iloc[i]
                atr_window = df["atr_percent"].iloc[i-lookback:i]
                vol_percentile = stats.percentileofscore(atr_window, atr_pct)
                high_volatility = vol_percentile >= vol_threshold
            
            # If not high volatility, keep the signal
            if not high_volatility:
                continue
                
            # Get strategy name and category
            strategy_name = None
            if "strategy_name" in df.columns:
                strategy_name = df["strategy_name"].iloc[i]
                
            # Determine strategy category
            strategy_category = "unknown"
            if strategy_name:
                for category, strategies in strategy_categories.items():
                    if strategy_name in strategies:
                        strategy_category = category
                        break
            
            # Apply high volatility rules based on strategy category
            if strategy_category in high_vol_rules:
                # If the rule is False, filter out the signal
                if not high_vol_rules[strategy_category]:
                    filtered_signals.iloc[i] = 0
            
            # Additional check for extreme volatility
            if "atr_percent" in df.columns and df["atr_percent"].iloc[i] > 5.0:
                # In extreme volatility, only allow breakout signals
                if strategy_category != "breakout":
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals