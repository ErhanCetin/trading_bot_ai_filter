"""
Trend following strategies for the trading system.
These strategies identify and follow market trends.
FIXED VERSION - Corrected indicator names to match actual outputs
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from signal_engine.signal_strategy_system import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Basic trend following strategy using multiple indicators."""
    
    name = "trend_following"
    display_name = "Trend Following Strategy"
    description = "Uses multiple indicators to identify and follow trends"
    category = "trend"
    
    default_params = {
        "adx_threshold": 25,
        "rsi_threshold": 50,
        "macd_threshold": 0,
        "ema_periods": [20, 50],
        "confirmation_count": 3
    }
    
    # FIXED: Corrected to match actual indicator outputs
    required_indicators = ["close", "adx", "di_pos", "di_neg", "rsi_14", "macd_line", "ema_20", "ema_50"]
    
    # FIXED: Optional indicators (can be missing)
    optional_indicators = ["market_regime", "atr_14", "atr_percent", "volume"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate conditions with proper weighting and categorization"""
        
        # Primary trend conditions (higher weight)
        primary_long = []
        primary_short = []
        
        # Secondary confirmation conditions (lower weight)  
        secondary_long = []
        secondary_short = []
        
        # Check ADX and directional indicators (PRIMARY)
        if self._has_valid_values(row, ["adx", "di_pos", "di_neg"]):
            adx_threshold = self.params.get("adx_threshold", 25)
            trend_strength = row["adx"] > adx_threshold
            
            if trend_strength:
                primary_long.append(row["di_pos"] > row["di_neg"])
                primary_short.append(row["di_neg"] > row["di_pos"])
        
        # Check RSI (SECONDARY)
        if self._has_valid_values(row, ["rsi_14"]):
            rsi_threshold = self.params.get("rsi_threshold", 50)
            secondary_long.append(row["rsi_14"] > rsi_threshold)
            secondary_short.append(row["rsi_14"] < rsi_threshold)
        
        # Check MACD (SECONDARY)
        if self._has_valid_values(row, ["macd_line"]):
            secondary_long.append(row["macd_line"] > 0)
            secondary_short.append(row["macd_line"] < 0)
        
        # Check EMA alignment (PRIMARY)
        ema_alignment = self._check_ema_alignment(row)
        if ema_alignment["valid"]:
            primary_long.append(ema_alignment["bullish"])
            primary_short.append(ema_alignment["bearish"])
        
        # Check market regime if available (OPTIONAL)
        if "market_regime" in row and not pd.isna(row["market_regime"]):
            if row["market_regime"] in ["strong_uptrend", "weak_uptrend"]:
                secondary_long.append(True)
            elif row["market_regime"] in ["strong_downtrend", "weak_downtrend"]:
                secondary_short.append(True)
        
        # Combine with weights (primary conditions count double)
        long_conditions = primary_long + primary_long + secondary_long
        short_conditions = primary_short + primary_short + secondary_short
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }
    
    def _has_valid_values(self, row: pd.Series, columns: List[str]) -> bool:
        """Check if all specified columns have valid (non-NaN, non-None) values"""
        return all(
            col in row and row[col] is not None and not pd.isna(row[col]) 
            for col in columns
        )
    
    def _check_ema_alignment(self, row: pd.Series) -> Dict[str, bool]:
        """Check EMA alignment for trend direction"""
        ema_periods = self.params.get("ema_periods", [20, 50])
        
        if len(ema_periods) < 2:
            return {"valid": False, "bullish": False, "bearish": False}
        
        ema_values = []
        for period in sorted(ema_periods):
            ema_col = f"ema_{period}"
            if not self._has_valid_values(row, [ema_col]):
                return {"valid": False, "bullish": False, "bearish": False}
            ema_values.append(row[ema_col])
        
        # Check alignment: fast EMA > slow EMA (bullish) or vice versa
        bullish_aligned = all(ema_values[i] > ema_values[i+1] 
                             for i in range(len(ema_values)-1))
        bearish_aligned = all(ema_values[i] < ema_values[i+1] 
                             for i in range(len(ema_values)-1))
        
        return {
            "valid": True,
            "bullish": bullish_aligned,
            "bearish": bearish_aligned
        }


class MultiTimeframeTrendStrategy(BaseStrategy):
    """Trend strategy that analyzes multiple timeframes."""
    
    name = "mtf_trend"
    display_name = "Multi-Timeframe Trend Strategy"
    description = "Identifies trends that are aligned across multiple timeframes"
    category = "trend"
    
    default_params = {
        "alignment_required": 0.8
    }
    
    # FIXED: Only basic requirement
    required_indicators = ["close"]
    optional_indicators = ["ema_alignment", "trend_alignment", "multi_timeframe_agreement"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate multi-timeframe trend signal conditions"""
        
        alignment_required = self.params.get("alignment_required", self.default_params["alignment_required"])
        
        long_conditions = []
        short_conditions = []
        
        # Check if we have EMA alignment indicator (from mtf_ema indicator)
        if "ema_alignment" in row and not pd.isna(row["ema_alignment"]):
            long_conditions.append(row["ema_alignment"] > alignment_required - 1)
            short_conditions.append(row["ema_alignment"] < 1 - alignment_required)
        
        # Check if we have trend alignment indicator (from trend_strength indicator)
        if "trend_alignment" in row and not pd.isna(row["trend_alignment"]):
            long_conditions.append(row["trend_alignment"] == 1)
            short_conditions.append(row["trend_alignment"] == -1)
        
        # Check if we have multi-timeframe agreement indicator (from trend_strength indicator)
        if "multi_timeframe_agreement" in row and not pd.isna(row["multi_timeframe_agreement"]):
            long_conditions.append(row["multi_timeframe_agreement"] > 0)
            short_conditions.append(row["multi_timeframe_agreement"] < 0)
        
        # Check EMAs directly if specific columns exist
        ema_cols = [col for col in row.index if col.startswith("ema_") and "_" in col]
        if ema_cols and "close" in row:
            above_count = sum(1 for col in ema_cols if row["close"] > row[col])
            alignment_pct = above_count / len(ema_cols)
            
            long_conditions.append(alignment_pct >= alignment_required)
            short_conditions.append(alignment_pct <= (1 - alignment_required))
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class AdaptiveTrendStrategy(BaseStrategy):
    """Trend strategy that adapts to current market conditions."""
    
    name = "adaptive_trend"
    display_name = "Adaptive Trend Strategy"
    description = "Adapts trend parameters based on current market conditions"
    category = "trend"
    
    default_params = {
        "adx_max_threshold": 40,
        "adx_min_threshold": 15
    }
    
    # FIXED: Corrected indicator names to match actual outputs
    required_indicators = ["adx", "di_pos", "di_neg", "atr_percent", "rsi_14"]
    optional_indicators = ["market_regime", "volatility_regime", "close"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate adaptive trend signal conditions"""
        
        adx_max = self.params.get("adx_max_threshold", self.default_params["adx_max_threshold"])
        adx_min = self.params.get("adx_min_threshold", self.default_params["adx_min_threshold"])
        
        long_conditions = []
        short_conditions = []
        
        # Determine current volatility level
        volatility_multiplier = 1.0
        
        # Check volatility regime if available
        if "volatility_regime" in row and not pd.isna(row["volatility_regime"]):
            if row["volatility_regime"] == "high":
                volatility_multiplier = 1.5
            elif row["volatility_regime"] == "low":
                volatility_multiplier = 0.7
        # Use atr_percent directly (available from ATRIndicator)
        elif "atr_percent" in row and not pd.isna(row["atr_percent"]):
            # atr_percent is calculated by ATRIndicator as (atr / close) * 100
            if row["atr_percent"] > 2.0:  # High volatility
                volatility_multiplier = 1.5
            elif row["atr_percent"] < 0.3:  # Low volatility
                volatility_multiplier = 0.7
        
        # Calculate adaptive ADX threshold
        adaptive_adx = max(adx_min, min(adx_max, adx_min * volatility_multiplier))
        
        # Check trend strength and direction
        if all(col in row and not pd.isna(row[col]) for col in ["adx", "di_pos", "di_neg"]):
            trend_strength = row["adx"] > adaptive_adx
            
            long_conditions.append(trend_strength and row["di_pos"] > row["di_neg"])
            short_conditions.append(trend_strength and row["di_neg"] > row["di_pos"])
            
            # Additional condition: trend strength vs. historical
            if i >= 50 and trend_strength:
                adx_history = df["adx"].iloc[i-50:i].dropna()
                if not adx_history.empty:
                    adx_percentile = (adx_history < row["adx"]).mean() * 100
                    strong_trend = adx_percentile > 80
                    long_conditions.append(strong_trend and row["di_pos"] > row["di_neg"])
                    short_conditions.append(strong_trend and row["di_neg"] > row["di_pos"])
        
        # Check RSI with adaptive thresholds
        if "rsi_14" in row and not pd.isna(row["rsi_14"]):
            rsi_middle = 50
            if volatility_multiplier > 1.2:  # High volatility
                long_conditions.append(row["rsi_14"] > rsi_middle + 5)
                short_conditions.append(row["rsi_14"] < rsi_middle - 5)
            elif volatility_multiplier < 0.8:  # Low volatility
                long_conditions.append(row["rsi_14"] > rsi_middle - 5)
                short_conditions.append(row["rsi_14"] < rsi_middle + 5)
            else:  # Normal volatility
                long_conditions.append(row["rsi_14"] > rsi_middle)
                short_conditions.append(row["rsi_14"] < rsi_middle)
        
        # Check market regime if available
        if "market_regime" in row and not pd.isna(row["market_regime"]):
            strong_trend_regimes = ["strong_uptrend", "strong_downtrend"]
            weak_trend_regimes = ["weak_uptrend", "weak_downtrend"]
            
            if row["market_regime"] in strong_trend_regimes:
                long_conditions.append(row["market_regime"] == "strong_uptrend")
                short_conditions.append(row["market_regime"] == "strong_downtrend")
            elif row["market_regime"] in weak_trend_regimes:
                # More conservative approach for weak trends
                if all(col in row and not pd.isna(row[col]) for col in ["adx", "di_pos", "di_neg", "rsi_14"]):
                    trend_confirms = 0
                    if row["adx"] > adaptive_adx:
                        trend_confirms += 1
                    if (row["di_pos"] > row["di_neg"] and row["market_regime"] == "weak_uptrend") or \
                       (row["di_neg"] > row["di_pos"] and row["market_regime"] == "weak_downtrend"):
                        trend_confirms += 1
                    if (row["rsi_14"] > 50 and row["market_regime"] == "weak_uptrend") or \
                       (row["rsi_14"] < 50 and row["market_regime"] == "weak_downtrend"):
                        trend_confirms += 1
                    
                    long_conditions.append(row["market_regime"] == "weak_uptrend" and trend_confirms >= 2)
                    short_conditions.append(row["market_regime"] == "weak_downtrend" and trend_confirms >= 2)
            else:
                # Avoid trend trading in ranging markets
                long_conditions.append(False)
                short_conditions.append(False)
        
        # Filter out None values
        long_conditions = [c for c in long_conditions if c is not None]
        short_conditions = [c for c in short_conditions if c is not None]
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }