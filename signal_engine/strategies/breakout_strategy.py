"""
Breakout strategies for the trading system.
These strategies identify breakouts from ranges, channels, or support/resistance levels.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from signal_engine.signal_strategy_system import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """Strategy that identifies breakouts based on volatility."""
    
    name = "volatility_breakout"
    display_name = "Volatility Breakout Strategy"
    description = "Identifies breakouts from volatility-based ranges"
    category = "breakout"
    
    default_params = {
        "atr_multiplier": 2.0,
        "lookback_period": 14,
        "volume_surge_factor": 1.5  # Volume increase factor required for confirmation
    }
    
    required_indicators = ["close", "atr"]
    optional_indicators = ["volume", "volume_ma", "bollinger_upper", "bollinger_lower", 
                          "volatility_regime", "keltner_upper", "keltner_lower"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate volatility breakout signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        atr_mult = self.params.get("atr_multiplier", self.default_params["atr_multiplier"])
        lookback = self.params.get("lookback_period", self.default_params["lookback_period"])
        vol_surge = self.params.get("volume_surge_factor", self.default_params["volume_surge_factor"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Ensure we have enough history
        if i < lookback:
            return {"long": [], "short": []}
        
        # Calculate breakout levels based on ATR if other volatility bands not available
        if "bollinger_upper" not in row or "bollinger_lower" not in row:
            recent_price = df["close"].iloc[i-1]
            atr_value = row["atr"] if "atr" in row else 0
            
            upper_band = recent_price + (atr_value * atr_mult)
            lower_band = recent_price - (atr_value * atr_mult)
        else:
            upper_band = row["bollinger_upper"]
            lower_band = row["bollinger_lower"]
        
        # Check for breakouts
        breakout_up = row["close"] > upper_band
        breakout_down = row["close"] < lower_band
        
        long_conditions.append(breakout_up)
        short_conditions.append(breakout_down)
        
        # Check if we have Keltner Channels (more robust for volatility breakouts)
        if "keltner_upper" in row and "keltner_lower" in row:
            keltner_breakout_up = row["close"] > row["keltner_upper"]
            keltner_breakout_down = row["close"] < row["keltner_lower"]
            
            long_conditions.append(keltner_breakout_up)
            short_conditions.append(keltner_breakout_down)
        
        # Check for volume confirmation
        if "volume" in row and "volume_ma" in row:
            volume_surge = row["volume"] > row["volume_ma"] * vol_surge
            
            # Volume should confirm the breakout
            long_conditions.append(volume_surge)
            short_conditions.append(volume_surge)
        
        # Consider volatility regime if available
        if "volatility_regime" in row:
            # Breakouts more significant after low volatility
            if row["volatility_regime"] == "low":
                # Add more weight to existing conditions
                if any(long_conditions):
                    long_conditions.extend(long_conditions)  # Double the weight
                if any(short_conditions):
                    short_conditions.extend(short_conditions)  # Double the weight
            
            # Be more cautious in high volatility regimes
            elif row["volatility_regime"] == "high":
                # Require more confirming factors
                if len(long_conditions) < 2:
                    long_conditions = []
                if len(short_conditions) < 2:
                    short_conditions = []
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class RangeBreakoutStrategy(BaseStrategy):
    """Strategy that identifies breakouts from price ranges."""
    
    name = "range_breakout"
    display_name = "Range Breakout Strategy"
    description = "Identifies breakouts from defined price ranges"
    category = "breakout"
    
    default_params = {
        "range_period": 20,  # Period to identify range
        "range_threshold": 0.03,  # Max range as percentage of price
        "breakout_factor": 1.005  # Factor for breakout (0.5% above/below range)
    }
    
    required_indicators = ["high", "low", "close"]
    optional_indicators = ["volume", "volume_ma", "market_regime", "bollinger_width"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate range breakout signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        range_period = self.params.get("range_period", self.default_params["range_period"])
        range_threshold = self.params.get("range_threshold", self.default_params["range_threshold"])
        breakout_factor = self.params.get("breakout_factor", self.default_params["breakout_factor"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Ensure we have enough history
        if i < range_period:
            return {"long": [], "short": []}
        
        # Check if we're in a range-bound market
        range_window = df.iloc[i-range_period:i]
        
        # Calculate price range as percentage
        price_range = (range_window["high"].max() - range_window["low"].min()) / row["close"]
        is_range_bound = price_range < range_threshold
        
        # Define range levels
        range_high = range_window["high"].max()
        range_low = range_window["low"].min()
        
        # Calculate breakout levels
        upside_breakout_level = range_high * breakout_factor
        downside_breakout_level = range_low / breakout_factor
        
        # Check for breakouts
        breakout_up = row["close"] > upside_breakout_level
        breakout_down = row["close"] < downside_breakout_level
        
        # Only consider if we were in a range
        if is_range_bound:
            long_conditions.append(breakout_up)
            short_conditions.append(breakout_down)
            
            # Additional check: range should be established (price near middle of range recently)
            avg_price = (range_high + range_low) / 2
            middle_range = (df["close"].iloc[i-5:i] - avg_price).abs().mean() / avg_price < 0.01
            
            if middle_range:
                long_conditions.append(breakout_up)  # Double weight if price was in middle of range
                short_conditions.append(breakout_down)  # Double weight if price was in middle of range
        
        # Check for volume confirmation
        if "volume" in row and "volume_ma" in row:
            volume_surge = row["volume"] > row["volume_ma"] * 1.5
            
            # Volume should confirm the breakout
            if volume_surge:
                long_conditions.append(breakout_up)
                short_conditions.append(breakout_down)
        
        # Consider Bollinger Band width if available
        if "bollinger_width" in row:
            # Breakouts are more significant after narrow bands
            tight_bands = row["bollinger_width"] < 0.03  # Tight bands threshold
            
            if tight_bands:
                if breakout_up:
                    long_conditions.append(True)  # Add weight to bullish breakout
                if breakout_down:
                    short_conditions.append(True)  # Add weight to bearish breakout
        
        # Consider market regime if available
        if "market_regime" in row:
            # Most effective in ranging markets
            if row["market_regime"] == "ranging":
                if any(long_conditions):
                    long_conditions.extend(long_conditions)  # Double the weight
                if any(short_conditions):
                    short_conditions.extend(short_conditions)  # Double the weight
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class SupportResistanceBreakoutStrategy(BaseStrategy):
    """Strategy that identifies breakouts from support/resistance levels."""
    
    name = "sr_breakout"
    display_name = "Support/Resistance Breakout Strategy"
    description = "Identifies breakouts from key support and resistance levels"
    category = "breakout"
    
    default_params = {
        "breakout_factor": 1.003,  # 0.3% beyond S/R level
        "level_strength_min": 2    # Minimum number of touches to consider a level valid
    }
    
    required_indicators = ["high", "low", "close"]
    optional_indicators = ["nearest_support", "nearest_resistance", "in_support_zone", 
                          "in_resistance_zone", "broke_support", "broke_resistance",
                          "volume", "volume_ma"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate support/resistance breakout signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        breakout_factor = self.params.get("breakout_factor", self.default_params["breakout_factor"])
        level_strength_min = self.params.get("level_strength_min", self.default_params["level_strength_min"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Ensure we have enough history
        if i < 50:  # Need enough data to establish support/resistance
            return {"long": [], "short": []}
        
        # Check if we have S/R indicators already calculated
        if all(ind in row for ind in ["broke_resistance", "broke_support"]):
            # Direct breakout indicators
            long_conditions.append(row["broke_resistance"])
            short_conditions.append(row["broke_support"])
            
        elif all(ind in row for ind in ["nearest_resistance", "nearest_support"]):
            # Calculate breakouts from nearest levels
            resistance_level = row["nearest_resistance"]
            support_level = row["nearest_support"]
            
            if not pd.isna(resistance_level):
                resistance_breakout = row["close"] > resistance_level * breakout_factor
                long_conditions.append(resistance_breakout)
                
            if not pd.isna(support_level):
                support_breakout = row["close"] < support_level / breakout_factor
                short_conditions.append(support_breakout)
                
        # If S/R indicators not available, identify key levels manually
        else:
            # Find potential pivot highs and lows
            highs = []
            lows = []
            
            for j in range(2, min(i-2, 50)):
                # Pivot high (higher than 2 bars before and after)
                if (df["high"].iloc[i-j] > df["high"].iloc[i-j-1] and 
                    df["high"].iloc[i-j] > df["high"].iloc[i-j-2] and
                    df["high"].iloc[i-j] > df["high"].iloc[i-j+1] and
                    df["high"].iloc[i-j] > df["high"].iloc[i-j+2]):
                    highs.append(df["high"].iloc[i-j])
                
                # Pivot low (lower than 2 bars before and after)
                if (df["low"].iloc[i-j] < df["low"].iloc[i-j-1] and 
                    df["low"].iloc[i-j] < df["low"].iloc[i-j-2] and
                    df["low"].iloc[i-j] < df["low"].iloc[i-j+1] and
                    df["low"].iloc[i-j] < df["low"].iloc[i-j+2]):
                    lows.append(df["low"].iloc[i-j])
            
            # Group nearby levels (within 0.5%)
            def group_levels(levels, threshold=0.005):
                if not levels:
                    return []
                
                levels = sorted(levels)
                groups = [[levels[0]]]
                
                for level in levels[1:]:
                    if level / groups[-1][0] - 1 < threshold:
                        groups[-1].append(level)
                    else:
                        groups.append([level])
                
                return [sum(group) / len(group) for group in groups]
            
            resistance_levels = group_levels(highs)
            support_levels = group_levels(lows)
            
            # Count touches for each level
            def count_touches(level, price_series, threshold=0.003):
                return sum((price_series - level).abs() / level < threshold)
            
            # Find key levels with multiple touches
            key_resistances = [level for level in resistance_levels 
                             if count_touches(level, df["high"].iloc[:i], 0.003) >= level_strength_min]
            
            key_supports = [level for level in support_levels 
                          if count_touches(level, df["low"].iloc[:i], 0.003) >= level_strength_min]
            
            # Check for breakouts
            for level in key_resistances:
                if row["close"] > level * breakout_factor:
                    long_conditions.append(True)
                    
            for level in key_supports:
                if row["close"] < level / breakout_factor:
                    short_conditions.append(True)
        
        # Check for volume confirmation
        if "volume" in row and "volume_ma" in row:
            volume_surge = row["volume"] > row["volume_ma"] * 1.5
            
            # Only confirm breakouts with volume
            if not volume_surge:
                long_conditions = []
                short_conditions = []
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }