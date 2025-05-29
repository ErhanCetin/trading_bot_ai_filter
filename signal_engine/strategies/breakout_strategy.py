"""
Breakout strategies for the trading system.
These strategies identify breakouts from ranges, channels, or support/resistance levels.
FIXED VERSION - Corrected indicator names to match actual outputs
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
        "volume_surge_factor": 1.5
    }
    
    # FIXED: Corrected to match actual indicator outputs (ATRIndicator produces atr_14, atr_50, etc.)
    required_indicators = ["close", "atr_14"]
    optional_indicators = ["volume", "sma_20", "bollinger_upper", "bollinger_lower", 
                          "volatility_regime", "keltner_upper", "keltner_lower"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate volatility breakout signal conditions"""
        
        atr_mult = self.params.get("atr_multiplier", self.default_params["atr_multiplier"])
        lookback = self.params.get("lookback_period", self.default_params["lookback_period"])
        vol_surge = self.params.get("volume_surge_factor", self.default_params["volume_surge_factor"])
        
        long_conditions = []
        short_conditions = []
        
        if i < lookback:
            return {"long": [], "short": []}
        
        # Calculate breakout levels based on ATR if other volatility bands not available
        if not ("bollinger_upper" in row and "bollinger_lower" in row and 
                not pd.isna(row["bollinger_upper"]) and not pd.isna(row["bollinger_lower"])):
            recent_price = df["close"].iloc[i-1]
            atr_value = row.get("atr_14", 0)  # FIXED: Use atr_14 instead of atr
            
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
        
        # Check if we have Keltner Channels
        if ("keltner_upper" in row and "keltner_lower" in row and 
            not pd.isna(row["keltner_upper"]) and not pd.isna(row["keltner_lower"])):
            keltner_breakout_up = row["close"] > row["keltner_upper"]
            keltner_breakout_down = row["close"] < row["keltner_lower"]
            
            long_conditions.append(keltner_breakout_up)
            short_conditions.append(keltner_breakout_down)
        
        # Check for volume confirmation - FIXED: Use sma_20 instead of volume_ma
        if ("volume" in row and "sma_20" in row and 
            not pd.isna(row["volume"]) and not pd.isna(row["sma_20"])):
            # Calculate volume MA manually if not available
            volume_ma = df["volume"].iloc[max(0, i-19):i+1].mean()
            volume_surge = row["volume"] > volume_ma * vol_surge
            long_conditions.append(volume_surge)
            short_conditions.append(volume_surge)
        
        # Consider volatility regime if available
        if "volatility_regime" in row and not pd.isna(row["volatility_regime"]):
            if row["volatility_regime"] == "low":
                # Add more weight to existing conditions
                if any(long_conditions):
                    long_conditions.extend(long_conditions)
                if any(short_conditions):
                    short_conditions.extend(short_conditions)
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
        "range_period": 20,
        "range_threshold": 0.03,
        "breakout_factor": 1.005
    }
    
    # FIXED: These are basic OHLC columns that always exist
    required_indicators = ["high", "low", "close"]
    optional_indicators = ["volume", "sma_20", "market_regime", "bollinger_width", "open"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate range breakout signal conditions"""
        
        range_period = self.params.get("range_period", self.default_params["range_period"])
        range_threshold = self.params.get("range_threshold", self.default_params["range_threshold"])
        breakout_factor = self.params.get("breakout_factor", self.default_params["breakout_factor"])
        
        long_conditions = []
        short_conditions = []
        
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
            
            # Additional check: range should be established
            avg_price = (range_high + range_low) / 2
            recent_closes = df["close"].iloc[max(0, i-5):i]
            middle_range = (recent_closes - avg_price).abs().mean() / avg_price < 0.01
            
            if middle_range:
                long_conditions.append(breakout_up)
                short_conditions.append(breakout_down)
        
        # Check for volume confirmation - FIXED: Calculate volume MA manually
        if "volume" in row and not pd.isna(row["volume"]):
            volume_ma = df["volume"].iloc[max(0, i-19):i+1].mean()
            volume_surge = row["volume"] > volume_ma * 1.5
            
            if volume_surge:
                long_conditions.append(breakout_up)
                short_conditions.append(breakout_down)
        
        # Consider Bollinger Band width if available
        if "bollinger_width" in row and not pd.isna(row["bollinger_width"]):
            tight_bands = row["bollinger_width"] < 0.03
            
            if tight_bands:
                if breakout_up:
                    long_conditions.append(True)
                if breakout_down:
                    short_conditions.append(True)
        
        # Consider market regime if available
        if "market_regime" in row and not pd.isna(row["market_regime"]):
            if row["market_regime"] == "ranging":
                if any(long_conditions):
                    long_conditions.extend(long_conditions)
                if any(short_conditions):
                    short_conditions.extend(short_conditions)
        
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
        "breakout_factor": 1.003,
        "level_strength_min": 2
    }
    
    # FIXED: These are basic OHLC columns that always exist
    required_indicators = ["high", "low", "close"]
    optional_indicators = ["nearest_support", "nearest_resistance", "in_support_zone", 
                          "in_resistance_zone", "broke_support", "broke_resistance",
                          "volume", "sma_20"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """Generate support/resistance breakout signal conditions"""
        
        breakout_factor = self.params.get("breakout_factor", self.default_params["breakout_factor"])
        level_strength_min = self.params.get("level_strength_min", self.default_params["level_strength_min"])
        
        long_conditions = []
        short_conditions = []
        
        if i < 50:
            return {"long": [], "short": []}
        
        # Check if we have S/R indicators already calculated (from support_resistance indicator)
        if (all(ind in row for ind in ["broke_resistance", "broke_support"]) and
            not pd.isna(row["broke_resistance"]) and not pd.isna(row["broke_support"])):
            long_conditions.append(bool(row["broke_resistance"]))
            short_conditions.append(bool(row["broke_support"]))
            
        elif (all(ind in row for ind in ["nearest_resistance", "nearest_support"]) and
              not pd.isna(row["nearest_resistance"]) and not pd.isna(row["nearest_support"])):
            resistance_level = row["nearest_resistance"]
            support_level = row["nearest_support"]
            
            if not pd.isna(resistance_level):
                resistance_breakout = row["close"] > resistance_level * breakout_factor
                long_conditions.append(resistance_breakout)
                
            if not pd.isna(support_level):
                support_breakout = row["close"] < support_level / breakout_factor
                short_conditions.append(support_breakout)
        
        else:
            # Manual S/R detection (simplified version)
            try:
                # Find potential pivot highs and lows
                highs = []
                lows = []
                
                for j in range(2, min(i-2, 50)):
                    if (j >= 2 and j < len(df) - 2 and
                        df["high"].iloc[i-j] > df["high"].iloc[i-j-1] and 
                        df["high"].iloc[i-j] > df["high"].iloc[i-j-2] and
                        df["high"].iloc[i-j] > df["high"].iloc[i-j+1] and
                        df["high"].iloc[i-j] > df["high"].iloc[i-j+2]):
                        highs.append(df["high"].iloc[i-j])
                    
                    if (j >= 2 and j < len(df) - 2 and
                        df["low"].iloc[i-j] < df["low"].iloc[i-j-1] and 
                        df["low"].iloc[i-j] < df["low"].iloc[i-j-2] and
                        df["low"].iloc[i-j] < df["low"].iloc[i-j+1] and
                        df["low"].iloc[i-j] < df["low"].iloc[i-j+2]):
                        lows.append(df["low"].iloc[i-j])
                
                # Simple level grouping and breakout detection
                if highs:
                    key_resistance = max(highs)  # Simplified - use highest resistance
                    if row["close"] > key_resistance * breakout_factor:
                        long_conditions.append(True)
                        
                if lows:
                    key_support = min(lows)  # Simplified - use lowest support
                    if row["close"] < key_support / breakout_factor:
                        short_conditions.append(True)
                        
            except (IndexError, KeyError):
                pass
        
        # Check for volume confirmation - FIXED: Calculate volume MA manually
        if "volume" in row and not pd.isna(row["volume"]):
            volume_ma = df["volume"].iloc[max(0, i-19):i+1].mean()
            volume_surge = row["volume"] > volume_ma * 1.5
            
            # Only confirm breakouts with volume
            if not volume_surge:
                long_conditions = []
                short_conditions = []
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }