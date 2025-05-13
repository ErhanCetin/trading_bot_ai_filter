"""
Feature engineering indicators for the trading system.
These indicators create complex features that capture different aspects of market behavior.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class PriceActionIndicator(BaseIndicator):
    """Analyzes price action patterns and candlestick characteristics."""
    
    name = "price_action"
    display_name = "Price Action"
    description = "Analyzes candlestick patterns and price action characteristics"
    category = "pattern"
    
    default_params = {}
    
    requires_columns = ["open", "high", "low", "close"]
    output_columns = [
        "body_size", "upper_shadow", "lower_shadow", "body_position", 
        "range_size", "body_range_ratio", "engulfing_pattern", "doji_pattern",
        "hammer_pattern", "shooting_star_pattern"
    ]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action features and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with price action features added
        """
        result_df = df.copy()
        
        # Calculate basic candle components
        result_df["body_size"] = abs(result_df["close"] - result_df["open"])
        result_df["range_size"] = result_df["high"] - result_df["low"]
        
        # Avoid division by zero
        result_df["range_size"] = result_df["range_size"].replace(0, np.nan)
        
        # Upper and lower shadows as percentage of range
        result_df["upper_shadow"] = (result_df["high"] - result_df[["open", "close"]].max(axis=1)) / result_df["range_size"]
        result_df["lower_shadow"] = (result_df[["open", "close"]].min(axis=1) - result_df["low"]) / result_df["range_size"]
        
        # Body position within range (0.5 = middle, 0 = bottom, 1 = top)
        min_price = result_df[["open", "close"]].min(axis=1)
        max_price = result_df[["open", "close"]].max(axis=1)
        result_df["body_position"] = (
            (min_price - result_df["low"]) / result_df["range_size"]
        )
        
        # Body to range ratio (1 = full body, 0 = doji)
        result_df["body_range_ratio"] = result_df["body_size"] / result_df["range_size"]
        
        # Initialize pattern columns
        result_df["engulfing_pattern"] = 0  # 1 for bullish, -1 for bearish
        result_df["doji_pattern"] = 0       # 1 for doji
        result_df["hammer_pattern"] = 0     # 1 for hammer, -1 for inverted hammer
        result_df["shooting_star_pattern"] = 0  # 1 for shooting star
        
        # Detect patterns
        for i in range(1, len(result_df)):
            curr = result_df.iloc[i]
            prev = result_df.iloc[i-1]
            
            # Engulfing pattern
            curr_bullish = curr["close"] > curr["open"]
            prev_bullish = prev["close"] > prev["open"]
            
            if (curr_bullish and not prev_bullish and  # Current bullish, previous bearish
                curr["open"] <= prev["close"] and      # Current open below previous close
                curr["close"] >= prev["open"]):        # Current close above previous open
                result_df.loc[result_df.index[i], "engulfing_pattern"] = 1  # Bullish engulfing
                
            elif (not curr_bullish and prev_bullish and  # Current bearish, previous bullish
                  curr["open"] >= prev["close"] and      # Current open above previous close
                  curr["close"] <= prev["open"]):        # Current close below previous open
                result_df.loc[result_df.index[i], "engulfing_pattern"] = -1  # Bearish engulfing
            
            # Doji pattern (very small body)
            if curr["body_range_ratio"] < 0.1:  # Body is less than 10% of range
                result_df.loc[result_df.index[i], "doji_pattern"] = 1
            
            # Hammer pattern (small body at top, long lower shadow)
            if (curr["body_range_ratio"] < 0.3 and  # Small body
                curr["body_position"] > 0.7 and     # Body near top
                curr["lower_shadow"] > 0.6):        # Long lower shadow
                result_df.loc[result_df.index[i], "hammer_pattern"] = 1
            
            # Inverted hammer (small body at bottom, long upper shadow)
            if (curr["body_range_ratio"] < 0.3 and  # Small body
                curr["body_position"] < 0.3 and     # Body near bottom
                curr["upper_shadow"] > 0.6):        # Long upper shadow
                result_df.loc[result_df.index[i], "hammer_pattern"] = -1
            
            # Shooting star (small body at bottom, long upper shadow)
            if (curr["body_range_ratio"] < 0.3 and    # Small body
                curr["body_position"] < 0.3 and       # Body near bottom
                curr["upper_shadow"] > 0.6 and        # Long upper shadow
                not prev_bullish):                    # Previous candle was bearish
                result_df.loc[result_df.index[i], "shooting_star_pattern"] = 1
        
        return result_df


class VolumePriceIndicator(BaseIndicator):
    """Analyzes volume in relation to price movements."""
    
    name = "volume_price"
    display_name = "Volume-Price Analysis"
    description = "Analyzes volume in relation to price movements"
    category = "volume"
    
    default_params = {
        "volume_ma_period": 20,
        "price_ma_period": 20
    }
    
    requires_columns = ["close", "volume"]
    output_columns = [
        "volume_ma", "volume_ratio", "obv", "price_volume_trend", 
        "volume_oscillator", "positive_volume_index", "negative_volume_index",
        "volume_price_confirmation"
    ]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-price relationship features and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volume-price features added
        """
        result_df = df.copy()
        
        # Get parameters
        volume_ma_period = self.params.get("volume_ma_period", self.default_params["volume_ma_period"])
        price_ma_period = self.params.get("price_ma_period", self.default_params["price_ma_period"])
        
        # Calculate basic volume metrics
        result_df["volume_ma"] = result_df["volume"].rolling(window=volume_ma_period).mean()
        result_df["volume_ratio"] = result_df["volume"] / result_df["volume_ma"]
        
        # On-Balance Volume (OBV)
        result_df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            close=result_df["close"],
            volume=result_df["volume"]
        ).on_balance_volume()
        
        # Price-Volume Trend (PVT)
        price_change = result_df["close"].pct_change()
        result_df["price_volume_trend"] = (price_change * result_df["volume"]).cumsum()
        
        # Volume Oscillator (percentage difference between fast and slow volume MAs)
        fast_vol_ma = result_df["volume"].rolling(window=int(volume_ma_period/2)).mean()
        slow_vol_ma = result_df["volume_ma"]  # Already calculated above
        result_df["volume_oscillator"] = ((fast_vol_ma - slow_vol_ma) / slow_vol_ma) * 100
        
        # Positive Volume Index (PVI)
        result_df["positive_volume_index"] = 1000.0  # Starting value
        for i in range(1, len(result_df)):
            if result_df["volume"].iloc[i] > result_df["volume"].iloc[i-1]:
                result_df.loc[result_df.index[i], "positive_volume_index"] = (
                    result_df["positive_volume_index"].iloc[i-1] * 
                    (1 + (result_df["close"].iloc[i] / result_df["close"].iloc[i-1] - 1))
                )
            else:
                result_df.loc[result_df.index[i], "positive_volume_index"] = result_df["positive_volume_index"].iloc[i-1]
        
        # Negative Volume Index (NVI)
        result_df["negative_volume_index"] = 1000.0  # Starting value
        for i in range(1, len(result_df)):
            if result_df["volume"].iloc[i] < result_df["volume"].iloc[i-1]:
                result_df.loc[result_df.index[i], "negative_volume_index"] = (
                    result_df["negative_volume_index"].iloc[i-1] * 
                    (1 + (result_df["close"].iloc[i] / result_df["close"].iloc[i-1] - 1))
                )
            else:
                result_df.loc[result_df.index[i], "negative_volume_index"] = result_df["negative_volume_index"].iloc[i-1]
        
        # Volume-Price Confirmation
        # 1 = confirming up, -1 = confirming down, 0 = divergence/neutral
        result_df["volume_price_confirmation"] = 0
        
        for i in range(1, len(result_df)):
            price_up = result_df["close"].iloc[i] > result_df["close"].iloc[i-1]
            volume_up = result_df["volume"].iloc[i] > result_df["volume"].iloc[i-1]
            
            if price_up and volume_up:
                result_df.loc[result_df.index[i], "volume_price_confirmation"] = 1  # Strong up confirmation
            elif not price_up and not volume_up:
                result_df.loc[result_df.index[i], "volume_price_confirmation"] = -1  # Strong down confirmation
        
        return result_df


class MomentumFeatureIndicator(BaseIndicator):
    """Creates advanced momentum-based features."""
    
    name = "momentum_features"
    display_name = "Momentum Features"
    description = "Creates advanced momentum-based features and metrics"
    category = "momentum"
    
    default_params = {
        "lookback_periods": [3, 5, 10, 20, 50]
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based features and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with momentum features added
        """
        result_df = df.copy()
        
        # Get parameters
        lookback_periods = self.params.get("lookback_periods", self.default_params["lookback_periods"])
        
        # Clear output columns list and build it dynamically
        self.output_columns = []
        
        # Momentum (rate of change)
        for period in lookback_periods:
            col_name = f"momentum_{period}"
            result_df[col_name] = result_df["close"].pct_change(periods=period) * 100
            self.output_columns.append(col_name)
        
        # Calculate momentum acceleration (change in momentum)
        for period in lookback_periods:
            momentum_col = f"momentum_{period}"
            accel_col = f"momentum_accel_{period}"
            result_df[accel_col] = result_df[momentum_col].diff()
            self.output_columns.append(accel_col)
        
        # Calculate rate of change of volume
        if "volume" in result_df.columns:
            for period in lookback_periods:
                col_name = f"volume_roc_{period}"
                result_df[col_name] = result_df["volume"].pct_change(periods=period) * 100
                self.output_columns.append(col_name)
        
        # Momentum divergence (price making new highs/lows but momentum isn't)
        # This requires a rolling window calculation
        window = 20  # Look for divergence within last 20 bars
        
        if len(result_df) > window:
            # Initialize divergence columns
            result_df["price_new_high"] = False
            result_df["price_new_low"] = False
            result_df["momentum_new_high"] = False
            result_df["momentum_new_low"] = False
            result_df["bullish_divergence"] = False
            result_df["bearish_divergence"] = False
            
            # Add these to output columns
            self.output_columns.extend([
                "price_new_high", "price_new_low", 
                "momentum_new_high", "momentum_new_low",
                "bullish_divergence", "bearish_divergence"
            ])
            
            # For each row, check if it's making a new high/low within the window
            for i in range(window, len(result_df)):
                window_slice = result_df.iloc[i-window:i]
                
                # Check if current price is making a new high/low
                if result_df["close"].iloc[i] > window_slice["close"].max():
                    result_df.loc[result_df.index[i], "price_new_high"] = True
                
                if result_df["close"].iloc[i] < window_slice["close"].min():
                    result_df.loc[result_df.index[i], "price_new_low"] = True
                
                # Check momentum new highs/lows
                for period in lookback_periods:
                    momentum_col = f"momentum_{period}"
                    
                    if result_df[momentum_col].iloc[i] > window_slice[momentum_col].max():
                        result_df.loc[result_df.index[i], "momentum_new_high"] = True
                    
                    if result_df[momentum_col].iloc[i] < window_slice[momentum_col].min():
                        result_df.loc[result_df.index[i], "momentum_new_low"] = True
                
                # Bullish divergence: price making new low but momentum isn't
                if result_df["price_new_low"].iloc[i] and not result_df["momentum_new_low"].iloc[i]:
                    result_df.loc[result_df.index[i], "bullish_divergence"] = True
                
                # Bearish divergence: price making new high but momentum isn't
                if result_df["price_new_high"].iloc[i] and not result_df["momentum_new_high"].iloc[i]:
                    result_df.loc[result_df.index[i], "bearish_divergence"] = True
        
        return result_df


class SupportResistanceIndicator(BaseIndicator):
    """Identifies support and resistance levels."""
    
    name = "support_resistance"
    display_name = "Support and Resistance"
    description = "Identifies support and resistance levels and proximity to price"
    category = "price"
    
    default_params = {
        "window_size": 20,
        "num_touches": 2,
        "threshold_percentage": 0.1,
        "zone_width": 0.5  # As percentage of price
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = [
        "nearest_support", "nearest_resistance",
        "support_distance", "resistance_distance",
        "in_support_zone", "in_resistance_zone",
        "broke_support", "broke_resistance"
    ]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify support and resistance levels and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with support and resistance features added
        """
        result_df = df.copy()
        
        # Get parameters
        window_size = self.params.get("window_size", self.default_params["window_size"])
        num_touches = self.params.get("num_touches", self.default_params["num_touches"])
        threshold_percentage = self.params.get("threshold_percentage", self.default_params["threshold_percentage"])
        zone_width = self.params.get("zone_width", self.default_params["zone_width"])
        
        # Initialize columns
        result_df["nearest_support"] = np.nan
        result_df["nearest_resistance"] = np.nan
        result_df["support_distance"] = np.nan
        result_df["resistance_distance"] = np.nan
        result_df["in_support_zone"] = False
        result_df["in_resistance_zone"] = False
        result_df["broke_support"] = False
        result_df["broke_resistance"] = False
        
        # We need at least window_size bars to identify S/R levels
        if len(result_df) < window_size:
            return result_df
        
        # Identify support and resistance for each row
        for i in range(window_size, len(result_df)):
            # Get recent price history
            window = result_df.iloc[i-window_size:i]
            
            # Current price
            current_price = result_df["close"].iloc[i]
            
            # Calculate threshold
            threshold = current_price * threshold_percentage / 100
            
            # Find potential pivot points
            pivots = self._find_pivots(window)
            
            # Count touches for each pivot
            pivot_touches = {}
            for pivot in pivots:
                touches = self._count_touches(window, pivot, threshold)
                if touches >= num_touches:
                    pivot_touches[pivot] = touches
            
            # Separate supports and resistances
            supports = [p for p in pivot_touches.keys() if p < current_price]
            resistances = [p for p in pivot_touches.keys() if p > current_price]
            
            # Find nearest support and resistance
            if supports:
                nearest_support = max(supports)
                result_df.loc[result_df.index[i], "nearest_support"] = nearest_support
                result_df.loc[result_df.index[i], "support_distance"] = (
                    (current_price - nearest_support) / nearest_support * 100
                )
                
                # Check if in support zone
                support_zone_width = nearest_support * zone_width / 100
                if current_price <= nearest_support + support_zone_width:
                    result_df.loc[result_df.index[i], "in_support_zone"] = True
                
                # Check if support was broken
                if i > 0:
                    prev_price = result_df["close"].iloc[i-1]
                    if prev_price > nearest_support and current_price < nearest_support:
                        result_df.loc[result_df.index[i], "broke_support"] = True
            
            if resistances:
                nearest_resistance = min(resistances)
                result_df.loc[result_df.index[i], "nearest_resistance"] = nearest_resistance
                result_df.loc[result_df.index[i], "resistance_distance"] = (
                    (nearest_resistance - current_price) / current_price * 100
                )
                
                # Check if in resistance zone
                resistance_zone_width = nearest_resistance * zone_width / 100
                if current_price >= nearest_resistance - resistance_zone_width:
                    result_df.loc[result_df.index[i], "in_resistance_zone"] = True
                
                # Check if resistance was broken
                if i > 0:
                    prev_price = result_df["close"].iloc[i-1]
                    if prev_price < nearest_resistance and current_price > nearest_resistance:
                        result_df.loc[result_df.index[i], "broke_resistance"] = True
        
        return result_df
    
    def _find_pivots(self, df: pd.DataFrame) -> List[float]:
        """Find pivot highs and lows in the window."""
        pivot_high_idx = self._pivot_high(df)
        pivot_low_idx = self._pivot_low(df)
        
        pivots = []
        for idx in pivot_high_idx:
            pivots.append(df["high"].iloc[idx])
        
        for idx in pivot_low_idx:
            pivots.append(df["low"].iloc[idx])
        
        return pivots
    
    def _pivot_high(self, df: pd.DataFrame, left_bars=2, right_bars=2) -> List[int]:
        """Find pivot highs."""
        highs = df["high"].values
        pivot_idx = []
        
        for i in range(left_bars, len(highs) - right_bars):
            if highs[i] > max(highs[i-left_bars:i]) and highs[i] > max(highs[i+1:i+right_bars+1]):
                pivot_idx.append(i)
        
        return pivot_idx
    
    def _pivot_low(self, df: pd.DataFrame, left_bars=2, right_bars=2) -> List[int]:
        """Find pivot lows."""
        lows = df["low"].values
        pivot_idx = []
        
        for i in range(left_bars, len(lows) - right_bars):
            if lows[i] < min(lows[i-left_bars:i]) and lows[i] < min(lows[i+1:i+right_bars+1]):
                pivot_idx.append(i)
        
        return pivot_idx
    
    def _count_touches(self, df: pd.DataFrame, level: float, threshold: float) -> int:
        """Count how many times price touched a level within threshold."""
        touches = 0
        
        for i in range(len(df)):
            # Check if price touched level
            if (df["low"].iloc[i] <= level + threshold and 
                df["high"].iloc[i] >= level - threshold):
                touches += 1
        
        return touches