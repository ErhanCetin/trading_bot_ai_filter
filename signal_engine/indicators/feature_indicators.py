"""
Feature engineering indicators for the trading system with Smart Dependencies.
These indicators create complex features that capture different aspects of market behavior.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class PriceActionIndicator(BaseIndicator):
    """Analyzes price action patterns and candlestick characteristics - Pure Price Analysis."""
    
    name = "price_action"
    display_name = "Price Action"
    description = "Analyzes candlestick patterns and price action characteristics"
    category = "pattern"
    
    # NO DEPENDENCIES - Pure price analysis
    dependencies = []
    
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
        
        # Validate required columns
        missing_cols = [col for col in self.requires_columns if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for Price Action: {missing_cols}")
        
        try:
            # Calculate basic candle components
            result_df["body_size"] = abs(result_df["close"] - result_df["open"])
            result_df["range_size"] = result_df["high"] - result_df["low"]
            
            # Avoid division by zero with improved handling
            result_df["range_size"] = result_df["range_size"].replace(0, np.nan)
            
            # Upper and lower shadows as percentage of range
            max_oc = result_df[["open", "close"]].max(axis=1)
            min_oc = result_df[["open", "close"]].min(axis=1)
            
            result_df["upper_shadow"] = (result_df["high"] - max_oc) / result_df["range_size"]
            result_df["lower_shadow"] = (min_oc - result_df["low"]) / result_df["range_size"]
            
            # Body position within range (0.5 = middle, 0 = bottom, 1 = top)
            result_df["body_position"] = (min_oc - result_df["low"]) / result_df["range_size"]
            
            # Body to range ratio (1 = full body, 0 = doji)
            result_df["body_range_ratio"] = result_df["body_size"] / result_df["range_size"]
            
            # Initialize pattern columns
            result_df["engulfing_pattern"] = 0  # 1 for bullish, -1 for bearish
            result_df["doji_pattern"] = 0       # 1 for doji
            result_df["hammer_pattern"] = 0     # 1 for hammer, -1 for inverted hammer
            result_df["shooting_star_pattern"] = 0  # 1 for shooting star
            
            # Vectorized pattern detection for better performance
            self._detect_patterns_vectorized(result_df)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating Price Action features: {e}")
            
            # Initialize with default values on error
            for col in self.output_columns:
                if col not in result_df.columns:
                    result_df[col] = 0
        
        return result_df
    
    def _detect_patterns_vectorized(self, df: pd.DataFrame) -> None:
        """Vectorized pattern detection for better performance."""
        
        # Current and previous candle info
        curr_bullish = df["close"] > df["open"]
        prev_bullish = df["close"].shift(1) > df["open"].shift(1)
        
        # Engulfing pattern detection
        curr_open = df["open"]
        curr_close = df["close"]
        prev_open = df["open"].shift(1)
        prev_close = df["close"].shift(1)
        
        # Bullish engulfing
        bullish_engulf = (
            curr_bullish & ~prev_bullish &
            (curr_open <= prev_close) &
            (curr_close >= prev_open)
        )
        df.loc[bullish_engulf, "engulfing_pattern"] = 1
        
        # Bearish engulfing
        bearish_engulf = (
            ~curr_bullish & prev_bullish &
            (curr_open >= prev_close) &
            (curr_close <= prev_open)
        )
        df.loc[bearish_engulf, "engulfing_pattern"] = -1
        
        # Doji pattern (vectorized)
        doji_mask = df["body_range_ratio"] < 0.1
        df.loc[doji_mask, "doji_pattern"] = 1
        
        # Hammer pattern (vectorized)
        hammer_mask = (
            (df["body_range_ratio"] < 0.3) &
            (df["body_position"] > 0.7) &
            (df["lower_shadow"] > 0.6)
        )
        df.loc[hammer_mask, "hammer_pattern"] = 1
        
        # Inverted hammer (vectorized)
        inv_hammer_mask = (
            (df["body_range_ratio"] < 0.3) &
            (df["body_position"] < 0.3) &
            (df["upper_shadow"] > 0.6)
        )
        df.loc[inv_hammer_mask, "hammer_pattern"] = -1
        
        # Shooting star (vectorized)
        shooting_star_mask = (
            (df["body_range_ratio"] < 0.3) &
            (df["body_position"] < 0.3) &
            (df["upper_shadow"] > 0.6) &
            ~prev_bullish
        )
        df.loc[shooting_star_mask, "shooting_star_pattern"] = 1


class VolumePriceIndicator(BaseIndicator):
    """Analyzes volume in relation to price movements with Smart Dependencies."""
    
    name = "volume_price"
    display_name = "Volume-Price Analysis"
    description = "Analyzes volume in relation to price movements"
    category = "volume"
    
    # SMART DEPENDENCIES - Need price moving average for comparison
    dependencies = ["sma_20"]  # Will auto-resolve to SMA indicator
    
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
        
        # Validate required columns
        missing_cols = [col for col in self.requires_columns if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for Volume-Price: {missing_cols}")
        
        # Smart dependency validation
        price_ma_col = f"sma_{price_ma_period}"
        if price_ma_col not in result_df.columns:
            # Fallback: Calculate SMA manually
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing dependency {price_ma_col} for Volume-Price Analysis. Calculating manually.")
            
            result_df[price_ma_col] = result_df["close"].rolling(window=price_ma_period).mean()
        
        try:
            # Calculate basic volume metrics
            result_df["volume_ma"] = result_df["volume"].rolling(window=volume_ma_period).mean()
            
            # Avoid division by zero
            volume_ma_safe = result_df["volume_ma"].replace(0, np.nan)
            result_df["volume_ratio"] = result_df["volume"] / volume_ma_safe
            
            # On-Balance Volume (OBV) with improved calculation
            result_df["obv"] = ta.volume.OnBalanceVolumeIndicator(
                close=result_df["close"],
                volume=result_df["volume"]
            ).on_balance_volume()
            
            # Price-Volume Trend (PVT) with better error handling
            price_change = result_df["close"].pct_change().fillna(0)
            result_df["price_volume_trend"] = (price_change * result_df["volume"]).cumsum()
            
            # Volume Oscillator (percentage difference between fast and slow volume MAs)
            fast_vol_ma = result_df["volume"].rolling(window=max(1, volume_ma_period//2)).mean()
            slow_vol_ma = result_df["volume_ma"]
            
            slow_vol_ma_safe = slow_vol_ma.replace(0, np.nan)
            result_df["volume_oscillator"] = ((fast_vol_ma - slow_vol_ma) / slow_vol_ma_safe) * 100
            
            # Vectorized PVI and NVI calculation for better performance
            self._calculate_volume_indices_vectorized(result_df)
            
            # Volume-Price Confirmation (vectorized)
            self._calculate_volume_price_confirmation_vectorized(result_df)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating Volume-Price features: {e}")
            
            # Initialize with default values on error
            for col in self.output_columns:
                if col not in result_df.columns:
                    result_df[col] = 0
        
        return result_df
    
    def _calculate_volume_indices_vectorized(self, df: pd.DataFrame) -> None:
        """Vectorized calculation of PVI and NVI for better performance."""
        
        # Initialize indices
        df["positive_volume_index"] = 1000.0
        df["negative_volume_index"] = 1000.0
        
        # Calculate returns
        returns = df["close"].pct_change().fillna(0)
        volume_change = df["volume"].diff()
        
        # Positive Volume Index
        pvi_mask = volume_change > 0
        pvi_multiplier = 1 + returns
        pvi_multiplier[~pvi_mask] = 1  # No change when volume decreases
        
        df["positive_volume_index"] = (df["positive_volume_index"].iloc[0] * 
                                     pvi_multiplier.cumprod())
        
        # Negative Volume Index
        nvi_mask = volume_change < 0
        nvi_multiplier = 1 + returns
        nvi_multiplier[~nvi_mask] = 1  # No change when volume increases
        
        df["negative_volume_index"] = (df["negative_volume_index"].iloc[0] * 
                                     nvi_multiplier.cumprod())
    
    def _calculate_volume_price_confirmation_vectorized(self, df: pd.DataFrame) -> None:
        """Vectorized volume-price confirmation calculation."""
        
        # Initialize confirmation column
        df["volume_price_confirmation"] = 0
        
        # Price and volume direction
        price_up = df["close"] > df["close"].shift(1)
        volume_up = df["volume"] > df["volume"].shift(1)
        
        # Strong confirmation conditions
        strong_up = price_up & volume_up
        strong_down = ~price_up & ~volume_up
        
        df.loc[strong_up, "volume_price_confirmation"] = 1
        df.loc[strong_down, "volume_price_confirmation"] = -1


class MomentumFeatureIndicator(BaseIndicator):
    """Creates advanced momentum-based features with Smart Dependencies."""
    
    name = "momentum_features"
    display_name = "Momentum Features"
    description = "Creates advanced momentum-based features and metrics"
    category = "momentum"
    
    # SMART DEPENDENCIES - Need RSI for momentum comparison
    dependencies = ["rsi_14"]  # Will auto-resolve to RSI indicator
    
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
        
        # Smart dependency validation
        if "rsi_14" not in result_df.columns:
            # Fallback: Calculate RSI manually
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Missing dependency rsi_14 for Momentum Features. Calculating manually.")
            
            result_df["rsi_14"] = ta.momentum.RSIIndicator(
                close=result_df["close"],
                window=14
            ).rsi()
        
        try:
            # Vectorized momentum calculation for better performance
            for period in lookback_periods:
                col_name = f"momentum_{period}"
                result_df[col_name] = result_df["close"].pct_change(periods=period) * 100
                self.output_columns.append(col_name)
            
            # Calculate momentum acceleration (vectorized)
            for period in lookback_periods:
                momentum_col = f"momentum_{period}"
                accel_col = f"momentum_accel_{period}"
                result_df[accel_col] = result_df[momentum_col].diff()
                self.output_columns.append(accel_col)
            
            # Calculate rate of change of volume (if available)
            if "volume" in result_df.columns:
                for period in lookback_periods:
                    col_name = f"volume_roc_{period}"
                    result_df[col_name] = result_df["volume"].pct_change(periods=period) * 100
                    self.output_columns.append(col_name)
            
            # Momentum divergence analysis (optimized)
            self._calculate_momentum_divergence_optimized(result_df, lookback_periods)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating Momentum features: {e}")
            
            # Initialize with default values on error
            for col in self.output_columns:
                if col not in result_df.columns:
                    result_df[col] = 0
        
        return result_df
    
    def _calculate_momentum_divergence_optimized(self, df: pd.DataFrame, periods: List[int]) -> None:
        """Optimized momentum divergence calculation."""
        
        window = 20  # Look for divergence within last 20 bars
        
        if len(df) <= window:
            return
        
        # Initialize divergence columns
        divergence_cols = [
            "price_new_high", "price_new_low", 
            "momentum_new_high", "momentum_new_low",
            "bullish_divergence", "bearish_divergence"
        ]
        
        for col in divergence_cols:
            df[col] = False
            self.output_columns.append(col)
        
        # Vectorized rolling max/min for price
        price_rolling_max = df["close"].rolling(window=window, min_periods=1).max()
        price_rolling_min = df["close"].rolling(window=window, min_periods=1).min()
        
        # Price new highs/lows
        df["price_new_high"] = df["close"] >= price_rolling_max
        df["price_new_low"] = df["close"] <= price_rolling_min
        
        # Momentum new highs/lows (using primary momentum period)
        if periods:
            primary_period = periods[len(periods) // 2]  # Use middle period
            momentum_col = f"momentum_{primary_period}"
            
            if momentum_col in df.columns:
                momentum_rolling_max = df[momentum_col].rolling(window=window, min_periods=1).max()
                momentum_rolling_min = df[momentum_col].rolling(window=window, min_periods=1).min()
                
                df["momentum_new_high"] = df[momentum_col] >= momentum_rolling_max
                df["momentum_new_low"] = df[momentum_col] <= momentum_rolling_min
                
                # Divergence detection
                df["bullish_divergence"] = df["price_new_low"] & ~df["momentum_new_low"]
                df["bearish_divergence"] = df["price_new_high"] & ~df["momentum_new_high"]


class SupportResistanceIndicator(BaseIndicator):
    """Identifies support and resistance levels with Smart Dependencies."""
    
    name = "support_resistance"
    display_name = "Support and Resistance"
    description = "Identifies support and resistance levels and proximity to price"
    category = "price"
    
    # SMART DEPENDENCIES - Need ATR for dynamic threshold calculation
    dependencies = ["atr_14"]  # Will auto-resolve to ATR indicator
    
    default_params = {
        "window_size": 20,
        "num_touches": 2,
        "threshold_percentage": 0.1,
        "zone_width": 0.5,  # As percentage of price
        "atr_period": 14    # For dynamic threshold
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
        atr_period = self.params.get("atr_period", self.default_params["atr_period"])
        
        # Smart dependency validation
        atr_col = f"atr_{atr_period}"
        if atr_col not in result_df.columns:
            # Fallback: Calculate ATR manually
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing dependency {atr_col} for Support/Resistance. Calculating manually.")
            
            result_df[atr_col] = ta.volatility.AverageTrueRange(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=atr_period
            ).average_true_range()
        
        # Initialize columns
        for col in self.output_columns:
            if col.endswith("_distance"):
                result_df[col] = np.nan
            elif col.startswith("nearest_"):
                result_df[col] = np.nan
            else:
                result_df[col] = False
        
        # We need at least window_size bars to identify S/R levels
        if len(result_df) < window_size:
            return result_df
        
        try:
            # Optimized support/resistance calculation
            self._calculate_support_resistance_optimized(result_df, window_size, num_touches, 
                                                       threshold_percentage, zone_width, atr_col)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating Support/Resistance: {e}")
        
        return result_df
    
    def _calculate_support_resistance_optimized(self, df: pd.DataFrame, window_size: int, 
                                              num_touches: int, threshold_pct: float, 
                                              zone_width: float, atr_col: str) -> None:
        """Optimized support/resistance calculation using vectorized operations."""
        
        # Pre-calculate rolling highs and lows for pivot detection
        rolling_high = df["high"].rolling(window=5, center=True).max()
        rolling_low = df["low"].rolling(window=5, center=True).min()
        
        # Identify pivot points (vectorized)
        pivot_highs = (df["high"] == rolling_high) & (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
        pivot_lows = (df["low"] == rolling_low) & (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))
        
        # Process each row for S/R identification
        for i in range(window_size, len(df)):
            try:
                current_price = df["close"].iloc[i]
                
                # Dynamic threshold using ATR (smart dependency!)
                atr_value = df[atr_col].iloc[i]
                if pd.isna(atr_value) or atr_value == 0:
                    threshold = current_price * threshold_pct / 100
                else:
                    threshold = max(atr_value * 0.5, current_price * threshold_pct / 100)
                
                # Get recent pivot points
                window_slice = df.iloc[i-window_size:i]
                
                # Extract pivot levels
                pivot_high_levels = window_slice.loc[pivot_highs.iloc[i-window_size:i], "high"].values
                pivot_low_levels = window_slice.loc[pivot_lows.iloc[i-window_size:i], "low"].values
                
                # Find significant levels (touched multiple times)
                significant_levels = []
                
                for level in np.concatenate([pivot_high_levels, pivot_low_levels]):
                    if pd.isna(level):
                        continue
                    
                    # Count touches within threshold
                    touches = self._count_touches_vectorized(window_slice, level, threshold)
                    if touches >= num_touches:
                        significant_levels.append(level)
                
                if not significant_levels:
                    continue
                
                # Separate supports and resistances
                supports = [level for level in significant_levels if level < current_price]
                resistances = [level for level in significant_levels if level > current_price]
                
                # Update support information
                if supports:
                    nearest_support = max(supports)
                    df.loc[df.index[i], "nearest_support"] = nearest_support
                    df.loc[df.index[i], "support_distance"] = (current_price - nearest_support) / nearest_support * 100
                    
                    # Support zone and breakout detection
                    support_zone_width = nearest_support * zone_width / 100
                    df.loc[df.index[i], "in_support_zone"] = current_price <= nearest_support + support_zone_width
                    
                    if i > 0:
                        prev_price = df["close"].iloc[i-1]
                        df.loc[df.index[i], "broke_support"] = (prev_price > nearest_support and current_price < nearest_support)
                
                # Update resistance information
                if resistances:
                    nearest_resistance = min(resistances)
                    df.loc[df.index[i], "nearest_resistance"] = nearest_resistance
                    df.loc[df.index[i], "resistance_distance"] = (nearest_resistance - current_price) / current_price * 100
                    
                    # Resistance zone and breakout detection
                    resistance_zone_width = nearest_resistance * zone_width / 100
                    df.loc[df.index[i], "in_resistance_zone"] = current_price >= nearest_resistance - resistance_zone_width
                    
                    if i > 0:
                        prev_price = df["close"].iloc[i-1]
                        df.loc[df.index[i], "broke_resistance"] = (prev_price < nearest_resistance and current_price > nearest_resistance)
                        
            except Exception as e:
                continue
    
    def _count_touches_vectorized(self, df: pd.DataFrame, level: float, threshold: float) -> int:
        """Vectorized touch counting for better performance."""
        
        # Check if price touched level within threshold
        touched = (df["low"] <= level + threshold) & (df["high"] >= level - threshold)
        return touched.sum()
    
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
            if (len(highs[i-left_bars:i]) > 0 and highs[i] > max(highs[i-left_bars:i]) and
                len(highs[i+1:i+right_bars+1]) > 0 and highs[i] > max(highs[i+1:i+right_bars+1])):
                pivot_idx.append(i)
        
        return pivot_idx
    
    def _pivot_low(self, df: pd.DataFrame, left_bars=2, right_bars=2) -> List[int]:
        """Find pivot lows."""
        lows = df["low"].values
        pivot_idx = []
        
        for i in range(left_bars, len(lows) - right_bars):
            if (len(lows[i-left_bars:i]) > 0 and lows[i] < min(lows[i-left_bars:i]) and
                len(lows[i+1:i+right_bars+1]) > 0 and lows[i] < min(lows[i+1:i+right_bars+1])):
                pivot_idx.append(i)
        
        return pivot_idx