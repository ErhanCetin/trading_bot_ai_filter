"""
Advanced technical indicators for the trading system with Smart Dependencies.
These indicators build upon basic indicators and provide more sophisticated analysis.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class AdaptiveRSIIndicator(BaseIndicator):
    """Calculates Adaptive RSI that adjusts period based on volatility with Smart Dependencies."""
    
    name = "adaptive_rsi"
    display_name = "Adaptive RSI"
    description = "RSI that adapts to market volatility using ATR"
    category = "momentum"
    
    # SMART DEPENDENCIES - Need ATR for volatility measurement
    dependencies = ["atr_14"]  # Will auto-resolve to ATR indicator
    
    default_params = {
        "base_period": 14,
        "volatility_window": 100,
        "min_period": 5,
        "max_period": 30,
        "apply_to": "close",
        "atr_period": 14  # For dependency matching
    }
    
    requires_columns = ["close", "high", "low"]
    output_columns = ["adaptive_rsi", "adaptive_rsi_period"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Adaptive RSI and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Adaptive RSI columns added
        """
        result_df = df.copy()
        
        # Get parameters
        base_period = self.params.get("base_period", self.default_params["base_period"])
        volatility_window = self.params.get("volatility_window", self.default_params["volatility_window"])
        min_period = self.params.get("min_period", self.default_params["min_period"])
        max_period = self.params.get("max_period", self.default_params["max_period"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        atr_period = self.params.get("atr_period", self.default_params["atr_period"])
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # Smart dependency validation
        atr_col = f"atr_{atr_period}"
        if atr_col not in result_df.columns:
            # Fallback: Calculate ATR manually
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing dependency {atr_col} for AdaptiveRSI. Calculating manually.")
            
            result_df[atr_col] = ta.volatility.AverageTrueRange(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=atr_period
            ).average_true_range()
        
        # Calculate ATR as percentage of price (using dependency)
        atr_pct = result_df[atr_col] / result_df["close"] * 100
        
        # Initialize adaptive period column
        result_df["adaptive_rsi_period"] = base_period
        
        # Need at least volatility_window data points for percentile calculation
        if len(result_df) > volatility_window:
            for i in range(volatility_window, len(result_df)):
                # Calculate ATR percentile over the lookback window
                atr_window = atr_pct.iloc[i-volatility_window:i]
                current_atr = atr_pct.iloc[i]
                
                # Convert to percentile rank (0-1) with error handling
                try:
                    valid_atr = atr_window.dropna()
                    if len(valid_atr) > 0 and not pd.isna(current_atr):
                        percentile = (valid_atr < current_atr).mean()
                    else:
                        percentile = 0.5  # Default to middle
                except:
                    percentile = 0.5  # Default to middle if calculation fails
                
                # Map percentile to RSI period
                # High volatility (high percentile) -> Shorter period (more responsive)
                # Low volatility (low percentile) -> Longer period (less noise)
                period_range = max_period - min_period
                adaptive_period = max_period - (period_range * percentile)
                
                # Ensure integer period within bounds
                adaptive_period = max(min_period, min(max_period, int(adaptive_period)))
                result_df.loc[result_df.index[i], "adaptive_rsi_period"] = adaptive_period
        
        # Calculate RSI for each row based on its adaptive period
        # Note: This is computationally intensive as we calculate a separate RSI for each row
        result_df["adaptive_rsi"] = np.nan
        
        for i in range(len(result_df)):
            period = int(result_df["adaptive_rsi_period"].iloc[i])
            
            # Need at least 'period' rows of data
            if i >= period:
                try:
                    price_window = result_df[price_column].iloc[i-period:i+1]
                    
                    # Calculate RSI manually for this window
                    delta = price_window.diff().dropna()
                    if len(delta) == 0:
                        continue
                    
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    
                    # Use simple moving average for adaptive calculation
                    avg_gain = gain.mean()
                    avg_loss = loss.mean()
                    
                    if avg_loss == 0:
                        rsi = 100
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    
                    result_df.loc[result_df.index[i], "adaptive_rsi"] = rsi
                    
                except Exception as e:
                    # Skip this calculation but continue
                    continue
        
        return result_df


class MultitimeframeEMAIndicator(BaseIndicator):
    """Calculates EMAs across multiple timeframes and their relationships with Smart Dependencies."""
    
    name = "mtf_ema"
    display_name = "Multi-timeframe EMA"
    description = "Calculates EMAs on multiple timeframes and their alignment"
    category = "trend"
    
    # SMART DEPENDENCIES - Need base EMAs
    dependencies = ["ema_20"]  # Will auto-resolve to base EMA indicator
    
    default_params = {
        "period": 20,
        "timeframes": [1, 4, 12, 24],  # Multiples of the base timeframe
        "apply_to": "close"
    }
    
    requires_columns = ["close", "open_time"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multi-timeframe EMAs and their relationships.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with MTF EMA columns added
        """
        result_df = df.copy()
        
        # Get parameters
        period = self.params.get("period", self.default_params["period"])
        timeframes = self.params.get("timeframes", self.default_params["timeframes"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Validate columns
        missing_cols = []
        if price_column not in result_df.columns:
            missing_cols.append(price_column)
        if "open_time" not in result_df.columns:
            missing_cols.append("open_time")
        
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in dataframe")
        
        # Smart dependency validation
        base_ema_col = f"ema_{period}"
        if base_ema_col not in result_df.columns:
            # Fallback: Calculate base EMA manually
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing dependency {base_ema_col} for Multi-timeframe EMA. Calculating manually.")
            
            result_df[base_ema_col] = ta.trend.EMAIndicator(
                close=result_df[price_column], 
                window=period
            ).ema_indicator()
        
        # Clear and initialize output columns
        self.output_columns = []
        
        # Base EMA is already available (from dependency or fallback)
        base_ema_name = f"ema_{period}"
        if base_ema_name not in self.output_columns:
            self.output_columns.append(base_ema_name)
        
        # Ensure open_time is in datetime format for resampling
        if not pd.api.types.is_datetime64_any_dtype(result_df["open_time"]):
            try:
                result_df["open_time"] = pd.to_datetime(result_df["open_time"], unit='ms')
            except:
                result_df["open_time"] = pd.to_datetime(result_df["open_time"])
        
        # Set open_time as index for resampling
        result_df_resampled = result_df.set_index('open_time')
        
        # Calculate EMAs for higher timeframes
        for tf_multiplier in timeframes[1:]:  # Skip the first (base) timeframe
            try:
                # Resample to higher timeframe
                resampled = result_df_resampled.resample(f'{tf_multiplier}min').agg({
                    price_column: 'last',
                    'high': 'max',
                    'low': 'min',
                    'open': 'first',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(resampled) == 0:
                    continue
                
                # Calculate EMA on the resampled data
                tf_ema = ta.trend.EMAIndicator(
                    close=resampled[price_column], 
                    window=period
                ).ema_indicator()
                
                # Create DataFrame for merging back
                tf_ema_df = pd.DataFrame(tf_ema, columns=[f'ema_{period}_{tf_multiplier}x'])
                
                # Reindex to original timeframe and forward fill
                tf_ema_df = tf_ema_df.reindex(result_df_resampled.index, method='ffill')
                
                # Add to result dataframe
                col_name = f'ema_{period}_{tf_multiplier}x'
                result_df[col_name] = tf_ema_df.values.flatten()
                
                # Add to output columns
                if col_name not in self.output_columns:
                    self.output_columns.append(col_name)
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error calculating EMA for timeframe {tf_multiplier}x: {e}")
                continue
        
        # Calculate alignment metrics
        ema_cols = [col for col in result_df.columns if col.startswith('ema_') and col in self.output_columns]
        n_emas = len(ema_cols)
        
        if n_emas > 1:
            # Create alignment score column
            result_df['ema_alignment'] = 0
            
            # Add to output columns
            if 'ema_alignment' not in self.output_columns:
                self.output_columns.append('ema_alignment')
            
            for i in range(len(result_df)):
                try:
                    # Get EMAs for this row (filter out NaN values)
                    ema_values = []
                    for col in ema_cols:
                        val = result_df[col].iloc[i]
                        if not pd.isna(val):
                            ema_values.append(val)
                    
                    if len(ema_values) < 2:
                        continue
                    
                    # Count ascending vs descending pairs
                    ascending_count = 0
                    descending_count = 0
                    total_pairs = len(ema_values) - 1
                    
                    for j in range(1, len(ema_values)):
                        if ema_values[j] > ema_values[j-1]:
                            ascending_count += 1
                        elif ema_values[j] < ema_values[j-1]:
                            descending_count += 1
                    
                    # Calculate alignment score (-1 to 1)
                    if total_pairs > 0:
                        if ascending_count > descending_count:
                            alignment = ascending_count / total_pairs
                        else:
                            alignment = -descending_count / total_pairs
                        
                        result_df.loc[result_df.index[i], 'ema_alignment'] = alignment
                        
                except Exception as e:
                    continue
        
        return result_df


class HeikinAshiIndicator(BaseIndicator):
    """Calculates Heikin Ashi candlesticks - Pure Price Transformation."""
    
    name = "heikin_ashi"
    display_name = "Heikin Ashi"
    description = "Calculates Heikin Ashi candlesticks for smoother trend visualization"
    category = "price_transformation"
    
    # NO DEPENDENCIES - Pure price transformation
    dependencies = []
    
    default_params = {}
    
    requires_columns = ["open", "high", "low", "close"]
    output_columns = ["ha_open", "ha_high", "ha_low", "ha_close", "ha_trend"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin Ashi candlesticks and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Heikin Ashi columns added
        """
        result_df = df.copy()
        
        # Validate required columns
        missing_cols = [col for col in self.requires_columns if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for Heikin Ashi: {missing_cols}")
        
        # Initialize HA columns
        result_df["ha_close"] = (result_df["open"] + result_df["high"] + 
                                result_df["low"] + result_df["close"]) / 4
        
        # Initialize HA open column
        result_df["ha_open"] = 0.0
        
        # For the first row, HA open equals regular open
        if len(result_df) > 0:
            result_df.loc[result_df.index[0], "ha_open"] = result_df["open"].iloc[0]
        
        # Calculate HA open for the rest of the rows
        for i in range(1, len(result_df)):
            try:
                prev_ha_open = result_df["ha_open"].iloc[i-1]
                prev_ha_close = result_df["ha_close"].iloc[i-1]
                
                # HA Open = (Previous HA Open + Previous HA Close) / 2
                ha_open = (prev_ha_open + prev_ha_close) / 2
                result_df.loc[result_df.index[i], "ha_open"] = ha_open
                
            except Exception as e:
                # Use regular open as fallback
                result_df.loc[result_df.index[i], "ha_open"] = result_df["open"].iloc[i]
        
        # Calculate HA high and low
        result_df["ha_high"] = result_df[["high", "ha_open", "ha_close"]].max(axis=1)
        result_df["ha_low"] = result_df[["low", "ha_open", "ha_close"]].min(axis=1)
        
        # Calculate trend direction based on HA (1 for bullish, -1 for bearish)
        result_df["ha_trend"] = np.where(
            result_df["ha_close"] >= result_df["ha_open"], 
            1, 
            -1
        )
        
        return result_df


class SupertrendIndicator(BaseIndicator):
    """Calculates the Supertrend indicator with Smart Dependencies."""
    
    name = "supertrend"
    display_name = "Supertrend"
    description = "Trend following indicator combining ATR with price action"
    category = "trend"
    
    # SMART DEPENDENCIES - Need ATR for volatility measurement
    dependencies = ["atr_10"]  # Will auto-resolve to ATR indicator
    
    default_params = {
        "atr_period": 10,
        "atr_multiplier": 3.0
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["supertrend", "supertrend_direction", "supertrend_upper", "supertrend_lower"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Supertrend columns added
        """
        result_df = df.copy()
        
        # Get parameters
        atr_period = self.params.get("atr_period", self.default_params["atr_period"])
        multiplier = self.params.get("atr_multiplier", self.default_params["atr_multiplier"])
        
        # Smart dependency validation
        atr_col = f"atr_{atr_period}"
        if atr_col not in result_df.columns:
            # Fallback: Calculate ATR manually
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing dependency {atr_col} for Supertrend. Calculating manually.")
            
            result_df[atr_col] = ta.volatility.AverageTrueRange(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=atr_period
            ).average_true_range()
        
        # Calculate basic upper and lower bands using dependency ATR
        hl2 = (result_df["high"] + result_df["low"]) / 2
        
        # Initial upper and lower bands
        basic_upperband = hl2 + (multiplier * result_df[atr_col])
        basic_lowerband = hl2 - (multiplier * result_df[atr_col])
        
        # Initialize supertrend columns
        result_df["supertrend_direction"] = True  # True = bullish, False = bearish
        result_df["supertrend_upper"] = 0.0
        result_df["supertrend_lower"] = 0.0
        result_df["supertrend"] = 0.0
        
        # Calculate Supertrend - iterate through the dataframe
        for i in range(len(result_df)):
            try:
                current_close = result_df["close"].iloc[i]
                current_upper = basic_upperband.iloc[i]
                current_lower = basic_lowerband.iloc[i]
                
                if i == 0:
                    # First row initialization
                    result_df.loc[result_df.index[i], "supertrend_upper"] = current_upper
                    result_df.loc[result_df.index[i], "supertrend_lower"] = current_lower
                    result_df.loc[result_df.index[i], "supertrend_direction"] = True
                    result_df.loc[result_df.index[i], "supertrend"] = current_lower
                else:
                    # Previous values
                    prev_close = result_df["close"].iloc[i-1]
                    prev_upper = result_df["supertrend_upper"].iloc[i-1]
                    prev_lower = result_df["supertrend_lower"].iloc[i-1]
                    prev_direction = result_df["supertrend_direction"].iloc[i-1]
                    
                    # Calculate final bands
                    final_upper = current_upper if (current_upper < prev_upper) or (prev_close > prev_upper) else prev_upper
                    final_lower = current_lower if (current_lower > prev_lower) or (prev_close < prev_lower) else prev_lower
                    
                    # Determine direction
                    if prev_direction and current_close < final_lower:
                        direction = False  # Bearish
                    elif not prev_direction and current_close > final_upper:
                        direction = True   # Bullish
                    else:
                        direction = prev_direction
                    
                    # Set supertrend value
                    supertrend_value = final_lower if direction else final_upper
                    
                    # Update dataframe
                    result_df.loc[result_df.index[i], "supertrend_upper"] = final_upper
                    result_df.loc[result_df.index[i], "supertrend_lower"] = final_lower
                    result_df.loc[result_df.index[i], "supertrend_direction"] = direction
                    result_df.loc[result_df.index[i], "supertrend"] = supertrend_value
                    
            except Exception as e:
                # Log error but continue
                if i < 5:  # Only log first few errors
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error calculating Supertrend at index {i}: {e}")
                continue
        
        return result_df


class IchimokuIndicator(BaseIndicator):
    """Calculates Ichimoku Cloud indicator - Pure Price Analysis."""
    
    name = "ichimoku"
    display_name = "Ichimoku Cloud"
    description = "Japanese charting technique providing comprehensive trend analysis"
    category = "trend"
    
    # NO DEPENDENCIES - Pure price analysis
    dependencies = []
    
    default_params = {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "displacement": 26
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span", "cloud_strength"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicator and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Ichimoku Cloud columns added
        """
        result_df = df.copy()
        
        # Get parameters
        tenkan_period = self.params.get("tenkan_period", self.default_params["tenkan_period"])
        kijun_period = self.params.get("kijun_period", self.default_params["kijun_period"])
        senkou_b_period = self.params.get("senkou_b_period", self.default_params["senkou_b_period"])
        displacement = self.params.get("displacement", self.default_params["displacement"])
        
        # Validate required columns
        missing_cols = [col for col in self.requires_columns if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for Ichimoku: {missing_cols}")
        
        try:
            # Calculate Tenkan-sen (Conversion Line)
            # (highest high + lowest low) / 2 for the past tenkan_period
            tenkan_high = result_df["high"].rolling(window=tenkan_period, min_periods=1).max()
            tenkan_low = result_df["low"].rolling(window=tenkan_period, min_periods=1).min()
            result_df["tenkan_sen"] = (tenkan_high + tenkan_low) / 2
            
            # Calculate Kijun-sen (Base Line)
            # (highest high + lowest low) / 2 for the past kijun_period
            kijun_high = result_df["high"].rolling(window=kijun_period, min_periods=1).max()
            kijun_low = result_df["low"].rolling(window=kijun_period, min_periods=1).min()
            result_df["kijun_sen"] = (kijun_high + kijun_low) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            # (Tenkan-sen + Kijun-sen) / 2, displaced forward
            senkou_a = (result_df["tenkan_sen"] + result_df["kijun_sen"]) / 2
            result_df["senkou_span_a"] = senkou_a.shift(displacement)
            
            # Calculate Senkou Span B (Leading Span B)
            # (highest high + lowest low) / 2 for senkou_b_period, displaced forward
            senkou_b_high = result_df["high"].rolling(window=senkou_b_period, min_periods=1).max()
            senkou_b_low = result_df["low"].rolling(window=senkou_b_period, min_periods=1).min()
            senkou_b = (senkou_b_high + senkou_b_low) / 2
            result_df["senkou_span_b"] = senkou_b.shift(displacement)
            
            # Calculate Chikou Span (Lagging Span)
            # Current closing price, displaced backward
            result_df["chikou_span"] = result_df["close"].shift(-displacement)
            
            # Calculate cloud strength and direction
            # Positive: bullish cloud (Senkou A > Senkou B)
            # Negative: bearish cloud (Senkou A < Senkou B)
            # Magnitude: difference between spans
            result_df["cloud_strength"] = result_df["senkou_span_a"] - result_df["senkou_span_b"]
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating Ichimoku components: {e}")
            
            # Initialize with NaN if calculation fails
            for col in self.output_columns:
                result_df[col] = np.nan
        
        return result_df