"""
Advanced technical indicators for the trading system.
These indicators build upon basic indicators and provide more sophisticated analysis.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class AdaptiveRSIIndicator(BaseIndicator):
    """Calculates Adaptive RSI that adjusts period based on volatility."""
    
    name = "adaptive_rsi"
    display_name = "Adaptive RSI"
    description = "RSI that adapts to market volatility"
    category = "momentum"
    
    default_params = {
        "base_period": 14,
        "volatility_window": 100,
        "min_period": 5,
        "max_period": 30,
        "apply_to": "close"
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
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # First, calculate ATR to measure volatility
        atr = ta.volatility.average_true_range(
            high=result_df["high"],
            low=result_df["low"],
            close=result_df["close"],
            window=14
        )
        
        # Calculate ATR as percentage of price
        atr_pct = atr / result_df["close"] * 100
        
        # Initialize adaptive period column
        result_df["adaptive_rsi_period"] = base_period
        
        # Need at least volatility_window data points for percentile calculation
        if len(result_df) > volatility_window:
            for i in range(volatility_window, len(result_df)):
                # Calculate ATR percentile over the lookback window
                atr_window = atr_pct.iloc[i-volatility_window:i]
                current_atr = atr_pct.iloc[i]
                
                # Convert to percentile rank (0-1)
                try:
                    percentile = (atr_window < current_atr).mean()
                except:
                    percentile = 0.5  # Default to middle if calculation fails
                
                # Map percentile to RSI period
                # High volatility (high percentile) -> Shorter period
                # Low volatility (low percentile) -> Longer period
                period_range = max_period - min_period
                adaptive_period = max_period - (period_range * percentile)
                
                # Ensure integer period
                result_df.loc[result_df.index[i], "adaptive_rsi_period"] = int(adaptive_period)
        
        # Calculate RSI for each row based on its adaptive period
        # Note: This is computationally intensive as we calculate a separate RSI for each row
        for i in range(len(result_df)):
            period = int(result_df["adaptive_rsi_period"].iloc[i])
            
            # Need at least 'period' rows of data
            if i >= period:
                price_window = result_df[price_column].iloc[i-period:i+1]
                
                # Calculate RSI manually for this window
                delta = price_window.diff().dropna()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=period).mean().iloc[-1]
                avg_loss = loss.rolling(window=period).mean().iloc[-1]
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                result_df.loc[result_df.index[i], "adaptive_rsi"] = rsi
            else:
                result_df.loc[result_df.index[i], "adaptive_rsi"] = np.nan
        
        return result_df


class MultitimeframeEMAIndicator(BaseIndicator):
    """Calculates EMAs across multiple timeframes and their relationships."""
    
    name = "mtf_ema"
    display_name = "Multi-timeframe EMA"
    description = "Calculates EMAs on multiple timeframes and their alignment"
    category = "trend"
    
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
        if price_column not in result_df.columns or "open_time" not in result_df.columns:
            missing = []
            if price_column not in result_df.columns:
                missing.append(price_column)
            if "open_time" not in result_df.columns:
                missing.append("open_time")
            raise ValueError(f"Columns {missing} not found in dataframe")
        
        # Ensure open_time is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(result_df["open_time"]):
            result_df["open_time"] = pd.to_datetime(result_df["open_time"], unit='ms')
        
        # Calculate EMA for the base timeframe
        base_ema_name = f"ema_{period}"
        result_df[base_ema_name] = ta.trend.EMAIndicator(
            close=result_df[price_column], 
            window=period
        ).ema_indicator()
        
        # Add to output columns
        if base_ema_name not in self.output_columns:
            self.output_columns.append(base_ema_name)
        
        # Calculate EMAs for higher timeframes
        for tf_multiplier in timeframes[1:]:  # Skip the first (base) timeframe
            # Resample to higher timeframe
            resampled = result_df.resample(
                f'{tf_multiplier}T',  # Assuming base timeframe is in minutes
                on='open_time'
            ).agg({
                price_column: 'last',
                'high': 'max',
                'low': 'min',
                'open': 'first',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate EMA on the resampled data
            tf_ema = ta.trend.EMAIndicator(
                close=resampled[price_column], 
                window=period
            ).ema_indicator()
            
            # Merge back to the original timeframe
            tf_ema_df = pd.DataFrame(tf_ema)
            tf_ema_df.columns = [f'ema_{period}_{tf_multiplier}x']
            
            # Forward fill to have values on all rows
            tf_ema_df = tf_ema_df.reindex(result_df['open_time']).ffill()
            
            # Add to original dataframe
            result_df[f'ema_{period}_{tf_multiplier}x'] = tf_ema_df.values
            
            # Add to output columns
            if f'ema_{period}_{tf_multiplier}x' not in self.output_columns:
                self.output_columns.append(f'ema_{period}_{tf_multiplier}x')
        
        # Calculate alignment metrics
        # The alignment value will be between -1 and 1:
        # 1: All EMAs are perfectly aligned in ascending order (bullish)
        # -1: All EMAs are perfectly aligned in descending order (bearish)
        # 0: No clear alignment
        
        ema_cols = [col for col in result_df.columns if col.startswith('ema_')]
        n_emas = len(ema_cols)
        
        if n_emas > 1:
            # Create alignment score column
            result_df['ema_alignment'] = 0
            
            for i in range(len(result_df)):
                # Get EMAs for this row
                ema_values = [result_df[col].iloc[i] for col in ema_cols]
                
                # Count how many are in ascending order
                ascending_count = 0
                for j in range(1, len(ema_values)):
                    if ema_values[j] > ema_values[j-1]:
                        ascending_count += 1
                
                # Count how many are in descending order
                descending_count = 0
                for j in range(1, len(ema_values)):
                    if ema_values[j] < ema_values[j-1]:
                        descending_count += 1
                
                # Normalize to [-1, 1]
                max_possible = n_emas - 1
                if ascending_count > descending_count:
                    alignment = ascending_count / max_possible
                else:
                    alignment = -descending_count / max_possible
                
                result_df.loc[result_df.index[i], 'ema_alignment'] = alignment
            
            # Add to output columns
            if 'ema_alignment' not in self.output_columns:
                self.output_columns.append('ema_alignment')
        
        return result_df


class HeikinAshiIndicator(BaseIndicator):
    """Calculates Heikin Ashi candlesticks."""
    
    name = "heikin_ashi"
    display_name = "Heikin Ashi"
    description = "Calculates Heikin Ashi candlesticks for smoother trend visualization"
    category = "price_transformation"
    
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
        
        # Initialize HA columns
        result_df["ha_close"] = (result_df["open"] + result_df["high"] + 
                                result_df["low"] + result_df["close"]) / 4
        
        # For the first row, HA open equals regular open
        if len(result_df) > 0:
            result_df.loc[result_df.index[0], "ha_open"] = result_df["open"].iloc[0]
        
        # Calculate HA open for the rest of the rows
        for i in range(1, len(result_df)):
            result_df.loc[result_df.index[i], "ha_open"] = (
                result_df["ha_open"].iloc[i-1] + result_df["ha_close"].iloc[i-1]
            ) / 2
        
        # Calculate HA high and low
        result_df["ha_high"] = result_df[["high", "ha_open", "ha_close"]].max(axis=1)
        result_df["ha_low"] = result_df[["low", "ha_open", "ha_close"]].min(axis=1)
        
        # Calculate trend direction based on HA (1 for bullish, -1 for bearish)
        result_df["ha_trend"] = np.where(
            result_df["ha_close"] > result_df["ha_open"], 
            1, 
            -1
        )
        
        return result_df


class SupertrendIndicator(BaseIndicator):
    """Calculates the Supertrend indicator."""
    
    name = "supertrend"
    display_name = "Supertrend"
    description = "Trend following indicator combining ATR with price action"
    category = "trend"
    
    default_params = {
        "atr_period": 10,
        "atr_multiplier": 3.0
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["supertrend", "supertrend_direction", "supertrend_upper", "supertrend_lower"]
    
# supertrend method continues...
    
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
        
        # Calculate ATR
        atr = ta.volatility.average_true_range(
            high=result_df["high"],
            low=result_df["low"],
            close=result_df["close"],
            window=atr_period
        )
        
        # Calculate basic upper and lower bands
        hl2 = (result_df["high"] + result_df["low"]) / 2
        
        # Initial upper and lower bands
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)
        
        # Initialize supertrend direction as True (bullish)
        result_df["supertrend_direction"] = True
        
        # Create supertrend columns
        result_df["supertrend_upper"] = 0.0
        result_df["supertrend_lower"] = 0.0
        result_df["supertrend"] = 0.0
        
        # Calculate Supertrend - we need to iterate through the dataframe
        for i in range(1, len(result_df)):
            # Current values
            curr_close = result_df["close"].iloc[i]
            curr_upper = final_upperband.iloc[i]
            curr_lower = final_lowerband.iloc[i]
            
            # Previous values
            prev_close = result_df["close"].iloc[i-1]
            prev_upper = final_upperband.iloc[i-1]
            prev_lower = final_lowerband.iloc[i-1]
            prev_supertrend = result_df["supertrend"].iloc[i-1]
            prev_direction = result_df["supertrend_direction"].iloc[i-1]
            
            # Calculate current direction
            if prev_close > prev_upper:
                curr_direction = True  # Bullish
            elif prev_close < prev_lower:
                curr_direction = False  # Bearish
            else:
                curr_direction = prev_direction
            
            # Calculate current supertrend value
            if curr_direction:
                curr_supertrend = curr_lower
            else:
                curr_supertrend = curr_upper
            
            # Update values in the dataframe
            result_df.loc[result_df.index[i], "supertrend_direction"] = curr_direction
            result_df.loc[result_df.index[i], "supertrend"] = curr_supertrend
            result_df.loc[result_df.index[i], "supertrend_upper"] = curr_upper
            result_df.loc[result_df.index[i], "supertrend_lower"] = curr_lower
        
        return result_df


class IchimokuIndicator(BaseIndicator):
    """Calculates Ichimoku Cloud indicator."""
    
    name = "ichimoku"
    display_name = "Ichimoku Cloud"
    description = "Japanese charting technique that provides more data points than standard candlestick charts"
    category = "trend"
    
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
        
        # Calculate Tenkan-sen (Conversion Line)
        # (highest high + lowest low) / 2 for the past tenkan_period
        tenkan_high = result_df["high"].rolling(window=tenkan_period).max()
        tenkan_low = result_df["low"].rolling(window=tenkan_period).min()
        result_df["tenkan_sen"] = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        # (highest high + lowest low) / 2 for the past kijun_period
        kijun_high = result_df["high"].rolling(window=kijun_period).max()
        kijun_low = result_df["low"].rolling(window=kijun_period).min()
        result_df["kijun_sen"] = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        # (Tenkan-sen + Kijun-sen) / 2, plotted displacement periods in the future
        result_df["senkou_span_a"] = ((result_df["tenkan_sen"] + result_df["kijun_sen"]) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        # (highest high + lowest low) / 2 for the past senkou_b_period, plotted displacement periods in the future
        senkou_b_high = result_df["high"].rolling(window=senkou_b_period).max()
        senkou_b_low = result_df["low"].rolling(window=senkou_b_period).min()
        result_df["senkou_span_b"] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        # Current closing price, plotted displacement periods in the past
        result_df["chikou_span"] = result_df["close"].shift(-displacement)
        
        # Calculate cloud strength and direction
        # Positive value indicates bullish cloud (Senkou A > Senkou B)
        # Negative value indicates bearish cloud (Senkou A < Senkou B)
        # Magnitude represents the difference between Senkou A and B
        result_df["cloud_strength"] = result_df["senkou_span_a"] - result_df["senkou_span_b"]
        
        return result_df