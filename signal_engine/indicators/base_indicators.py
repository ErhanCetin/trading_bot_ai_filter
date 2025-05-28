"""
Base technical indicators for the trading system.
These are standard indicators used in technical analysis.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class EMAIndicator(BaseIndicator):
    """Calculates Exponential Moving Average."""
    
    name = "ema"
    display_name = "Exponential Moving Average"
    description = "Calculates EMA with configurable periods"
    category = "trend"
    
    default_params = {
        "periods": [9, 21, 50, 200],  # Multiple periods by default
        "apply_to": "close"
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA values for multiple periods.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with ema columns added
        """
        result_df = df.copy()
        
        # Get parameters
        periods = self.params.get("periods", self.default_params["periods"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # Calculate EMAs for all periods
        for period in periods:
            col_name = f"ema_{period}"
            result_df[col_name] = ta.trend.EMAIndicator(
                close=result_df[price_column], 
                window=period
            ).ema_indicator()
            
            # Update output columns
            if col_name not in self.output_columns:
                self.output_columns.append(col_name)
        
        return result_df


class SMAIndicator(BaseIndicator):
    """Calculates Simple Moving Average."""
    
    name = "sma"
    display_name = "Simple Moving Average"
    description = "Calculates SMA with configurable periods"
    category = "trend"
    
    default_params = {
        "periods": [10, 20, 50, 200],  # Multiple periods by default
        "apply_to": "close"
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMA values for multiple periods.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with sma columns added
        """
        result_df = df.copy()
        
        # Get parameters
        periods = self.params.get("periods", self.default_params["periods"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # Calculate SMAs for all periods
        for period in periods:
            col_name = f"sma_{period}"
            result_df[col_name] = ta.trend.SMAIndicator(
                close=result_df[price_column], 
                window=period
            ).sma_indicator()
            
            # Update output columns
            if col_name not in self.output_columns:
                self.output_columns.append(col_name)
        
        return result_df


class RSIIndicator(BaseIndicator):
    """Calculates Relative Strength Index."""
    
    name = "rsi"
    display_name = "Relative Strength Index"
    description = "Measures the speed and change of price movements"
    category = "momentum"
    
    default_params = {
        "periods": [7, 14, 21],  # Multiple periods
        "apply_to": "close"
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for multiple periods.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with RSI column added
        """
        result_df = df.copy()
        
        # Get parameters
        periods = self.params.get("periods", self.default_params["periods"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # Calculate RSI for all periods
        for period in periods:
            col_name = f"rsi_{period}"
            result_df[col_name] = ta.momentum.RSIIndicator(
                close=result_df[price_column],
                window=period
            ).rsi()
            
            # Update output columns
            if col_name not in self.output_columns:
                self.output_columns.append(col_name)
        
        # Add standard aliases for commonly used periods
        if 14 in periods:
            result_df["rsi_14"] = result_df["rsi_14"]  # Explicit alias for clarity
        
        return result_df


class MACDIndicator(BaseIndicator):
    """Calculates Moving Average Convergence Divergence."""
    
    name = "macd"
    display_name = "MACD"
    description = "Moving Average Convergence Divergence indicator"
    category = "trend"
    
    default_params = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "apply_to": "close"
    }
    
    requires_columns = ["close"]
    output_columns = ["macd_line", "macd_signal", "macd_histogram", "macd_crossover"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with MACD columns added
        """
        result_df = df.copy()
        
        # Get parameters
        fast_period = self.params.get("fast_period", self.default_params["fast_period"])
        slow_period = self.params.get("slow_period", self.default_params["slow_period"])
        signal_period = self.params.get("signal_period", self.default_params["signal_period"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            close=result_df[price_column],
            window_slow=slow_period,
            window_fast=fast_period,
            window_sign=signal_period
        )
        
        result_df["macd_line"] = macd_indicator.macd()
        result_df["macd_signal"] = macd_indicator.macd_signal()
        result_df["macd_histogram"] = macd_indicator.macd_diff()
        
        # Calculate MACD crossover signal
        result_df["macd_crossover"] = 0
        
        # Crossover detection (1 for bullish, -1 for bearish, 0 for no crossover)
        result_df["macd_crossover"] = 0
        macd_diff = result_df["macd_line"] - result_df["macd_signal"]
        prev_diff = macd_diff.shift(1)

        # Vectorized crossover detection
        result_df.loc[(prev_diff < 0) & (macd_diff > 0), "macd_crossover"] = 1   # Bullish
        result_df.loc[(prev_diff > 0) & (macd_diff < 0), "macd_crossover"] = -1  # Bearish
                
        return result_df


# base_indicators.py'de ATRIndicator güncellenmesi

class ATRIndicator(BaseIndicator):
    """Calculates Average True Range for volatility measurement."""
    
    name = "atr"
    display_name = "Average True Range"
    description = "Measures market volatility across multiple periods"
    category = "volatility"
    
    default_params = {
        "windows": [14, 50],  # Multiple periods like other indicators
        "calculate_percent": True  # Also calculate ATR as percentage
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR for multiple periods and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with ATR columns added
        """
        result_df = df.copy()
        
        # Get parameters
        windows = self.params.get("windows", self.default_params["windows"])
        calculate_percent = self.params.get("calculate_percent", self.default_params["calculate_percent"])
        
        # Clear output columns list
        self.output_columns = []
        
        # Calculate ATR for each window
        for window in windows:
            atr_col = f"atr_{window}"
            
            # Calculate ATR
            indicator = ta.volatility.AverageTrueRange(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=window
            )
            
            result_df[atr_col] = indicator.average_true_range()
            self.output_columns.append(atr_col)
            
            # Calculate ATR as percentage of price if requested
            if calculate_percent:
                atr_pct_col = f"atr_{window}_percent"
                result_df[atr_pct_col] = result_df[atr_col] / result_df["close"] * 100
                self.output_columns.append(atr_pct_col)
        
        # Add backward compatibility for default single ATR
        if 14 in windows:
            result_df["atr"] = result_df["atr_14"]  # Default ATR alias
            result_df["atr_percent"] = result_df["atr_14_percent"] if calculate_percent else result_df["atr_14"] / result_df["close"] * 100
            
            # Add these to output columns for compatibility
            if "atr" not in self.output_columns:
                self.output_columns.append("atr")
            if "atr_percent" not in self.output_columns:
                self.output_columns.append("atr_percent")
        
        return result_df


class StochasticIndicator(BaseIndicator):
    """Calculates Stochastic Oscillator."""
    
    name = "stochastic"
    display_name = "Stochastic Oscillator"
    description = "Compares a particular closing price to a range of prices over time"
    category = "momentum"
    
    default_params = {
        "window": 14,
        "smooth_window": 3,
        "d_window": 3
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["stoch_k", "stoch_d", "stoch_crossover"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Stochastic columns added
        """
        result_df = df.copy()
        
        # Get parameters
        window = self.params.get("window", self.default_params["window"])
        smooth_window = self.params.get("smooth_window", self.default_params["smooth_window"])
        d_window = self.params.get("d_window", self.default_params["d_window"])
        
        # Calculate Stochastic
        indicator = ta.momentum.StochasticOscillator(
            high=result_df["high"],
            low=result_df["low"],
            close=result_df["close"],
            window=window,
            smooth_window=smooth_window
        )
        
        result_df["stoch_k"] = indicator.stoch()
        result_df["stoch_d"] = indicator.stoch_signal()
        
        # Calculate Stochastic crossover signal
        result_df["stoch_crossover"] = 0
        
        # Crossover detection (1 for bullish, -1 for bearish, 0 for no crossover)
        for i in range(1, len(result_df)):
            prev_k = result_df["stoch_k"].iloc[i-1]
            prev_d = result_df["stoch_d"].iloc[i-1]
            curr_k = result_df["stoch_k"].iloc[i]
            curr_d = result_df["stoch_d"].iloc[i]
            
            if prev_k < prev_d and curr_k > curr_d:
                result_df.loc[result_df.index[i], "stoch_crossover"] = 1  # Bullish crossover
            elif prev_k > prev_d and curr_k < curr_d:
                result_df.loc[result_df.index[i], "stoch_crossover"] = -1  # Bearish crossover
        
        return result_df

class BollingerBandsIndicator(BaseIndicator):
    """Calculates Bollinger Bands."""
    
    name = "bollinger"
    display_name = "Bollinger Bands"
    description = "Volatility bands placed above and below a moving average"
    category = "volatility"
    
    default_params = {
        "window": 20,
        "window_dev": 2.0,
        "apply_to": "close"
    }
    
    requires_columns = ["close"]
    output_columns = [
        "bollinger_upper", "bollinger_middle", "bollinger_lower", 
        "bollinger_width", "bollinger_pct_b"  # bollinger_width standardized
    ]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        result_df = df.copy()
        
        # Get parameters
        window = self.params.get("window", self.default_params["window"])
        window_dev = self.params.get("window_dev", self.default_params["window_dev"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Validate columns
        if price_column not in result_df.columns:
            raise ValueError(f"Column {price_column} not found in dataframe")
        
        # Calculate Bollinger Bands
        indicator = ta.volatility.BollingerBands(
            close=result_df[price_column],
            window=window,
            window_dev=window_dev
        )
        
        result_df["bollinger_upper"] = indicator.bollinger_hband()
        result_df["bollinger_middle"] = indicator.bollinger_mavg()
        result_df["bollinger_lower"] = indicator.bollinger_lband()
        result_df["bollinger_width"] = indicator.bollinger_wband()  # Standardized name
        result_df["bollinger_pct_b"] = indicator.bollinger_pband()
        
        return result_df    