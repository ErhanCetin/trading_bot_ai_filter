"""
Statistical indicators for the trading system.
These indicators use statistical methods to analyze price action.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional
from scipy import stats

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class ZScoreIndicator(BaseIndicator):
    """Calculates Z-Score for various metrics."""
    
    name = "zscore"
    display_name = "Z-Score"
    description = "Calculates Z-Score showing standard deviations from mean"
    category = "statistical"
    
    default_params = {
        "window": 100,
        "apply_to": ["close", "volume", "rsi_14", "macd"]
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Z-Score for various metrics and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Z-Score columns added
        """
        result_df = df.copy()
        
        # Get parameters
        window = self.params.get("window", self.default_params["window"])
        apply_to = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Clear output columns list
        self.output_columns = []
        
        # Calculate Z-Score for each specified column
        for column in apply_to:
            # Check if column exists
            if column not in result_df.columns:
                continue
                
            z_col = f"{column}_zscore"
            
            # Calculate Z-Score
            rolling_mean = result_df[column].rolling(window=window).mean()
            rolling_std = result_df[column].rolling(window=window).std()
            
            result_df[z_col] = (result_df[column] - rolling_mean) / rolling_std
            
            # Add to output columns
            self.output_columns.append(z_col)
            
            # Add percentile rank (0-100)
            percentile_col = f"{column}_percentile"
            result_df[percentile_col] = result_df[column].rolling(window=window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]),
                raw=False
            )
            
            # Add to output columns
            self.output_columns.append(percentile_col)
        
        return result_df


class KeltnerChannelIndicator(BaseIndicator):
    """Calculates Keltner Channel."""
    
    name = "keltner"
    display_name = "Keltner Channel"
    description = "Volatility-based bands using EMA and ATR"
    category = "volatility"
    
    default_params = {
        "ema_window": 20,
        "atr_window": 10,
        "atr_multiplier": 2.0,
        "apply_to": "close"
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["keltner_middle", "keltner_upper", "keltner_lower", "keltner_width", "keltner_position"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channel and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Keltner Channel columns added
        """
        result_df = df.copy()
        
        # Get parameters
        ema_window = self.params.get("ema_window", self.default_params["ema_window"])
        atr_window = self.params.get("atr_window", self.default_params["atr_window"])
        atr_multiplier = self.params.get("atr_multiplier", self.default_params["atr_multiplier"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Calculate EMA (middle line)
        result_df["keltner_middle"] = ta.trend.EMAIndicator(
            close=result_df[price_column],
            window=ema_window
        ).ema_indicator()
        
        # Calculate ATR
        result_df["atr"] = ta.volatility.AverageTrueRange(
            high=result_df["high"],
            low=result_df["low"],
            close=result_df["close"],
            window=atr_window
        ).average_true_range()
        
        # Calculate upper and lower bands
        result_df["keltner_upper"] = result_df["keltner_middle"] + (atr_multiplier * result_df["atr"])
        result_df["keltner_lower"] = result_df["keltner_middle"] - (atr_multiplier * result_df["atr"])
        
        # Calculate channel width as percentage of middle line
        result_df["keltner_width"] = (
            (result_df["keltner_upper"] - result_df["keltner_lower"]) / 
            result_df["keltner_middle"]
        ) * 100
        
        # Calculate price position within channel (0 = lower band, 1 = upper band)
        result_df["keltner_position"] = (
            (result_df[price_column] - result_df["keltner_lower"]) / 
            (result_df["keltner_upper"] - result_df["keltner_lower"])
        )
        
        return result_df


class StandardDeviationIndicator(BaseIndicator):
    """Calculates Standard Deviation based indicators."""
    
    name = "std_deviation"
    display_name = "Standard Deviation"
    description = "Standard deviation based indicators for volatility analysis"
    category = "statistical"
    
    default_params = {
        "windows": [5, 20, 50],
        "apply_to": "close",
        "annualization_factor": 252  # For annual volatility calculation
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate standard deviation based indicators and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with standard deviation columns added
        """
        result_df = df.copy()
        
        # Get parameters
        windows = self.params.get("windows", self.default_params["windows"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        annualization_factor = self.params.get("annualization_factor", self.default_params["annualization_factor"])
        
        # Clear output columns list
        self.output_columns = []
        
        # Calculate returns if not already present
        if "returns" not in result_df.columns:
            result_df["returns"] = result_df[price_column].pct_change()
        
        # Calculate standard deviation for each window
        for window in windows:
            # Standard deviation of returns
            std_col = f"std_{window}"
            result_df[std_col] = result_df["returns"].rolling(window=window).std()
            self.output_columns.append(std_col)
            
            # Annualized volatility
            vol_col = f"volatility_{window}"
            result_df[vol_col] = result_df[std_col] * np.sqrt(annualization_factor)
            self.output_columns.append(vol_col)
            
            # Relative volatility (current vol compared to longer-term vol)
            if window < max(windows):
                rel_vol_col = f"rel_vol_{window}_{max(windows)}"
                result_df[rel_vol_col] = result_df[std_col] / result_df[f"std_{max(windows)}"]
                self.output_columns.append(rel_vol_col)
        
        # Add volatility regimes
        # Use the medium window for this
        med_window = windows[len(windows) // 2]
        std_col = f"std_{med_window}"
        
        # Calculate percentile of current volatility
        result_df["volatility_percentile"] = result_df[std_col].rolling(window=100).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]),
            raw=False
        )
        self.output_columns.append("volatility_percentile")
        
        # Categorize volatility regime
        result_df["volatility_regime"] = "normal"
        result_df.loc[result_df["volatility_percentile"] >= 80, "volatility_regime"] = "high"
        result_df.loc[result_df["volatility_percentile"] <= 20, "volatility_regime"] = "low"
        self.output_columns.append("volatility_regime")
        
        return result_df


class LinearRegressionIndicator(BaseIndicator):
    """Calculates linear regression based indicators."""
    
    name = "linear_regression"
    display_name = "Linear Regression"
    description = "Linear regression based indicators for trend analysis"
    category = "statistical"
    
    default_params = {
        "windows": [20, 50, 100],
        "apply_to": "close",
        "forecast_periods": 5
    }
    
    requires_columns = ["close"]
    output_columns = []  # Will be dynamically generated
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate linear regression based indicators and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with linear regression columns added
        """
        result_df = df.copy()
        
        # Get parameters
        windows = self.params.get("windows", self.default_params["windows"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        forecast_periods = self.params.get("forecast_periods", self.default_params["forecast_periods"])
        
        # Clear output columns list
        self.output_columns = []
        
        # Calculate linear regression for each window
        for window in windows:
            # Skip if we don't have enough data
            if len(result_df) < window:
                continue
                
            # Calculate regression values
            reg_slope_col = f"reg_slope_{window}"
            reg_intercept_col = f"reg_intercept_{window}"
            reg_r2_col = f"reg_r2_{window}"
            reg_line_col = f"reg_line_{window}"
            reg_dev_col = f"reg_deviation_{window}"
            reg_forecast_col = f"reg_forecast_{window}"
            
            # Initialize columns
            result_df[reg_slope_col] = np.nan
            result_df[reg_intercept_col] = np.nan
            result_df[reg_r2_col] = np.nan
            result_df[reg_line_col] = np.nan
            result_df[reg_dev_col] = np.nan
            result_df[reg_forecast_col] = np.nan
            
            # Add to output columns
            self.output_columns.extend([
                reg_slope_col, reg_intercept_col, reg_r2_col, 
                reg_line_col, reg_dev_col, reg_forecast_col
            ])
            
            # Calculate regression for each point
            for i in range(window, len(result_df)):
                # Get price window
                price_window = result_df[price_column].iloc[i-window:i].values
                
                # Create X array (0 to window-1)
                x = np.arange(window)
                
                # Fit linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, price_window)
                
                # Store results
                result_df.loc[result_df.index[i], reg_slope_col] = slope
                result_df.loc[result_df.index[i], reg_intercept_col] = intercept
                result_df.loc[result_df.index[i], reg_r2_col] = r_value ** 2
                
                # Calculate regression line value (where the line is at the last point)
                reg_line = intercept + slope * (window - 1)
                result_df.loc[result_df.index[i], reg_line_col] = reg_line
                
                # Calculate deviation from regression line
                actual_price = result_df[price_column].iloc[i-1]
                deviation = (actual_price - reg_line) / reg_line * 100
                result_df.loc[result_df.index[i], reg_dev_col] = deviation
                
                # Calculate forecast (where the line will be in N periods)
                forecast = intercept + slope * (window - 1 + forecast_periods)
                result_df.loc[result_df.index[i], reg_forecast_col] = forecast
        
        # Add consolidated features
        
        # Average slope across all windows
        if len(windows) > 0:
            slope_cols = [f"reg_slope_{window}" for window in windows]
            existing_slope_cols = [col for col in slope_cols if col in result_df.columns]
            
            if existing_slope_cols:
                result_df["avg_slope"] = result_df[existing_slope_cols].mean(axis=1)
                self.output_columns.append("avg_slope")
                
                # Slope direction (1 for up, -1 for down)
                result_df["slope_direction"] = np.sign(result_df["avg_slope"])
                self.output_columns.append("slope_direction")
                
                # Slope is significant (above threshold)
                result_df["significant_slope"] = (abs(result_df["avg_slope"]) > 0.001).astype(int)
                self.output_columns.append("significant_slope")
        
        return result_df