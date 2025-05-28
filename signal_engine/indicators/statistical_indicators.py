"""
Statistical indicators for the trading system with Smart Dependencies.
These indicators use statistical methods to analyze price action.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional
from scipy import stats

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class ZScoreIndicator(BaseIndicator):
    """Calculates Z-Score for various metrics with Smart Dependencies."""
    
    name = "zscore"
    display_name = "Z-Score"
    description = "Calculates Z-Score showing standard deviations from mean"
    category = "statistical"
    
    # SMART DEPENDENCIES - Will auto-resolve to appropriate indicators
    dependencies = ["rsi_14", "macd_line"]  # Common indicators to apply Z-Score to
    
    default_params = {
        "window": 100,
        "apply_to": ["close", "volume", "rsi_14", "macd_line"]  # Can reference dependency columns
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
            # Check if column exists (dependencies should ensure this)
            if column not in result_df.columns:
                # Log warning but continue with other columns
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Column '{column}' not found for Z-Score calculation. Skipping.")
                continue
                
            z_col = f"{column}_zscore"
            
            # Calculate Z-Score with better error handling
            try:
                rolling_mean = result_df[column].rolling(window=window, min_periods=1).mean()
                rolling_std = result_df[column].rolling(window=window, min_periods=1).std()
                
                # Avoid division by zero
                rolling_std = rolling_std.replace(0, np.nan)
                result_df[z_col] = (result_df[column] - rolling_mean) / rolling_std
                
                # Handle infinite values
                result_df[z_col] = result_df[z_col].replace([np.inf, -np.inf], np.nan)
                
                # Add to output columns
                self.output_columns.append(z_col)
                
                # Add percentile rank (0-100) with improved calculation
                percentile_col = f"{column}_percentile"
                result_df[percentile_col] = result_df[column].rolling(
                    window=window, min_periods=10
                ).apply(
                    lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) if len(x.dropna()) >= 10 else 50,
                    raw=False
                )
                
                # Add to output columns
                self.output_columns.append(percentile_col)
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error calculating Z-Score for {column}: {e}")
        
        return result_df


class KeltnerChannelIndicator(BaseIndicator):
    """Calculates Keltner Channel with Smart Dependencies."""
    
    name = "keltner"
    display_name = "Keltner Channel"
    description = "Volatility-based bands using EMA and ATR"
    category = "volatility"
    
    # SMART DEPENDENCIES - Need EMA and ATR
    dependencies = ["ema_20", "atr_10"]  # Will auto-resolve to ema and atr indicators
    
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
        
        # Smart validation: Check if dependencies are available
        required_ema = f"ema_{ema_window}"
        required_atr = f"atr_{atr_window}"
        
        missing_deps = []
        if required_ema not in result_df.columns:
            missing_deps.append(required_ema)
        if required_atr not in result_df.columns:
            missing_deps.append(required_atr)
        
        if missing_deps:
            # Fallback: Calculate manually if dependencies not available
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing dependencies {missing_deps} for Keltner Channel. Calculating manually.")
            
            # Calculate EMA manually
            result_df[required_ema] = ta.trend.EMAIndicator(
                close=result_df[price_column],
                window=ema_window
            ).ema_indicator()
            
            # Calculate ATR manually
            result_df[required_atr] = ta.volatility.AverageTrueRange(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=atr_window
            ).average_true_range()
        
        # Use dependency columns (now guaranteed to exist)
        result_df["keltner_middle"] = result_df[required_ema]
        
        # Calculate upper and lower bands
        result_df["keltner_upper"] = result_df["keltner_middle"] + (atr_multiplier * result_df[required_atr])
        result_df["keltner_lower"] = result_df["keltner_middle"] - (atr_multiplier * result_df[required_atr])
        
        # Calculate channel width as percentage of middle line
        result_df["keltner_width"] = (
            (result_df["keltner_upper"] - result_df["keltner_lower"]) / 
            result_df["keltner_middle"]
        ) * 100
        
        # Calculate price position within channel (0 = lower band, 1 = upper band)
        channel_range = result_df["keltner_upper"] - result_df["keltner_lower"]
        channel_range = channel_range.replace(0, np.nan)  # Avoid division by zero
        
        result_df["keltner_position"] = (
            (result_df[price_column] - result_df["keltner_lower"]) / channel_range
        )
        
        return result_df


class StandardDeviationIndicator(BaseIndicator):
    """Calculates Standard Deviation based indicators with Smart Dependencies."""
    
    name = "std_deviation"
    display_name = "Standard Deviation"
    description = "Standard deviation based indicators for volatility analysis"
    category = "statistical"
    
    # NO DEPENDENCIES - This is a base statistical indicator
    dependencies = []
    
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
        # Clean copy to start with
        result_df = df.copy()
        
        # Get parameters with validation
        windows = self.params.get("windows", self.default_params["windows"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        annualization_factor = self.params.get("annualization_factor", self.default_params["annualization_factor"])
        
        # Validate parameters
        if not isinstance(windows, list) or len(windows) == 0:
            windows = self.default_params["windows"]
        
        # Clear output columns
        self.output_columns = []
        
        # Validate required columns exist
        if price_column not in result_df.columns:
            if "close" in result_df.columns:
                price_column = "close"  # Fallback to close
            else:
                # Cannot calculate
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Required column '{price_column}' not found for StandardDeviationIndicator")
                return result_df
        
        # Calculate returns with improved error handling
        try:
            result_df["returns"] = result_df[price_column].pct_change()
            # Clean infinite and NaN values
            result_df["returns"] = result_df["returns"].replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating returns: {e}")
            return result_df
        
        # Calculate standard deviation for all windows
        std_columns = {}
        for window in windows:
            std_col = f"std_{window}"
            try:
                # Calculate rolling standard deviation with minimum periods
                result_df[std_col] = result_df["returns"].rolling(
                    window=window, min_periods=max(1, window // 4)
                ).std().fillna(0)
                
                std_columns[window] = std_col
                self.output_columns.append(std_col)
                
                # Annualized volatility
                vol_col = f"volatility_{window}"
                result_df[vol_col] = result_df[std_col] * np.sqrt(annualization_factor)
                self.output_columns.append(vol_col)
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error calculating std for window {window}: {e}")
        
        # Calculate relative volatility (current vs longer-term)
        for window in windows:
            if window < max(windows) and window in std_columns:
                rel_vol_col = f"rel_vol_{window}_{max(windows)}"
                max_std_col = std_columns.get(max(windows))
                
                if max_std_col and max_std_col in result_df.columns:
                    # Avoid division by zero
                    denominator = result_df[max_std_col].replace(0, np.nan)
                    result_df[rel_vol_col] = (result_df[std_columns[window]] / denominator).fillna(1.0)
                    self.output_columns.append(rel_vol_col)
        
        # Add volatility regime analysis
        if len(windows) > 0 and len(result_df) >= 100:
            med_window = windows[len(windows) // 2] if len(windows) > 1 else windows[0]
            std_col = std_columns.get(med_window)
            
            if std_col and std_col in result_df.columns:
                try:
                    # Volatility percentile calculation with improved error handling
                    def safe_percentile(x):
                        clean_x = x.dropna()
                        if len(clean_x) < 5:  # Need minimum data points
                            return 50
                        current_val = clean_x.iloc[-1] if len(clean_x) > 0 else 0
                        if pd.isna(current_val):
                            return 50
                        return 100 * (clean_x < current_val).mean()
                    
                    result_df["volatility_percentile"] = result_df[std_col].rolling(
                        window=min(100, len(result_df)), min_periods=10
                    ).apply(safe_percentile, raw=False).fillna(50)
                    
                    self.output_columns.append("volatility_percentile")
                    
                    # Volatility regime classification
                    result_df["volatility_regime"] = "normal"
                    result_df.loc[result_df["volatility_percentile"] >= 80, "volatility_regime"] = "high"
                    result_df.loc[result_df["volatility_percentile"] <= 20, "volatility_regime"] = "low"
                    
                    self.output_columns.append("volatility_regime")
                    
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error calculating volatility regime: {e}")
        
        return result_df


class LinearRegressionIndicator(BaseIndicator):
    """Calculates linear regression based indicators with Smart Dependencies."""
    
    name = "linear_regression"
    display_name = "Linear Regression"
    description = "Linear regression based indicators for trend analysis"
    category = "statistical"
    
    # NO DEPENDENCIES - Works directly with price data
    dependencies = []
    
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
        
        # Get parameters with validation
        windows = self.params.get("windows", self.default_params["windows"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        forecast_periods = self.params.get("forecast_periods", self.default_params["forecast_periods"])
        
        # Validate parameters
        if not isinstance(windows, list) or len(windows) == 0:
            windows = self.default_params["windows"]
        
        # Clear output columns list
        self.output_columns = []
        
        # Validate price column
        if price_column not in result_df.columns:
            price_column = "close"  # Fallback
        
        # Calculate linear regression for each window
        for window in windows:
            # Skip if insufficient data
            if len(result_df) < window:
                continue
                
            # Initialize columns for this window
            reg_slope_col = f"reg_slope_{window}"
            reg_intercept_col = f"reg_intercept_{window}"
            reg_r2_col = f"reg_r2_{window}"
            reg_line_col = f"reg_line_{window}"
            reg_dev_col = f"reg_deviation_{window}"
            reg_forecast_col = f"reg_forecast_{window}"
            
            # Initialize with NaN
            for col in [reg_slope_col, reg_intercept_col, reg_r2_col, reg_line_col, reg_dev_col, reg_forecast_col]:
                result_df[col] = np.nan
                self.output_columns.append(col)
            
            # Calculate regression for each point with sufficient history
            for i in range(window, len(result_df)):
                try:
                    # Get price window
                    price_window = result_df[price_column].iloc[i-window:i].values
                    
                    # Skip if insufficient valid data
                    if len(price_window) < window // 2:
                        continue
                    
                    # Create X array (0 to window-1)
                    x = np.arange(len(price_window))
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(price_window)
                    if valid_mask.sum() < 3:  # Need at least 3 points
                        continue
                    
                    x_clean = x[valid_mask]
                    y_clean = price_window[valid_mask]
                    
                    # Fit linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                    
                    # Store results
                    result_df.loc[result_df.index[i], reg_slope_col] = slope
                    result_df.loc[result_df.index[i], reg_intercept_col] = intercept
                    result_df.loc[result_df.index[i], reg_r2_col] = r_value ** 2
                    
                    # Calculate regression line value (where the line is at the last point)
                    reg_line = intercept + slope * (len(x_clean) - 1)
                    result_df.loc[result_df.index[i], reg_line_col] = reg_line
                    
                    # Calculate deviation from regression line
                    actual_price = result_df[price_column].iloc[i-1]
                    if not pd.isna(actual_price) and reg_line != 0:
                        deviation = (actual_price - reg_line) / reg_line * 100
                        result_df.loc[result_df.index[i], reg_dev_col] = deviation
                    
                    # Calculate forecast (where the line will be in N periods)
                    forecast = intercept + slope * (len(x_clean) - 1 + forecast_periods)
                    result_df.loc[result_df.index[i], reg_forecast_col] = forecast
                    
                except Exception as e:
                    # Log error but continue
                    if i == window:  # Only log first error to avoid spam
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Error in linear regression for window {window}: {e}")
                    continue
        
        # Add consolidated features if we have multiple windows
        if len(windows) > 0:
            slope_cols = [f"reg_slope_{window}" for window in windows]
            existing_slope_cols = [col for col in slope_cols if col in result_df.columns]
            
            if existing_slope_cols:
                try:
                    # Average slope across all windows
                    result_df["avg_slope"] = result_df[existing_slope_cols].mean(axis=1)
                    self.output_columns.append("avg_slope")
                    
                    # Slope direction (1 for up, -1 for down)
                    result_df["slope_direction"] = np.sign(result_df["avg_slope"])
                    self.output_columns.append("slope_direction")
                    
                    # Significant slope (above threshold)
                    result_df["significant_slope"] = (abs(result_df["avg_slope"]) > 0.001).astype(int)
                    self.output_columns.append("significant_slope")
                    
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error calculating consolidated slope features: {e}")
        
        return result_df