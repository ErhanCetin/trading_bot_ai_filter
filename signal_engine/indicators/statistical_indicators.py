"""
Statistical indicators for the trading system with Smart Dependencies - FIXED VERSION.
These indicators use statistical methods to analyze price action.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional
from scipy import stats

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class ZScoreIndicator(BaseIndicator):
    """Calculates Z-Score for various metrics with Smart Dependencies - FIXED."""
    
    name = "zscore"
    display_name = "Z-Score"
    description = "Calculates Z-Score showing standard deviations from mean"
    category = "statistical"
    
    # SMART DEPENDENCIES - Will auto-resolve to appropriate indicators
    dependencies = ["rsi_14", "macd_line"]  # Common indicators to apply Z-Score to
    
    default_params = {
        "window": 100,
        "apply_to": ["close", "volume"]  # Start with basic columns that always exist
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
        
        # Get parameters with validation
        window = self.params.get("window", self.default_params["window"])
        apply_to = self.params.get("apply_to", self.default_params["apply_to"])
        
        # Ensure apply_to is a list
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        elif not isinstance(apply_to, list):
            apply_to = self.default_params["apply_to"]
        
        # Clear output columns list
        self.output_columns = []
        
        # Add dependency columns if they exist (optional enhancement)
        available_dependency_columns = []
        for dep_col in ["rsi_14", "macd_line"]:
            if dep_col in result_df.columns:
                available_dependency_columns.append(dep_col)
        
        # Combine base columns with available dependency columns
        all_columns_to_process = list(apply_to) + available_dependency_columns
        
        # Remove duplicates while preserving order
        seen = set()
        columns_to_process = []
        for col in all_columns_to_process:
            if col not in seen:
                columns_to_process.append(col)
                seen.add(col)
        
        # Calculate Z-Score for each specified column
        for column in columns_to_process:
            # Check if column exists
            if column not in result_df.columns:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Column '{column}' not found for Z-Score calculation. Skipping.")
                continue
            
            # Skip if column has insufficient valid data
            valid_data = result_df[column].dropna()
            if len(valid_data) < max(10, window // 10):  # Need at least 10 points or 10% of window
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Column '{column}' has insufficient data for Z-Score. Skipping.")
                continue
                
            z_col = f"{column}_zscore"
            percentile_col = f"{column}_percentile"
            
            # Calculate Z-Score with better error handling
            try:
                # Use minimum periods to avoid too many NaN values
                min_periods = max(5, window // 10)  # At least 5 periods, or 10% of window
                
                rolling_mean = result_df[column].rolling(window=window, min_periods=min_periods).mean()
                rolling_std = result_df[column].rolling(window=window, min_periods=min_periods).std()
                
                # Avoid division by zero - replace zero std with small value
                rolling_std = rolling_std.replace(0, 1e-8)
                
                # Calculate Z-Score
                z_scores = (result_df[column] - rolling_mean) / rolling_std
                
                # Handle infinite values and extreme outliers
                z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
                
                # Cap extreme Z-scores at ±10 to prevent outliers
                z_scores = z_scores.clip(-10, 10)
                
                result_df[z_col] = z_scores
                self.output_columns.append(z_col)
                
                # Add percentile rank with improved calculation
                try:
                    def safe_percentile_rank(series):
                        """Safe percentile calculation with error handling."""
                        clean_data = series.dropna()
                        if len(clean_data) < 3:  # Need at least 3 points
                            return 50.0  # Return median percentile
                        
                        current_value = clean_data.iloc[-1]
                        if pd.isna(current_value):
                            return 50.0
                        
                        # Calculate percentile rank
                        percentile = (clean_data < current_value).sum() / len(clean_data) * 100
                        return min(100, max(0, percentile))  # Ensure 0-100 range
                    
                    result_df[percentile_col] = result_df[column].rolling(
                        window=window, min_periods=min_periods
                    ).apply(safe_percentile_rank, raw=False)
                    
                    self.output_columns.append(percentile_col)
                    
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error calculating percentile for {column}: {e}")
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error calculating Z-Score for {column}: {e}")
                continue
        
        return result_df


class KeltnerChannelIndicator(BaseIndicator):
    """Calculates Keltner Channel with Smart Dependencies - FIXED."""
    
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
        
        # Validate required columns
        missing_cols = [col for col in self.requires_columns if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for Keltner Channel: {missing_cols}")
        
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
            
            try:
                # Calculate EMA manually
                if required_ema not in result_df.columns:
                    result_df[required_ema] = ta.trend.EMAIndicator(
                        close=result_df[price_column],
                        window=ema_window
                    ).ema_indicator()
                
                # Calculate ATR manually
                if required_atr not in result_df.columns:
                    result_df[required_atr] = ta.volatility.AverageTrueRange(
                        high=result_df["high"],
                        low=result_df["low"],
                        close=result_df["close"],
                        window=atr_window
                    ).average_true_range()
            except Exception as e:
                logger.error(f"Error calculating manual dependencies for Keltner: {e}")
                raise
        
        try:
            # Use dependency columns (now guaranteed to exist)
            result_df["keltner_middle"] = result_df[required_ema]
            
            # Calculate upper and lower bands
            result_df["keltner_upper"] = result_df["keltner_middle"] + (atr_multiplier * result_df[required_atr])
            result_df["keltner_lower"] = result_df["keltner_middle"] - (atr_multiplier * result_df[required_atr])
            
            # Calculate channel width as percentage of middle line
            middle_safe = result_df["keltner_middle"].replace(0, np.nan)
            result_df["keltner_width"] = (
                (result_df["keltner_upper"] - result_df["keltner_lower"]) / middle_safe
            ) * 100
            
            # Calculate price position within channel (0 = lower band, 1 = upper band)
            channel_range = result_df["keltner_upper"] - result_df["keltner_lower"]
            channel_range = channel_range.replace(0, np.nan)  # Avoid division by zero
            
            result_df["keltner_position"] = (
                (result_df[price_column] - result_df["keltner_lower"]) / channel_range
            )
            
            # Clean up any extreme values
            result_df["keltner_position"] = result_df["keltner_position"].clip(-2, 3)  # Allow some overshoot
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error calculating Keltner Channel: {e}")
            
            # Initialize with NaN on error
            for col in self.output_columns:
                result_df[col] = np.nan
        
        return result_df


class StandardDeviationIndicator(BaseIndicator):
    """Calculates Standard Deviation based indicators - FIXED VERSION."""
    
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
        result_df = df.copy()
        
        # Get parameters with validation
        windows = self.params.get("windows", self.default_params["windows"])
        price_column = self.params.get("apply_to", self.default_params["apply_to"])
        annualization_factor = self.params.get("annualization_factor", self.default_params["annualization_factor"])
        
        # Validate parameters
        if not isinstance(windows, list) or len(windows) == 0:
            windows = self.default_params["windows"]
        
        # Validate price column
        if price_column not in result_df.columns:
            if "close" in result_df.columns:
                price_column = "close"
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Required column '{price_column}' not found for StandardDeviationIndicator")
                return result_df
        
        # Clear output columns
        self.output_columns = []
        
        try:
            # Calculate returns with improved error handling
            result_df["returns"] = result_df[price_column].pct_change()
            
            # Clean infinite and extreme values
            result_df["returns"] = result_df["returns"].replace([np.inf, -np.inf], np.nan)
            
            # Cap extreme returns at ±50% to prevent outliers from affecting calculations
            result_df["returns"] = result_df["returns"].clip(-0.5, 0.5)
            
            # Fill NaN with 0 for first row
            result_df["returns"] = result_df["returns"].fillna(0)
            
            # Calculate standard deviation for all windows
            std_columns = {}
            for window in windows:
                if window <= 0 or window > len(result_df):
                    continue  # Skip invalid window sizes
                
                std_col = f"std_{window}"
                vol_col = f"volatility_{window}"
                
                try:
                    # Calculate rolling standard deviation with appropriate minimum periods
                    min_periods = max(2, min(5, window // 4))  # At least 2, max 5, or 25% of window
                    
                    rolling_std = result_df["returns"].rolling(
                        window=window, min_periods=min_periods
                    ).std()
                    
                    # Handle NaN and zero values
                    rolling_std = rolling_std.fillna(0)
                    result_df[std_col] = rolling_std
                    
                    std_columns[window] = std_col
                    self.output_columns.append(std_col)
                    
                    # Annualized volatility
                    if annualization_factor > 0:
                        result_df[vol_col] = rolling_std * np.sqrt(annualization_factor)
                        self.output_columns.append(vol_col)
                    
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error calculating std for window {window}: {e}")
                    continue
            
            # Calculate relative volatility (current vs longer-term)
            if len(std_columns) > 1:
                sorted_windows = sorted(std_columns.keys())
                
                for i, window in enumerate(sorted_windows[:-1]):  # All except last
                    max_window = sorted_windows[-1]  # Largest window
                    
                    if window in std_columns and max_window in std_columns:
                        rel_vol_col = f"rel_vol_{window}_{max_window}"
                        
                        try:
                            # Avoid division by zero
                            denominator = result_df[std_columns[max_window]].replace(0, 1e-8)
                            result_df[rel_vol_col] = (result_df[std_columns[window]] / denominator)
                            
                            # Cap extreme ratios
                            result_df[rel_vol_col] = result_df[rel_vol_col].clip(0, 10)
                            
                            self.output_columns.append(rel_vol_col)
                        except Exception as e:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Error calculating relative volatility {rel_vol_col}: {e}")
            
            # Add volatility regime analysis
            if len(std_columns) > 0 and len(result_df) >= 50:  # Need reasonable amount of data
                # Use middle window for regime analysis
                window_keys = sorted(std_columns.keys())
                med_window = window_keys[len(window_keys) // 2]
                std_col = std_columns[med_window]
                
                try:
                    # Improved volatility percentile calculation
                    def safe_vol_percentile(x):
                        clean_x = x.dropna()
                        if len(clean_x) < 10:  # Need minimum data points
                            return 50.0
                        current_val = clean_x.iloc[-1] if len(clean_x) > 0 else 0
                        if pd.isna(current_val) or current_val == 0:
                            return 50.0
                        
                        # Calculate percentile
                        percentile = (clean_x < current_val).sum() / len(clean_x) * 100
                        return min(100, max(0, percentile))
                    
                    lookback_window = min(100, len(result_df))
                    result_df["volatility_percentile"] = result_df[std_col].rolling(
                        window=lookback_window, min_periods=20
                    ).apply(safe_vol_percentile, raw=False)
                    
                    # Fill NaN values with 50 (median percentile)
                    result_df["volatility_percentile"] = result_df["volatility_percentile"].fillna(50.0)
                    
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
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in StandardDeviationIndicator: {e}")
        
        return result_df


class LinearRegressionIndicator(BaseIndicator):
    """Calculates linear regression based indicators - FIXED VERSION."""
    
    name = "linear_regression"
    display_name = "Linear Regression"
    description = "Linear regression based indicators for trend analysis"
    category = "statistical"
    
    # NO DEPENDENCIES - Works directly with price data
    dependencies = []
    
    default_params = {
        "windows": [20, 50],  # Reduced default windows for better performance
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
        
        # Filter out invalid windows
        valid_windows = [w for w in windows if isinstance(w, int) and 5 <= w <= len(result_df)]
        if not valid_windows:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No valid windows for LinearRegressionIndicator")
            return result_df
        
        # Clear output columns list
        self.output_columns = []
        
        # Validate price column
        if price_column not in result_df.columns:
            price_column = "close"  # Fallback
        
        try:
            # Calculate linear regression for each valid window
            for window in valid_windows:
                # Skip if insufficient data
                if len(result_df) < window + 5:  # Need some extra data
                    continue
                    
                # Initialize columns for this window
                reg_slope_col = f"reg_slope_{window}"
                reg_r2_col = f"reg_r2_{window}"
                reg_line_col = f"reg_line_{window}"
                reg_dev_col = f"reg_deviation_{window}"
                
                # Initialize with NaN
                for col in [reg_slope_col, reg_r2_col, reg_line_col, reg_dev_col]:
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
                        if valid_mask.sum() < max(3, window // 4):  # Need sufficient points
                            continue
                        
                        x_clean = x[valid_mask]
                        y_clean = price_window[valid_mask]
                        
                        # Skip if all prices are the same (would cause division by zero)
                        if len(set(y_clean)) < 2:
                            continue
                        
                        # Fit linear regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                        
                        # Validate results
                        if not np.isfinite(slope) or not np.isfinite(r_value):
                            continue
                        
                        # Store results
                        result_df.loc[result_df.index[i], reg_slope_col] = slope
                        result_df.loc[result_df.index[i], reg_r2_col] = max(0, min(1, r_value ** 2))  # Ensure 0-1 range
                        
                        # Calculate regression line value (where the line is at the last point)
                        reg_line = intercept + slope * (len(x_clean) - 1)
                        if np.isfinite(reg_line):
                            result_df.loc[result_df.index[i], reg_line_col] = reg_line
                            
                            # Calculate deviation from regression line
                            actual_price = result_df[price_column].iloc[i-1]
                            if not pd.isna(actual_price) and reg_line != 0:
                                deviation = (actual_price - reg_line) / abs(reg_line) * 100
                                # Cap extreme deviations
                                deviation = max(-100, min(100, deviation))
                                result_df.loc[result_df.index[i], reg_dev_col] = deviation
                        
                    except Exception as e:
                        # Log error but continue
                        if i == window:  # Only log first error to avoid spam
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Error in linear regression for window {window}: {e}")
                        continue
            
            # Add consolidated features if we have multiple windows
            if len(valid_windows) > 0:
                slope_cols = [f"reg_slope_{window}" for window in valid_windows]
                existing_slope_cols = [col for col in slope_cols if col in result_df.columns]
                
                if existing_slope_cols:
                    try:
                        # Average slope across all windows
                        slope_data = result_df[existing_slope_cols]
                        result_df["avg_slope"] = slope_data.mean(axis=1, skipna=True)
                        self.output_columns.append("avg_slope")
                        
                        # Slope direction (1 for up, -1 for down, 0 for flat)
                        result_df["slope_direction"] = np.sign(result_df["avg_slope"])
                        self.output_columns.append("slope_direction")
                        
                        # Significant slope (above threshold)
                        slope_threshold = result_df[price_column].std() * 0.001  # Dynamic threshold
                        result_df["significant_slope"] = (abs(result_df["avg_slope"]) > slope_threshold).astype(int)
                        self.output_columns.append("significant_slope")
                        
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Error calculating consolidated slope features: {e}")
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in LinearRegressionIndicator: {e}")
        
        return result_df