"""
Enhanced base strength calculator with comprehensive error handling and validation.
FIXED VERSION - Robust validation and performance improvements.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseStrengthCalculator(ABC):
    """Enhanced base class for all signal strength calculators."""
    
    name = "base_strength"
    display_name = "Base Strength Calculator"
    description = "Base class for all strength calculators"
    category = "base"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strength calculator with parameters and validation.
        
        Args:
            params: Optional parameters to override defaults
        """
        self.params = self.default_params.copy() if hasattr(self, 'default_params') else {}
        if params:
            self.params.update(params)
        
        # Initialize logging for this calculator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate strength values for signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values (typically 1 for long, -1 for short, 0 for no signal)
            
        Returns:
            Series with signal strength values (0-100 scale, 0 = weakest, 100 = strongest)
        """
        pass
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Enhanced dataframe validation with detailed error reporting.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic DataFrame validation
            if df is None or df.empty:
                self.logger.warning("DataFrame is None or empty")
                return False
            
            # Check for required indicators
            required_columns = getattr(self, 'required_indicators', [])
            
            if required_columns:
                missing_required = [col for col in required_columns if col not in df.columns]
                if missing_required:
                    self.logger.warning(f"Missing required indicators: {missing_required}")
                    return False
            
            # Check for optional indicators (log warnings but don't fail)
            optional_columns = getattr(self, 'optional_indicators', [])
            if optional_columns:
                missing_optional = [col for col in optional_columns if col not in df.columns]
                if missing_optional:
                    self.logger.debug(f"Missing optional indicators: {missing_optional}")
            
            # Validate data quality
            if len(df) < 10:  # Minimum data requirement
                self.logger.warning(f"Insufficient data: {len(df)} rows, minimum 10 required")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating DataFrame: {e}")
            return False
    
    def validate_signals(self, signals: pd.Series) -> bool:
        """
        Validate signals series.
        
        Args:
            signals: Series with signal values
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if signals is None or signals.empty:
                self.logger.warning("Signals series is None or empty")
                return False
            
            # Check for valid signal values
            unique_signals = signals.dropna().unique()
            valid_signal_range = all(-10 <= val <= 10 for val in unique_signals)
            
            if not valid_signal_range:
                self.logger.warning(f"Signals contain values outside expected range [-10, 10]: {unique_signals}")
            
            # Check if there are any non-zero signals
            non_zero_signals = (signals != 0).sum()
            if non_zero_signals == 0:
                self.logger.info("No non-zero signals found")
            else:
                self.logger.debug(f"Found {non_zero_signals} non-zero signals out of {len(signals)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signals: {e}")
            return False
    
    def preprocess_data(self, df: pd.DataFrame, signals: pd.Series) -> Tuple[pd.DataFrame, pd.Series, bool]:
        """
        Preprocess data with error handling and cleaning.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Tuple of (cleaned_df, cleaned_signals, is_valid)
        """
        try:
            # Align indices
            common_index = df.index.intersection(signals.index)
            
            if len(common_index) == 0:
                self.logger.error("No common indices between DataFrame and signals")
                return df, signals, False
            
            df_clean = df.loc[common_index].copy()
            signals_clean = signals.loc[common_index].copy()
            
            # Basic data cleaning
            # Remove rows where all indicator values are NaN
            indicator_cols = [col for col in df_clean.columns if col not in ['open_time', 'timestamp']]
            
            if indicator_cols:
                all_nan_mask = df_clean[indicator_cols].isna().all(axis=1)
                if all_nan_mask.any():
                    self.logger.debug(f"Removing {all_nan_mask.sum()} rows with all NaN indicator values")
                    df_clean = df_clean[~all_nan_mask]
                    signals_clean = signals_clean[~all_nan_mask]
            
            # Validate final data
            if len(df_clean) < 5:
                self.logger.warning(f"Insufficient data after cleaning: {len(df_clean)} rows")
                return df_clean, signals_clean, False
            
            return df_clean, signals_clean, True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return df, signals, False
    
    def calculate_with_validation(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate strength with comprehensive validation and error handling.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values
        """
        # Get fallback strength
        fallback_strength = self.params.get('fallback_strength', 50)
        
        try:
            # Validate inputs
            if not self.validate_signals(signals):
                self.logger.warning("Invalid signals, returning fallback strength")
                return pd.Series(fallback_strength, index=signals.index)
            
            # Preprocess data
            df_clean, signals_clean, is_valid = self.preprocess_data(df, signals)
            
            if not is_valid:
                self.logger.warning("Data preprocessing failed, returning fallback strength")
                return pd.Series(fallback_strength, index=signals.index)
            
            # Validate DataFrame
            if not self.validate_dataframe(df_clean):
                self.logger.warning("DataFrame validation failed, using fallback for signals")
                strength = pd.Series(0, index=signals.index)
                strength.loc[signals != 0] = fallback_strength
                return strength
            
            # Call the actual calculation method
            result = self.calculate(df_clean, signals_clean)
            
            # Validate result
            if not self._validate_result(result, signals):
                self.logger.warning("Result validation failed, returning fallback strength")
                return pd.Series(fallback_strength, index=signals.index)
            
            # Ensure result has same index as original signals
            if not result.index.equals(signals.index):
                result = result.reindex(signals.index, fill_value=0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in strength calculation: {e}")
            # Return fallback strength for all signals
            strength = pd.Series(0, index=signals.index)
            strength.loc[signals != 0] = fallback_strength
            return strength
    
    def _validate_result(self, result: pd.Series, original_signals: pd.Series) -> bool:
        """
        Validate calculation result.
        
        Args:
            result: Calculated strength series
            original_signals: Original signals series
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if result is None or result.empty:
                self.logger.warning("Result is None or empty")
                return False
            
            # Check value range
            if not result.between(0, 100).all():
                out_of_range = result[(result < 0) | (result > 100)]
                self.logger.warning(f"Result contains values outside [0, 100] range: {len(out_of_range)} values")
                return False
            
            # Check for excessive NaN values
            nan_count = result.isna().sum()
            if nan_count > len(result) * 0.5:  # More than 50% NaN
                self.logger.warning(f"Result contains too many NaN values: {nan_count}/{len(result)}")
                return False
            
            # Check that strength is only assigned to signal positions
            signal_positions = original_signals != 0
            non_signal_strength = result[~signal_positions & (result != 0)]
            
            if len(non_signal_strength) > 0:
                self.logger.debug(f"Found {len(non_signal_strength)} non-zero strengths at non-signal positions")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating result: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get calculator information and current parameters.
        
        Returns:
            Dictionary with calculator info
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "required_indicators": getattr(self, 'required_indicators', []),
            "optional_indicators": getattr(self, 'optional_indicators', []),
            "current_params": self.params.copy(),
            "default_params": getattr(self, 'default_params', {})
        }
    
    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update calculator parameters.
        
        Args:
            new_params: New parameters to update
        """
        if new_params:
            self.params.update(new_params)
            self.logger.debug(f"Updated parameters: {new_params}")


class BasicStrengthCalculator(BaseStrengthCalculator):
    """
    Basic strength calculator that provides simple rule-based strength calculation.
    Used as a fallback when more sophisticated calculators are not available.
    """
    
    name = "basic_strength"
    display_name = "Basic Strength Calculator"
    description = "Simple rule-based strength calculation"
    category = "basic"
    
    default_params = {
        "base_strength": 60,
        "rsi_weight": 0.3,
        "trend_weight": 0.4,
        "volatility_weight": 0.3,
        "rsi_thresholds": {"overbought": 70, "oversold": 30},
        "trend_threshold": 25  # ADX threshold for trend strength
    }
    
    required_indicators = []
    optional_indicators = ["rsi_14", "adx", "atr_percent", "market_regime"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate basic strength using simple rules.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values
        """
        # Initialize with base strength
        base_strength = self.params.get("base_strength", 60)
        strength = pd.Series(0, index=signals.index)
        
        # Get parameters
        rsi_weight = self.params.get("rsi_weight", 0.3)
        trend_weight = self.params.get("trend_weight", 0.4)
        vol_weight = self.params.get("volatility_weight", 0.3)
        rsi_thresholds = self.params.get("rsi_thresholds", {"overbought": 70, "oversold": 30})
        trend_threshold = self.params.get("trend_threshold", 25)
        
        # Only calculate for signals
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        
        for i in signal_indices:
            try:
                signal_strength = base_strength
                is_long = signals.loc[i] > 0
                
                # RSI component
                if "rsi_14" in df.columns and not pd.isna(df["rsi_14"].loc[i]):
                    rsi = df["rsi_14"].loc[i]
                    
                    if is_long:
                        # For long signals, prefer RSI not overbought
                        if rsi < rsi_thresholds["overbought"]:
                            rsi_bonus = (rsi_thresholds["overbought"] - rsi) / rsi_thresholds["overbought"] * 20
                        else:
                            rsi_bonus = -10  # Penalty for overbought
                    else:
                        # For short signals, prefer RSI not oversold
                        if rsi > rsi_thresholds["oversold"]:
                            rsi_bonus = (rsi - rsi_thresholds["oversold"]) / (100 - rsi_thresholds["oversold"]) * 20
                        else:
                            rsi_bonus = -10  # Penalty for oversold
                    
                    signal_strength += rsi_bonus * rsi_weight
                
                # Trend component
                if "adx" in df.columns and not pd.isna(df["adx"].loc[i]):
                    adx = df["adx"].loc[i]
                    
                    if adx > trend_threshold:
                        trend_bonus = min(20, (adx - trend_threshold) / trend_threshold * 20)
                        signal_strength += trend_bonus * trend_weight
                
                # Volatility component (inverse - prefer lower volatility)
                if "atr_percent" in df.columns and not pd.isna(df["atr_percent"].loc[i]):
                    atr_pct = df["atr_percent"].loc[i]
                    
                    if atr_pct < 1.0:  # Low volatility bonus
                        vol_bonus = (1.0 - atr_pct) * 10
                    elif atr_pct > 3.0:  # High volatility penalty
                        vol_bonus = -10
                    else:
                        vol_bonus = 0
                    
                    signal_strength += vol_bonus * vol_weight
                
                # Market regime bonus
                if "market_regime" in df.columns and not pd.isna(df["market_regime"].loc[i]):
                    regime = str(df["market_regime"].loc[i]).lower()
                    
                    if is_long and regime in ["strong_uptrend", "weak_uptrend"]:
                        signal_strength += 10
                    elif not is_long and regime in ["strong_downtrend", "weak_downtrend"]:
                        signal_strength += 10
                    elif regime == "ranging":
                        signal_strength -= 5  # Slight penalty for ranging market
                
                # Ensure bounds
                strength.loc[i] = round(max(0, min(100, signal_strength)))
                
            except Exception as e:
                self.logger.debug(f"Error calculating strength for index {i}: {e}")
                strength.loc[i] = base_strength
        
        return strength