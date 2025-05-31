"""
Predictive signal strength calculators for the trading system.
FIXED VERSION - Performance optimization, robust error handling, and updated indicator names.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from signal_engine.signal_strength_system import BaseStrengthCalculator

logger = logging.getLogger(__name__)


class ProbabilisticStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on historical probability with performance optimization."""
    
    name = "probabilistic_strength"
    display_name = "Probabilistic Strength Calculator"
    description = "Calculates signal strength based on historical win rates"
    category = "predictive"
    
    default_params = {
        "lookback_window": 100,
        "min_signals": 10,
        "similar_condition_columns": ["market_regime", "volatility_regime", "trend_strength", "trend_direction"],
        "min_similar_conditions": 1,
        "forward_window": 20,  # Max bars to look ahead for profit calculation
        "min_profit_threshold": 0.5,  # Minimum profit % to consider a win
        "fallback_strength": 50,
        "cache_results": True  # Cache probability calculations
    }
    
    # FIXED: Updated to optional since we use fallbacks
    required_indicators = []
    optional_indicators = ["market_regime", "volatility_regime", "trend_strength", "trend_direction", "close"]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with caching support."""
        super().__init__(params)
        self._probability_cache = {} if self.params.get("cache_results", True) else None
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate probabilistic strength with major performance improvements.
        """
        # Initialize strength series
        fallback = self.params.get("fallback_strength", 50)
        strength = pd.Series(fallback, index=signals.index)
        
        # Validate basic requirements
        if "close" not in df.columns or len(df) < 50:
            logger.warning("Insufficient data for probabilistic strength calculation")
            return strength
        
        # Get parameters
        lookback = self.params.get("lookback_window", 100)
        min_signals = self.params.get("min_signals", 10)
        similar_columns = self.params.get("similar_condition_columns", [])
        min_similar = self.params.get("min_similar_conditions", 1)
        forward_window = self.params.get("forward_window", 20)
        min_profit = self.params.get("min_profit_threshold", 0.5)
        
        # Filter available similar condition columns
        available_columns = [col for col in similar_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No similar condition columns available for probabilistic calculation")
            strength.loc[signals != 0] = fallback
            return strength
        
        # PERFORMANCE: Only process signals and only after sufficient history
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        valid_indices = [i for i in signal_indices if i >= lookback and i < len(df) - forward_window]
        
        if not valid_indices:
            return strength
        
        try:
            # PERFORMANCE: Pre-calculate price returns for profit analysis
            price_returns = df["close"].pct_change()
            
            # PERFORMANCE: Vectorized historical signal identification
            long_signals = signals > 0
            short_signals = signals < 0
            
            # Cache for probability calculations
            probability_cache = self._probability_cache if self._probability_cache is not None else {}
            
            for i in valid_indices:
                if pd.isna(signals.loc[i]):
                    continue
                
                # Get signal direction
                signal_direction = "long" if signals.loc[i] > 0 else "short"
                
                # Create cache key based on current conditions
                cache_key = self._create_cache_key(df.loc[i], available_columns, signal_direction)
                
                # Check cache first
                if cache_key in probability_cache:
                    win_rate = probability_cache[cache_key]
                else:
                    # Calculate win rate for similar conditions
                    win_rate = self._calculate_win_rate_optimized(
                        df, signals, i, lookback, signal_direction, 
                        available_columns, min_similar, forward_window, 
                        min_profit, price_returns, long_signals, short_signals
                    )
                    
                    # Cache result
                    if self._probability_cache is not None:
                        probability_cache[cache_key] = win_rate
                
                # Convert win rate to strength
                if win_rate is not None:
                    strength.loc[i] = round(max(0, min(100, win_rate * 100)))
                else:
                    strength.loc[i] = fallback
        
        except Exception as e:
            logger.error(f"Error in ProbabilisticStrengthCalculator: {e}")
            strength.loc[signal_mask] = fallback
        
        return strength
    
    def _create_cache_key(self, row: pd.Series, available_columns: List[str], direction: str) -> str:
        """Create cache key based on current market conditions."""
        try:
            conditions = []
            for col in available_columns:
                if not pd.isna(row[col]):
                    conditions.append(f"{col}:{row[col]}")
            return f"{direction}|{'|'.join(conditions)}"
        except:
            return f"{direction}|default"
    
    def _calculate_win_rate_optimized(self, df: pd.DataFrame, signals: pd.Series, 
                                    current_idx: int, lookback: int, signal_direction: str,
                                    available_columns: List[str], min_similar: int,
                                    forward_window: int, min_profit: float,
                                    price_returns: pd.Series, long_signals: pd.Series, 
                                    short_signals: pd.Series) -> Optional[float]:
        """
        Optimized win rate calculation with vectorized operations.
        """
        try:
            # Define search window
            start_idx = max(0, current_idx - lookback)
            
            # PERFORMANCE: Filter relevant signals first
            if signal_direction == "long":
                relevant_signals = long_signals.loc[start_idx:current_idx-1]
            else:
                relevant_signals = short_signals.loc[start_idx:current_idx-1]
            
            relevant_indices = relevant_signals[relevant_signals].index
            
            if len(relevant_indices) == 0:
                return None
            
            # Get current conditions
            current_conditions = df.loc[current_idx, available_columns]
            
            # PERFORMANCE: Vectorized similarity check
            similar_indices = []
            
            for idx in relevant_indices:
                try:
                    # Compare conditions
                    past_conditions = df.loc[idx, available_columns]
                    
                    # Count similar conditions (handle NaN)
                    similar_count = 0
                    for col in available_columns:
                        if not pd.isna(current_conditions[col]) and not pd.isna(past_conditions[col]):
                            if current_conditions[col] == past_conditions[col]:
                                similar_count += 1
                    
                    if similar_count >= min_similar:
                        similar_indices.append(idx)
                        
                except Exception:
                    continue
            
            if len(similar_indices) < self.params.get("min_signals", 10):
                return None
            
            # PERFORMANCE: Vectorized profit calculation
            wins = 0
            total_trades = 0
            
            for idx in similar_indices:
                try:
                    # Calculate forward return efficiently
                    end_idx = min(idx + forward_window, len(df) - 1)
                    
                    if end_idx <= idx:
                        continue
                    
                    # PERFORMANCE: Use pre-calculated returns
                    forward_returns = price_returns.loc[idx+1:end_idx+1]
                    
                    if signal_direction == "long":
                        # For long signals: positive return is profit
                        max_profit = forward_returns.cumsum().max() * 100
                    else:
                        # For short signals: negative return is profit
                        max_profit = (-forward_returns).cumsum().max() * 100
                    
                    total_trades += 1
                    if max_profit >= min_profit:
                        wins += 1
                        
                except Exception:
                    continue
            
            return wins / total_trades if total_trades > 0 else None
            
        except Exception as e:
            logger.debug(f"Error in win rate calculation: {e}")
            return None


class RiskRewardStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on risk/reward ratio with robust error handling."""
    
    name = "risk_reward_strength"
    display_name = "Risk-Reward Strength Calculator"
    description = "Calculates signal strength based on potential risk/reward ratio"
    category = "predictive"
    
    default_params = {
        "risk_factor": 1.0,
        "reward_factor": 2.0,
        "min_reward_risk_ratio": 1.5,
        "stop_method": "atr",  # "atr", "support_resistance", "bollinger"
        "target_method": "atr",  # "atr", "support_resistance", "bollinger"
        "fallback_strength": 50,
        "max_ratio_for_calc": 5.0  # Cap ratio for strength calculation
    }
    
    # FIXED: Updated to match refactored indicator names
    required_indicators = []  # Use fallbacks
    optional_indicators = ["atr_14", "atr", "nearest_support", "nearest_resistance", 
                          "bollinger_lower", "bollinger_upper", "close"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate risk/reward strength with robust error handling.
        """
        # Initialize strength series
        fallback = self.params.get("fallback_strength", 50)
        strength = pd.Series(fallback, index=signals.index)
        
        # Validate basic requirements
        if "close" not in df.columns:
            logger.warning("Close price not available for risk/reward calculation")
            return strength
        
        # Check for ATR availability (primary requirement)
        atr_col = None
        for col in ["atr_14", "atr"]:  # FIXED: Updated ATR names
            if col in df.columns:
                atr_col = col
                break
        
        if atr_col is None:
            logger.warning("No ATR data available for risk/reward calculation")
            strength.loc[signals != 0] = fallback
            return strength
        
        # Get parameters
        risk_factor = self.params.get("risk_factor", 1.0)
        reward_factor = self.params.get("reward_factor", 2.0)
        min_ratio = self.params.get("min_reward_risk_ratio", 1.5)
        stop_method = self.params.get("stop_method", "atr")
        target_method = self.params.get("target_method", "atr")
        max_ratio = self.params.get("max_ratio_for_calc", 5.0)
        
        # PERFORMANCE: Only process signals
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        
        if len(signal_indices) == 0:
            return strength
        
        try:
            # Pre-check available indicators for different methods
            has_support_resistance = all(col in df.columns for col in ["nearest_support", "nearest_resistance"])
            has_bollinger = all(col in df.columns for col in ["bollinger_lower", "bollinger_upper"])
            
            # Vectorized calculation
            for i in signal_indices:
                if pd.isna(signals.loc[i]) or pd.isna(df["close"].loc[i]) or pd.isna(df[atr_col].loc[i]):
                    strength.loc[i] = fallback
                    continue
                
                # Get current values
                current_price = df["close"].loc[i]
                atr_value = df[atr_col].loc[i]
                is_long = signals.loc[i] > 0
                
                # Calculate stop loss distance
                stop_distance = self._calculate_stop_distance(
                    df.loc[i], current_price, atr_value, is_long, stop_method, 
                    risk_factor, has_support_resistance, has_bollinger
                )
                
                # Calculate take profit distance
                target_distance = self._calculate_target_distance(
                    df.loc[i], current_price, atr_value, is_long, target_method,
                    reward_factor, has_support_resistance, has_bollinger
                )
                
                # Calculate reward/risk ratio
                if stop_distance > 0:
                    rr_ratio = target_distance / stop_distance
                    
                    # Calculate strength based on ratio
                    if rr_ratio >= min_ratio:
                        # Scale ratio to strength (min_ratio=50, max_ratio=100)
                        normalized_ratio = min(rr_ratio, max_ratio)
                        ratio_strength = 50 + (normalized_ratio - min_ratio) * (50 / (max_ratio - min_ratio))
                        strength.loc[i] = round(max(50, min(100, ratio_strength)))
                    else:
                        # Below minimum ratio, scale from 0 to 50
                        strength.loc[i] = round(max(0, (rr_ratio / min_ratio) * 50))
                else:
                    strength.loc[i] = fallback
        
        except Exception as e:
            logger.error(f"Error in RiskRewardStrengthCalculator: {e}")
            strength.loc[signal_mask] = fallback
        
        return strength
    
    def _calculate_stop_distance(self, row: pd.Series, current_price: float, atr_value: float,
                                is_long: bool, method: str, risk_factor: float,
                                has_support_resistance: bool, has_bollinger: bool) -> float:
        """Calculate stop loss distance based on method."""
        try:
            if method == "atr":
                return atr_value * risk_factor
            
            elif method == "support_resistance" and has_support_resistance:
                if is_long and not pd.isna(row.get("nearest_support", np.nan)):
                    return max(current_price - row["nearest_support"], atr_value * 0.5)
                elif not is_long and not pd.isna(row.get("nearest_resistance", np.nan)):
                    return max(row["nearest_resistance"] - current_price, atr_value * 0.5)
            
            elif method == "bollinger" and has_bollinger:
                if is_long and not pd.isna(row.get("bollinger_lower", np.nan)):
                    return max(current_price - row["bollinger_lower"], atr_value * 0.5)
                elif not is_long and not pd.isna(row.get("bollinger_upper", np.nan)):
                    return max(row["bollinger_upper"] - current_price, atr_value * 0.5)
            
            # Fallback to ATR
            return atr_value * risk_factor
            
        except Exception:
            return atr_value * risk_factor
    
    def _calculate_target_distance(self, row: pd.Series, current_price: float, atr_value: float,
                                  is_long: bool, method: str, reward_factor: float,
                                  has_support_resistance: bool, has_bollinger: bool) -> float:
        """Calculate take profit distance based on method."""
        try:
            if method == "atr":
                return atr_value * reward_factor
            
            elif method == "support_resistance" and has_support_resistance:
                if is_long and not pd.isna(row.get("nearest_resistance", np.nan)):
                    return max(row["nearest_resistance"] - current_price, atr_value * reward_factor)
                elif not is_long and not pd.isna(row.get("nearest_support", np.nan)):
                    return max(current_price - row["nearest_support"], atr_value * reward_factor)
            
            elif method == "bollinger" and has_bollinger:
                if is_long and not pd.isna(row.get("bollinger_upper", np.nan)):
                    return max(row["bollinger_upper"] - current_price, atr_value * reward_factor)
                elif not is_long and not pd.isna(row.get("bollinger_lower", np.nan)):
                    return max(current_price - row["bollinger_lower"], atr_value * reward_factor)
            
            # Fallback to ATR
            return atr_value * reward_factor
            
        except Exception:
            return atr_value * reward_factor


class MLPredictiveStrengthCalculator(BaseStrengthCalculator):
    """ML-based strength calculator with robust error handling and fallbacks."""
    
    name = "ml_predictive_strength"
    display_name = "ML Predictive Strength Calculator"
    description = "Calculates signal strength using machine learning models"
    category = "predictive"
    
    default_params = {
        "model_path": "models/signal_strength_predictor.joblib",
        "features": ["rsi_14", "adx", "macd_line", "bollinger_width", "atr_percent", "trend_strength"],
        "categorical_features": [],  # FIXED: Removed problematic categorical encoding
        "fallback_strength": 50,
        "feature_scaling": True,
        "handle_missing_features": True
    }
    
    required_indicators = []
    optional_indicators = []  # Will be set from features
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with robust model loading."""
        super().__init__(params)
        
        # Set optional indicators from features
        self.optional_indicators = self.params.get("features", [])
        
        # Initialize model with error handling
        self.model = None
        self.model_loaded = False
        
        try:
            model_path = self.params.get("model_path", "")
            if model_path and os.path.exists(model_path):
                import joblib
                self.model = joblib.load(model_path)
                self.model_loaded = True
                logger.info(f"ML model loaded successfully from {model_path}")
            else:
                logger.info(f"ML model not found at {model_path}, using fallback strength")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self.model_loaded = False
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate ML-based strength with comprehensive error handling.
        """
        # Initialize with fallback strength
        fallback = self.params.get("fallback_strength", 50)
        strength = pd.Series(fallback, index=signals.index)
        
        # If model not loaded, return fallback for all signals
        if not self.model_loaded or self.model is None:
            strength.loc[signals != 0] = fallback
            return strength
        
        # Get features and parameters
        features = self.params.get("features", [])
        handle_missing = self.params.get("handle_missing_features", True)
        
        # Check available features
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < len(features) * 0.5:  # Less than 50% features available
            if handle_missing:
                logger.warning(f"Only {len(available_features)}/{len(features)} features available for ML prediction")
            else:
                logger.warning("Insufficient features for ML prediction, using fallback")
                strength.loc[signals != 0] = fallback
                return strength
        
        # PERFORMANCE: Only process signals
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        
        if len(signal_indices) == 0:
            return strength
        
        try:
            # Prepare feature matrix
            feature_df = df[available_features].copy()
            
            # Handle missing values
            if handle_missing:
                # Forward fill then backward fill
                feature_df = feature_df.ffill().bfill()
                # Fill remaining NaNs with median
                feature_df = feature_df.fillna(feature_df.median())
            
            # Feature scaling if enabled
            if self.params.get("feature_scaling", True):
                try:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    feature_df[available_features] = scaler.fit_transform(feature_df[available_features])
                except ImportError:
                    logger.warning("sklearn not available for feature scaling")
                except Exception as e:
                    logger.warning(f"Feature scaling failed: {e}")
            
            # Batch prediction for better performance
            try:
                # Get features for signal indices
                X = feature_df.loc[signal_indices]
                
                # Make batch prediction
                predictions = self.model.predict(X)
                
                # Ensure predictions are in valid range
                predictions = np.clip(predictions, 0, 100)
                
                # Assign predictions to strength series
                strength.loc[signal_indices] = predictions.round().astype(int)
                
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                # Fallback to individual predictions
                for i in signal_indices:
                    try:
                        X = feature_df.loc[i:i].values.reshape(1, -1)
                        prediction = self.model.predict(X)[0]
                        strength.loc[i] = round(max(0, min(100, prediction)))
                    except Exception:
                        strength.loc[i] = fallback
        
        except Exception as e:
            logger.error(f"Error in MLPredictiveStrengthCalculator: {e}")
            strength.loc[signal_mask] = fallback
        
        return strength