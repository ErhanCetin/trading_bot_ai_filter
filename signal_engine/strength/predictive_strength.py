"""
Predictive signal strength calculators for the trading system.
These calculators estimate signal strength based on future potential.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import joblib
import os

from signal_engine.signal_strength_system import BaseStrengthCalculator

logger = logging.getLogger(__name__)


class ProbabilisticStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on historical probability of success."""
    
    name = "probabilistic_strength"
    display_name = "Probabilistic Strength Calculator"
    description = "Calculates signal strength based on historical win rates"
    category = "predictive"
    
    default_params = {
        "lookback_window": 100,  # Window to evaluate historical performance
        "min_signals": 10,       # Minimum number of signals needed for calculation
        "similar_condition_columns": [
            "market_regime", "volatility_regime", 
            "trend_strength", "trend_direction"
        ],
        "min_similar_conditions": 1  # Minimum number of similar conditions needed
    }
    
    required_indicators = ["market_regime"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on historical probability.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values (0-100)
        """
        # Initialize strength series with zeros
        strength = pd.Series(0, index=signals.index)
        
        # Get parameters
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        min_signals = self.params.get("min_signals", self.default_params["min_signals"])
        similar_columns = self.params.get("similar_condition_columns", self.default_params["similar_condition_columns"])
        min_similar = self.params.get("min_similar_conditions", self.default_params["min_similar_conditions"])
        
        # Filter available similar condition columns
        available_columns = [col for col in similar_columns if col in df.columns]
        
        # Ensure we have enough history and at least one condition column
        if len(df) <= lookback or not available_columns:
            return strength
        
        # Calculate probabilities for each signal
        for i in range(lookback, len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Get signal direction
            signal_direction = "long" if signals.iloc[i] > 0 else "short"
            
            # Check historical performance under similar conditions
            similar_signals = []
            similar_wins = 0
            
            # Define what constitutes a win
            # For long signals: future price > entry price
            # For short signals: future price < entry price
            
            # Analyze past signals
            for j in range(i - lookback, i):
                if signals.iloc[j] == 0:
                    continue
                
                # Only consider signals in same direction
                past_direction = "long" if signals.iloc[j] > 0 else "short"
                if past_direction != signal_direction:
                    continue
                
                # Check similarity of conditions
                similar_count = 0
                for col in available_columns:
                    if df[col].iloc[i] == df[col].iloc[j]:
                        similar_count += 1
                
                # If enough conditions are similar, include in analysis
                if similar_count >= min_similar:
                    # Check if this signal was a winner
                    entry_price = df["close"].iloc[j]
                    
                    # Define forward window for profit calculation
                    # Look ahead at most 20 bars or until the end of the data
                    forward_window = min(20, len(df) - j - 1)
                    
                    # Track maximum profit/loss in forward window
                    max_profit = 0
                    for k in range(1, forward_window + 1):
                        future_price = df["close"].iloc[j + k]
                        
                        if signal_direction == "long":
                            profit_pct = (future_price / entry_price - 1) * 100
                        else:  # short
                            profit_pct = (1 - future_price / entry_price) * 100
                        
                        max_profit = max(max_profit, profit_pct)
                    
                    # Consider it a win if profit is positive
                    is_win = max_profit > 0
                    similar_signals.append(is_win)
                    
                    if is_win:
                        similar_wins += 1
            
            # Calculate win rate
            if len(similar_signals) >= min_signals:
                win_rate = similar_wins / len(similar_signals)
                
                # Convert win rate to strength value (0-100)
                strength.iloc[i] = round(win_rate * 100)
            else:
                # Not enough similar signals
                strength.iloc[i] = 50  # Neutral strength
        
        return strength


class RiskRewardStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on potential risk/reward ratio."""
    
    name = "risk_reward_strength"
    display_name = "Risk-Reward Strength Calculator"
    description = "Calculates signal strength based on potential risk/reward ratio"
    category = "predictive"
    
    default_params = {
        "risk_factor": 1.0,         # Multiplier for stop loss distance
        "reward_factor": 1.0,       # Multiplier for take profit distance
        "min_reward_risk_ratio": 1.5,  # Minimum reward/risk ratio for a signal
        "stop_method": "atr",       # Method to calculate stop distance
        "target_method": "support_resistance"  # Method to calculate target distance
    }
    
    required_indicators = ["atr"]
    optional_indicators = ["nearest_support", "nearest_resistance", "bollinger_lower", "bollinger_upper"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on risk/reward ratio.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values (0-100)
        """
        # Initialize strength series with zeros
        strength = pd.Series(0, index=signals.index)
        
        # Get parameters
        risk_factor = self.params.get("risk_factor", self.default_params["risk_factor"])
        reward_factor = self.params.get("reward_factor", self.default_params["reward_factor"])
        min_ratio = self.params.get("min_reward_risk_ratio", self.default_params["min_reward_risk_ratio"])
        stop_method = self.params.get("stop_method", self.default_params["stop_method"])
        target_method = self.params.get("target_method", self.default_params["target_method"])
        
        # Ensure we have the necessary indicators
        if "atr" not in df.columns:
            return strength
        
        # Calculate risk and reward for each signal
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Get current price and signal direction
            current_price = df["close"].iloc[i]
            is_long = signals.iloc[i] > 0
            
            # Calculate stop loss distance
            stop_distance = 0
            
            if stop_method == "atr":
                # Use ATR for stop distance
                stop_distance = df["atr"].iloc[i] * risk_factor
            
            elif stop_method == "support_resistance":
                # Use nearest support/resistance
                if is_long and "nearest_support" in df.columns:
                    support_price = df["nearest_support"].iloc[i]
                    if not pd.isna(support_price):
                        stop_distance = current_price - support_price
                        
                elif not is_long and "nearest_resistance" in df.columns:
                    resistance_price = df["nearest_resistance"].iloc[i]
                    if not pd.isna(resistance_price):
                        stop_distance = resistance_price - current_price
            
            elif stop_method == "bollinger":
                # Use Bollinger Bands for stop distance
                if is_long and "bollinger_lower" in df.columns:
                    stop_distance = current_price - df["bollinger_lower"].iloc[i]
                    
                elif not is_long and "bollinger_upper" in df.columns:
                    stop_distance = df["bollinger_upper"].iloc[i] - current_price
            
            # Ensure stop distance is positive and non-zero
            stop_distance = max(stop_distance, df["atr"].iloc[i] * 0.5)
            
            # Calculate take profit distance
            target_distance = 0
            
            if target_method == "atr":
                # Use ATR multiplier for target distance
                target_distance = df["atr"].iloc[i] * reward_factor * 2
            
            elif target_method == "support_resistance":
                # Use nearest support/resistance
                if is_long and "nearest_resistance" in df.columns:
                    resistance_price = df["nearest_resistance"].iloc[i]
                    if not pd.isna(resistance_price):
                        target_distance = resistance_price - current_price
                        
                elif not is_long and "nearest_support" in df.columns:
                    support_price = df["nearest_support"].iloc[i]
                    if not pd.isna(support_price):
                        target_distance = current_price - support_price
            
            elif target_method == "bollinger":
                # Use Bollinger Bands for target distance
                if is_long and "bollinger_upper" in df.columns:
                    target_distance = df["bollinger_upper"].iloc[i] - current_price
                    
                elif not is_long and "bollinger_lower" in df.columns:
                    target_distance = current_price - df["bollinger_lower"].iloc[i]
            
            # Ensure target distance is positive and non-zero
            target_distance = max(target_distance, df["atr"].iloc[i] * reward_factor * 2)
            
            # Calculate reward/risk ratio
            if stop_distance > 0:
                reward_risk_ratio = target_distance / stop_distance
            else:
                reward_risk_ratio = 0
            
            # Calculate strength based on reward/risk ratio
            if reward_risk_ratio >= min_ratio:
                # Scale from min_ratio (50) to 4.0 (100)
                normalized_ratio = min(reward_risk_ratio, 4.0)
                ratio_strength = 50 + (normalized_ratio - min_ratio) * (50 / (4.0 - min_ratio))
                strength.iloc[i] = round(ratio_strength)
            else:
                # Below minimum ratio
                strength.iloc[i] = round(reward_risk_ratio / min_ratio * 50)
        
        return strength


class MLPredictiveStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength using ML prediction models."""
    
    name = "ml_predictive_strength"
    display_name = "ML Predictive Strength Calculator"
    description = "Calculates signal strength using machine learning models"
    category = "predictive"
    
    default_params = {
        "model_path": "models/signal_strength_predictor.joblib",
        "features": [
            "rsi_14", "adx", "macd_line", "market_regime_encoded", 
            "bollinger_width", "atr_percent", "trend_strength"
        ],
        "categorical_features": ["market_regime_encoded"],
        "fallback_strength": 50  # Default strength if model unavailable
    }
    
    required_indicators = []  # Will be populated from features
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with model loading and dynamic required indicators."""
        super().__init__(params)
        
        # Set required indicators based on parameters
        self.required_indicators = self.params.get("features", self.default_params["features"])
        
        # Initialize model
        self.model = None
        try:
            model_path = self.params.get("model_path", self.default_params["model_path"])
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")
            else:
                logger.warning(f"ML model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength using ML prediction model.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values (0-100)
        """
        # Initialize strength series with zeros
        strength = pd.Series(0, index=signals.index)
        
        # Check if model is loaded
        if self.model is None:
            # Fall back to default strength
            fallback = self.params.get("fallback_strength", self.default_params["fallback_strength"])
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    strength.iloc[i] = fallback
            return strength
        
        # Get features and parameters
        features = self.params.get("features", self.default_params["features"])
        categorical_features = self.params.get("categorical_features", self.default_params["categorical_features"])
        
        # Check if all required features are available
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            logger.warning(f"No features available for ML prediction, using fallback strength")
            fallback = self.params.get("fallback_strength", self.default_params["fallback_strength"])
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    strength.iloc[i] = fallback
            return strength
        
        # Prepare feature dataframe
        feature_df = df[available_features].copy()
        
        # Handle categorical features (one-hot encoding)
        for cat_feature in categorical_features:
            if cat_feature in feature_df.columns:
                # Simple one-hot encoding for categorical features
                unique_values = df[cat_feature].unique()
                for value in unique_values:
                    feature_df[f"{cat_feature}_{value}"] = (df[cat_feature] == value).astype(int)
                feature_df.drop(cat_feature, axis=1, inplace=True)
        
        # Make predictions for each signal
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Get features for this row
            X = feature_df.iloc[i:i+1]
            
            try:
                # Get prediction
                prediction = self.model.predict(X)[0]
                
                # Ensure prediction is in the 0-100 range
                strength.iloc[i] = round(max(0, min(100, prediction)))
                
            except Exception as e:
                logger.error(f"Error making prediction for row {i}: {e}")
                strength.iloc[i] = self.params.get("fallback_strength", self.default_params["fallback_strength"])
        
        return strength
                