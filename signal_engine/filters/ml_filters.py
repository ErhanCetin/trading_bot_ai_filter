"""
Machine learning based filters for the trading system - FIXED VERSION.
These filters use ML models to evaluate signal quality.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import joblib
import os
from datetime import datetime

from signal_engine.signal_filter_system import BaseFilter

logger = logging.getLogger(__name__)


class ProbabilisticSignalFilter(BaseFilter):
    """Filter that uses probabilistic model predictions to filter signals - FIXED."""
    
    name = "probabilistic_signal_filter"
    display_name = "Probabilistic Signal Filter"
    description = "Uses probability-based predictions to filter signals"
    category = "ml"
    
    default_params = {
        "model_path": "models/signal_classifier.joblib",
        "probability_threshold": 0.6,  # Minimum probability for a valid signal
        "features": [
            "rsi_14", "adx", "macd_line", "ema_alignment", 
            "trend_strength", "volatility_percentile", "market_regime"  # FIXED: market_regime_encoded -> market_regime
        ],
        "categorical_features": ["market_regime"],  # FIXED: Will be encoded internally
        "fallback_to_raw_signals": True  # Use raw signals if model can't be loaded
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
    
    def _encode_market_regime(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        🆕 YENI: market_regime string değerlerini encode eder
        
        Args:
            regime_series: market_regime string sütunu
            
        Returns:
            Encoded regime DataFrame
        """
        # Regime mapping - consistent with MarketRegimeIndicator
        regime_mapping = {
            "strong_uptrend": 1,
            "weak_uptrend": 2, 
            "ranging": 3,
            "weak_downtrend": 4,
            "strong_downtrend": 5,
            "volatile": 6,
            "overbought": 7,
            "oversold": 8,
            "unknown": 0
        }
        
        # Encode to numeric
        encoded = regime_series.map(regime_mapping).fillna(0)
        
        # Create one-hot encoding DataFrame
        unique_regimes = regime_series.dropna().unique()
        regime_df = pd.DataFrame(index=regime_series.index)
        
        for regime in unique_regimes:
            if regime and not pd.isna(regime):
                col_name = f"market_regime_{regime}"
                regime_df[col_name] = (regime_series == regime).astype(int)
        
        return regime_df
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply ML prediction filter to signals - FIXED VERSION.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Check if model is loaded
        if self.model is None:
            # Fall back to raw signals if configured
            if self.params.get("fallback_to_raw_signals", self.default_params["fallback_to_raw_signals"]):
                logger.warning("No ML model available, returning raw signals")
                return signals
            else:
                # Filter out all signals
                logger.warning("No ML model available, filtering out all signals")
                return pd.Series(0, index=signals.index)
        
        # Get features and parameters
        features = self.params.get("features", self.default_params["features"])
        probability_threshold = self.params.get("probability_threshold", self.default_params["probability_threshold"])
        categorical_features = self.params.get("categorical_features", self.default_params["categorical_features"])
        
        # Check if all required features are available
        if not all(feature in df.columns for feature in features):
            missing = [f for f in features if f not in df.columns]
            logger.warning(f"Missing features for ML model: {missing}, using raw signals")
            return signals
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Prepare feature dataframe - FIXED VERSION
        feature_df = df[features].copy()
        
        # Handle categorical features (improved encoding)
        for cat_feature in categorical_features:
            if cat_feature in feature_df.columns:
                if cat_feature == "market_regime":
                    # 🆕 FIXED: Proper market_regime encoding
                    regime_encoded = self._encode_market_regime(df[cat_feature])
                    
                    # Add encoded columns to feature_df
                    for col in regime_encoded.columns:
                        feature_df[col] = regime_encoded[col]
                    
                    # Remove original categorical column
                    feature_df.drop(cat_feature, axis=1, inplace=True)
                else:
                    # Generic one-hot encoding for other categorical features
                    unique_values = df[cat_feature].unique()
                    for value in unique_values:
                        if value and not pd.isna(value):
                            feature_df[f"{cat_feature}_{value}"] = (df[cat_feature] == value).astype(int)
                    feature_df.drop(cat_feature, axis=1, inplace=True)
        
        # Make predictions for each row
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Get features for this row
            X = feature_df.iloc[i:i+1]
            
            try:
                # Get prediction probability
                if hasattr(self.model, 'predict_proba'):
                    # For classifiers that provide probability
                    proba = self.model.predict_proba(X)
                    
                    # Get probability for the appropriate class
                    # Assuming class 1 = long, class 2 = short, class 0 = no signal
                    if current_signal > 0:  # Long
                        class_idx = 1
                    else:  # Short
                        class_idx = 2
                    
                    # Check if we have enough classes in the model
                    if proba.shape[1] > class_idx:
                        signal_probability = proba[0, class_idx]
                    else:
                        # Fall back to binary classification (class 1 = signal)
                        signal_probability = proba[0, 1] if proba.shape[1] > 1 else 0
                else:
                    # For models without predict_proba, use raw prediction
                    pred = self.model.predict(X)[0]
                    signal_probability = 1.0 if pred == current_signal else 0.0
                
                # Filter out signals below probability threshold
                if signal_probability < probability_threshold:
                    filtered_signals.iloc[i] = 0
                    
            except Exception as e:
                logger.error(f"Error making prediction for row {i}: {e}")
        
        return filtered_signals


class PatternRecognitionFilter(BaseFilter):
    """Filter that uses pattern recognition models to filter signals - UNCHANGED."""
    
    name = "pattern_recognition_filter"
    display_name = "Pattern Recognition Filter"
    description = "Uses pattern recognition to identify high-quality signals"
    category = "ml"
    
    default_params = {
        "model_path": "models/pattern_recognizer.joblib",
        "lookback_window": 10,  # Number of bars to include in pattern
        "confidence_threshold": 0.7,  # Minimum confidence for a valid pattern
        "features": [
            "close", "high", "low", "volume", 
            "rsi_14", "macd_line", "bollinger_width"  # ✅ UYUMLU - değişiklik yok
        ],
        "normalize_features": True,  # Whether to normalize features
        "fallback_to_raw_signals": True  # Use raw signals if model can't be loaded
    }
    
    required_indicators = ["close"]  # Will be extended based on features
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with model loading and dynamic required indicators."""
        super().__init__(params)
        
        # Set required indicators based on parameters
        features = self.params.get("features", self.default_params["features"])
        self.required_indicators = list(set(features))
        
        # Initialize model
        self.model = None
        try:
            model_path = self.params.get("model_path", self.default_params["model_path"])
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Loaded pattern recognition model from {model_path}")
            else:
                logger.warning(f"Pattern recognition model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading pattern recognition model: {e}")
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply pattern recognition filter to signals - UNCHANGED.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Check if model is loaded
        if self.model is None:
            # Fall back to raw signals if configured
            if self.params.get("fallback_to_raw_signals", self.default_params["fallback_to_raw_signals"]):
                logger.warning("No pattern recognition model available, returning raw signals")
                return signals
            else:
                # Filter out all signals
                logger.warning("No pattern recognition model available, filtering out all signals")
                return pd.Series(0, index=signals.index)
        
        # Get features and parameters
        features = self.params.get("features", self.default_params["features"])
        confidence_threshold = self.params.get("confidence_threshold", self.default_params["confidence_threshold"])
        lookback_window = self.params.get("lookback_window", self.default_params["lookback_window"])
        normalize = self.params.get("normalize_features", self.default_params["normalize_features"])
        
        # Check if all required features are available
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            logger.warning(f"No features available for pattern recognition, using raw signals")
            return signals
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Need enough history for pattern recognition
        if len(df) <= lookback_window:
            logger.warning(f"Not enough history for pattern recognition, using raw signals")
            return signals
        
        # Apply pattern recognition filter
        for i in range(lookback_window, len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Extract pattern window
            pattern_window = df[available_features].iloc[i-lookback_window:i]
            
            # Normalize if specified
            if normalize:
                # Normalize each feature to 0-1 range
                pattern_window = (pattern_window - pattern_window.min()) / (pattern_window.max() - pattern_window.min())
            
            # Flatten pattern window to feature vector
            X = pattern_window.values.flatten().reshape(1, -1)
            
            try:
                # Get prediction confidence
                confidence = 0.0
                
                if hasattr(self.model, 'predict_proba'):
                    # For classifiers that provide probability
                    proba = self.model.predict_proba(X)
                    
                    # Get confidence for the appropriate class
                    if current_signal > 0:  # Long
                        class_idx = 1 % proba.shape[1]  # Ensure it's a valid index
                    else:  # Short
                        class_idx = 2 % proba.shape[1]  # Ensure it's a valid index
                    
                    confidence = proba[0, class_idx]
                else:
                    # For models without predict_proba, use raw prediction
                    pred = self.model.predict(X)[0]
                    
                    # Convert prediction to confidence based on the signal direction
                    if (current_signal > 0 and pred > 0) or (current_signal < 0 and pred < 0):
                        confidence = 1.0
                    else:
                        confidence = 0.0
                
                # Filter out signals below confidence threshold
                if confidence < confidence_threshold:
                    filtered_signals.iloc[i] = 0
                    
            except Exception as e:
                logger.error(f"Error in pattern recognition for row {i}: {e}")
        
        return filtered_signals


class PerformanceClassifierFilter(BaseFilter):
    """Filter that uses historical performance to classify and filter signals - FIXED."""
    
    name = "performance_classifier_filter"
    display_name = "Performance Classifier Filter"
    description = "Classifies signals based on historical performance"
    category = "ml"
    
    default_params = {
        "model_path": "models/performance_classifier.joblib",
        "performance_threshold": 0.6,  # Minimum performance score for a valid signal
        "features": [
            "rsi_14", "adx", "macd_line", "bollinger_width", 
            "atr_percent", "market_regime"  # FIXED: market_regime_encoded -> market_regime
        ],
        "categorical_features": ["market_regime"],  # FIXED: Will be encoded internally
        "fallback_to_raw_signals": True,  # Use raw signals if model can't be loaded
        "historical_lookback": 100  # Bars to look back for historical performance
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
                logger.info(f"Loaded performance classifier model from {model_path}")
            else:
                logger.warning(f"Performance classifier model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading performance classifier model: {e}")
        
        # Initialize performance tracking
        self.historical_signals = {
            "long": [],
            "short": []
        }
    
    def _encode_market_regime(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        🆕 YENI: market_regime string değerlerini encode eder - Performance için optimize edilmiş
        """
        # Regime mapping - consistent with MarketRegimeIndicator
        regime_mapping = {
            "strong_uptrend": 1,
            "weak_uptrend": 2, 
            "ranging": 3,
            "weak_downtrend": 4,
            "strong_downtrend": 5,
            "volatile": 6,
            "overbought": 7,
            "oversold": 8,
            "unknown": 0
        }
        
        # Encode to numeric
        encoded = regime_series.map(regime_mapping).fillna(0)
        
        # For performance classification, use one-hot encoding
        unique_regimes = regime_series.dropna().unique()
        regime_df = pd.DataFrame(index=regime_series.index)
        
        for regime in unique_regimes:
            if regime and not pd.isna(regime):
                col_name = f"market_regime_{regime}"
                regime_df[col_name] = (regime_series == regime).astype(int)
        
        return regime_df
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply performance classifier filter to signals - FIXED VERSION.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Check if enough data is available
        if len(df) < 2:
            return signals
        
        # Get parameters
        features = self.params.get("features", self.default_params["features"])
        performance_threshold = self.params.get("performance_threshold", self.default_params["performance_threshold"])
        categorical_features = self.params.get("categorical_features", self.default_params["categorical_features"])
        lookback = self.params.get("historical_lookback", self.default_params["historical_lookback"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # If model is available, use ML approach
        if self.model is not None and all(feature in df.columns for feature in features):
            # Prepare feature dataframe - FIXED VERSION
            feature_df = df[features].copy()
            
            # Handle categorical features (improved encoding)
            for cat_feature in categorical_features:
                if cat_feature in feature_df.columns:
                    if cat_feature == "market_regime":
                        # 🆕 FIXED: Proper market_regime encoding
                        regime_encoded = self._encode_market_regime(df[cat_feature])
                        
                        # Add encoded columns to feature_df
                        for col in regime_encoded.columns:
                            feature_df[col] = regime_encoded[col]
                        
                        # Remove original categorical column
                        feature_df.drop(cat_feature, axis=1, inplace=True)
                    else:
                        # Generic one-hot encoding for other categorical features
                        unique_values = df[cat_feature].unique()
                        for value in unique_values:
                            if value and not pd.isna(value):
                                feature_df[f"{cat_feature}_{value}"] = (df[cat_feature] == value).astype(int)
                        feature_df.drop(cat_feature, axis=1, inplace=True)
            
            # Make predictions for each row
            for i in range(len(df)):
                current_signal = signals.iloc[i]
                
                # Skip if no signal
                if current_signal == 0:
                    continue
                
                # Get features for this row
                X = feature_df.iloc[i:i+1]
                
                try:
                    # Get performance prediction
                    if hasattr(self.model, 'predict_proba'):
                        # For classifiers that provide probability
                        proba = self.model.predict_proba(X)
                        
                        # Get probability for the "good performance" class
                        # Assuming binary classification where class 1 = good performance
                        performance_score = proba[0, 1] if proba.shape[1] > 1 else 0.5
                    else:
                        # For regression models, use prediction directly
                        performance_score = self.model.predict(X)[0]
                        
                        # Normalize to 0-1 range if needed
                        if performance_score < 0 or performance_score > 1:
                            performance_score = max(0, min(1, performance_score))
                    
                    # Filter out signals below performance threshold
                    if performance_score < performance_threshold:
                        filtered_signals.iloc[i] = 0
                        
                except Exception as e:
                    logger.error(f"Error making performance prediction for row {i}: {e}")
            
            return filtered_signals
        
        # If model not available or missing features, use historical performance approach
        logger.info("Using historical performance approach for filtering")
        
        # Update historical performance tracking
        for i in range(1, len(df)):
            previous_signal = signals.iloc[i-1]
            
            # Skip if no signal
            if previous_signal == 0:
                continue
            
            # Check performance of previous signal
            entry_price = df["close"].iloc[i-1]
            current_price = df["close"].iloc[i]
            
            # Calculate return
            if previous_signal > 0:  # Long
                returns = (current_price / entry_price - 1) * 100
                # Track signal for historical performance
                self.historical_signals["long"].append({
                    "entry_index": i-1,
                    "returns": returns,
                    "features": {col: df[col].iloc[i-1] for col in df.columns if col in features}
                })
            else:  # Short
                returns = (1 - current_price / entry_price) * 100
                # Track signal for historical performance
                self.historical_signals["short"].append({
                    "entry_index": i-1,
                    "returns": returns,
                    "features": {col: df[col].iloc[i-1] for col in df.columns if col in features}
                })
                
            # Keep history limited to lookback window
            self.historical_signals["long"] = self.historical_signals["long"][-lookback:]
            self.historical_signals["short"] = self.historical_signals["short"][-lookback:]
        
        # Filter signals based on historical performance
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Determine signal direction
            signal_type = "long" if current_signal > 0 else "short"
            
            # Get similar historical signals
            if not self.historical_signals[signal_type]:
                # No historical data, keep the signal
                continue
            
            # Find similar signals based on feature similarity
            similar_signals = []
            
            for hist_signal in self.historical_signals[signal_type]:
                # Calculate feature similarity score
                similarity_score = 0
                feature_count = 0
                
                for feature in features:
                    if feature in hist_signal["features"] and feature in df.columns:
                        feature_count += 1
                        # Calculate normalized difference
                        current_value = df[feature].iloc[i]
                        hist_value = hist_signal["features"][feature]
                        
                        # Handle categorical features
                        if feature in categorical_features:
                            # Binary similarity (1 if same, 0 if different)
                            similarity_score += 1 if current_value == hist_value else 0
                        else:
                            # For numerical features, calculate normalized distance
                            # Get feature range from dataframe
                            feature_min = df[feature].min()
                            feature_max = df[feature].max()
                            feature_range = feature_max - feature_min
                            
                            if feature_range > 0:
                                normalized_diff = abs(current_value - hist_value) / feature_range
                                similarity_score += 1 - normalized_diff
                
                # Calculate average similarity
                avg_similarity = similarity_score / feature_count if feature_count > 0 else 0
                
                # If similar enough, include in analysis
                if avg_similarity > 0.7:  # At least 70% similar
                    similar_signals.append(hist_signal)
            
            # If we have enough similar signals, calculate average performance
            if len(similar_signals) >= 5:  # At least 5 similar signals
                avg_returns = sum(signal["returns"] for signal in similar_signals) / len(similar_signals)
                
                # Calculate win rate
                win_count = sum(1 for signal in similar_signals if signal["returns"] > 0)
                win_rate = win_count / len(similar_signals)
                
                # Define performance score as combination of win rate and average returns
                performance_score = (win_rate * 0.7) + (max(0, min(1, avg_returns / 5)) * 0.3)
                
                # Filter out signals below performance threshold
                if performance_score < performance_threshold:
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals