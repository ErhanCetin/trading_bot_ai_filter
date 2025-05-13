"""
Predictor classes for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SignalPredictor:
    """Makes predictions for trading signals using trained models."""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signal predictor.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            config: Configuration parameters
        """
        self.config = {
            'probability_threshold': 0.6,
            'features': [],
            'categorical_features': [],
            'signal_mapping': {
                0: 0,    # No signal
                1: 1,    # Long signal
                2: -1    # Short signal
            }
        }
        
        if config:
            self.config.update(config)
        
        # Load model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Load or create scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = StandardScaler()
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions for each row in the dataframe.
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            Series with signal predictions
        """
        if self.model is None:
            logger.error("No model loaded")
            return pd.Series(0, index=df.index)
        
        # Get features
        features = self.config.get('features', [])
        
        # Check if all required features are available
        if not features:
            logger.error("No features specified in configuration")
            return pd.Series(0, index=df.index)
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return pd.Series(0, index=df.index)
        
        # Prepare data
        X = df[features].copy()
        
        # Handle categorical features (one-hot encoding)
        categorical_features = self.config.get('categorical_features', [])
        for cat_feature in categorical_features:
            if cat_feature in X.columns:
                # Simple one-hot encoding for categorical features
                unique_values = df[cat_feature].unique()
                for value in unique_values:
                    X[f"{cat_feature}_{value}"] = (df[cat_feature] == value).astype(int)
                X.drop(cat_feature, axis=1, inplace=True)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        signal_mapping = self.config.get('signal_mapping', {0: 0, 1: 1, 2: -1})
        threshold = self.config.get('probability_threshold', 0.6)
        
        try:
            # If model has predict_proba method, use it
            if hasattr(self.model, 'predict_proba'):
                # Get class probabilities
                proba = self.model.predict_proba(X_scaled)
                
                # Apply threshold to probabilities
                pred = np.zeros(len(df), dtype=int)
                
                # Default to no signal (class 0)
                max_prob_class = np.argmax(proba, axis=1)
                
                for i in range(len(pred)):
                    class_idx = max_prob_class[i]
                    if class_idx > 0 and proba[i, class_idx] >= threshold:
                        pred[i] = class_idx
            else:
                # Use direct prediction
                pred = self.model.predict(X_scaled)
                
            # Map predictions to signal values
            signals = pd.Series([signal_mapping.get(p, 0) for p in pred], index=df.index)
            return signals
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.Series(0, index=df.index)
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make probability predictions for each row in the dataframe.
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            DataFrame with class probabilities
        """
        if self.model is None:
            logger.error("No model loaded")
            return pd.DataFrame(index=df.index)
        
        # Get features
        features = self.config.get('features', [])
        
        # Check if all required features are available
        if not features:
            logger.error("No features specified in configuration")
            return pd.DataFrame(index=df.index)
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return pd.DataFrame(index=df.index)
        
        # Prepare data
        X = df[features].copy()
        
        # Handle categorical features (one-hot encoding)
        categorical_features = self.config.get('categorical_features', [])
        for cat_feature in categorical_features:
            if cat_feature in X.columns:
                # Simple one-hot encoding for categorical features
                unique_values = df[cat_feature].unique()
                for value in unique_values:
                    X[f"{cat_feature}_{value}"] = (df[cat_feature] == value).astype(int)
                X.drop(cat_feature, axis=1, inplace=True)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        try:
            # Check if model has predict_proba method
            if hasattr(self.model, 'predict_proba'):
                # Get class probabilities
                proba = self.model.predict_proba(X_scaled)
                
                # Create DataFrame with probabilities
                class_names = ['no_signal', 'long', 'short']
                if proba.shape[1] <= len(class_names):
                    class_names = class_names[:proba.shape[1]]
                else:
                    class_names = [f'class_{i}' for i in range(proba.shape[1])]
                
                proba_df = pd.DataFrame(proba, index=df.index, columns=class_names)
                return proba_df
            else:
                logger.warning("Model does not support probability predictions")
                return pd.DataFrame(index=df.index)
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            return pd.DataFrame(index=df.index)


class StrengthPredictor:
    """Makes predictions for signal strength using trained models."""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strength predictor.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            config: Configuration parameters
        """
        self.config = {
            'features': [],
            'categorical_features': [],
            'default_strength': 50,
            'min_strength': 0,
            'max_strength': 100
        }
        
        if config:
            self.config.update(config)
        
        # Load model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Load or create scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = StandardScaler()
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Make strength predictions for each signal.
        
        Args:
            df: DataFrame with feature data
            signals: Series with signal values
            
        Returns:
            Series with strength predictions
        """
        # Initialize strength series with default value
        default_strength = self.config.get('default_strength', 50)
        strength = pd.Series(0, index=df.index)
        
        # Set default strength for all signals
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                strength.iloc[i] = default_strength
        
        if self.model is None:
            logger.warning("No model loaded, using default strength")
            return strength
        
        # Get features
        features = self.config.get('features', [])
        
        # Check if all required features are available
        if not features:
            logger.error("No features specified in configuration")
            return strength
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return strength
        
        # Prepare data
        X = df[features].copy()
        
        # Handle categorical features (one-hot encoding)
        categorical_features = self.config.get('categorical_features', [])
        for cat_feature in categorical_features:
            if cat_feature in X.columns:
                # Simple one-hot encoding for categorical features
                unique_values = df[cat_feature].unique()
                for value in unique_values:
                    X[f"{cat_feature}_{value}"] = (df[cat_feature] == value).astype(int)
                X.drop(cat_feature, axis=1, inplace=True)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        try:
            # Make predictions
            pred = self.model.predict(X_scaled)
            
            # Ensure predictions are within range
            min_strength = self.config.get('min_strength', 0)
            max_strength = self.config.get('max_strength', 100)
            pred = np.clip(pred, min_strength, max_strength)
            
            # Update strength for signals
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    strength.iloc[i] = round(pred[i])
            
            return strength
            
        except Exception as e:
            logger.error(f"Error making strength predictions: {e}")
            return strength