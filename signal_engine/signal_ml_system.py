"""
Machine Learning management system for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import os
import joblib
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class MLManager:
    """ML modellerinin yönetimini ve tahminlerin yapılmasını koordine eden sınıf."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the ML manager.
        
        Args:
            model_dir: Directory where models are stored
        """
        self.model_dir = model_dir
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Available models cache
        self._signal_models = {}
        self._strength_models = {}
        self._anomaly_models = {}
        
        # Initialize ML components on demand
        self._model_trainer = None
        self._feature_selector = None
        self._signal_predictor = None
        self._strength_predictor = None
    
    def get_predictions(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions from ML models.
        
        Args:
            df: DataFrame with indicator data
            config: ML configuration
            
        Returns:
            Dictionary with prediction results
        """
        results = {}
        
        # Get configuration parameters
        signal_model = config.get("signal_model")
        strength_model = config.get("strength_model")
        anomaly_model = config.get("anomaly_model")
        signals = config.get("signals")
        
        # Load signal predictor if needed
        if signal_model:
            signal_model_path = self._get_model_path(signal_model, "signal")
            if os.path.exists(signal_model_path):
                try:
                    # Import and use the SignalPredictor
                    from signal_engine.ml.predictors import SignalPredictor
                    
                    predictor_config = config.get("signal_predictor_config", {})
                    predictor = SignalPredictor(signal_model_path, predictor_config)
                    
                    # Make predictions
                    signals = predictor.predict(df)
                    results["predicted_signals"] = signals
                    
                    # Get probabilities if available
                    try:
                        probabilities = predictor.predict_proba(df)
                        results["signal_probabilities"] = probabilities
                    except:
                        pass
                    
                except Exception as e:
                    logger.error(f"Error making signal predictions: {e}")
        
        # Load strength predictor if needed
        if strength_model and signals is not None:
            strength_model_path = self._get_model_path(strength_model, "strength")
            if os.path.exists(strength_model_path):
                try:
                    # Import and use the StrengthPredictor
                    from signal_engine.ml.predictors import StrengthPredictor
                    
                    predictor_config = config.get("strength_predictor_config", {})
                    predictor = StrengthPredictor(strength_model_path, predictor_config)
                    
                    # Make predictions
                    strength = predictor.predict(df, signals)
                    results["predicted_strength"] = strength
                    
                except Exception as e:
                    logger.error(f"Error making strength predictions: {e}")
        
        # Load anomaly detector if needed
        if anomaly_model:
            anomaly_model_path = self._get_model_path(anomaly_model, "anomaly")
            if os.path.exists(anomaly_model_path):
                try:
                    # Load the model
                    model = joblib.load(anomaly_model_path)
                    
                    # Get features for anomaly detection
                    features = config.get("anomaly_features", [])
                    if not features:
                        logger.warning("No features specified for anomaly detection")
                        return results
                    
                    X = df[features].copy()
                    
                    # Make predictions
                    anomalies = model.predict(X)
                    results["anomalies"] = pd.Series(anomalies, index=df.index)
                    
                except Exception as e:
                    logger.error(f"Error detecting anomalies: {e}")
        
        return results
    
    def train_model(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ML models.
        
        Args:
            df: DataFrame with indicator and target data
            config: Training configuration
            
        Returns:
            Dictionary with training results
        """
        # Initialize results
        results = {
            "status": "error",
            "message": "Unknown error",
            "metrics": {},
            "model_path": None
        }
        
        # Get configuration parameters
        model_type = config.get("model_type", "signal")  # signal, strength, or anomaly
        model_name = config.get("model_name", f"{model_type}_{int(time.time())}")
        features = config.get("features")
        target = config.get("target")
        
        # Validate required parameters
        if not features or not target:
            results["message"] = "Missing required parameters: features and target"
            return results
        
        # Check if features and target exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            results["message"] = f"Missing features in dataframe: {missing_features}"
            return results
            
        if target not in df.columns:
            results["message"] = f"Target column '{target}' not found in dataframe"
            return results
        
        try:
            # Import ModelTrainer
            from signal_engine.ml.model_trainer import ModelTrainer
            
            # Initialize trainer if not already done
            if self._model_trainer is None:
                self._model_trainer = ModelTrainer()
            
            # Train model based on type
            if model_type == "signal":
                model, metrics = self._model_trainer.train_signal_classifier(
                    df=df,
                    features=features,
                    target_column=target,
                    model_name=config.get("algorithm", "random_forest"),
                    grid_search=config.get("grid_search", False)
                )
                
                # Save model
                from signal_engine.ml.utils import save_model
                model_path, _ = save_model(model, model_name, model_dir=self._get_model_dir("signal"))
                
            elif model_type == "strength":
                model, metrics = self._model_trainer.train_strength_regressor(
                    df=df,
                    features=features,
                    target_column=target,
                    model_name=config.get("algorithm", "random_forest"),
                    grid_search=config.get("grid_search", False)
                )
                
                # Save model
                from signal_engine.ml.utils import save_model
                model_path, _ = save_model(model, model_name, model_dir=self._get_model_dir("strength"))
                
            elif model_type == "anomaly":
                # Import specialized tools if needed
                from sklearn.ensemble import IsolationForest
                
                # Create and train anomaly detection model
                model = IsolationForest(
                    contamination=config.get("contamination", 0.05),
                    random_state=config.get("random_state", 42)
                )
                
                # Fit model
                X = df[features]
                model.fit(X)
                
                # Simple metrics for anomaly detection
                metrics = {
                    "n_samples": len(X),
                    "n_features": len(features),
                    "contamination": config.get("contamination", 0.05)
                }
                
                # Save model
                model_path = os.path.join(self._get_model_dir("anomaly"), f"{model_name}.joblib")
                joblib.dump(model, model_path)
                
            else:
                results["message"] = f"Unknown model type: {model_type}"
                return results
            
            # Update results
            results.update({
                "status": "success",
                "message": f"Successfully trained {model_type} model",
                "metrics": metrics,
                "model_path": model_path
            })
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            results["message"] = f"Error training model: {str(e)}"
        
        return results
    
    def select_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select features for ML models.
        
        Args:
            df: DataFrame with indicator and target data
            config: Feature selection configuration
            
        Returns:
            Dictionary with feature selection results
        """
        # Initialize results
        results = {
            "status": "error",
            "message": "Unknown error",
            "selected_features": []
        }
        
        # Get configuration parameters
        features = config.get("features")
        target = config.get("target")
        methods = config.get("methods", ["variance_threshold", "feature_importance"])
        
        # Validate required parameters
        if not features or not target:
            results["message"] = "Missing required parameters: features and target"
            return results
        
        # Check if features and target exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            results["message"] = f"Missing features in dataframe: {missing_features}"
            return results
            
        if target not in df.columns:
            results["message"] = f"Target column '{target}' not found in dataframe"
            return results
        
        try:
            # Import FeatureSelector
            from signal_engine.ml.feature_selector import FeatureSelector
            
            # Initialize selector if not already done
            if self._feature_selector is None:
                self._feature_selector = FeatureSelector()
            
            # Select features
            selected_features = self._feature_selector.select_features(
                df=df,
                features=features,
                target_column=target,
                methods=methods
            )
            
            # Update results
            results.update({
                "status": "success",
                "message": f"Successfully selected {len(selected_features)} features",
                "selected_features": selected_features
            })
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            results["message"] = f"Error selecting features: {str(e)}"
        
        return results
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        Get a list of available models by type.
        
        Returns:
            Dictionary of model types to list of model names
        """
        result = {
            "signal": [],
            "strength": [],
            "anomaly": []
        }
        
        # List signal models
        signal_dir = self._get_model_dir("signal")
        if os.path.exists(signal_dir):
            result["signal"] = [f[:-7] for f in os.listdir(signal_dir) if f.endswith(".joblib")]
        
        # List strength models
        strength_dir = self._get_model_dir("strength")
        if os.path.exists(strength_dir):
            result["strength"] = [f[:-7] for f in os.listdir(strength_dir) if f.endswith(".joblib")]
        
        # List anomaly models
        anomaly_dir = self._get_model_dir("anomaly")
        if os.path.exists(anomaly_dir):
            result["anomaly"] = [f[:-7] for f in os.listdir(anomaly_dir) if f.endswith(".joblib")]
        
        return result
    
    def _get_model_dir(self, model_type: str) -> str:
        """Get directory for a specific model type."""
        model_dir = os.path.join(self.model_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def _get_model_path(self, model_name: str, model_type: str) -> str:
        """Get full path for a model."""
        return os.path.join(self._get_model_dir(model_type), f"{model_name}.joblib")