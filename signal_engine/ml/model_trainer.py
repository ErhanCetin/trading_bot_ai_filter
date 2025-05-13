"""
Model training functionality for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from .utils import prepare_data

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains machine learning models for signal prediction and evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration parameters for model training
        """
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'models': {
                'random_forest': {
                    'class': RandomForestClassifier,
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'min_samples_split': 10,
                        'min_samples_leaf': 4,
                        'random_state': 42
                    },
                    'grid_search': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'gradient_boosting': {
                    'class': GradientBoostingClassifier,
                    'params': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 5,
                        'random_state': 42
                    },
                    'grid_search': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                },
                'xgboost': {
                    'class': xgb.XGBClassifier,
                    'params': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 5,
                        'random_state': 42
                    },
                    'grid_search': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                },
                'logistic_regression': {
                    'class': LogisticRegression,
                    'params': {
                        'C': 1.0,
                        'random_state': 42,
                        'max_iter': 1000
                    },
                    'grid_search': {
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['liblinear', 'lbfgs']
                    }
                }
            },
            'regression_models': {
                'random_forest': {
                    'class': RandomForestRegressor,
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'min_samples_split': 10,
                        'min_samples_leaf': 4,
                        'random_state': 42
                    },
                    'grid_search': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'xgboost_regressor': {
                    'class': xgb.XGBRegressor,
                    'params': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 5,
                        'random_state': 42
                    },
                    'grid_search': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                }
            }
        }
        
        if config:
            # Update configuration with provided values
            self._update_config(config)
            
        self.scaler = StandardScaler()
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration dictionary with new values."""
        for key, value in config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                self._update_dict(self.config[key], value)
            else:
                self.config[key] = value
    
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def train_signal_classifier(self, df: pd.DataFrame, features: List[str], 
                              target_column: str, model_name: str = 'random_forest',
                              grid_search: bool = False) -> Tuple[Any, Dict[str, float]]:
        """
        Train a classifier model for signal prediction.
        
        Args:
            df: DataFrame with features and target
            features: List of feature column names
            target_column: Name of the target column
            model_name: Name of the model to train
            grid_search: Whether to use grid search for hyperparameter optimization
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        logger.info(f"Training {model_name} classifier for signal prediction")
        
        # Prepare data
        X, y, X_train, X_test, y_train, y_test = prepare_data(
            df, features, target_column, self.config['test_size'], self.config['random_state']
        )
        
        # Standardize features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_name not in self.config['models']:
            raise ValueError(f"Model {model_name} not found in configuration")
            
        model_config = self.config['models'][model_name]
        model_class = model_config['class']
        model_params = model_config['params']
        
        # Train model
        if grid_search and 'grid_search' in model_config:
            logger.info(f"Performing grid search for {model_name}")
            model = GridSearchCV(
                model_class(**model_params),
                model_config['grid_search'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = model.best_estimator_
            logger.info(f"Best parameters: {model.best_params_}")
        else:
            # Train with default parameters
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            best_model = model
        
        # Evaluate model
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return best_model, metrics
    
    def train_strength_regressor(self, df: pd.DataFrame, features: List[str], 
                               target_column: str, model_name: str = 'random_forest',
                               grid_search: bool = False) -> Tuple[Any, Dict[str, float]]:
        """
        Train a regression model for signal strength prediction.
        
        Args:
            df: DataFrame with features and target
            features: List of feature column names
            target_column: Name of the target column
            model_name: Name of the model to train
            grid_search: Whether to use grid search for hyperparameter optimization
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        logger.info(f"Training {model_name} regressor for strength prediction")
        
        # Prepare data
        X, y, X_train, X_test, y_train, y_test = prepare_data(
            df, features, target_column, self.config['test_size'], self.config['random_state']
        )
        
        # Standardize features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_name not in self.config['regression_models']:
            raise ValueError(f"Regression model {model_name} not found in configuration")
            
        model_config = self.config['regression_models'][model_name]
        model_class = model_config['class']
        model_params = model_config['params']
        
        # Train model
        if grid_search and 'grid_search' in model_config:
            logger.info(f"Performing grid search for {model_name}")
            model = GridSearchCV(
                model_class(**model_params),
                model_config['grid_search'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = model.best_estimator_
            logger.info(f"Best parameters: {model.best_params_}")
        else:
            # Train with default parameters
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            best_model = model
        
        # Evaluate model
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'mean_squared_error': mean_squared_error(y_test, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': best_model.score(X_test_scaled, y_test)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return best_model, metrics
    
    def save_model(self, model: Any, model_name: str, model_dir: str = 'models') -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            model_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model filename
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None