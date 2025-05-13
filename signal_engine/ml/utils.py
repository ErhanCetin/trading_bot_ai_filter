"""
Utility functions for the ML module.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def prepare_data(df: pd.DataFrame, features: List[str], target_column: str,
               test_size: float = 0.2, random_state: int = 42
               ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target_column: Name of the target column
        test_size: Test set size (0-1)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X, y, X_train, X_test, y_train, y_test)
    """
    # Ensure all features are in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise ValueError("No features available in dataframe")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Prepare features and target
    X = df[available_features].copy()
    y = df[target_column].copy()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X, y, X_train, X_test, y_train, y_test


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                 is_classifier: bool = True) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        is_classifier: Whether the model is a classifier
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    if is_classifier:
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
    else:
        metrics = {
            'mean_squared_error': mean_squared_error(y_test, y_pred),
            'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }
    
    return metrics


def save_model(model: Any, model_name: str, model_dir: str = 'models',
             scaler: Optional[Any] = None) -> Tuple[str, Optional[str]]:
    """
    Save a trained model and optional scaler to disk.
    
    Args:
        model: Trained model to save
        model_name: Name of the model
        model_dir: Directory to save the model
        scaler: Optional fitted scaler
        
    Returns:
        Tuple of (model_path, scaler_path)
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model filename
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler if provided
    scaler_path = None
    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    return model_path, scaler_path


def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                        class_names: Optional[List[str]] = None, figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix for a classifier.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test targets
        class_names: List of class names
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: Any, feature_names: List[str], figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance for a model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        figsize: Figure size
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importance = model.feature_importances_
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


def plot_learning_curves(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       train_sizes: Optional[List[float]] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot learning curves for a model.
    
    Args:
        model: Model object with fit and predict methods
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        train_sizes: List of training sizes to evaluate
        figsize: Figure size
    """
    from sklearn.model_selection import learning_curve
    
    # Default train sizes
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, train_sizes=train_sizes, cv=5, scoring='accuracy'
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def preprocess_data(df: pd.DataFrame, categorical_features: List[str] = [],
                  numerical_features: List[str] = []) -> pd.DataFrame:
    """
    Preprocess data for model training.
    
    Args:
        df: DataFrame to preprocess
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        
    Returns:
        Preprocessed DataFrame
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Handle categorical features
    for feature in categorical_features:
        if feature in result_df.columns:
            # One-hot encoding
            dummies = pd.get_dummies(result_df[feature], prefix=feature, drop_first=False)
            result_df = pd.concat([result_df.drop(feature, axis=1), dummies], axis=1)
    
    # Handle numerical features
    for feature in numerical_features:
        if feature in result_df.columns:
            # Fill missing values with median
            result_df[feature] = result_df[feature].fillna(result_df[feature].median())
    
    return result_df


def encode_market_regime(df: pd.DataFrame, regime_col: str = 'market_regime') -> pd.DataFrame:
    """
    Encode market regime column as numerical values.
    
    Args:
        df: DataFrame with market_regime column
        regime_col: Name of the regime column
        
    Returns:
        DataFrame with encoded market_regime
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Check if market_regime column exists
    if regime_col not in result_df.columns:
        logger.warning(f"Column '{regime_col}' not found in dataframe")
        return result_df
    
    # Define mapping
    regime_mapping = {
        'strong_uptrend': 3,
        'weak_uptrend': 2,
        'ranging': 0,
        'weak_downtrend': -2,
        'strong_downtrend': -3,
        'volatile': 1,
        'overbought': 4,
        'oversold': -4,
        'unknown': 0
    }
    
    # Create encoded column
    result_df[f'{regime_col}_encoded'] = result_df[regime_col].map(regime_mapping).fillna(0)
    
    return result_df