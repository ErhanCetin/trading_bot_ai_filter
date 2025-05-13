"""
Feature selection functionality for the trading system.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects the most important features for model training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature selector.
        
        Args:
            config: Configuration parameters for feature selection
        """
        self.config = {
            'random_state': 42,
            'methods': {
                'variance_threshold': 0.01,
                'select_k_best': 10,
                'rfe': 10,
                'feature_importance': 0.01,
                'correlation_threshold': 0.8
            }
        }
        
        if config:
            self.config.update(config)
    
    def select_features(self, df: pd.DataFrame, features: List[str], 
                      target_column: str, methods: Optional[List[str]] = None) -> List[str]:
        """
        Select features using multiple methods.
        
        Args:
            df: DataFrame with features and target
            features: List of feature column names
            target_column: Name of the target column
            methods: Methods to use for feature selection
            
        Returns:
            List of selected feature names
        """
        # Default to all methods if none specified
        if methods is None:
            methods = list(self.config['methods'].keys())
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Track feature scores
        feature_scores = pd.DataFrame(index=features)
        
        # Apply each method
        for method in methods:
            if method == 'variance_threshold':
                scores = self._variance_threshold(X)
                feature_scores[method] = scores
                
            elif method == 'select_k_best':
                scores = self._select_k_best(X, y)
                feature_scores[method] = scores
                
            elif method == 'rfe':
                scores = self._recursive_feature_elimination(X, y)
                feature_scores[method] = scores
                
            elif method == 'feature_importance':
                scores = self._feature_importance(X, y)
                feature_scores[method] = scores
                
            elif method == 'correlation_threshold':
                # This method directly returns selected features, not scores
                corr_selected = self._correlation_threshold(X)
                feature_scores[method] = [1 if f in corr_selected else 0 for f in features]
        
        # Calculate overall score by averaging
        feature_scores['average'] = feature_scores.mean(axis=1)
        
        # Sort features by overall score
        feature_scores = feature_scores.sort_values('average', ascending=False)
        
        # Select top K features
        k = self.config['methods'].get('select_k_best', 10)
        selected_features = feature_scores.index[:k].tolist()
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_features
    
    def _variance_threshold(self, X: pd.DataFrame) -> pd.Series:
        """
        Select features based on variance threshold.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Series of variance scores
        """
        # Calculate variance for each feature
        variances = X.var()
        
        # Normalize to 0-1 range
        if variances.max() > variances.min():
            normalized = (variances - variances.min()) / (variances.max() - variances.min())
        else:
            normalized = variances / variances.max() if variances.max() > 0 else variances
        
        return normalized
    
    def _select_k_best(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Select features using SelectKBest.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Series of feature scores
        """
        # Handle categorical target
        if y.dtype == 'object' or y.dtype.name == 'category':
            selector = SelectKBest(mutual_info_classif, k='all')
        else:
            selector = SelectKBest(f_classif, k='all')
        
        # Fit selector
        selector.fit(X, y)
        
        # Get scores
        scores = pd.Series(selector.scores_, index=X.columns)
        
        # Normalize scores
        if scores.max() > scores.min():
            normalized = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized = scores / scores.max() if scores.max() > 0 else scores
        
        return normalized
    
    def _recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Series of feature importance scores
        """
        # Initialize model for RFE
        model = RandomForestClassifier(
            n_estimators=100, random_state=self.config['random_state']
        )
        
        # Initialize RFE
        rfe = RFE(
            estimator=model,
            n_features_to_select=min(self.config['methods'].get('rfe', 10), len(X.columns)),
            step=1
        )
        
        # Fit RFE
        try:
            rfe.fit(X, y)
            
            # Get ranking (1 = selected)
            ranks = pd.Series(rfe.ranking_, index=X.columns)
            
            # Convert ranking to scores (higher is better)
            max_rank = ranks.max()
            scores = (max_rank - ranks + 1) / max_rank
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in RFE feature selection: {e}")
            # Return default scores
            return pd.Series(1, index=X.columns)
    
    def _feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Select features using feature importance from tree-based models.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Series of feature importance scores
        """
        # Initialize models
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
        gb = GradientBoostingClassifier(n_estimators=100, random_state=self.config['random_state'])
        
        # Initialize scores
        rf_scores = pd.Series(0, index=X.columns)
        gb_scores = pd.Series(0, index=X.columns)
        
        # Fit Random Forest
        try:
            rf.fit(X, y)
            rf_scores = pd.Series(rf.feature_importances_, index=X.columns)
        except Exception as e:
            logger.error(f"Error in Random Forest feature importance: {e}")
        
        # Fit Gradient Boosting
        try:
            gb.fit(X, y)
            gb_scores = pd.Series(gb.feature_importances_, index=X.columns)
        except Exception as e:
            logger.error(f"Error in Gradient Boosting feature importance: {e}")
        
        # Average the scores
        scores = (rf_scores + gb_scores) / 2
        
        # Normalize scores
        if scores.max() > scores.min():
            normalized = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized = scores / scores.max() if scores.max() > 0 else scores
        
        return normalized
    
    def _correlation_threshold(self, X: pd.DataFrame) -> List[str]:
        """
        Select features by removing highly correlated features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Extract upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
        # Find features with correlation greater than threshold
        threshold = self.config['methods'].get('correlation_threshold', 0.8)
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Return features to keep
        return [col for col in X.columns if col not in to_drop]
    
    def plot_feature_importance(self, feature_scores: pd.Series, title: str = "Feature Importance",
                              figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_scores: Series of feature importance scores
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.barplot(x=feature_scores.values, y=feature_scores.index)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str],
                              figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features
            features: List of feature column names
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        corr = df[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()