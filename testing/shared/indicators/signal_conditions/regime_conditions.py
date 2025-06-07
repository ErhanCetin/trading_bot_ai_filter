"""
Regime Signal Conditions - Phase 4
Condition checkers for market regime indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from .base_conditions import BaseConditionChecker

logger = logging.getLogger(__name__)


class MarketRegimeConditionChecker(BaseConditionChecker):
    """Handles market regime conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['regime_filter', 'regime_duration', 'regime_strength', 'regime_transition']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check market regime conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'regime_filter':
            return self._check_regime_filter(row, condition)
        elif condition_type == 'regime_duration':
            return self._check_regime_duration(row, condition)
        elif condition_type == 'regime_strength':
            return self._check_regime_strength(row, condition)
        elif condition_type == 'regime_transition':
            return self._check_regime_transition(row, condition)
        
        return False
    
    def _check_regime_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check market regime filter."""
        allowed_regimes = condition.get('allowed_regimes', ['strong_uptrend', 'weak_uptrend'])
        
        # FIX: Use actual column name
        if 'market_regime' in row.index:
            current_regime = row['market_regime']
            if pd.isna(current_regime):
                return True
            return current_regime in allowed_regimes
        
        return True
    
    def _check_regime_duration(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check regime duration."""
        min_duration = condition.get('min_duration', 5)
        max_duration = condition.get('max_duration', 100)
        
        if 'regime_duration' in row.index:
            duration = row['regime_duration']
            if pd.isna(duration):
                return True
            return min_duration <= duration <= max_duration
        
        return True
    
    def _check_regime_strength(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check regime strength."""
        min_strength = condition.get('min_strength', 50)
        
        if 'regime_strength' in row.index:
            strength = row['regime_strength']
            if pd.isna(strength):
                return True
            return strength >= min_strength
        
        return True
    
    def _check_regime_transition(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check regime transition."""
        transition_type = condition.get('transition_type', 'any')
        
        # This would need historical data for proper transition detection
        # Simplified implementation
        if 'regime_duration' in row.index:
            duration = row['regime_duration']
            if pd.isna(duration):
                return True
            
            # Consider transition if duration is very short (new regime)
            if transition_type == 'new':
                return duration <= 3
            elif transition_type == 'stable':
                return duration > 10
            else:  # any
                return True
        
        return True


class VolatilityRegimeConditionChecker(BaseConditionChecker):
    """Handles volatility regime conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['volatility_regime_filter', 'volatility_percentile', 'volatility_trend', 'volatility_ratio']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility regime conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'volatility_regime_filter':
            return self._check_volatility_regime_filter(row, condition)
        elif condition_type == 'volatility_percentile':
            return self._check_volatility_percentile(row, condition)
        elif condition_type == 'volatility_trend':
            return self._check_volatility_trend(row, condition)
        elif condition_type == 'volatility_ratio':
            return self._check_volatility_ratio(row, condition)
        
        return False
    
    def _check_volatility_regime_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility regime filter."""
        allowed_regimes = condition.get('allowed_regimes', ['normal', 'high'])
        
        # FIX: Use actual column name  
        if 'volatility_regime' in row.index:
            current_regime = row['volatility_regime']
            if pd.isna(current_regime):
                return True
            return current_regime in allowed_regimes
        
        return True
    
    def _check_volatility_percentile(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility percentile."""
        min_percentile = condition.get('min_percentile', 25)
        max_percentile = condition.get('max_percentile', 75)
        
        if 'volatility_percentile' in row.index:
            percentile = row['volatility_percentile']
            if pd.isna(percentile):
                return True
            return min_percentile <= percentile <= max_percentile
        
        return True
    
    def _check_volatility_trend(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility trend."""
        trend_direction = condition.get('trend_direction', 'increasing')
        
        if 'volatility_trend' in row.index:
            trend = row['volatility_trend']
            if pd.isna(trend):
                return True
            
            if trend_direction == 'increasing':
                return trend > 0
            elif trend_direction == 'decreasing':
                return trend < 0
            else:  # stable
                return abs(trend) < 0.1
        
        return True
    
    def _check_volatility_ratio(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility ratio."""
        min_ratio = condition.get('min_ratio', 0.8)
        max_ratio = condition.get('max_ratio', 1.2)
        
        if 'volatility_ratio' in row.index:
            ratio = row['volatility_ratio']
            if pd.isna(ratio):
                return True
            return min_ratio <= ratio <= max_ratio
        
        return True


class TrendStrengthConditionChecker(BaseConditionChecker):
    """Handles trend strength conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['trend_strength_filter', 'trend_direction_filter', 'trend_alignment', 'trend_health']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend strength conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'trend_strength_filter':
            return self._check_trend_strength_filter(row, condition)
        elif condition_type == 'trend_direction_filter':
            return self._check_trend_direction_filter(row, condition)
        elif condition_type == 'trend_alignment':
            return self._check_trend_alignment(row, condition)
        elif condition_type == 'trend_health':
            return self._check_trend_health(row, condition)
        
        return False
    
    def _check_trend_strength_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend strength filter."""
        min_strength = condition.get('min_strength', 50)
        
        if 'trend_strength' in row.index:
            strength = row['trend_strength']
            if pd.isna(strength):
                return True
            return strength >= min_strength
        
        return True
    
    def _check_trend_direction_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend direction filter."""
        required_direction = condition.get('direction', 1)  # 1 for up, -1 for down
        
        if 'trend_direction' in row.index:
            direction = row['trend_direction']
            if pd.isna(direction):
                return True
            return direction == required_direction
        
        return True
    
    def _check_trend_alignment(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend alignment."""
        min_alignment = condition.get('min_alignment', 0.5)
        
        if 'trend_alignment' in row.index:
            alignment = row['trend_alignment']
            if pd.isna(alignment):
                return True
            return abs(alignment) >= min_alignment
        
        return True
    
    def _check_trend_health(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend health."""
        min_health = condition.get('min_health', 60)
        
        if 'trend_health' in row.index:
            health = row['trend_health']
            if pd.isna(health):
                return True
            return health >= min_health
        
        return True


class StatisticalConditionChecker(BaseConditionChecker):
    """Handles statistical indicator conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['zscore_threshold', 'percentile_filter', 'statistical_outlier', 'mean_reversion']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check statistical conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'zscore_threshold':
            return self._check_zscore_threshold(row, condition)
        elif condition_type == 'percentile_filter':
            return self._check_percentile_filter(row, condition)
        elif condition_type == 'statistical_outlier':
            return self._check_statistical_outlier(row, condition)
        elif condition_type == 'mean_reversion':
            return self._check_mean_reversion(row, condition)
        
        return False
    
    def _check_zscore_threshold(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Z-score threshold."""
        column = condition.get('column', 'close_zscore')
        threshold = condition.get('threshold', 2.0)
        operator = condition.get('operator', '>=')
        
        if column in row.index:
            zscore_value = row[column]
            if pd.isna(zscore_value):
                return True
            
            if operator == '>=':
                return abs(zscore_value) >= threshold
            elif operator == '<=':
                return abs(zscore_value) <= threshold
            elif operator == '>':
                return abs(zscore_value) > threshold
            elif operator == '<':
                return abs(zscore_value) < threshold
        
        return True
    
    def _check_percentile_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check percentile filter."""
        column = condition.get('column', 'close_percentile')
        min_percentile = condition.get('min_percentile', 10)
        max_percentile = condition.get('max_percentile', 90)
        
        if column in row.index:
            percentile_value = row[column]
            if pd.isna(percentile_value):
                return True
            return min_percentile <= percentile_value <= max_percentile
        
        return True
    
    def _check_statistical_outlier(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check statistical outlier conditions."""
        column = condition.get('column', 'close_zscore')
        outlier_threshold = condition.get('outlier_threshold', 3.0)
        
        if column in row.index:
            value = row[column]
            if pd.isna(value):
                return True
            return abs(value) >= outlier_threshold
        
        return True
    
    def _check_mean_reversion(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check mean reversion conditions."""
        reversion_type = condition.get('reversion_type', 'extreme')
        
        # Look for Z-score columns to assess mean reversion potential
        zscore_cols = [col for col in row.index if col.endswith('_zscore')]
        
        if not zscore_cols:
            return True
        
        # Check if any Z-score indicates extreme values (mean reversion opportunity)
        for col in zscore_cols:
            zscore_value = row[col]
            if not pd.isna(zscore_value):
                if reversion_type == 'extreme' and abs(zscore_value) > 2.0:
                    return True
                elif reversion_type == 'moderate' and 1.0 <= abs(zscore_value) <= 2.0:
                    return True
        
        return False