"""
Feature Signal Conditions - Phase 3
Condition checkers for feature engineering indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from .base_conditions import BaseConditionChecker

logger = logging.getLogger(__name__)


class PriceActionConditionChecker(BaseConditionChecker):
    """Handles price action pattern conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['candlestick_pattern', 'body_size_filter', 'shadow_analysis', 'price_range_filter']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check price action conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'candlestick_pattern':
            return self._check_candlestick_pattern(row, condition)
        elif condition_type == 'body_size_filter':
            return self._check_body_size_filter(row, condition)
        elif condition_type == 'shadow_analysis':
            return self._check_shadow_analysis(row, condition)
        elif condition_type == 'price_range_filter':
            return self._check_price_range_filter(row, condition)
        
        return False
    

    def _check_candlestick_pattern(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check specific candlestick patterns."""
        pattern_type = condition.get('pattern', 'doji')
        
        # FIX: Use actual column names from indicator
        if pattern_type == 'doji' and 'doji_pattern' in row.index:
            return bool(row['doji_pattern'])
        elif pattern_type == 'engulfing' and 'engulfing_pattern' in row.index:
            return abs(row['engulfing_pattern']) > 0  # FIX: Handle 1/-1 values
        elif pattern_type == 'hammer' and 'hammer_pattern' in row.index:
            return bool(row['hammer_pattern'])
        elif pattern_type == 'shooting_star' and 'shooting_star_pattern' in row.index:
            return bool(row['shooting_star_pattern'])
        
        return True  # Changed from False to True for missing patterns
    
    def _check_body_size_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check body size conditions."""
        min_body_size = condition.get('min_body_size', 0.01)
        max_body_size = condition.get('max_body_size', 0.1)
        
        if 'body_size' in row.index and 'range_size' in row.index:
            body_size = row['body_size']
            range_size = row['range_size']
            
            if pd.isna(body_size) or pd.isna(range_size) or range_size == 0:
                return True
            
            body_ratio = body_size / range_size
            return min_body_size <= body_ratio <= max_body_size
        
        return True
    
    def _check_shadow_analysis(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check shadow length conditions."""
        shadow_type = condition.get('shadow_type', 'upper')
        min_ratio = condition.get('min_ratio', 0.3)
        
        # FIX: Use actual column names
        if shadow_type == 'upper' and 'upper_shadow' in row.index:
            shadow_ratio = row['upper_shadow']
            if pd.isna(shadow_ratio):
                return True
            return shadow_ratio >= min_ratio
        elif shadow_type == 'lower' and 'lower_shadow' in row.index:
            shadow_ratio = row['lower_shadow']
            if pd.isna(shadow_ratio):
                return True
            return shadow_ratio >= min_ratio
        
        return True
    
    def _check_price_range_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check price range conditions."""
        min_range_pct = condition.get('min_range_pct', 0.5)
        
        if 'daily_range' in row.index:
            range_pct = row['daily_range'] * 100  # Convert to percentage
            if pd.isna(range_pct):
                return True
            return range_pct >= min_range_pct
        
        return True


class VolumePriceConditionChecker(BaseConditionChecker):
    """Handles volume-price relationship conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['volume_price_confirmation', 'obv_direction', 'price_volume_trend', 'volume_oscillator']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volume-price conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'volume_price_confirmation':
            return self._check_volume_price_confirmation(row, condition)
        elif condition_type == 'obv_direction':
            return self._check_obv_direction(row, condition)
        elif condition_type == 'price_volume_trend':
            return self._check_price_volume_trend(row, condition)
        elif condition_type == 'volume_oscillator':
            return self._check_volume_oscillator(row, condition)
        
        return False
    
    def _check_volume_price_confirmation(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volume-price confirmation."""
        confirmation_type = condition.get('confirmation_type', 'positive')
        
        if 'volume_price_confirmation' in row.index:
            confirmation_value = row['volume_price_confirmation']
            if pd.isna(confirmation_value):
                return True
            
            if confirmation_type == 'positive':
                return confirmation_value > 0
            elif confirmation_type == 'negative':
                return confirmation_value < 0
            else:  # any
                return abs(confirmation_value) > 0
        
        return True
    
    def _check_obv_direction(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check OBV direction."""
        direction = condition.get('direction', 'up')
        
        if 'obv' in row.index:
            # This would need historical data for proper direction calculation
            # Simplified implementation
            obv_value = row['obv']
            if pd.isna(obv_value):
                return True
            
            # Simplified: assume positive OBV growth = up direction
            return obv_value > 0 if direction == 'up' else obv_value < 0
        
        return True
    
    def _check_price_volume_trend(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check price volume trend indicator."""
        trend_direction = condition.get('trend_direction', 'bullish')
        
        if 'price_volume_trend' in row.index:
            pvt_value = row['price_volume_trend']
            if pd.isna(pvt_value):
                return True
            
            if trend_direction == 'bullish':
                return pvt_value > 0
            else:
                return pvt_value < 0
        
        return True
    
    def _check_volume_oscillator(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volume oscillator conditions."""
        oscillator_level = condition.get('level', 'overbought')
        threshold = condition.get('threshold', 5.0)
        
        if 'volume_oscillator' in row.index:
            oscillator_value = row['volume_oscillator']
            if pd.isna(oscillator_value):
                return True
            
            if oscillator_level == 'overbought':
                return oscillator_value >= threshold
            elif oscillator_level == 'oversold':
                return oscillator_value <= -threshold
            else:  # neutral
                return abs(oscillator_value) < threshold
        
        return True


class MomentumConditionChecker(BaseConditionChecker):
    """Handles momentum feature conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['momentum_threshold', 'momentum_acceleration', 'momentum_divergence', 'momentum_consistency']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check momentum conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'momentum_threshold':
            return self._check_momentum_threshold(row, condition)
        elif condition_type == 'momentum_acceleration':
            return self._check_momentum_acceleration(row, condition)
        elif condition_type == 'momentum_divergence':
            return self._check_momentum_divergence(row, condition)
        elif condition_type == 'momentum_consistency':
            return self._check_momentum_consistency(row, condition)
        
        return False
    
    def _check_momentum_threshold(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check momentum threshold conditions."""
        period = condition.get('period', 5)
        threshold = condition.get('threshold', 2.0)
        operator = condition.get('operator', '>=')
        
        # FIX: Use actual column names from momentum_features
        momentum_col = f'momentum_{period}'
        
        if momentum_col in row.index:
            momentum_value = row[momentum_col]
            if pd.isna(momentum_value):
                return True
            
            if operator == '>=':
                return abs(momentum_value) >= threshold
            elif operator == '>':
                return abs(momentum_value) > threshold
            elif operator == '<=':
                return abs(momentum_value) <= threshold
            elif operator == '<':
                return abs(momentum_value) < threshold
        
        return True
    
    def _check_momentum_acceleration(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check momentum acceleration."""
        period = condition.get('period', 5)
        acceleration_type = condition.get('acceleration_type', 'increasing')
        
        # FIX: Use actual column names
        accel_col = f'momentum_accel_{period}'
        
        if accel_col in row.index:
            accel_value = row[accel_col]
            if pd.isna(accel_value):
                return True
            
            if acceleration_type == 'increasing':
                return accel_value > 0
            else:
                return accel_value < 0
        
        return True
    
    def _check_momentum_divergence(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check momentum divergence conditions."""
        divergence_type = condition.get('divergence_type', 'bullish')
        
        # FIX: Use actual column names
        if divergence_type == 'bullish' and 'bullish_divergence' in row.index:
            return bool(row['bullish_divergence'])
        elif divergence_type == 'bearish' and 'bearish_divergence' in row.index:
            return bool(row['bearish_divergence'])
        
        return True
    
    def _check_momentum_consistency(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check momentum consistency across periods."""
        periods = condition.get('periods', [5, 10, 20])
        consistency_threshold = condition.get('consistency_threshold', 0.7)
        
        momentum_cols = [f'momentum_{period}' for period in periods]
        available_cols = [col for col in momentum_cols if col in row.index]
        
        if len(available_cols) < 2:
            return True
        
        # Check how many momentum indicators agree on direction
        positive_count = 0
        negative_count = 0
        
        for col in available_cols:
            momentum_value = row[col]
            if not pd.isna(momentum_value):
                if momentum_value > 0:
                    positive_count += 1
                else:
                    negative_count += 1
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return True
        
        consistency = max(positive_count, negative_count) / total_indicators
        return consistency >= consistency_threshold


class SupportResistanceConditionChecker(BaseConditionChecker):
    """Handles support and resistance conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['near_support', 'near_resistance', 'support_break', 'resistance_break', 'sr_zone']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check support/resistance conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'near_support':
            return self._check_near_support(row, condition)
        elif condition_type == 'near_resistance':
            return self._check_near_resistance(row, condition)
        elif condition_type == 'support_break':
            return self._check_support_break(row, condition)
        elif condition_type == 'resistance_break':
            return self._check_resistance_break(row, condition)
        elif condition_type == 'sr_zone':
            return self._check_sr_zone(row, condition)
        
        return False
    
    def _check_near_support(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check if price is near support."""
        max_distance_pct = condition.get('max_distance_pct', 2.0)
        
        if 'support_distance' in row.index:
            distance = row['support_distance']
            if pd.isna(distance):
                return True
            return abs(distance) <= max_distance_pct
        
        return True
    
    def _check_near_resistance(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check if price is near resistance."""
        max_distance_pct = condition.get('max_distance_pct', 2.0)
        
        if 'resistance_distance' in row.index:
            distance = row['resistance_distance']
            if pd.isna(distance):
                return True
            return abs(distance) <= max_distance_pct
        
        return True
    
    def _check_support_break(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check support breakout."""
        if 'broke_support' in row.index:
            return bool(row['broke_support'])
        
        return True
    
    def _check_resistance_break(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check resistance breakout."""
        if 'broke_resistance' in row.index:
            return bool(row['broke_resistance'])
        
        return True
    
    def _check_sr_zone(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check if price is in support/resistance zone."""
        zone_type = condition.get('zone_type', 'support')
        
        zone_col = f'in_{zone_type}_zone'
        
        if zone_col in row.index:
            return bool(row[zone_col])
        
        return True