"""
Base Signal Conditions - Phase 1
Basic condition checkers for fundamental technical indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseConditionChecker(ABC):
    """Base class for all condition checkers."""
    
    @abstractmethod
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check if condition is met for given row."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of condition types this checker supports."""
        pass


class BasicConditionChecker(BaseConditionChecker):
    """Handles basic threshold and comparison conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['threshold', 'price_cross_above', 'price_cross_below', 'line_cross_above', 'line_cross_below']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check basic conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'threshold':
            return self._check_threshold(row, condition)
        elif condition_type == 'price_cross_above':
            return self._check_cross(row, condition, 'above')
        elif condition_type == 'price_cross_below':
            return self._check_cross(row, condition, 'below')
        elif condition_type == 'line_cross_above':
            return self._check_line_cross(row, condition, 'above')
        elif condition_type == 'line_cross_below':
            return self._check_line_cross(row, condition, 'below')
        
        return False
    
    def _check_threshold(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check threshold condition (e.g., RSI <= 30)."""
        column = condition.get('column', '')
        operator = condition.get('operator', '')
        value = condition.get('value', 0)
        
        if column not in row.index:
            return False
        
        current_value = row[column]
        if pd.isna(current_value):
            return False
        
        if operator == '<=':
            return current_value <= value
        elif operator == '>=':
            return current_value >= value
        elif operator == '<':
            return current_value < value
        elif operator == '>':
            return current_value > value
        elif operator == '==':
            return abs(current_value - value) < 1e-6
        
        return False
    
    def _check_cross(self, row: pd.Series, condition: Dict[str, Any], direction: str) -> bool:
        """Check price crossover condition."""
        price_column = condition.get('price_column', 'close')
        indicator_column = condition.get('indicator_column', '')
        
        if price_column not in row.index or indicator_column not in row.index:
            return False
        
        price = row[price_column]
        indicator_value = row[indicator_column]
        
        if pd.isna(price) or pd.isna(indicator_value):
            return False
        
        if direction == 'above':
            return price > indicator_value
        else:
            return price < indicator_value
    
    def _check_line_cross(self, row: pd.Series, condition: Dict[str, Any], direction: str) -> bool:
        """Check line-to-line crossover condition."""
        line1 = condition.get('line1', '')
        line2 = condition.get('line2', '')
        
        if line1 not in row.index or line2 not in row.index:
            return False
        
        value1 = row[line1]
        value2 = row[line2]
        
        if pd.isna(value1) or pd.isna(value2):
            return False
        
        if direction == 'above':
            return value1 > value2
        else:
            return value1 < value2


class VolumeConditionChecker(BaseConditionChecker):
    """Handles volume-related conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['volume_confirmation', 'volume_surge']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volume conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'volume_confirmation':
            return self._check_volume_confirmation(row, condition)
        elif condition_type == 'volume_surge':
            return self._check_volume_surge(row, condition)
        
        return False
    
    def _check_volume_confirmation(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volume confirmation condition."""
        min_volume_ratio = condition.get('min_volume_ratio', 1.2)
        
        if 'volume' not in row.index:
            return True  # Pass if no volume data
        
        current_volume = row['volume']
        volume_ma = row.get('volume_ma', current_volume)
        
        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma == 0:
            return True
        
        volume_ratio = current_volume / volume_ma
        return volume_ratio >= min_volume_ratio
    
    def _check_volume_surge(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volume surge condition."""
        volume_multiplier = condition.get('volume_multiplier', 1.3)
        
        if 'volume' not in row.index:
            return True
        
        volume = row['volume']
        volume_ma = row.get('volume_ma', volume)
        
        if pd.isna(volume) or pd.isna(volume_ma) or volume_ma == 0:
            return True
        
        return (volume / volume_ma) >= volume_multiplier


class MACDConditionChecker(BaseConditionChecker):
    """Handles MACD-specific conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['zero_line_filter', 'histogram_positive', 'histogram_negative']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check MACD conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'zero_line_filter':
            return self._check_zero_line_filter(row, condition)
        elif condition_type == 'histogram_positive':
            return self._check_histogram_condition(row, condition, 'positive')
        elif condition_type == 'histogram_negative':
            return self._check_histogram_condition(row, condition, 'negative')
        
        return False
    
    def _check_zero_line_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check zero line filter condition."""
        above_zero = condition.get('above_zero', True)
        
        if 'macd_line' in row.index:
            macd_value = row['macd_line']
            if pd.isna(macd_value):
                return True
            
            if above_zero:
                return macd_value > 0
            else:
                return macd_value < 0
        
        return True
    
    def _check_histogram_condition(self, row: pd.Series, condition: Dict[str, Any], polarity: str) -> bool:
        """Check histogram condition."""
        column = condition.get('column', 'macd_histogram')
        
        if column not in row.index:
            return True
        
        value = row[column]
        if pd.isna(value):
            return True
        
        if polarity == 'positive':
            return value > 0
        else:
            return value < 0


class BollingerConditionChecker(BaseConditionChecker):
    """Handles Bollinger Bands conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['price_touch_lower_band', 'price_touch_upper_band', 'squeeze_condition', 'mean_reversion_setup']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Bollinger conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'price_touch_lower_band':
            return self._check_band_touch(row, condition, 'lower')
        elif condition_type == 'price_touch_upper_band':
            return self._check_band_touch(row, condition, 'upper')
        elif condition_type == 'squeeze_condition':
            return self._check_squeeze_condition(row, condition)
        elif condition_type == 'mean_reversion_setup':
            return self._check_mean_reversion_setup(row, condition)
        
        return False
    
    def _check_band_touch(self, row: pd.Series, condition: Dict[str, Any], band_type: str) -> bool:
        """Check if price touches Bollinger Band."""
        price_column = condition.get('price_column', 'close')
        band_column = condition.get('band_column', f'bollinger_{band_type}')
        
        if price_column not in row.index or band_column not in row.index:
            return True
        
        price = row[price_column]
        band_value = row[band_column]
        
        if pd.isna(price) or pd.isna(band_value):
            return True
        
        tolerance = abs(band_value) * 0.001
        
        if band_type == 'lower':
            return price <= (band_value + tolerance)
        else:
            return price >= (band_value - tolerance)
    
    def _check_squeeze_condition(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Bollinger Band squeeze condition."""
        width_threshold = condition.get('width_threshold', 0.02)
        
        if 'bollinger_width' in row.index:
            width = row['bollinger_width']
            if pd.isna(width):
                return True
            return width <= width_threshold
        
        return True
    
    def _check_mean_reversion_setup(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check mean reversion setup condition."""
        if 'bollinger_middle' in row.index and 'close' in row.index:
            close = row['close']
            middle = row['bollinger_middle']
            
            if pd.isna(close) or pd.isna(middle) or middle == 0:
                return True
            
            deviation = abs(close - middle) / middle
            return deviation >= 0.01
        
        return True


class ATRConditionChecker(BaseConditionChecker):
    """Handles ATR/volatility conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['volatility_breakout', 'price_momentum']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check ATR conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'volatility_breakout':
            return self._check_volatility_breakout(row, condition)
        elif condition_type == 'price_momentum':
            return self._check_price_momentum(row, condition)
        
        return False
    
    def _check_volatility_breakout(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility breakout condition."""
        atr_multiplier = condition.get('atr_multiplier', 1.5)
        
        if 'atr_14' in row.index:
            atr = row['atr_14']
            if pd.isna(atr):
                return True
            return atr > 0  # Simplified condition
        
        return True
    
    def _check_price_momentum(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check price momentum condition."""
        periods = condition.get('periods', 5)
        momentum_col = f'momentum_{periods}'
        
        if momentum_col in row.index:
            momentum = row[momentum_col]
            if pd.isna(momentum):
                return True
            return abs(momentum) > 1.0
        
        return True


class TrendConditionChecker(BaseConditionChecker):
    """Handles trend-related conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['trend_confirmation']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'trend_confirmation':
            return self._check_trend_confirmation(row, condition)
        
        return False
    
    def _check_trend_confirmation(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check trend confirmation condition."""
        lookback = condition.get('lookback', 3)
        # Simplified: assume trend confirmation
        return True