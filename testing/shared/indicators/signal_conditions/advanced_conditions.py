"""
Advanced Signal Conditions - Phase 2
Condition checkers for advanced technical indicators
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from .base_conditions import BaseConditionChecker

logger = logging.getLogger(__name__)


class SupertrendConditionChecker(BaseConditionChecker):
    """Handles Supertrend-specific conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['supertrend_direction_change', 'supertrend_band_break']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Supertrend conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'supertrend_direction_change':
            return self._check_direction_change(row, condition)
        elif condition_type == 'supertrend_band_break':
            return self._check_band_break(row, condition)
        
        return False
    
    def _check_direction_change(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check if Supertrend direction changed."""
        direction = condition.get('direction', 'bullish')
        
        if 'supertrend_direction' in row.index:
            current_direction = row['supertrend_direction']
            if pd.isna(current_direction):
                return True
            
            if direction == 'bullish':
                return bool(current_direction)  # True for bullish
            else:
                return not bool(current_direction)  # False for bearish
        
        return True
    
    def _check_band_break(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check if price breaks Supertrend bands."""
        break_type = condition.get('break_type', 'above')
        
        required_cols = ['close', 'supertrend']
        if not all(col in row.index for col in required_cols):
            return True
        
        price = row['close']
        supertrend = row['supertrend']
        
        if pd.isna(price) or pd.isna(supertrend):
            return True
        
        if break_type == 'above':
            return price > supertrend
        else:
            return price < supertrend


class HeikinAshiConditionChecker(BaseConditionChecker):
    """Handles Heikin Ashi-specific conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['ha_trend_consistency', 'ha_candle_pattern', 'ha_momentum']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Heikin Ashi conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'ha_trend_consistency':
            return self._check_trend_consistency(row, condition)
        elif condition_type == 'ha_candle_pattern':
            return self._check_candle_pattern(row, condition)
        elif condition_type == 'ha_momentum':
            return self._check_momentum(row, condition)
        
        return False
    
    def _check_trend_consistency(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check HA trend consistency."""
        required_trend = condition.get('trend', 1)  # 1 for bullish, -1 for bearish
        
        if 'ha_trend' in row.index:
            current_trend = row['ha_trend']
            if pd.isna(current_trend):
                return True
            return current_trend == required_trend
        
        return True
    
    def _check_candle_pattern(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check HA candle patterns."""
        pattern_type = condition.get('pattern', 'doji')
        
        ha_cols = ['ha_open', 'ha_high', 'ha_low', 'ha_close']
        if not all(col in row.index for col in ha_cols):
            return True
        
        ha_open = row['ha_open']
        ha_close = row['ha_close']
        ha_high = row['ha_high']
        ha_low = row['ha_low']
        
        if any(pd.isna(val) for val in [ha_open, ha_close, ha_high, ha_low]):
            return True
        
        body_size = abs(ha_close - ha_open)
        range_size = ha_high - ha_low
        
        if range_size == 0:
            return True
        
        if pattern_type == 'doji':
            return (body_size / range_size) < 0.1
        elif pattern_type == 'strong_bull':
            return (ha_close > ha_open) and ((body_size / range_size) > 0.7)
        elif pattern_type == 'strong_bear':
            return (ha_close < ha_open) and ((body_size / range_size) > 0.7)
        
        return True
    
    def _check_momentum(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check HA momentum."""
        momentum_type = condition.get('momentum', 'increasing')
        
        # Simplified momentum check
        if 'ha_close' in row.index and 'ha_open' in row.index:
            ha_close = row['ha_close']
            ha_open = row['ha_open']
            
            if pd.isna(ha_close) or pd.isna(ha_open):
                return True
            
            if momentum_type == 'increasing':
                return ha_close > ha_open
            else:
                return ha_close < ha_open
        
        return True


class IchimokuConditionChecker(BaseConditionChecker):
    """Handles Ichimoku Cloud-specific conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['cloud_position', 'cloud_strength_filter', 'kumo_twist', 'tenkan_kijun_cross']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Ichimoku conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'cloud_position':
            return self._check_cloud_position(row, condition)
        elif condition_type == 'cloud_strength_filter':
            return self._check_cloud_strength(row, condition)
        elif condition_type == 'kumo_twist':
            return self._check_kumo_twist(row, condition)
        elif condition_type == 'tenkan_kijun_cross':
            return self._check_tk_cross(row, condition)
        
        return False
    
    def _check_cloud_position(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check price position relative to cloud."""
        position = condition.get('position', 'above')
        
        cloud_cols = ['senkou_span_a', 'senkou_span_b', 'close']
        if not all(col in row.index for col in cloud_cols):
            return True
        
        price = row['close']
        span_a = row['senkou_span_a']
        span_b = row['senkou_span_b']
        
        if any(pd.isna(val) for val in [price, span_a, span_b]):
            return True
        
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 'above':
            return price > cloud_top
        elif position == 'below':
            return price < cloud_bottom
        elif position == 'inside':
            return cloud_bottom <= price <= cloud_top
        
        return True
    
    def _check_cloud_strength(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check cloud strength filter."""
        min_strength = condition.get('min_strength', 0)
        
        if 'cloud_strength' in row.index:
            strength = row['cloud_strength']
            if pd.isna(strength):
                return True
            return abs(strength) >= min_strength
        
        return True
    
    def _check_kumo_twist(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check for Kumo twist (cloud color change)."""
        twist_type = condition.get('twist_type', 'bullish')
        
        if 'cloud_strength' in row.index:
            strength = row['cloud_strength']
            if pd.isna(strength):
                return True
            
            if twist_type == 'bullish':
                return strength > 0
            else:
                return strength < 0
        
        return True
    
    def _check_tk_cross(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Tenkan-Kijun cross."""
        cross_type = condition.get('cross_type', 'golden')
        
        tk_cols = ['tenkan_sen', 'kijun_sen']
        if not all(col in row.index for col in tk_cols):
            return True
        
        tenkan = row['tenkan_sen']
        kijun = row['kijun_sen']
        
        if pd.isna(tenkan) or pd.isna(kijun):
            return True
        
        if cross_type == 'golden':  # Bullish cross
            return tenkan > kijun
        else:  # Death cross
            return tenkan < kijun


class AdaptiveRSIConditionChecker(BaseConditionChecker):
    """Handles Adaptive RSI-specific conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['adaptive_period_filter', 'volatility_regime_filter', 'adaptive_threshold']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check Adaptive RSI conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'adaptive_period_filter':
            return self._check_period_filter(row, condition)
        elif condition_type == 'volatility_regime_filter':
            return self._check_volatility_regime(row, condition)
        elif condition_type == 'adaptive_threshold':
            return self._check_adaptive_threshold(row, condition)
        
        return False
    
    def _check_period_filter(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check adaptive period filter."""
        max_period = condition.get('max_period', 25)
        min_period = condition.get('min_period', 5)
        
        if 'adaptive_rsi_period' in row.index:
            period = row['adaptive_rsi_period']
            if pd.isna(period):
                return True
            return min_period <= period <= max_period
        
        return True
    
    def _check_volatility_regime(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check volatility regime for adaptive RSI."""
        regime = condition.get('regime', 'high')
        
        # Check if we have ATR data for volatility assessment
        if 'atr_14' in row.index and 'close' in row.index:
            atr = row['atr_14']
            close = row['close']
            
            if pd.isna(atr) or pd.isna(close) or close == 0:
                return True
            
            atr_pct = (atr / close) * 100
            
            if regime == 'high':
                return atr_pct > 2.0  # High volatility threshold
            elif regime == 'low':
                return atr_pct < 1.0  # Low volatility threshold
            else:  # medium
                return 1.0 <= atr_pct <= 2.0
        
        return True
    
    def _check_adaptive_threshold(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check adaptive threshold based on period."""
        base_threshold = condition.get('base_threshold', 30)
        operator = condition.get('operator', '<=')
        
        if 'adaptive_rsi' not in row.index or 'adaptive_rsi_period' not in row.index:
            return True
        
        rsi_value = row['adaptive_rsi']
        period = row['adaptive_rsi_period']
        
        if pd.isna(rsi_value) or pd.isna(period):
            return True
        
        # Adjust threshold based on period
        # Shorter periods (high volatility) -> more extreme thresholds
        # Longer periods (low volatility) -> less extreme thresholds
        adjustment = (period - 14) * 0.5  # Adjust ±0.5 per period difference
        adjusted_threshold = base_threshold + adjustment
        
        # Ensure threshold stays within bounds
        if base_threshold <= 50:  # Oversold threshold
            adjusted_threshold = max(10, min(40, adjusted_threshold))
        else:  # Overbought threshold
            adjusted_threshold = max(60, min(90, adjusted_threshold))
        
        if operator == '<=':
            return rsi_value <= adjusted_threshold
        elif operator == '>=':
            return rsi_value >= adjusted_threshold
        elif operator == '<':
            return rsi_value < adjusted_threshold
        elif operator == '>':
            return rsi_value > adjusted_threshold
        
        return True


class MTFEMAConditionChecker(BaseConditionChecker):
    """Handles Multi-Timeframe EMA conditions."""
    
    def get_supported_types(self) -> List[str]:
        return ['ema_alignment_strength', 'timeframe_consistency', 'alignment_momentum']
    
    def check(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check MTF EMA conditions."""
        condition_type = condition.get('type', '')
        
        if condition_type == 'ema_alignment_strength':
            return self._check_alignment_strength(row, condition)
        elif condition_type == 'timeframe_consistency':
            return self._check_timeframe_consistency(row, condition)
        elif condition_type == 'alignment_momentum':
            return self._check_alignment_momentum(row, condition)
        
        return False
    
    def _check_alignment_strength(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check EMA alignment strength."""
        min_strength = condition.get('min_strength', 0.5)
        
        if 'ema_alignment' in row.index:
            alignment = row['ema_alignment']
            if pd.isna(alignment):
                return True
            return abs(alignment) >= min_strength
        
        return True
    
    def _check_timeframe_consistency(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check consistency across timeframes."""
        consistency_threshold = condition.get('consistency_threshold', 0.8)
        
        # Look for EMA columns across different timeframes
        ema_cols = [col for col in row.index if col.startswith('ema_') and '_' in col and 'x' in col]
        
        if len(ema_cols) < 2:
            return True
        
        price = row.get('close', 0)
        if pd.isna(price) or price == 0:
            return True
        
        # Check how many EMAs are on the same side of price
        above_count = 0
        below_count = 0
        
        for col in ema_cols:
            ema_value = row.get(col, price)
            if not pd.isna(ema_value):
                if price > ema_value:
                    above_count += 1
                else:
                    below_count += 1
        
        total_emas = above_count + below_count
        if total_emas == 0:
            return True
        
        consistency = max(above_count, below_count) / total_emas
        return consistency >= consistency_threshold
    
    def _check_alignment_momentum(self, row: pd.Series, condition: Dict[str, Any]) -> bool:
        """Check alignment momentum."""
        momentum_direction = condition.get('direction', 'increasing')
        
        if 'ema_alignment' in row.index:
            alignment = row['ema_alignment']
            if pd.isna(alignment):
                return True
            
            # Simplified momentum check (would need historical data for proper implementation)
            if momentum_direction == 'increasing':
                return alignment > 0
            else:
                return alignment < 0
        
        return True