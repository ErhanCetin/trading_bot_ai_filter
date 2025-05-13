
"""
regime_filters.py dosyası, piyasa rejimi, volatilite rejimi ve 
trend gücüne göre sinyalleri filtreleyen üç önemli filtre sınıfı içerir.
 Bu filtreler, stratejilerin ürettiği sinyalleri, 
 mevcut piyasa koşullarına göre süzerek, yanlış sinyalleri elimine etmeyi amaçlar.
"""
"""
Market regime-based filters for the trading system.
These filters evaluate signals based on the current market regime.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from signal_engine.filters.base_filter import AdvancedFilterRule


class MarketRegimeFilter(AdvancedFilterRule):
    """Filter signals based on market regime."""
    
    name = "market_regime"
    display_name = "Market Regime Filter"
    description = "Filters signals based on compatibility with current market regime"
    category = "regime"
    
    default_params = {
        "regime_signal_map": {
            "strong_uptrend": {"long": True, "short": False},
            "weak_uptrend": {"long": True, "short": False},
            "strong_downtrend": {"long": False, "short": True},
            "weak_downtrend": {"long": False, "short": True},
            "ranging": {"long": True, "short": True},
            "volatile": {"long": True, "short": True},
            "overbought": {"long": False, "short": True},
            "oversold": {"long": True, "short": False},
            "unknown": {"long": True, "short": True}
        }
    }
    
    required_indicators = ["market_regime"]
    
    def check_rule(self, df: pd.DataFrame, row: pd.Series, i: int, signal_type: str) -> bool:
        """
        Check if the signal is compatible with the current market regime.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            signal_type: Type of signal ('long' or 'short')
            
        Returns:
            True if rule passes, False otherwise
        """
        # Get regime-signal mapping
        regime_map = self.params.get("regime_signal_map", self.default_params["regime_signal_map"])
        
        # Get current regime
        regime = row.get("market_regime", "unknown")
        
        # Check if signal type is allowed in this regime
        return regime_map.get(regime, {}).get(signal_type, True)


class VolatilityRegimeFilter(AdvancedFilterRule):
    """Filter signals based on volatility regime."""
    
    name = "volatility_regime"
    display_name = "Volatility Regime Filter"
    description = "Filters signals based on compatibility with current volatility regime"
    category = "regime"
    
    default_params = {
        "high_volatility_filter": {
            "min_strength": 7,  # Minimum signal strength in high volatility
            "atr_threshold": 2.0  # ATR factor for high volatility
        },
        "low_volatility_filter": {
            "min_strength": 4,  # Minimum signal strength in low volatility
            "atr_threshold": 0.5  # ATR factor for low volatility
        }
    }
    
    required_indicators = ["atr_percent", "signal_strength"]
    optional_indicators = ["volatility_regime"]
    
    def check_rule(self, df: pd.DataFrame, row: pd.Series, i: int, signal_type: str) -> bool:
        """
        Check if the signal meets volatility regime criteria.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            signal_type: Type of signal ('long' or 'short')
            
        Returns:
            True if rule passes, False otherwise
        """
        # Get parameters
        high_vol = self.params.get("high_volatility_filter", self.default_params["high_volatility_filter"])
        low_vol = self.params.get("low_volatility_filter", self.default_params["low_volatility_filter"])
        
        # Get current volatility regime or calculate from ATR
        volatility_regime = "normal"
        if "volatility_regime" in row:
            volatility_regime = row["volatility_regime"]
        elif "atr_percent" in row:
            avg_atr = 1.0  # Assume 1% ATR is average
            if row["atr_percent"] > avg_atr * high_vol["atr_threshold"]:
                volatility_regime = "high"
            elif row["atr_percent"] < avg_atr * low_vol["atr_threshold"]:
                volatility_regime = "low"
        
        # Apply filter based on regime
        if volatility_regime == "high":
            # In high volatility, require stronger signals
            return row["signal_strength"] >= high_vol["min_strength"]
        elif volatility_regime == "low":
            # In low volatility, allow weaker signals
            return row["signal_strength"] >= low_vol["min_strength"]
        else:
            # In normal volatility, use default
            return True


class TrendStrengthFilter(AdvancedFilterRule):
    """Filter signals based on trend strength."""
    
    name = "trend_strength"
    display_name = "Trend Strength Filter"
    description = "Filters signals based on the strength of the current trend"
    category = "regime"
    
    default_params = {
        "adx_threshold": 25,  # Minimum ADX for trend following
        "signal_compatibility": {
            "long": ["strong_uptrend", "weak_uptrend"],
            "short": ["strong_downtrend", "weak_downtrend"]
        }
    }
    
    required_indicators = ["adx"]
    optional_indicators = ["di_pos", "di_neg", "market_regime", "trend_direction", "trend_health"]
    
    def check_rule(self, df: pd.DataFrame, row: pd.Series, i: int, signal_type: str) -> bool:
        """
        Check if the signal aligns with the current trend.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            signal_type: Type of signal ('long' or 'short')
            
        Returns:
            True if rule passes, False otherwise
        """
        # Get parameters
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        signal_compatibility = self.params.get("signal_compatibility", self.default_params["signal_compatibility"])
        
        # Check trend strength via ADX
        if "adx" in row:
            # Weak trend, only allow counter-trend signals
            if row["adx"] < adx_threshold:
                # In weak trend, allow counter-trend signals
                if signal_type == "long" and "di_neg" in row and "di_pos" in row:
                    return row["di_neg"] >= row["di_pos"]  # Allow long if in downtrend
                elif signal_type == "short" and "di_neg" in row and "di_pos" in row:
                    return row["di_pos"] >= row["di_neg"]  # Allow short if in uptrend
            else:
                # Strong trend, only allow trend-following signals
                if signal_type == "long" and "di_neg" in row and "di_pos" in row:
                    return row["di_pos"] > row["di_neg"]  # Allow long if in uptrend
                elif signal_type == "short" and "di_neg" in row and "di_pos" in row:
                    return row["di_neg"] > row["di_pos"]  # Allow short if in downtrend
        
        # Check market regime compatibility
        if "market_regime" in row:
            compatible_regimes = signal_compatibility.get(signal_type, [])
            
            # If signal is compatible with current regime, allow it
            if row["market_regime"] in compatible_regimes:
                return True
            
            # Otherwise, require stronger confirmation
            if "signal_strength" in row:
                return row["signal_strength"] >= 8  # Require very strong signal
        
        # Check trend health if available
        if "trend_health" in row and "trend_direction" in row:
            # For long signals, trend should be healthy and up
            if signal_type == "long":
                return row["trend_health"] > 50 and row["trend_direction"] > 0
            # For short signals, trend should be healthy and down
            elif signal_type == "short":
                return row["trend_health"] > 50 and row["trend_direction"] < 0
        
        return True  # Default to passing filter