
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

from signal_engine.signal_filter_system import BaseFilter


class MarketRegimeFilter(BaseFilter):
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
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply market regime filter to signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Validate dataframe
        if not self.validate_dataframe(df):
            return signals
        
        # Get regime-signal mapping
        regime_map = self.params.get("regime_signal_map", self.default_params["regime_signal_map"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Apply filtering
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Get current regime
            regime = df["market_regime"].iloc[i] if "market_regime" in df.columns else "unknown"
            
            # Determine signal type
            signal_type = "long" if current_signal > 0 else "short"
            
            # Check if signal type is allowed in this regime
            if not regime_map.get(regime, {}).get(signal_type, True):
                filtered_signals.iloc[i] = 0
        
        return filtered_signals


class VolatilityRegimeFilter(BaseFilter):
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
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply volatility regime filter to signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Validate dataframe
        if not self.validate_dataframe(df):
            return signals
        
        # Get parameters
        high_vol = self.params.get("high_volatility_filter", self.default_params["high_volatility_filter"])
        low_vol = self.params.get("low_volatility_filter", self.default_params["low_volatility_filter"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Apply filtering
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Get current volatility regime or calculate from ATR
            volatility_regime = "normal"
            if "volatility_regime" in df.columns:
                volatility_regime = df["volatility_regime"].iloc[i]
            elif "atr_percent" in df.columns:
                avg_atr = 1.0  # Assume 1% ATR is average
                atr_percent = df["atr_percent"].iloc[i]
                if atr_percent > avg_atr * high_vol["atr_threshold"]:
                    volatility_regime = "high"
                elif atr_percent < avg_atr * low_vol["atr_threshold"]:
                    volatility_regime = "low"
            
            # Get signal strength
            signal_strength = df.get("signal_strength", pd.Series(5)).iloc[i]
            
            # Apply filter based on regime
            if volatility_regime == "high":
                # In high volatility, require stronger signals
                if signal_strength < high_vol["min_strength"]:
                    filtered_signals.iloc[i] = 0
            elif volatility_regime == "low":
                # In low volatility, allow weaker signals
                if signal_strength < low_vol["min_strength"]:
                    filtered_signals.iloc[i] = 0
        
        return filtered_signals


class TrendStrengthFilter(BaseFilter):
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
    
    def apply(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply trend strength filter to signals.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Filtered signals series
        """
        # Validate dataframe
        if not self.validate_dataframe(df):
            return signals
        
        # Get parameters
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        signal_compatibility = self.params.get("signal_compatibility", self.default_params["signal_compatibility"])
        
        # Create a copy of the signals to modify
        filtered_signals = signals.copy()
        
        # Apply filtering
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            # Skip if no signal
            if current_signal == 0:
                continue
            
            # Determine signal type
            signal_type = "long" if current_signal > 0 else "short"
            
            # Check trend strength via ADX
            if "adx" in df.columns:
                adx = df["adx"].iloc[i]
                
                # Weak trend, only allow counter-trend signals
                if adx < adx_threshold:
                    # In weak trend, allow counter-trend signals
                    if signal_type == "long" and "di_neg" in df.columns and "di_pos" in df.columns:
                        if not (df["di_neg"].iloc[i] >= df["di_pos"].iloc[i]):  # Don't allow long if not in downtrend
                            filtered_signals.iloc[i] = 0
                    elif signal_type == "short" and "di_neg" in df.columns and "di_pos" in df.columns:
                        if not (df["di_pos"].iloc[i] >= df["di_neg"].iloc[i]):  # Don't allow short if not in uptrend
                            filtered_signals.iloc[i] = 0
                else:
                    # Strong trend, only allow trend-following signals
                    if signal_type == "long" and "di_neg" in df.columns and "di_pos" in df.columns:
                        if not (df["di_pos"].iloc[i] > df["di_neg"].iloc[i]):  # Don't allow long if not in uptrend
                            filtered_signals.iloc[i] = 0
                    elif signal_type == "short" and "di_neg" in df.columns and "di_pos" in df.columns:
                        if not (df["di_neg"].iloc[i] > df["di_pos"].iloc[i]):  # Don't allow short if not in downtrend
                            filtered_signals.iloc[i] = 0
            
            # Check market regime compatibility
            if "market_regime" in df.columns:
                regime = df["market_regime"].iloc[i]
                compatible_regimes = signal_compatibility.get(signal_type, [])
                
                # If signal is not compatible with current regime, check signal strength
                if regime not in compatible_regimes:
                    # Require stronger confirmation
                    if "signal_strength" in df.columns:
                        if df["signal_strength"].iloc[i] < 8:  # Require very strong signal
                            filtered_signals.iloc[i] = 0
            
            # Check trend health if available
            if "trend_health" in df.columns and "trend_direction" in df.columns:
                trend_health = df["trend_health"].iloc[i]
                trend_direction = df["trend_direction"].iloc[i]
                
                # For long signals, trend should be healthy and up
                if signal_type == "long":
                    if not (trend_health > 50 and trend_direction > 0):
                        filtered_signals.iloc[i] = 0
                # For short signals, trend should be healthy and down
                elif signal_type == "short":
                    if not (trend_health > 50 and trend_direction < 0):
                        filtered_signals.iloc[i] = 0
        
        return filtered_signals