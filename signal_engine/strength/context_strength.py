"""
Context-aware signal strength calculators for the trading system.
These calculators estimate signal strength based on market context and indicator confirmation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from signal_engine.signal_strength_system import BaseStrengthCalculator

logger = logging.getLogger(__name__)


class MarketContextStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on market context and regime."""
    
    name = "market_context_strength"
    display_name = "Market Context Strength Calculator"
    description = "Calculates signal strength based on market conditions"
    category = "context"
    
    default_params = {
        "regime_weights": {
            "strong_uptrend": {"long": 90, "short": 20},
            "weak_uptrend": {"long": 70, "short": 40},
            "ranging": {"long": 50, "short": 50},
            "weak_downtrend": {"long": 40, "short": 70},
            "strong_downtrend": {"long": 20, "short": 90},
            "volatile": {"long": 40, "short": 40},
            "overbought": {"long": 30, "short": 80},
            "oversold": {"long": 80, "short": 30}
        },
        "volatility_adjustment": True,  # Adjust strength based on volatility
        "trend_health_adjustment": True  # Adjust strength based on trend health
    }
    
    required_indicators = ["market_regime"]
    optional_indicators = ["regime_strength", "volatility_regime", "volatility_percentile", "trend_health"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on market context.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values (0-100)
        """
        # Initialize strength series with zeros
        strength = pd.Series(0, index=signals.index)
        
        # Validate dataframe
        if not self.validate_dataframe(df):
            return strength
        
        # Get parameters
        regime_weights = self.params.get("regime_weights", self.default_params["regime_weights"])
        vol_adjust = self.params.get("volatility_adjustment", self.default_params["volatility_adjustment"])
        trend_adjust = self.params.get("trend_health_adjustment", self.default_params["trend_health_adjustment"])
        
        # Calculate strength based on market context
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Determine signal direction
            direction = "long" if signals.iloc[i] > 0 else "short"
            
            # Base strength from market regime
            regime = df["market_regime"].iloc[i]
            
            # Get base strength value from regime weight
            if regime in regime_weights:
                base_strength = regime_weights[regime][direction]
            else:
                # Default to neutral
                base_strength = 50
            
            # Adjust by regime strength if available
            if "regime_strength" in df.columns:
                regime_strength = df["regime_strength"].iloc[i] / 100  # Normalize to 0-1
                # Stronger regime = more confidence in the weightings
                base_strength = 50 + (base_strength - 50) * regime_strength
            
            # Volatility adjustment
            if vol_adjust:
                vol_modifier = 1.0  # Default modifier
                
                if "volatility_regime" in df.columns:
                    vol_regime = df["volatility_regime"].iloc[i]
                    
                    # Adjust based on volatility regime
                    if vol_regime == "high":
                        # Reduce strength in high volatility
                        vol_modifier = 0.8
                    elif vol_regime == "low":
                        # Increase strength in low volatility for trend signals
                        if (direction == "long" and regime in ["strong_uptrend", "weak_uptrend"]) or \
                           (direction == "short" and regime in ["strong_downtrend", "weak_downtrend"]):
                            vol_modifier = 1.2
                
                elif "volatility_percentile" in df.columns:
                    # Scale from 0-100 to 0.8-1.2
                    vol_percentile = df["volatility_percentile"].iloc[i]
                    vol_modifier = 1.2 - (vol_percentile / 100) * 0.4
                
                # Apply volatility adjustment
                base_strength = base_strength * vol_modifier
            
            # Trend health adjustment
            if trend_adjust and "trend_health" in df.columns:
                trend_health = df["trend_health"].iloc[i] / 100  # Normalize to 0-1
                
                # In trending regimes, trend health is important
                if regime in ["strong_uptrend", "weak_uptrend", "strong_downtrend", "weak_downtrend"]:
                    # Adjust strength by trend health
                    # If trend is healthy, maintain or increase strength
                    # If trend is unhealthy, reduce strength
                    trend_modifier = 0.5 + (trend_health * 0.5)  # Scale from 0.5 to 1.0
                    base_strength = base_strength * trend_modifier
            
            # Ensure strength is within 0-100 range
            strength.iloc[i] = round(max(0, min(100, base_strength)))
        
        return strength


class IndicatorConfirmationStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on indicator confirmations."""
    
    name = "indicator_confirmation_strength"
    display_name = "Indicator Confirmation Strength Calculator"
    description = "Calculates signal strength based on indicator confirmations"
    category = "context"
    
    default_params = {
        "indicators": {
            "long": {
                "rsi_14": {"condition": "above", "value": 50, "weight": 1.0},
                "macd_line": {"condition": "above", "value": 0, "weight": 1.0},
                "ema_alignment": {"condition": "above", "value": 0, "weight": 1.5},
                "trend_strength": {"condition": "above", "value": 30, "weight": 1.2},
                "obv": {"condition": "rising", "periods": 3, "weight": 1.0}
            },
            "short": {
                "rsi_14": {"condition": "below", "value": 50, "weight": 1.0},
                "macd_line": {"condition": "below", "value": 0, "weight": 1.0},
                "ema_alignment": {"condition": "below", "value": 0, "weight": 1.5},
                "trend_strength": {"condition": "above", "value": 30, "weight": 1.2},
                "obv": {"condition": "falling", "periods": 3, "weight": 1.0}
            }
        },
        "base_strength": 50,  # Base strength value
        "confirmation_value": 5  # Strength points per confirmation
    }
    
    required_indicators = []  # Will be populated dynamically
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with dynamic required indicators."""
        super().__init__(params)
        
        # Extract required indicators from parameters
        indicators = self.params.get("indicators", self.default_params["indicators"])
        required = set()
        
        for direction in ["long", "short"]:
            for indicator in indicators[direction].keys():
                required.add(indicator)
        
        self.required_indicators = list(required)
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on indicator confirmations.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values (0-100)
        """
        # Initialize strength series
        base_strength = self.params.get("base_strength", self.default_params["base_strength"])
        strength = pd.Series(0, index=signals.index)
        
        # Get parameters
        indicators = self.params.get("indicators", self.default_params["indicators"])
        conf_value = self.params.get("confirmation_value", self.default_params["confirmation_value"])
        
        # Calculate indicator-based strength
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Determine signal direction
            direction = "long" if signals.iloc[i] > 0 else "short"
            
            # Start with base strength
            signal_strength = base_strength
            
            # Check each indicator
            total_weight = 0
            weighted_confirmations = 0
            
            for indicator, criteria in indicators[direction].items():
                if indicator not in df.columns:
                    continue
                    
                weight = criteria.get("weight", 1.0)
                total_weight += weight
                
                # Check if indicator confirms the signal
                if criteria["condition"] == "above":
                    if df[indicator].iloc[i] > criteria["value"]:
                        weighted_confirmations += weight
                        
                elif criteria["condition"] == "below":
                    if df[indicator].iloc[i] < criteria["value"]:
                        weighted_confirmations += weight
                        
                elif criteria["condition"] == "equal":
                    if df[indicator].iloc[i] == criteria["value"]:
                        weighted_confirmations += weight
                        
                elif criteria["condition"] in ["rising", "falling"]:
                    # Check trend over specified periods
                    periods = criteria.get("periods", 3)
                    
                    if i >= periods:
                        # Check if indicator is consistently rising/falling
                        is_rising = True
                        is_falling = True
                        
                        for j in range(1, periods):
                            if df[indicator].iloc[i-j] >= df[indicator].iloc[i-j+1]:
                                is_rising = False
                            if df[indicator].iloc[i-j] <= df[indicator].iloc[i-j+1]:
                                is_falling = False
                        
                        if criteria["condition"] == "rising" and is_rising:
                            weighted_confirmations += weight
                        elif criteria["condition"] == "falling" and is_falling:
                            weighted_confirmations += weight
            
            # Calculate confirmation percentage
            if total_weight > 0:
                confirmation_pct = weighted_confirmations / total_weight
                
                # Adjust strength based on confirmations
                signal_strength = base_strength + (confirmation_pct * (100 - base_strength))
            
            # Ensure strength is within 0-100 range
            strength.iloc[i] = round(max(0, min(100, signal_strength)))
        
        return strength


class MultiTimeframeStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on multi-timeframe agreement."""
    
    name = "mtf_strength"
    display_name = "Multi-Timeframe Strength Calculator"
    description = "Calculates signal strength based on multi-timeframe agreement"
    category = "context"
    
    default_params = {
        "base_strength": 50,  # Base strength value
        "mtf_agreement_value": 10,  # Strength points for MTF agreement
        "alignment_multiplier": 1.5  # Multiplier for full alignment
    }
    
    required_indicators = ["close"]
    optional_indicators = ["mtf_ema_alignment", "trend_alignment", "multi_timeframe_agreement", 
                          "ema_20_1x", "ema_20_4x", "ema_20_12x", "ema_20_24x"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on multi-timeframe agreement.
        
        Args:
            df: DataFrame with indicator data
            signals: Series with signal values
            
        Returns:
            Series with signal strength values (0-100)
        """
        # Initialize strength series
        base_strength = self.params.get("base_strength", self.default_params["base_strength"])
        strength = pd.Series(0, index=signals.index)
        
        # Get parameters
        mtf_value = self.params.get("mtf_agreement_value", self.default_params["mtf_agreement_value"])
        align_mult = self.params.get("alignment_multiplier", self.default_params["alignment_multiplier"])
        
        # Calculate multi-timeframe strength
        for i in range(len(df)):
            # Skip if no signal
            if signals.iloc[i] == 0:
                continue
            
            # Determine signal direction
            is_long = signals.iloc[i] > 0
            
            # Start with base strength
            signal_strength = base_strength
            
            # Check MTF alignment indicators
            if "mtf_ema_alignment" in df.columns:
                mtf_alignment = df["mtf_ema_alignment"].iloc[i]
                
                if is_long and mtf_alignment > 0:
                    # Positive alignment is good for long signals
                    # Scale from 0 to 1 to 0 to mtf_value
                    signal_strength += mtf_alignment * mtf_value
                elif not is_long and mtf_alignment < 0:
                    # Negative alignment is good for short signals
                    # Scale from -1 to 0 to mtf_value to 0
                    signal_strength += abs(mtf_alignment) * mtf_value
            
            # Check trend alignment
            if "trend_alignment" in df.columns:
                trend_alignment = df["trend_alignment"].iloc[i]
                
                if (is_long and trend_alignment == 1) or (not is_long and trend_alignment == -1):
                    # Perfect alignment, apply multiplier
                    signal_strength = signal_strength * align_mult
            
            # Check MTF agreement
            if "multi_timeframe_agreement" in df.columns:
                mtf_agreement = df["multi_timeframe_agreement"].iloc[i]
                
                if (is_long and mtf_agreement > 0) or (not is_long and mtf_agreement < 0):
                    # Agreement across timeframes, boost strength
                    signal_strength += mtf_value
            
            # Check direct MTF EMAs
            mtf_ema_cols = [col for col in df.columns if col.startswith("ema_") and "_" in col]
            
            if mtf_ema_cols and "close" in df.columns:
                # Count how many EMAs the price is above/below
                above_count = 0
                total_emas = len(mtf_ema_cols)
                
                for col in mtf_ema_cols:
                    if df["close"].iloc[i] > df[col].iloc[i]:
                        above_count += 1
                
                # Calculate alignment percentage
                alignment_pct = above_count / total_emas
                
                if is_long:
                    # Long signals stronger with more EMAs below price
                    signal_strength += alignment_pct * mtf_value * 2
                else:
                    # Short signals stronger with more EMAs above price
                    signal_strength += (1 - alignment_pct) * mtf_value * 2
            
            # Ensure strength is within 0-100 range
            strength.iloc[i] = round(max(0, min(100, signal_strength)))
        
        return strength