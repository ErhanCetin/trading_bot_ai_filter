"""
Context-aware signal strength calculators for the trading system.
FIXED VERSION - Updated indicator names, performance optimization, and robust error handling.
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
            "oversold": {"long": 80, "short": 30},
            "unknown": {"long": 50, "short": 50}  # Fallback for unknown regimes
        },
        "volatility_adjustment": True,
        "trend_health_adjustment": True,
        "fallback_strength": 50
    }
    
    # FIXED: Updated to match refactored indicator names
    required_indicators = []  # No strict requirements, use fallbacks
    optional_indicators = ["market_regime", "regime_strength", "volatility_regime", 
                          "volatility_percentile", "trend_health", "atr_percent"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on market context with robust error handling.
        """
        # Initialize strength series with fallback values
        fallback = self.params.get("fallback_strength", 50)
        strength = pd.Series(fallback, index=signals.index)
        
        # Get parameters
        regime_weights = self.params.get("regime_weights", self.default_params["regime_weights"])
        vol_adjust = self.params.get("volatility_adjustment", True)
        trend_adjust = self.params.get("trend_health_adjustment", True)
        
        # PERFORMANCE: Vectorized operations where possible
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        
        if len(signal_indices) == 0:
            return strength
        
        # ROBUST ERROR HANDLING: Check available columns
        has_regime = "market_regime" in df.columns
        has_regime_strength = "regime_strength" in df.columns
        has_vol_regime = "volatility_regime" in df.columns
        has_vol_percentile = "volatility_percentile" in df.columns
        has_trend_health = "trend_health" in df.columns
        has_atr_percent = "atr_percent" in df.columns
        
        try:
            # Vectorized calculation for signals
            for i in signal_indices:
                # Skip if signal is NaN
                if pd.isna(signals.loc[i]):
                    continue
                
                # Determine signal direction
                direction = "long" if signals.loc[i] > 0 else "short"
                
                # Base strength from market regime
                base_strength = fallback
                
                if has_regime and not pd.isna(df["market_regime"].loc[i]):
                    regime = str(df["market_regime"].loc[i]).lower().strip()
                    
                    if regime in regime_weights:
                        base_strength = regime_weights[regime][direction]
                    else:
                        # Unknown regime, use default
                        base_strength = regime_weights["unknown"][direction]
                
                # Adjust by regime strength if available
                if has_regime_strength and not pd.isna(df["regime_strength"].loc[i]):
                    regime_strength_val = df["regime_strength"].loc[i]
                    if 0 <= regime_strength_val <= 100:  # Validate range
                        regime_strength_norm = regime_strength_val / 100
                        base_strength = 50 + (base_strength - 50) * regime_strength_norm
                
                # VOLATILITY ADJUSTMENT - Multiple fallback options
                if vol_adjust:
                    vol_modifier = 1.0
                    
                    if has_vol_regime and not pd.isna(df["volatility_regime"].loc[i]):
                        vol_regime = str(df["volatility_regime"].loc[i]).lower()
                        
                        if vol_regime == "high":
                            vol_modifier = 0.8  # Reduce strength in high volatility
                        elif vol_regime == "low":
                            # Increase strength for trend signals in low volatility
                            if (direction == "long" and regime in ["strong_uptrend", "weak_uptrend"]) or \
                               (direction == "short" and regime in ["strong_downtrend", "weak_downtrend"]):
                                vol_modifier = 1.2
                    
                    elif has_vol_percentile and not pd.isna(df["volatility_percentile"].loc[i]):
                        vol_percentile = df["volatility_percentile"].loc[i]
                        if 0 <= vol_percentile <= 100:  # Validate range
                            # Scale from 0-100 to 0.8-1.2
                            vol_modifier = 1.2 - (vol_percentile / 100) * 0.4
                    
                    elif has_atr_percent and not pd.isna(df["atr_percent"].loc[i]):
                        # Fallback: Use ATR percentage for volatility assessment
                        atr_pct = df["atr_percent"].loc[i]
                        if atr_pct > 2.0:  # High volatility
                            vol_modifier = 0.8
                        elif atr_pct < 0.5:  # Low volatility
                            vol_modifier = 1.1
                    
                    # Apply volatility adjustment with bounds
                    base_strength = base_strength * max(0.5, min(1.5, vol_modifier))
                
                # TREND HEALTH ADJUSTMENT
                if trend_adjust and has_trend_health and not pd.isna(df["trend_health"].loc[i]):
                    trend_health_val = df["trend_health"].loc[i]
                    
                    if 0 <= trend_health_val <= 100:  # Validate range
                        trend_health_norm = trend_health_val / 100
                        
                        # Apply trend health adjustment only in trending regimes
                        if has_regime and not pd.isna(df["market_regime"].loc[i]):
                            regime = str(df["market_regime"].loc[i]).lower()
                            
                            if regime in ["strong_uptrend", "weak_uptrend", "strong_downtrend", "weak_downtrend"]:
                                # Scale from 0.6 to 1.1 based on trend health
                                trend_modifier = 0.6 + (trend_health_norm * 0.5)
                                base_strength = base_strength * trend_modifier
                
                # Ensure strength is within bounds and not NaN
                final_strength = max(0, min(100, base_strength))
                if not pd.isna(final_strength):
                    strength.loc[i] = round(final_strength)
                else:
                    strength.loc[i] = fallback
        
        except Exception as e:
            logger.error(f"Error in MarketContextStrengthCalculator: {e}")
            # Return fallback strengths for all signals
            strength.loc[signal_mask] = fallback
        
        return strength


class IndicatorConfirmationStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on indicator confirmations with updated indicator names."""
    
    name = "indicator_confirmation_strength"
    display_name = "Indicator Confirmation Strength Calculator"
    description = "Calculates signal strength based on indicator confirmations"
    category = "context"
    
    default_params = {
        "indicators": {
            "long": {
                "rsi_14": {"condition": "above", "value": 50, "weight": 1.0},
                "macd_line": {"condition": "above", "value": 0, "weight": 1.0},
                "ema_alignment": {"condition": "above", "value": 0, "weight": 1.5},  # From mtf_ema indicator
                "trend_strength": {"condition": "above", "value": 30, "weight": 1.2},
                "adx": {"condition": "above", "value": 25, "weight": 1.0}  # FIXED: Updated name
            },
            "short": {
                "rsi_14": {"condition": "below", "value": 50, "weight": 1.0},
                "macd_line": {"condition": "below", "value": 0, "weight": 1.0},
                "ema_alignment": {"condition": "below", "value": 0, "weight": 1.5},
                "trend_strength": {"condition": "above", "value": 30, "weight": 1.2},
                "adx": {"condition": "above", "value": 25, "weight": 1.0}
            }
        },
        "base_strength": 50,
        "confirmation_value": 5,
        "fallback_strength": 50
    }
    
    required_indicators = []  # Will be populated dynamically
    optional_indicators = []  # Will be populated dynamically
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize with dynamic indicator lists."""
        super().__init__(params)
        
        # Extract all indicators from parameters
        indicators = self.params.get("indicators", self.default_params["indicators"])
        all_indicators = set()
        
        for direction in ["long", "short"]:
            for indicator in indicators[direction].keys():
                all_indicators.add(indicator)
        
        self.optional_indicators = list(all_indicators)
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength with performance optimization and error handling.
        """
        # Initialize with base strength
        base_strength = self.params.get("base_strength", 50)
        fallback = self.params.get("fallback_strength", 50)
        strength = pd.Series(base_strength, index=signals.index)
        
        # Get parameters
        indicators = self.params.get("indicators", self.default_params["indicators"])
        conf_value = self.params.get("confirmation_value", 5)
        
        # PERFORMANCE: Only process signals
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        
        if len(signal_indices) == 0:
            return strength
        
        try:
            # Pre-check available indicators
            available_indicators = {
                direction: {
                    indicator: criteria 
                    for indicator, criteria in indicators[direction].items()
                    if indicator in df.columns
                }
                for direction in ["long", "short"]
            }
            
            # Vectorized calculation
            for i in signal_indices:
                if pd.isna(signals.loc[i]):
                    continue
                
                # Determine signal direction
                direction = "long" if signals.loc[i] > 0 else "short"
                
                # Start with base strength
                signal_strength = base_strength
                
                # Check available indicators for this direction
                direction_indicators = available_indicators[direction]
                
                if not direction_indicators:
                    # No indicators available, use fallback
                    strength.loc[i] = fallback
                    continue
                
                total_weight = 0
                weighted_confirmations = 0
                
                # Check each available indicator
                for indicator, criteria in direction_indicators.items():
                    try:
                        weight = criteria.get("weight", 1.0)
                        total_weight += weight
                        
                        # Get indicator value with NaN check
                        indicator_value = df[indicator].loc[i]
                        if pd.isna(indicator_value):
                            continue
                        
                        # Check confirmation condition
                        is_confirmed = False
                        
                        if criteria["condition"] == "above":
                            is_confirmed = indicator_value > criteria["value"]
                        elif criteria["condition"] == "below":
                            is_confirmed = indicator_value < criteria["value"]
                        elif criteria["condition"] == "equal":
                            is_confirmed = abs(indicator_value - criteria["value"]) < 1e-6
                        elif criteria["condition"] in ["rising", "falling"]:
                            # PERFORMANCE: Check trend with vectorized operation
                            periods = criteria.get("periods", 3)
                            if i >= periods:
                                # Get recent values
                                recent_values = df[indicator].loc[i-periods+1:i+1]
                                
                                if len(recent_values) >= 2 and not recent_values.isna().any():
                                    if criteria["condition"] == "rising":
                                        is_confirmed = recent_values.is_monotonic_increasing
                                    else:  # falling
                                        is_confirmed = recent_values.is_monotonic_decreasing
                        
                        if is_confirmed:
                            weighted_confirmations += weight
                    
                    except Exception as e:
                        logger.debug(f"Error checking indicator {indicator}: {e}")
                        continue
                
                # Calculate final strength
                if total_weight > 0:
                    confirmation_pct = weighted_confirmations / total_weight
                    signal_strength = base_strength + (confirmation_pct * (100 - base_strength))
                else:
                    signal_strength = fallback
                
                # Ensure bounds
                strength.loc[i] = round(max(0, min(100, signal_strength)))
        
        except Exception as e:
            logger.error(f"Error in IndicatorConfirmationStrengthCalculator: {e}")
            strength.loc[signal_mask] = fallback
        
        return strength


class MultiTimeframeStrengthCalculator(BaseStrengthCalculator):
    """Calculates signal strength based on multi-timeframe agreement with updated indicator names."""
    
    name = "mtf_strength"
    display_name = "Multi-Timeframe Strength Calculator"
    description = "Calculates signal strength based on multi-timeframe agreement"
    category = "context"
    
    default_params = {
        "base_strength": 50,
        "mtf_agreement_value": 10,
        "alignment_multiplier": 1.5,
        "fallback_strength": 50
    }
    
    required_indicators = ["close"]
    # FIXED: Updated indicator names to match refactored system
    optional_indicators = ["ema_alignment", "trend_alignment", "multi_timeframe_agreement"]
    
    def calculate(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate multi-timeframe strength with performance optimization.
        """
        # Initialize strength series
        base_strength = self.params.get("base_strength", 50)
        fallback = self.params.get("fallback_strength", 50)
        strength = pd.Series(base_strength, index=signals.index)
        
        # Validate required indicators
        if "close" not in df.columns:
            logger.warning("Close price not available for MTF strength calculation")
            strength.loc[signals != 0] = fallback
            return strength
        
        # Get parameters
        mtf_value = self.params.get("mtf_agreement_value", 10)
        align_mult = self.params.get("alignment_multiplier", 1.5)
        
        # PERFORMANCE: Only process signals
        signal_mask = signals != 0
        signal_indices = signal_mask[signal_mask].index
        
        if len(signal_indices) == 0:
            return strength
        
        try:
            # Pre-check available indicators
            has_ema_alignment = "ema_alignment" in df.columns
            has_trend_alignment = "trend_alignment" in df.columns
            has_mtf_agreement = "multi_timeframe_agreement" in df.columns
            
            # Find MTF EMA columns (from mtf_ema indicator)
            mtf_ema_cols = [col for col in df.columns if col.startswith("ema_") and "_" in col]
            has_mtf_emas = len(mtf_ema_cols) > 0
            
            # Vectorized calculation
            for i in signal_indices:
                if pd.isna(signals.loc[i]) or pd.isna(df["close"].loc[i]):
                    strength.loc[i] = fallback
                    continue
                
                # Determine signal direction
                is_long = signals.loc[i] > 0
                current_price = df["close"].loc[i]
                
                # Start with base strength
                signal_strength = base_strength
                
                # EMA alignment check
                if has_ema_alignment and not pd.isna(df["ema_alignment"].loc[i]):
                    ema_alignment = df["ema_alignment"].loc[i]
                    
                    if is_long and ema_alignment > 0:
                        signal_strength += ema_alignment * mtf_value
                    elif not is_long and ema_alignment < 0:
                        signal_strength += abs(ema_alignment) * mtf_value
                
                # Trend alignment check
                if has_trend_alignment and not pd.isna(df["trend_alignment"].loc[i]):
                    trend_alignment = df["trend_alignment"].loc[i]
                    
                    if (is_long and trend_alignment == 1) or (not is_long and trend_alignment == -1):
                        signal_strength *= align_mult
                
                # MTF agreement check
                if has_mtf_agreement and not pd.isna(df["multi_timeframe_agreement"].loc[i]):
                    mtf_agreement = df["multi_timeframe_agreement"].loc[i]
                    
                    if (is_long and mtf_agreement > 0) or (not is_long and mtf_agreement < 0):
                        signal_strength += mtf_value
                
                # Direct MTF EMA analysis
                if has_mtf_emas:
                    try:
                        # PERFORMANCE: Vectorized EMA comparison
                        ema_values = df.loc[i, mtf_ema_cols]
                        valid_emas = ema_values.dropna()
                        
                        if len(valid_emas) > 0:
                            above_count = (current_price > valid_emas).sum()
                            alignment_pct = above_count / len(valid_emas)
                            
                            if is_long:
                                signal_strength += alignment_pct * mtf_value * 2
                            else:
                                signal_strength += (1 - alignment_pct) * mtf_value * 2
                    
                    except Exception as e:
                        logger.debug(f"Error in MTF EMA analysis: {e}")
                
                # Ensure bounds
                strength.loc[i] = round(max(0, min(100, signal_strength)))
        
        except Exception as e:
            logger.error(f"Error in MultiTimeframeStrengthCalculator: {e}")
            strength.loc[signal_mask] = fallback
        
        return strength