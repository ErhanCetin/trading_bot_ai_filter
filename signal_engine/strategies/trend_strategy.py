"""
Trend following strategies for the trading system.
These strategies identify and follow market trends.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from signal_engine.signal_strategy_system import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Basic trend following strategy using multiple indicators."""
    
    name = "trend_following"
    display_name = "Trend Following Strategy"
    description = "Uses multiple indicators to identify and follow trends"
    category = "trend"
    
    default_params = {
        "adx_threshold": 25,
        "rsi_threshold": 50,
        "macd_threshold": 0,
        "ema_periods": [20, 50],
        "confirmation_count": 3  # Number of confirming conditions required
    }
    
    required_indicators = ["adx", "di_pos", "di_neg", "rsi_14", "macd_line", 
                           "ema_20", "ema_50", "market_regime"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate trend following signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        rsi_threshold = self.params.get("rsi_threshold", self.default_params["rsi_threshold"])
        macd_threshold = self.params.get("macd_threshold", self.default_params["macd_threshold"])
        ema_periods = self.params.get("ema_periods", self.default_params["ema_periods"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Check market regime if available
        if "market_regime" in row and row["market_regime"] is not None:
            long_conditions.append(row["market_regime"] in ["strong_uptrend", "weak_uptrend"])
            short_conditions.append(row["market_regime"] in ["strong_downtrend", "weak_downtrend"])
        
        # Check ADX and directional indicators
        if all(key in row and row[key] is not None for key in ["adx", "di_pos", "di_neg"]):
            # Strong trend condition
            trend_strength = row["adx"] > adx_threshold
            
            # Trend direction conditions
            long_conditions.append(trend_strength and row["di_pos"] > row["di_neg"])
            short_conditions.append(trend_strength and row["di_neg"] > row["di_pos"])
        
        # Check RSI
        if "rsi_14" in row and row["rsi_14"] is not None:
            # Above/below center line
            long_conditions.append(row["rsi_14"] > rsi_threshold)
            short_conditions.append(row["rsi_14"] < rsi_threshold)
        
        # Check MACD
        if "macd_line" in row and row["macd_line"] is not None:
            # Above/below zero line
            long_conditions.append(row["macd_line"] > macd_threshold)
            short_conditions.append(row["macd_line"] < macd_threshold)
            
            # MACD trend
            if i > 0 and "macd_line" in df.columns:
                prev_macd = df["macd_line"].iloc[i-1]
                if prev_macd is not None and row["macd_line"] is not None:
                    long_conditions.append(row["macd_line"] > prev_macd)  # MACD rising
                    short_conditions.append(row["macd_line"] < prev_macd)  # MACD falling
        
        # Check EMAs
        # Ensure the EMAs are available
        ema_available = True
        for period in ema_periods:
            ema_col = f"ema_{period}"
            if ema_col not in row or row[ema_col] is None:
                ema_available = False
                break
        
        if ema_available and len(ema_periods) >= 2 and "close" in row and row["close"] is not None:
            # Price above/below EMAs
            for period in ema_periods:
                ema_col = f"ema_{period}"
                long_conditions.append(row["close"] > row[ema_col])
                short_conditions.append(row["close"] < row[ema_col])
            
            # EMA alignment (faster above/below slower)
            for j in range(len(ema_periods) - 1):
                fast_ema = f"ema_{ema_periods[j]}"
                slow_ema = f"ema_{ema_periods[j+1]}"
                long_conditions.append(row[fast_ema] > row[slow_ema])
                short_conditions.append(row[fast_ema] < row[slow_ema])
        
        # Additional condition: recent price direction
        if i > 0 and "close" in row and row["close"] is not None:
            prev_close = df["close"].iloc[i-1]
            if prev_close is not None:
                long_conditions.append(row["close"] > prev_close)
                short_conditions.append(row["close"] < prev_close)
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class MultiTimeframeTrendStrategy(BaseStrategy):
    """Trend strategy that analyzes multiple timeframes."""
    
    name = "mtf_trend"
    display_name = "Multi-Timeframe Trend Strategy"
    description = "Identifies trends that are aligned across multiple timeframes"
    category = "trend"
    
    default_params = {
        "alignment_required": 0.8  # Percentage of timeframes that must align
    }
    
    required_indicators = ["close"]
    optional_indicators = ["mtf_ema_alignment", "trend_alignment", "multi_timeframe_agreement"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate multi-timeframe trend signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        alignment_required = self.params.get("alignment_required", self.default_params["alignment_required"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Check if we have MTF EMA alignment indicator
        if "mtf_ema_alignment" in row:
            # MTF EMA alignment (values between -1 and 1)
            long_conditions.append(row["mtf_ema_alignment"] > alignment_required - 1)  # e.g., > 0.2 if required = 0.8
            short_conditions.append(row["mtf_ema_alignment"] < 1 - alignment_required)  # e.g., < -0.2 if required = 0.8
        
        # Check if we have trend alignment indicator
        if "trend_alignment" in row:
            # Trend alignment (values: 1 for uptrend, -1 for downtrend)
            long_conditions.append(row["trend_alignment"] == 1)
            short_conditions.append(row["trend_alignment"] == -1)
        
        # Check if we have multi-timeframe agreement indicator
        if "multi_timeframe_agreement" in row:
            # Multi-timeframe agreement (values: 1 for agreement, 0 for neutral, -1 for disagreement)
            long_conditions.append(row["multi_timeframe_agreement"] > 0)
            short_conditions.append(row["multi_timeframe_agreement"] < 0)
        
        # Check MTF EMAs directly if specific columns exist
        mtf_ema_cols = [col for col in row.index if col.startswith("ema_") and "_" in col]
        if mtf_ema_cols and "close" in row:
            # Count how many EMAs the price is above/below
            above_count = 0
            for col in mtf_ema_cols:
                if row["close"] > row[col]:
                    above_count += 1
            
            # Calculate alignment percentage
            alignment_pct = above_count / len(mtf_ema_cols)
            
            # Add conditions based on alignment percentage
            long_conditions.append(alignment_pct >= alignment_required)
            short_conditions.append(alignment_pct <= (1 - alignment_required))
            
            # Check if EMAs are aligned in the correct order
            if len(mtf_ema_cols) >= 2:
                # Extract timeframe multipliers from column names
                # Assumes format like "ema_period_timeframe"
                ema_multipliers = []
                for col in mtf_ema_cols:
                    parts = col.split("_")
                    if len(parts) >= 3 and parts[-1].endswith("x"):
                        try:
                            multiplier = float(parts[-1][:-1])  # Remove the 'x' at the end
                            ema_multipliers.append((col, multiplier))
                        except ValueError:
                            continue
                
                # Sort by multiplier (ascending)
                ema_multipliers.sort(key=lambda x: x[1])
                
                # Check if EMAs are aligned in ascending order (bullish)
                bullish_alignment = True
                for j in range(len(ema_multipliers) - 1):
                    if row[ema_multipliers[j][0]] < row[ema_multipliers[j+1][0]]:
                        bullish_alignment = False
                        break
                
                # Check if EMAs are aligned in descending order (bearish)
                bearish_alignment = True
                for j in range(len(ema_multipliers) - 1):
                    if row[ema_multipliers[j][0]] > row[ema_multipliers[j+1][0]]:
                        bearish_alignment = False
                        break
                
                long_conditions.append(bullish_alignment)
                short_conditions.append(bearish_alignment)
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class AdaptiveTrendStrategy(BaseStrategy):
    """Trend strategy that adapts to current market conditions."""
    
    name = "adaptive_trend"
    display_name = "Adaptive Trend Strategy"
    description = "Adapts trend parameters based on current market conditions"
    category = "trend"
    
    default_params = {
        "adx_max_threshold": 40,  # Maximum ADX threshold to use
        "adx_min_threshold": 15   # Minimum ADX threshold to use
    }
    
    required_indicators = ["adx", "di_pos", "di_neg", "atr_percent", "rsi_14"]
    optional_indicators = ["market_regime", "volatility_regime"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate adaptive trend signal conditions.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        adx_max = self.params.get("adx_max_threshold", self.default_params["adx_max_threshold"])
        adx_min = self.params.get("adx_min_threshold", self.default_params["adx_min_threshold"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Determine current volatility level
        volatility_multiplier = 1.0
        
        # Güvenli şekilde volatility_regime'i kontrol et
        if "volatility_regime" in row and row["volatility_regime"] is not None:
            if row["volatility_regime"] == "high":
                volatility_multiplier = 1.5  # Higher threshold in high volatility
            elif row["volatility_regime"] == "low":
                volatility_multiplier = 0.7  # Lower threshold in low volatility
        # Güvenli şekilde atr_percent'i kontrol et
        elif "atr_percent" in row and row["atr_percent"] is not None:
            # Use ATR as percentage of price to determine volatility
            # Assuming average ATR percent is around 0.5-1%
            if row["atr_percent"] > 2.0:  # High volatility
                volatility_multiplier = 1.5
            elif row["atr_percent"] < 0.3:  # Low volatility
                volatility_multiplier = 0.7
        
        # Calculate adaptive ADX threshold
        adaptive_adx = max(adx_min, min(adx_max, adx_min * volatility_multiplier))
        
        # Check trend strength and direction - Tüm değerlerin None olup olmadığını kontrol et
        if all(col in row and row[col] is not None for col in ["adx", "di_pos", "di_neg"]):
            trend_strength = row["adx"] > adaptive_adx
            
            long_conditions.append(trend_strength and row["di_pos"] > row["di_neg"])
            short_conditions.append(trend_strength and row["di_neg"] > row["di_pos"])
            
            # Additional condition: trend strength vs. historical
            if i >= 50 and trend_strength:
                # Verilerin None olmadığından emin ol
                adx_history = df["adx"].iloc[i-50:i].dropna()
                if not adx_history.empty:
                    adx_percentile = (adx_history < row["adx"]).mean() * 100
                    
                    # Strong trend condition (ADX in top 20%)
                    strong_trend = adx_percentile > 80
                    long_conditions.append(strong_trend and row["di_pos"] > row["di_neg"])
                    short_conditions.append(strong_trend and row["di_neg"] > row["di_pos"])
        
        # Check RSI with adaptive thresholds - Güvenli şekilde RSI'ı kontrol et
        if "rsi_14" in row and row["rsi_14"] is not None:
            # Adjust RSI thresholds based on volatility
            rsi_middle = 50
            if volatility_multiplier > 1.2:  # High volatility
                long_conditions.append(row["rsi_14"] > rsi_middle + 5)
                short_conditions.append(row["rsi_14"] < rsi_middle - 5)
            elif volatility_multiplier < 0.8:  # Low volatility
                long_conditions.append(row["rsi_14"] > rsi_middle - 5)
                short_conditions.append(row["rsi_14"] < rsi_middle + 5)
            else:  # Normal volatility
                long_conditions.append(row["rsi_14"] > rsi_middle)
                short_conditions.append(row["rsi_14"] < rsi_middle)
        
        # Check market regime if available - Güvenli şekilde market_regime'i kontrol et
        if "market_regime" in row and row["market_regime"] is not None:
            strong_trend_regimes = ["strong_uptrend", "strong_downtrend"]
            weak_trend_regimes = ["weak_uptrend", "weak_downtrend"]
            
            # More aggressive in strong trends
            if row["market_regime"] in strong_trend_regimes:
                long_conditions.append(row["market_regime"] == "strong_uptrend")
                short_conditions.append(row["market_regime"] == "strong_downtrend")
            # Less aggressive in weak trends
            elif row["market_regime"] in weak_trend_regimes:
                # Only add trend conditions if we have at least 3 confirming conditions
                if all(col in row and row[col] is not None for col in ["adx", "di_pos", "di_neg", "rsi_14"]):
                    trend_confirms = 0
                    # Count how many indicators confirm trend
                    if row["adx"] > adaptive_adx:
                        trend_confirms += 1
                    if (row["di_pos"] > row["di_neg"] and row["market_regime"] == "weak_uptrend") or \
                    (row["di_neg"] > row["di_pos"] and row["market_regime"] == "weak_downtrend"):
                        trend_confirms += 1
                    if (row["rsi_14"] > 50 and row["market_regime"] == "weak_uptrend") or \
                    (row["rsi_14"] < 50 and row["market_regime"] == "weak_downtrend"):
                        trend_confirms += 1
                    
                    long_conditions.append(row["market_regime"] == "weak_uptrend" and trend_confirms >= 2)
                    short_conditions.append(row["market_regime"] == "weak_downtrend" and trend_confirms >= 2)
            # Avoid trend trading in ranging markets
            else:
                long_conditions.append(False)
                short_conditions.append(False)
        
        # None değerlerini filtrele
        long_conditions = [c for c in long_conditions if c is not None]
        short_conditions = [c for c in short_conditions if c is not None]
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }