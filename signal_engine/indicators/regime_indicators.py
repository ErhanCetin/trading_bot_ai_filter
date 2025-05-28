"""
Market regime indicators for the trading system.
These indicators identify market conditions like trend, volatility regime, etc.
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, List, Optional
from enum import Enum

from signal_engine.signal_indicator_plugin_system import BaseIndicator


class MarketRegime(Enum):
    """Enum representing different market regimes."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    VOLATILE = "volatile"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    UNKNOWN = "unknown"


class MarketRegimeIndicator(BaseIndicator):
    """Identifies market regime (trend, range, volatility, etc.)."""
    
    name = "market_regime"
    display_name = "Market Regime"
    description = "Identifies current market regime (trend, range, volatility, etc.)"
    category = "regime"
    
    # SMART DEPENDENCIES - Column names that will be auto-resolved to indicators
    dependencies = ["adx", "di_pos", "di_neg", "bollinger_width", "rsi_14"]
    
    default_params = {
        "lookback_window": 50,  # Window to analyze for regime
        "adx_threshold": 25,    # Threshold for trend strength
        "bb_width_threshold": 0.05,  # BB width threshold for volatility
        "rsi_overbought": 70,   # RSI overbought threshold
        "rsi_oversold": 30,     # RSI oversold threshold
        "range_threshold": 0.03  # Price range threshold for ranging market
    }
    
    requires_columns = ["close", "high", "low"]
    output_columns = ["market_regime", "regime_duration", "regime_strength"]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market regime and add to dataframe.

        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with market regime columns added
        """
        result_df = df.copy()

        # Smart validation: Check that required columns exist
        required_columns = ["adx", "di_pos", "di_neg", "bollinger_width", "rsi_14"]
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"MarketRegimeIndicator requires columns: {missing_columns}. "
                f"These should be auto-calculated by dependency resolution. "
                f"Make sure you're using IndicatorManager.calculate_indicators()"
            )

        # Get parameters
        lookback_window = self.params.get("lookback_window", self.default_params["lookback_window"])
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        bb_width_threshold = self.params.get("bb_width_threshold", self.default_params["bb_width_threshold"])
        rsi_overbought = self.params.get("rsi_overbought", self.default_params["rsi_overbought"])
        rsi_oversold = self.params.get("rsi_oversold", self.default_params["rsi_oversold"])
        range_threshold = self.params.get("range_threshold", self.default_params["range_threshold"])

        # Initialize regime columns
        result_df["market_regime"] = None
        result_df["regime_duration"] = 0
        result_df["regime_strength"] = 0

        # Need at least lookback_window data points
        if len(result_df) < lookback_window:
            return result_df

        # Import logging and MarketRegime enum
        import logging
        logger = logging.getLogger(__name__)
        
        # Import MarketRegime enum (assuming it's defined in this file)
        from enum import Enum
        
        class MarketRegime(Enum):
            """Enum representing different market regimes."""
            STRONG_UPTREND = "strong_uptrend"
            WEAK_UPTREND = "weak_uptrend"
            STRONG_DOWNTREND = "strong_downtrend"
            WEAK_DOWNTREND = "weak_downtrend"
            RANGING = "ranging"
            VOLATILE = "volatile"
            OVERBOUGHT = "overbought"
            OVERSOLD = "oversold"
            UNKNOWN = "unknown"

        # Identify regime for each row
        for i in range(lookback_window, len(result_df)):
            window = result_df.iloc[i-lookback_window:i+1]
            
            # Calculate price range as percentage
            price_range = (window["high"].max() - window["low"].min()) / window["close"].iloc[-1]
            
            # Get current indicators - directly from dataframe (already calculated!)
            adx = window["adx"].iloc[-1]
            di_pos = window["di_pos"].iloc[-1]
            di_neg = window["di_neg"].iloc[-1]
            bb_width = window["bollinger_width"].iloc[-1]
            rsi = window["rsi_14"].iloc[-1]
            
            # Validate indicator values
            valid_indicators = (
                adx is not None and not pd.isna(adx) and
                di_pos is not None and not pd.isna(di_pos) and
                di_neg is not None and not pd.isna(di_neg) and
                bb_width is not None and not pd.isna(bb_width) and
                rsi is not None and not pd.isna(rsi)
            )
            
            if not valid_indicators:
                missing_values = []
                if adx is None or pd.isna(adx):
                    missing_values.append("ADX")
                if di_pos is None or pd.isna(di_pos):
                    missing_values.append("DI+")
                if di_neg is None or pd.isna(di_neg):
                    missing_values.append("DI-")
                if bb_width is None or pd.isna(bb_width):
                    missing_values.append("BB Width")
                if rsi is None or pd.isna(rsi):
                    missing_values.append("RSI")
                
                logger.warning(f"Missing/invalid indicators at index {i}: {missing_values}")
                continue
            
            # Determine trend direction
            is_uptrend = di_pos > di_neg
            
            # Determine if price is in a narrow range
            is_ranging = price_range < range_threshold
            
            # Initialize regime variables
            regime = None
            regime_strength = 0
            
            # Market regime identification logic
            if adx > adx_threshold:
                # We have a trend
                if is_uptrend:
                    if adx > adx_threshold * 1.5:
                        regime = MarketRegime.STRONG_UPTREND.value
                        regime_strength = min(100, int(adx))
                    else:
                        regime = MarketRegime.WEAK_UPTREND.value
                        regime_strength = min(100, int(adx))
                else:
                    if adx > adx_threshold * 1.5:
                        regime = MarketRegime.STRONG_DOWNTREND.value
                        regime_strength = min(100, int(adx))
                    else:
                        regime = MarketRegime.WEAK_DOWNTREND.value
                        regime_strength = min(100, int(adx))
            
            elif is_ranging:
                # Ranging market
                regime = MarketRegime.RANGING.value
                regime_strength = min(100, int((1 - price_range / range_threshold) * 100))
            
            elif bb_width > bb_width_threshold:
                # Volatile market
                regime = MarketRegime.VOLATILE.value
                regime_strength = min(100, int((bb_width / bb_width_threshold) * 100))
            
            elif rsi > rsi_overbought:
                # Overbought
                regime = MarketRegime.OVERBOUGHT.value
                regime_strength = min(100, int((rsi - rsi_overbought) / (100 - rsi_overbought) * 100))
            
            elif rsi < rsi_oversold:
                # Oversold
                regime = MarketRegime.OVERSOLD.value
                regime_strength = min(100, int((rsi_oversold - rsi) / rsi_oversold * 100))
            
            # If no regime condition is met, continue to next iteration
            if regime is None:
                logger.debug(f"No market regime condition met at index {i}")
                continue
            
            # Assign regime values
            result_df.loc[result_df.index[i], "market_regime"] = regime
            result_df.loc[result_df.index[i], "regime_strength"] = regime_strength
            
            # Calculate regime duration
            if i > 0 and result_df["market_regime"].iloc[i] == result_df["market_regime"].iloc[i-1]:
                result_df.loc[result_df.index[i], "regime_duration"] = result_df["regime_duration"].iloc[i-1] + 1
            else:
                result_df.loc[result_df.index[i], "regime_duration"] = 1

        return result_df


# regime_indicators.py'de VolatilityRegimeIndicator güncellenmesi

class VolatilityRegimeIndicator(BaseIndicator):
    """Identifies volatility regime and characteristics."""
    
    name = "volatility_regime"
    display_name = "Volatility Regime"
    description = "Identifies volatility regime and characteristics"
    category = "regime"
    
    # SMART DEPENDENCIES - ATR columns that will be auto-resolved
    dependencies = ["atr_14", "atr_50"]  # Will auto-resolve to ATR indicator
    
    default_params = {
        "lookback_window": 50,
        "atr_periods": [14, 50],
        "volatility_percentile": 75  # Percentile threshold for high volatility
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = ["volatility_regime", "volatility_percentile", "volatility_ratio", "volatility_trend"]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify volatility regime and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volatility regime columns added
        """
        result_df = df.copy()
        
        # Get parameters
        lookback_window = self.params.get("lookback_window", self.default_params["lookback_window"])
        atr_periods = self.params.get("atr_periods", self.default_params["atr_periods"])
        volatility_percentile = self.params.get("volatility_percentile", self.default_params["volatility_percentile"])
        
        # Smart validation: Check ATR columns exist (should be auto-calculated)
        required_atr_columns = [f"atr_{period}" for period in atr_periods]
        missing_columns = [col for col in required_atr_columns if col not in result_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"VolatilityRegimeIndicator requires ATR columns: {missing_columns}. "
                f"These should be auto-calculated by dependency resolution. "
                f"Make sure ATRIndicator is registered and IndicatorManager is used."
            )
        
        # Calculate ATR percentages (normalize by price)
        for period in atr_periods:
            atr_col = f"atr_{period}"
            atr_pct_col = f"{atr_col}_pct"
            
            # ATR as percentage of price (already calculated ATR values!)
            result_df[atr_pct_col] = result_df[atr_col] / result_df["close"] * 100
        
        # Use shortest ATR period for main volatility measure
        short_atr_col = f"atr_{atr_periods[0]}_pct"
        long_atr_col = f"atr_{atr_periods[-1]}_pct"
        
        # Initialize volatility regime columns
        result_df["volatility_regime"] = "normal"
        result_df["volatility_percentile"] = 0.0
        result_df["volatility_ratio"] = result_df[short_atr_col] / result_df[long_atr_col]
        result_df["volatility_trend"] = 0  # 1 for increasing, -1 for decreasing
        
        # Need at least lookback_window data points
        if len(result_df) < lookback_window:
            return result_df
        
        # Calculate volatility regime for each row
        for i in range(lookback_window, len(result_df)):
            window = result_df.iloc[i-lookback_window:i+1]
            
            # Calculate volatility percentile within window
            current_vol = window[short_atr_col].iloc[-1]
            
            # Handle NaN values in volatility calculation
            valid_vol_data = window[short_atr_col].dropna()
            if len(valid_vol_data) > 0 and not pd.isna(current_vol):
                percentile = ((valid_vol_data < current_vol).sum() / len(valid_vol_data)) * 100
            else:
                percentile = 50  # Default to middle if no valid data
            
            result_df.loc[result_df.index[i], "volatility_percentile"] = percentile
            
            # Determine volatility regime
            if percentile >= volatility_percentile:
                result_df.loc[result_df.index[i], "volatility_regime"] = "high"
            elif percentile <= (100 - volatility_percentile):
                result_df.loc[result_df.index[i], "volatility_regime"] = "low"
            else:
                result_df.loc[result_df.index[i], "volatility_regime"] = "normal"
            
            # Calculate volatility trend (5-period window)
            if i >= 5:
                vol_5_period = result_df[short_atr_col].iloc[i-5:i+1]
                
                # Check for monotonic trends (with NaN handling)
                valid_vol_trend = vol_5_period.dropna()
                if len(valid_vol_trend) >= 3:  # Need at least 3 points for trend
                    if valid_vol_trend.is_monotonic_increasing:
                        result_df.loc[result_df.index[i], "volatility_trend"] = 1
                    elif valid_vol_trend.is_monotonic_decreasing:
                        result_df.loc[result_df.index[i], "volatility_trend"] = -1
        
        return result_df

# regime_indicators.py'de TrendStrengthIndicator güncellenmesi

class TrendStrengthIndicator(BaseIndicator):
    """Analyzes trend strength and characteristics across multiple indicators."""
    
    name = "trend_strength"
    display_name = "Trend Strength"
    description = "Analyzes trend strength and characteristics across multiple indicators"
    category = "trend"
    
    # SMART DEPENDENCIES - Column names that will be auto-resolved
    dependencies = ["adx", "di_pos", "di_neg", "ema_20", "ema_50", "ema_200"]
    
    default_params = {
        "lookback_window": 50,
        "adx_threshold": 25,
        "ema_periods": [20, 50, 200]  # This will be used to validate dependencies
    }
    
    requires_columns = ["high", "low", "close"]
    output_columns = [
        "trend_strength", "trend_direction", "trend_alignment", 
        "multi_timeframe_agreement", "trend_health"
    ]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength and add to dataframe.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with trend strength columns added
        """
        result_df = df.copy()
        
        # Smart validation: Check required columns exist
        required_columns = ["adx", "di_pos", "di_neg", "ema_20", "ema_50", "ema_200"]
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"TrendStrengthIndicator requires columns: {missing_columns}. "
                f"These should be auto-calculated by dependency resolution. "
                f"Make sure you're using IndicatorManager.calculate_indicators()"
            )
        
        # Get parameters
        lookback_window = self.params.get("lookback_window", self.default_params["lookback_window"])
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        ema_periods = self.params.get("ema_periods", self.default_params["ema_periods"])
        
        # Initialize trend columns
        result_df["trend_strength"] = 0
        result_df["trend_direction"] = 0  # 1 for up, -1 for down
        result_df["trend_alignment"] = 0  # 1 for perfectly aligned, 0 for no alignment
        result_df["multi_timeframe_agreement"] = 0  # 1 for agreement, -1 for disagreement
        result_df["trend_health"] = 0  # 0-100 score
        
        # Need at least lookback_window data points
        if len(result_df) < lookback_window:
            return result_df
        
        # Calculate trend metrics for each row
        for i in range(lookback_window, len(result_df)):
            current_close = result_df["close"].iloc[i]
            
            # Trend strength from ADX (already calculated!)
            adx = result_df["adx"].iloc[i]
            di_pos = result_df["di_pos"].iloc[i]
            di_neg = result_df["di_neg"].iloc[i]
            
            # Trend strength as percentage of threshold
            if adx is not None and not pd.isna(adx) and adx_threshold > 0:
                trend_strength = min(100, int((adx / adx_threshold) * 100))
            else:
                trend_strength = 0
            result_df.loc[result_df.index[i], "trend_strength"] = trend_strength
            
            # Trend direction from DI+ and DI-
            if di_pos is not None and di_neg is not None and not pd.isna(di_pos) and not pd.isna(di_neg):
                if di_pos > di_neg:
                    result_df.loc[result_df.index[i], "trend_direction"] = 1
                else:
                    result_df.loc[result_df.index[i], "trend_direction"] = -1
            
            # Trend alignment from EMAs (already calculated!)
            ema_cols = ["ema_200", "ema_50", "ema_20"]  # Longest to shortest
            
            # Get EMA values with null checking
            ema_values = []
            valid_emas = True
            
            for col in ema_cols:
                if col in result_df.columns and not pd.isna(result_df[col].iloc[i]):
                    ema_values.append(result_df[col].iloc[i])
                else:
                    valid_emas = False
                    break
            
            # Calculate alignment if all EMAs are valid
            if valid_emas and len(ema_values) == len(ema_cols):
                # Check uptrend alignment (each EMA > previous longer-period EMA)
                uptrend_alignment = True
                for j in range(1, len(ema_values)):
                    if ema_values[j] <= ema_values[j-1]:
                        uptrend_alignment = False
                        break
                
                # Check downtrend alignment
                downtrend_alignment = True
                for j in range(1, len(ema_values)):
                    if ema_values[j] >= ema_values[j-1]:
                        downtrend_alignment = False
                        break
                
                # Set alignment value
                if uptrend_alignment:
                    result_df.loc[result_df.index[i], "trend_alignment"] = 1
                elif downtrend_alignment:
                    result_df.loc[result_df.index[i], "trend_alignment"] = -1
                
                # Multi-timeframe agreement: Price above/below all EMAs
                above_all_emas = True
                below_all_emas = True
                
                for ema_value in ema_values:
                    if current_close < ema_value:
                        above_all_emas = False
                    if current_close > ema_value:
                        below_all_emas = False
                
                if above_all_emas:
                    result_df.loc[result_df.index[i], "multi_timeframe_agreement"] = 1
                elif below_all_emas:
                    result_df.loc[result_df.index[i], "multi_timeframe_agreement"] = -1
                
                # Trend health: combined score of different metrics
                # Components:
                # 1. ADX strength (0-40 points)
                if adx is not None and not pd.isna(adx):
                    adx_score = min(40, int(adx * 40 / 50))  # Max 40 points at ADX=50
                else:
                    adx_score = 0
                
                # 2. Trend alignment (0-30 points)
                alignment_score = 30 if abs(result_df["trend_alignment"].iloc[i]) == 1 else 0
                
                # 3. Multi-timeframe agreement (0-30 points)
                mtf_score = 30 if abs(result_df["multi_timeframe_agreement"].iloc[i]) == 1 else 0
                
                # Combine scores
                trend_health = adx_score + alignment_score + mtf_score
                result_df.loc[result_df.index[i], "trend_health"] = trend_health
        
        return result_df