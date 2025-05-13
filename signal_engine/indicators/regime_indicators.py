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
        
        # Get parameters
        lookback_window = self.params.get("lookback_window", self.default_params["lookback_window"])
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        bb_width_threshold = self.params.get("bb_width_threshold", self.default_params["bb_width_threshold"])
        rsi_overbought = self.params.get("rsi_overbought", self.default_params["rsi_overbought"])
        rsi_oversold = self.params.get("rsi_oversold", self.default_params["rsi_oversold"])
        range_threshold = self.params.get("range_threshold", self.default_params["range_threshold"])
        
        # Calculate necessary indicators if they don't exist
        if "adx" not in result_df.columns:
            result_df["adx"] = ta.trend.ADXIndicator(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=14
            ).adx()
        
        if "di_pos" not in result_df.columns:
            result_df["di_pos"] = ta.trend.ADXIndicator(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=14
            ).adx_pos()
        
        if "di_neg" not in result_df.columns:
            result_df["di_neg"] = ta.trend.ADXIndicator(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=14
            ).adx_neg()
        
        if "bollinger_width" not in result_df.columns:
            bb = ta.volatility.BollingerBands(
                close=result_df["close"],
                window=20,
                window_dev=2
            )
            result_df["bollinger_width"] = bb.bollinger_wband()
        
        if "rsi_14" not in result_df.columns:
            result_df["rsi_14"] = ta.momentum.RSIIndicator(
                close=result_df["close"],
                window=14
            ).rsi()
        
        # Initialize regime column
        result_df["market_regime"] = MarketRegime.UNKNOWN.value
        result_df["regime_duration"] = 0
        result_df["regime_strength"] = 0
        
        # Need at least lookback_window data points
        if len(result_df) < lookback_window:
            return result_df
        
        # Identify regime for each row
        for i in range(lookback_window, len(result_df)):
            window = result_df.iloc[i-lookback_window:i+1]
            
            # Calculate price range as percentage
            price_range = (window["high"].max() - window["low"].min()) / window["close"].iloc[-1]
            
            # Get current indicators
            adx = window["adx"].iloc[-1]
            di_pos = window["di_pos"].iloc[-1]
            di_neg = window["di_neg"].iloc[-1]
            bb_width = window["bollinger_width"].iloc[-1]
            rsi = window["rsi_14"].iloc[-1]
            
            # Determine trend direction
            is_uptrend = di_pos > di_neg
            
            # Determine if price is in a narrow range
            is_ranging = price_range < range_threshold
            
            # Determine regime
            regime = MarketRegime.UNKNOWN.value
            regime_strength = 0
            
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
            
            result_df.loc[result_df.index[i], "market_regime"] = regime
            result_df.loc[result_df.index[i], "regime_strength"] = regime_strength
            
            # Calculate regime duration
            if i > 0 and result_df["market_regime"].iloc[i] == result_df["market_regime"].iloc[i-1]:
                result_df.loc[result_df.index[i], "regime_duration"] = result_df["regime_duration"].iloc[i-1] + 1
            else:
                result_df.loc[result_df.index[i], "regime_duration"] = 1
        
        return result_df


class VolatilityRegimeIndicator(BaseIndicator):
    """Identifies volatility regime and characteristics."""
    
    name = "volatility_regime"
    display_name = "Volatility Regime"
    description = "Identifies volatility regime and characteristics"
    category = "regime"
    
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
        
        # Calculate ATR for each period
        for period in atr_periods:
            atr_col = f"atr_{period}"
            result_df[atr_col] = ta.volatility.AverageTrueRange(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=period
            ).average_true_range()
            
            # Normalize ATR as percentage of price
            result_df[f"{atr_col}_pct"] = result_df[atr_col] / result_df["close"] * 100
        
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
            percentile = ((window[short_atr_col] < current_vol).sum() / len(window)) * 100
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
                if vol_5_period.is_monotonic_increasing:
                    result_df.loc[result_df.index[i], "volatility_trend"] = 1
                elif vol_5_period.is_monotonic_decreasing:
                    result_df.loc[result_df.index[i], "volatility_trend"] = -1
        
        return result_df


class TrendStrengthIndicator(BaseIndicator):
    """Analyzes trend strength and characteristics across multiple indicators."""
    
    name = "trend_strength"
    display_name = "Trend Strength"
    description = "Analyzes trend strength and characteristics across multiple indicators"
    category = "trend"
    
    default_params = {
        "lookback_window": 50,
        "adx_threshold": 25,
        "ema_periods": [20, 50, 200]
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
        
        # Get parameters
        lookback_window = self.params.get("lookback_window", self.default_params["lookback_window"])
        adx_threshold = self.params.get("adx_threshold", self.default_params["adx_threshold"])
        ema_periods = self.params.get("ema_periods", self.default_params["ema_periods"])
        
        # Calculate necessary indicators
        # ADX for trend strength
        if "adx" not in result_df.columns:
            adx_indicator = ta.trend.ADXIndicator(
                high=result_df["high"],
                low=result_df["low"],
                close=result_df["close"],
                window=14
            )
            result_df["adx"] = adx_indicator.adx()
            result_df["di_pos"] = adx_indicator.adx_pos()
            result_df["di_neg"] = adx_indicator.adx_neg()
        
        # EMAs for trend direction and alignment
        for period in ema_periods:
            ema_col = f"ema_{period}"
            if ema_col not in result_df.columns:
                result_df[ema_col] = ta.trend.EMAIndicator(
                    close=result_df["close"],
                    window=period
                ).ema_indicator()
        
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
            
            # Trend strength from ADX
            adx = result_df["adx"].iloc[i]
            di_pos = result_df["di_pos"].iloc[i]
            di_neg = result_df["di_neg"].iloc[i]
            
            # Trend strength as percentage of threshold
            trend_strength = min(100, int((adx / adx_threshold) * 100))
            result_df.loc[result_df.index[i], "trend_strength"] = trend_strength
            
            # Trend direction from DI+ and DI-
            if di_pos > di_neg:
                result_df.loc[result_df.index[i], "trend_direction"] = 1
            else:
                result_df.loc[result_df.index[i], "trend_direction"] = -1
            
            # Trend alignment from EMAs
            # Check if all EMAs are perfectly aligned (longest to shortest)
            ema_cols = [f"ema_{p}" for p in sorted(ema_periods, reverse=True)]
            ema_values = [result_df[col].iloc[i] for col in ema_cols]
            
            # Check uptrend alignment
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
                
            # Multi-timeframe agreement
            # Price above/below all EMAs
            above_all_emas = True
            below_all_emas = True
            
            for ema_col in ema_cols:
                ema_value = result_df[ema_col].iloc[i]
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
            adx_score = min(40, int(adx * 40 / 50))  # Max 40 points at ADX=50
            
            # 2. Trend alignment (0-30 points)
            alignment_score = 30 if abs(result_df["trend_alignment"].iloc[i]) == 1 else 0
            
            # 3. Multi-timeframe agreement (0-30 points)
            mtf_score = 30 if abs(result_df["multi_timeframe_agreement"].iloc[i]) == 1 else 0
            
            # Combine scores
            trend_health = adx_score + alignment_score + mtf_score
            result_df.loc[result_df.index[i], "trend_health"] = trend_health
        
        return result_df