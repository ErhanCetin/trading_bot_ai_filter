"""
Reversal strategies for the trading system.
These strategies identify potential market reversals.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from signal_engine.signal_strategy_system import BaseStrategy


class OverextendedReversalStrategy(BaseStrategy):
    """Strategy that looks for reversals when market is overextended."""
    
    name = "overextended_reversal"
    display_name = "Overextended Reversal Strategy"
    description = "Identifies reversal opportunities when market is overextended"
    category = "reversal"
    
    default_params = {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "bollinger_threshold": 1.0,  # Distance in standard deviations
        "consecutive_candles": 3      # Number of consecutive candles in same direction
    }
    
    required_indicators = ["rsi_14", "close"]
    optional_indicators = ["bollinger_upper", "bollinger_lower", "bollinger_pct_b",
                          "market_regime", "z_score"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate reversal signal conditions when market is overextended.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        rsi_ob = self.params.get("rsi_overbought", self.default_params["rsi_overbought"])
        rsi_os = self.params.get("rsi_oversold", self.default_params["rsi_oversold"])
        bb_threshold = self.params.get("bollinger_threshold", self.default_params["bollinger_threshold"])
        consec_candles = self.params.get("consecutive_candles", self.default_params["consecutive_candles"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Ensure we have enough history
        if i < consec_candles:
            return {"long": [], "short": []}
        
        # Check for oversold conditions (long reversal)
        if "rsi_14" in row:
            # RSI oversold condition
            long_conditions.append(row["rsi_14"] < rsi_os)
            
            # RSI overbought condition
            short_conditions.append(row["rsi_14"] > rsi_ob)
            
            # RSI bullish divergence
            if i > 5:
                price_lower_low = (df["close"].iloc[i] < df["close"].iloc[i-1] and 
                                  df["close"].iloc[i-1] < df["close"].iloc[i-2])
                rsi_higher_low = (df["rsi_14"].iloc[i] > df["rsi_14"].iloc[i-1] and 
                                 df["rsi_14"].iloc[i-1] > df["rsi_14"].iloc[i-2])
                long_conditions.append(price_lower_low and rsi_higher_low)
                
                # RSI bearish divergence
                price_higher_high = (df["close"].iloc[i] > df["close"].iloc[i-1] and 
                                    df["close"].iloc[i-1] > df["close"].iloc[i-2])
                rsi_lower_high = (df["rsi_14"].iloc[i] < df["rsi_14"].iloc[i-1] and 
                                 df["rsi_14"].iloc[i-1] < df["rsi_14"].iloc[i-2])
                short_conditions.append(price_higher_high and rsi_lower_high)
        
        # Check for Bollinger Band conditions
        if all(col in row for col in ["bollinger_upper", "bollinger_lower", "close"]):
            # Price below lower band (oversold)
            band_distance = (row["bollinger_lower"] - row["close"]) / row["bollinger_lower"]
            long_conditions.append(band_distance > 0 and band_distance < bb_threshold)
            
            # Price above upper band (overbought)
            band_distance = (row["close"] - row["bollinger_upper"]) / row["bollinger_upper"]
            short_conditions.append(band_distance > 0 and band_distance < bb_threshold)
        
        # Check for Bollinger %B
        if "bollinger_pct_b" in row:
            long_conditions.append(row["bollinger_pct_b"] < 0)  # Below lower band
            short_conditions.append(row["bollinger_pct_b"] > 1)  # Above upper band
        
        # Check for extreme Z-Score
        if "z_score" in row:
            long_conditions.append(row["z_score"] < -2)  # Extremely oversold
            short_conditions.append(row["z_score"] > 2)  # Extremely overbought
        
        # Check consecutive bearish/bullish candles
        bearish_count = 0
        bullish_count = 0
        
        for j in range(i - consec_candles + 1, i + 1):
            if df["close"].iloc[j] < df["open"].iloc[j]:  # Bearish candle
                bearish_count += 1
            elif df["close"].iloc[j] > df["open"].iloc[j]:  # Bullish candle
                bullish_count += 1
        
        # After consecutive bearish candles, expect bullish reversal
        long_conditions.append(bearish_count >= consec_candles)
        
        # After consecutive bullish candles, expect bearish reversal
        short_conditions.append(bullish_count >= consec_candles)
        
        # Consider market regime if available
        if "market_regime" in row:
            # More likely to reverse from overbought in weak uptrend
            long_conditions.append(row["market_regime"] == "overbought")
            
            # More likely to reverse from oversold in weak downtrend
            short_conditions.append(row["market_regime"] == "oversold")
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class PatternReversalStrategy(BaseStrategy):
    """Strategy that looks for reversal patterns in price action."""
    
    name = "pattern_reversal"
    display_name = "Pattern Reversal Strategy"
    description = "Identifies reversal opportunities based on candlestick patterns"
    category = "reversal"
    
    default_params = {}
    
    required_indicators = ["open", "high", "low", "close"]
    optional_indicators = ["engulfing_pattern", "hammer_pattern", "shooting_star_pattern", 
                          "doji_pattern", "market_regime"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate reversal signal conditions based on price patterns.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Ensure we have enough history
        if i < 2:
            return {"long": [], "short": []}
        
        # Check if pattern indicators are already calculated
        
        # Engulfing pattern
        if "engulfing_pattern" in row:
            long_conditions.append(row["engulfing_pattern"] == 1)  # Bullish engulfing
            short_conditions.append(row["engulfing_pattern"] == -1)  # Bearish engulfing
        
        # Hammer pattern
        if "hammer_pattern" in row:
            long_conditions.append(row["hammer_pattern"] == 1)  # Hammer
            short_conditions.append(row["hammer_pattern"] == -1)  # Inverted hammer
        
        # Shooting star pattern
        if "shooting_star_pattern" in row:
            short_conditions.append(row["shooting_star_pattern"] == 1)
        
        # Doji pattern
        if "doji_pattern" in row and i > 0:
            # Doji after a downtrend suggests potential reversal
            if row["doji_pattern"] and df["close"].iloc[i-1] < df["open"].iloc[i-1]:
                long_conditions.append(True)
            
            # Doji after an uptrend suggests potential reversal
            if row["doji_pattern"] and df["close"].iloc[i-1] > df["open"].iloc[i-1]:
                short_conditions.append(True)
        
        # If pattern indicators are not available, calculate basic patterns manually
        else:
            # Current and previous candles
            curr = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "body": abs(row["close"] - row["open"]),
                "range": row["high"] - row["low"],
                "is_bullish": row["close"] > row["open"]
            }
            
            prev = {
                "open": df["open"].iloc[i-1],
                "high": df["high"].iloc[i-1],
                "low": df["low"].iloc[i-1],
                "close": df["close"].iloc[i-1],
                "body": abs(df["close"].iloc[i-1] - df["open"].iloc[i-1]),
                "range": df["high"].iloc[i-1] - df["low"].iloc[i-1],
                "is_bullish": df["close"].iloc[i-1] > df["open"].iloc[i-1]
            }
            
            # Bullish engulfing pattern
            bullish_engulfing = (
                curr["is_bullish"] and 
                not prev["is_bullish"] and 
                curr["open"] <= prev["close"] and 
                curr["close"] > prev["open"]
            )
            long_conditions.append(bullish_engulfing)
            
            # Bearish engulfing pattern
            bearish_engulfing = (
                not curr["is_bullish"] and 
                prev["is_bullish"] and 
                curr["open"] >= prev["close"] and 
                curr["close"] < prev["open"]
            )
            short_conditions.append(bearish_engulfing)
            
            # Hammer pattern (small body at top, long lower shadow)
            hammer = (
                curr["body"] / curr["range"] < 0.3 and  # Small body
                (curr["high"] - max(curr["open"], curr["close"])) < curr["body"] and  # Short upper shadow
                (min(curr["open"], curr["close"]) - curr["low"]) > 2 * curr["body"]  # Long lower shadow
            )
            long_conditions.append(hammer and not curr["is_bullish"])  # More significant if bearish
            
# Shooting star pattern (small body at bottom, long upper shadow)
            shooting_star = (
                curr["body"] / curr["range"] < 0.3 and  # Small body
                (curr["high"] - max(curr["open"], curr["close"])) > 2 * curr["body"] and  # Long upper shadow
                (min(curr["open"], curr["close"]) - curr["low"]) < curr["body"]  # Short lower shadow
            )
            short_conditions.append(shooting_star and curr["is_bullish"])  # More significant if bullish
            
            # Doji pattern (very small body)
            doji = curr["body"] / curr["range"] < 0.1 if curr["range"] > 0 else False
            
            # Doji after a downtrend suggests potential reversal
            if doji and i > 1 and df["close"].iloc[i-1] < df["close"].iloc[i-2]:
                long_conditions.append(True)
            
            # Doji after an uptrend suggests potential reversal
            if doji and i > 1 and df["close"].iloc[i-1] > df["close"].iloc[i-2]:
                short_conditions.append(True)
        
        # Add market context if available
        if "market_regime" in row:
            # In oversold regime, patterns are more significant for long reversals
            if row["market_regime"] == "oversold":
                # Duplicate long conditions to give them more weight
                long_conditions.extend(long_conditions)
            
            # In overbought regime, patterns are more significant for short reversals
            elif row["market_regime"] == "overbought":
                # Duplicate short conditions to give them more weight
                short_conditions.extend(short_conditions)
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }


class DivergenceReversalStrategy(BaseStrategy):
    """Strategy that looks for divergences between price and oscillators."""
    
    name = "divergence_reversal"
    display_name = "Divergence Reversal Strategy"
    description = "Identifies reversals based on divergences between price and oscillators"
    category = "reversal"
    
    default_params = {
        "lookback_window": 5  # Window to look back for divergence
    }
    
    required_indicators = ["close", "rsi_14"]
    optional_indicators = ["macd_line", "bullish_divergence", "bearish_divergence", 
                          "stoch_k", "stoch_d", "cci", "market_regime"]
    
    def generate_conditions(self, df: pd.DataFrame, row: pd.Series, i: int) -> Dict[str, List[bool]]:
        """
        Generate reversal signal conditions based on divergences.
        
        Args:
            df: DataFrame with indicator data
            row: Current row (Series) being processed
            i: Index of the current row
            
        Returns:
            Dictionary with keys 'long', 'short' containing lists of boolean conditions
        """
        # Get parameters
        lookback = self.params.get("lookback_window", self.default_params["lookback_window"])
        
        # Initialize condition lists
        long_conditions = []
        short_conditions = []
        
        # Ensure we have enough history
        if i < lookback + 1:
            return {"long": [], "short": []}
        
        # Check if divergence indicators are already calculated
        if "bullish_divergence" in row:
            long_conditions.append(row["bullish_divergence"])
        
        if "bearish_divergence" in row:
            short_conditions.append(row["bearish_divergence"])
        
        # If not, calculate divergences manually
        else:
            # Find local price extremes
            price_window = df["close"].iloc[i-lookback:i+1]
            curr_price = price_window.iloc[-1]
            min_price_idx = price_window.idxmin()
            max_price_idx = price_window.idxmax()
            
            # RSI divergence
            if "rsi_14" in row:
                rsi_window = df["rsi_14"].iloc[i-lookback:i+1]
                curr_rsi = rsi_window.iloc[-1]
                min_rsi_idx = rsi_window.idxmin()
                max_rsi_idx = rsi_window.idxmax()
                
                # Bullish divergence: price makes lower low but RSI makes higher low
                if min_price_idx == price_window.index[-1] and min_rsi_idx != min_price_idx:
                    long_conditions.append(True)
                
                # Bearish divergence: price makes higher high but RSI makes lower high
                if max_price_idx == price_window.index[-1] and max_rsi_idx != max_price_idx:
                    short_conditions.append(True)
            
            # MACD divergence
            if "macd_line" in row:
                macd_window = df["macd_line"].iloc[i-lookback:i+1]
                curr_macd = macd_window.iloc[-1]
                min_macd_idx = macd_window.idxmin()
                max_macd_idx = macd_window.idxmax()
                
                # Bullish divergence: price makes lower low but MACD makes higher low
                if min_price_idx == price_window.index[-1] and min_macd_idx != min_price_idx:
                    long_conditions.append(True)
                
                # Bearish divergence: price makes higher high but MACD makes lower high
                if max_price_idx == price_window.index[-1] and max_macd_idx != max_price_idx:
                    short_conditions.append(True)
            
            # Stochastic divergence
            if "stoch_k" in row:
                stoch_window = df["stoch_k"].iloc[i-lookback:i+1]
                curr_stoch = stoch_window.iloc[-1]
                min_stoch_idx = stoch_window.idxmin()
                max_stoch_idx = stoch_window.idxmax()
                
                # Bullish divergence: price makes lower low but Stochastic makes higher low
                if min_price_idx == price_window.index[-1] and min_stoch_idx != min_price_idx:
                    long_conditions.append(True)
                
                # Bearish divergence: price makes higher high but Stochastic makes lower high
                if max_price_idx == price_window.index[-1] and max_stoch_idx != max_price_idx:
                    short_conditions.append(True)
                    
            # CCI divergence
            if "cci" in row:
                cci_window = df["cci"].iloc[i-lookback:i+1]
                curr_cci = cci_window.iloc[-1]
                min_cci_idx = cci_window.idxmin()
                max_cci_idx = cci_window.idxmax()
                
                # Bullish divergence: price makes lower low but CCI makes higher low
                if min_price_idx == price_window.index[-1] and min_cci_idx != min_price_idx:
                    long_conditions.append(True)
                
                # Bearish divergence: price makes higher high but CCI makes lower high
                if max_price_idx == price_window.index[-1] and max_cci_idx != max_price_idx:
                    short_conditions.append(True)
        
        # Add market context if available
        if "market_regime" in row:
            # Strengthen long signals in oversold market
            if row["market_regime"] == "oversold" and long_conditions:
                long_conditions.append(True)  # Double the weight
            
            # Strengthen short signals in overbought market
            elif row["market_regime"] == "overbought" and short_conditions:
                short_conditions.append(True)  # Double the weight
            
            # Weaken signals in strong trending markets
            elif row["market_regime"] in ["strong_uptrend", "strong_downtrend"]:
                # Reduce number of conditions
                if long_conditions:
                    long_conditions = [long_conditions[0]]
                if short_conditions:
                    short_conditions = [short_conditions[0]]
        
        return {
            "long": long_conditions,
            "short": short_conditions
        }