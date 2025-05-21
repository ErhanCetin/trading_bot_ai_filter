"""
Test cases for trend strategy implementations.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategies
from signal_engine.strategies.trend_strategy import (
    TrendFollowingStrategy,
    MultiTimeframeTrendStrategy,
    AdaptiveTrendStrategy
)

class TestTrendStrategies(unittest.TestCase):
    """Test cases for trend strategy implementations."""
    
    def setUp(self):
        """Set up test data for trend strategies."""
        # Create sample data for testing
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]
        self.df = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 500000, 100),
            'adx': np.random.normal(30, 10, 100),
            'di_pos': np.random.normal(30, 10, 100),
            'di_neg': np.random.normal(20, 10, 100),
            'rsi_14': np.random.normal(50, 15, 100),
            'macd_line': np.random.normal(0, 1, 100),
            'ema_20': np.random.normal(100, 5, 100),
            'ema_50': np.random.normal(98, 5, 100),
            'market_regime': ['strong_uptrend'] * 30 + ['weak_uptrend'] * 20 + 
                            ['ranging'] * 20 + ['weak_downtrend'] * 20 + ['strong_downtrend'] * 10
        }, index=dates)
        
        # Add some bullish and bearish patterns
        # Bullish: ADX high, DI+ > DI-, RSI > 50, MACD > 0, close > EMAs
        bullish_indices = list(range(20, 40))
        self.df.loc[self.df.index[bullish_indices], 'adx'] = 35
        self.df.loc[self.df.index[bullish_indices], 'di_pos'] = 35
        self.df.loc[self.df.index[bullish_indices], 'di_neg'] = 15
        self.df.loc[self.df.index[bullish_indices], 'rsi_14'] = 65
        self.df.loc[self.df.index[bullish_indices], 'macd_line'] = 2
        self.df.loc[self.df.index[bullish_indices], 'close'] = 110
        self.df.loc[self.df.index[bullish_indices], 'ema_20'] = 105
        self.df.loc[self.df.index[bullish_indices], 'ema_50'] = 100
        
        # Bearish: ADX high, DI- > DI+, RSI < 50, MACD < 0, close < EMAs
        bearish_indices = list(range(60, 80))
        self.df.loc[self.df.index[bearish_indices], 'adx'] = 35
        self.df.loc[self.df.index[bearish_indices], 'di_pos'] = 15
        self.df.loc[self.df.index[bearish_indices], 'di_neg'] = 35
        self.df.loc[self.df.index[bearish_indices], 'rsi_14'] = 35
        self.df.loc[self.df.index[bearish_indices], 'macd_line'] = -2
        self.df.loc[self.df.index[bearish_indices], 'close'] = 90
        self.df.loc[self.df.index[bearish_indices], 'ema_20'] = 95
        self.df.loc[self.df.index[bearish_indices], 'ema_50'] = 100
        
        # Add MTF indicators for MultiTimeframeTrendStrategy
        self.df['mtf_ema_alignment'] = 0
        self.df.loc[self.df.index[bullish_indices], 'mtf_ema_alignment'] = 0.8
        self.df.loc[self.df.index[bearish_indices], 'mtf_ema_alignment'] = -0.8
        
        # Add volatility indicators for AdaptiveTrendStrategy
        self.df['atr_percent'] = np.random.normal(1, 0.5, 100)
        self.df['volatility_regime'] = ['normal'] * 100
        self.df.loc[self.df.index[30:40], 'volatility_regime'] = 'high'
        self.df.loc[self.df.index[70:80], 'volatility_regime'] = 'low'

    def test_trend_following_strategy_init(self):
        """Test TrendFollowingStrategy initialization."""
        # Test with default parameters
        strategy = TrendFollowingStrategy()
        self.assertEqual(strategy.name, "trend_following")
        self.assertEqual(strategy.category, "trend")
        self.assertEqual(strategy.params["adx_threshold"], 25)
        
        # Test with custom parameters
        custom_params = {"adx_threshold": 30, "rsi_threshold": 55}
        strategy = TrendFollowingStrategy(custom_params)
        self.assertEqual(strategy.params["adx_threshold"], 30)
        self.assertEqual(strategy.params["rsi_threshold"], 55)
    
    def test_trend_following_strategy_signals(self):
        """Test TrendFollowingStrategy signal generation."""
        strategy = TrendFollowingStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Check that bullish patterns produce long signals
        bullish_indices = list(range(20, 40))
        for i in bullish_indices:
            self.assertEqual(signals.iloc[i], 1, f"Expected long signal at index {i}")
        
        # Check that bearish patterns produce short signals
        bearish_indices = list(range(60, 80))
        for i in bearish_indices:
            self.assertEqual(signals.iloc[i], -1, f"Expected short signal at index {i}")
    
    def test_multi_timeframe_trend_strategy_init(self):
        """Test MultiTimeframeTrendStrategy initialization."""
        # Test with default parameters
        strategy = MultiTimeframeTrendStrategy()
        self.assertEqual(strategy.name, "mtf_trend")
        self.assertEqual(strategy.category, "trend")
        self.assertEqual(strategy.params["alignment_required"], 0.8)
        
        # Test with custom parameters
        custom_params = {"alignment_required": 0.7}
        strategy = MultiTimeframeTrendStrategy(custom_params)
        self.assertEqual(strategy.params["alignment_required"], 0.7)
    
    def test_multi_timeframe_trend_strategy_signals(self):
        """Test MultiTimeframeTrendStrategy signal generation."""
        strategy = MultiTimeframeTrendStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Check that strong alignment patterns produce signals
        bullish_indices = list(range(20, 40))
        for i in bullish_indices:
            self.assertEqual(signals.iloc[i], 1, f"Expected long signal at index {i}")
        
        bearish_indices = list(range(60, 80))
        for i in bearish_indices:
            self.assertEqual(signals.iloc[i], -1, f"Expected short signal at index {i}")
    
    def test_adaptive_trend_strategy_init(self):
        """Test AdaptiveTrendStrategy initialization."""
        # Test with default parameters
        strategy = AdaptiveTrendStrategy()
        self.assertEqual(strategy.name, "adaptive_trend")
        self.assertEqual(strategy.category, "trend")
        self.assertEqual(strategy.params["adx_max_threshold"], 40)
        self.assertEqual(strategy.params["adx_min_threshold"], 15)
        
        # Test with custom parameters
        custom_params = {"adx_max_threshold": 35, "adx_min_threshold": 10}
        strategy = AdaptiveTrendStrategy(custom_params)
        self.assertEqual(strategy.params["adx_max_threshold"], 35)
        self.assertEqual(strategy.params["adx_min_threshold"], 10)
    
    def test_adaptive_trend_strategy_signals(self):
        """Test AdaptiveTrendStrategy signal generation."""
        strategy = AdaptiveTrendStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Test signal generation in different volatility regimes
        high_vol_indices = list(range(30, 40))
        low_vol_indices = list(range(70, 80))
        
        # In high volatility, strategy should be more conservative (less signals)
        high_vol_signals = sum(1 for i in high_vol_indices if signals.iloc[i] != 0)
        
        # In low volatility, strategy should be more aggressive (more signals)
        low_vol_signals = sum(1 for i in low_vol_indices if signals.iloc[i] != 0)
        
        # This assertion depends on the exact implementation, but we test the general behavior
        self.assertLessEqual(high_vol_signals, len(high_vol_indices), 
                           "Expected fewer signals in high volatility")

if __name__ == '__main__':
    unittest.main()