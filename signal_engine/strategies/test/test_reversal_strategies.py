"""
Test cases for reversal strategy implementations.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategies
from signal_engine.strategies.reversal_strategy import (
    OverextendedReversalStrategy,
    PatternReversalStrategy,
    DivergenceReversalStrategy
)

class TestReversalStrategies(unittest.TestCase):
    """Test cases for reversal strategy implementations."""
    
    def setUp(self):
        """Set up test data for reversal strategies."""
        # Create sample data for testing
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]
        self.df = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 500000, 100),
            'rsi_14': np.random.normal(50, 15, 100),
            'bollinger_upper': np.random.normal(110, 5, 100),
            'bollinger_lower': np.random.normal(90, 5, 100),
            'bollinger_pct_b': np.random.normal(0.5, 0.3, 100),
            'market_regime': ['normal'] * 100,
            'z_score': np.random.normal(0, 1, 100)
        }, index=dates)
        
        # Create overextended conditions for long signals
        oversold_indices = list(range(20, 30))
        self.df.loc[self.df.index[oversold_indices], 'rsi_14'] = 20  # Oversold RSI
        self.df.loc[self.df.index[oversold_indices], 'bollinger_pct_b'] = -0.2  # Below lower band
        self.df.loc[self.df.index[oversold_indices], 'z_score'] = -3  # Extremely oversold
        self.df.loc[self.df.index[oversold_indices], 'market_regime'] = 'oversold'
        
        # Create overextended conditions for short signals
        overbought_indices = list(range(60, 70))
        self.df.loc[self.df.index[overbought_indices], 'rsi_14'] = 80  # Overbought RSI
        self.df.loc[self.df.index[overbought_indices], 'bollinger_pct_b'] = 1.2  # Above upper band
        self.df.loc[self.df.index[overbought_indices], 'z_score'] = 3  # Extremely overbought
        self.df.loc[self.df.index[overbought_indices], 'market_regime'] = 'overbought'
        
        # Add candlestick pattern data for pattern reversal
        self.df['engulfing_pattern'] = 0
        self.df.loc[self.df.index[22:25], 'engulfing_pattern'] = 1  # Bullish engulfing
        self.df.loc[self.df.index[62:65], 'engulfing_pattern'] = -1  # Bearish engulfing
        
        self.df['hammer_pattern'] = 0
        self.df.loc[self.df.index[26:28], 'hammer_pattern'] = 1  # Hammer
        self.df.loc[self.df.index[66:68], 'hammer_pattern'] = -1  # Inverted hammer
        
        self.df['shooting_star_pattern'] = 0
        self.df.loc[self.df.index[67:69], 'shooting_star_pattern'] = 1  # Shooting star
        
        self.df['doji_pattern'] = 0
        self.df.loc[self.df.index[29:30], 'doji_pattern'] = 1  # Doji
        self.df.loc[self.df.index[69:70], 'doji_pattern'] = 1  # Doji
        
        # Add divergence data
        self.df['bullish_divergence'] = False
        self.df['bearish_divergence'] = False
        self.df.loc[self.df.index[25:30], 'bullish_divergence'] = True
        self.df.loc[self.df.index[65:70], 'bearish_divergence'] = True
        
        # Add MACD data for divergence
        self.df['macd_line'] = np.random.normal(0, 1, 100)
        
        # Create consecutive bearish/bullish candles
        for i in range(20, 23):  # 3 consecutive bearish candles
            self.df.loc[self.df.index[i], 'open'] = 105
            self.df.loc[self.df.index[i], 'close'] = 95
        
        for i in range(60, 63):  # 3 consecutive bullish candles
            self.df.loc[self.df.index[i], 'open'] = 95
            self.df.loc[self.df.index[i], 'close'] = 105

    def test_overextended_reversal_strategy_init(self):
        """Test OverextendedReversalStrategy initialization."""
        # Test with default parameters
        strategy = OverextendedReversalStrategy()
        self.assertEqual(strategy.name, "overextended_reversal")
        self.assertEqual(strategy.category, "reversal")
        self.assertEqual(strategy.params["rsi_overbought"], 70)
        self.assertEqual(strategy.params["rsi_oversold"], 30)
        
        # Test with custom parameters
        custom_params = {"rsi_overbought": 75, "rsi_oversold": 25}
        strategy = OverextendedReversalStrategy(custom_params)
        self.assertEqual(strategy.params["rsi_overbought"], 75)
        self.assertEqual(strategy.params["rsi_oversold"], 25)
    
    def test_overextended_reversal_strategy_signals(self):
        """Test OverextendedReversalStrategy signal generation."""
        strategy = OverextendedReversalStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Check that oversold conditions produce long signals
        oversold_indices = list(range(20, 30))
        long_signals_count = sum(1 for i in oversold_indices if signals.iloc[i] == 1)
        self.assertGreater(long_signals_count, 0, "Expected at least one long signal in oversold conditions")
        
        # Check that overbought conditions produce short signals
        overbought_indices = list(range(60, 70))
        short_signals_count = sum(1 for i in overbought_indices if signals.iloc[i] == -1)
        self.assertGreater(short_signals_count, 0, "Expected at least one short signal in overbought conditions")
    
    def test_pattern_reversal_strategy_init(self):
        """Test PatternReversalStrategy initialization."""
        # Test with default parameters
        strategy = PatternReversalStrategy()
        self.assertEqual(strategy.name, "pattern_reversal")
        self.assertEqual(strategy.category, "reversal")
        
        # Initialize with custom parameters (even though default_params is empty)
        custom_params = {"pattern_threshold": 0.5}
        strategy = PatternReversalStrategy(custom_params)
        self.assertEqual(strategy.params["pattern_threshold"], 0.5)
    
    def test_pattern_reversal_strategy_signals(self):
        """Test PatternReversalStrategy signal generation."""
        strategy = PatternReversalStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Check for bullish pattern signals (engulfing, hammer, doji)
        bullish_pattern_indices = list(range(22, 30))
        long_signals_count = sum(1 for i in bullish_pattern_indices if signals.iloc[i] == 1)
        self.assertGreater(long_signals_count, 0, "Expected at least one long signal from bullish patterns")
        
        # Check for bearish pattern signals (engulfing, inverted hammer, shooting star, doji)
        bearish_pattern_indices = list(range(62, 70))
        short_signals_count = sum(1 for i in bearish_pattern_indices if signals.iloc[i] == -1)
        self.assertGreater(short_signals_count, 0, "Expected at least one short signal from bearish patterns")
    
    def test_divergence_reversal_strategy_init(self):
        """Test DivergenceReversalStrategy initialization."""
        # Test with default parameters
        strategy = DivergenceReversalStrategy()
        self.assertEqual(strategy.name, "divergence_reversal")
        self.assertEqual(strategy.category, "reversal")
        self.assertEqual(strategy.params["lookback_window"], 5)
        
        # Test with custom parameters
        custom_params = {"lookback_window": 10}
        strategy = DivergenceReversalStrategy(custom_params)
        self.assertEqual(strategy.params["lookback_window"], 10)
    
    def test_divergence_reversal_strategy_signals(self):
        """Test DivergenceReversalStrategy signal generation."""
        strategy = DivergenceReversalStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Check that bullish divergence produces long signals
        bullish_div_indices = list(range(25, 30))
        long_signals_count = sum(1 for i in bullish_div_indices if signals.iloc[i] == 1)
        self.assertGreater(long_signals_count, 0, "Expected at least one long signal from bullish divergence")
        
        # Check that bearish divergence produces short signals
        bearish_div_indices = list(range(65, 70))
        short_signals_count = sum(1 for i in bearish_div_indices if signals.iloc[i] == -1)
        self.assertGreater(short_signals_count, 0, "Expected at least one short signal from bearish divergence")

if __name__ == '__main__':
    unittest.main()