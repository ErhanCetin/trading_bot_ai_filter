"""
Test cases for breakout strategy implementations.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategies
from signal_engine.strategies.breakout_strategy import (
    VolatilityBreakoutStrategy,
    RangeBreakoutStrategy,
    SupportResistanceBreakoutStrategy
)

class TestBreakoutStrategies(unittest.TestCase):
    """Test cases for breakout strategy implementations."""
    

    def setUp(self):
        """Set up test data for breakout strategies."""
        # Create sample data for testing
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]
        self.df = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 500000, 100),
            'atr': np.random.normal(2, 0.5, 100),
            'bollinger_upper': np.random.normal(110, 5, 100),
            'bollinger_lower': np.random.normal(90, 5, 100),
            'volume_ma': np.random.normal(1000000, 100000, 100),
            'volatility_regime': ['normal'] * 100,
            'market_regime': ['normal'] * 100,
        }, index=dates)
        
        # Volatility breakout bölümü aynı kalabilir
        volatility_breakout_indices = list(range(20, 30))
        for i in volatility_breakout_indices:
            if i % 2 == 0:  # Upside breakout
                self.df.loc[self.df.index[i], 'close'] = 115
                self.df.loc[self.df.index[i], 'volume'] = 2000000
            else:  # Downside breakout
                self.df.loc[self.df.index[i], 'close'] = 85
                self.df.loc[self.df.index[i], 'volume'] = 2000000
        
        # Create VERY CLEAR range bound conditions followed by breakouts
        # First, set up a tight range for a period
        start_range = 40
        end_range = 55
        range_bound_indices = list(range(start_range, end_range))
        
        # Define a narrow range
        range_center = 100
        range_half_width = 1.5  # Very tight 3-point range: 98.5-101.5
        
        for i in range_bound_indices:
            # Create a very tight range
            self.df.loc[self.df.index[i], 'open'] = np.random.uniform(range_center - range_half_width, 
                                                                range_center + range_half_width)
            self.df.loc[self.df.index[i], 'close'] = np.random.uniform(range_center - range_half_width, 
                                                                    range_center + range_half_width)
            self.df.loc[self.df.index[i], 'high'] = self.df.loc[self.df.index[i], 'close'] + np.random.uniform(0, 0.5)
            self.df.loc[self.df.index[i], 'low'] = self.df.loc[self.df.index[i], 'close'] - np.random.uniform(0, 0.5)
            
            # Set volume to be normal
            self.df.loc[self.df.index[i], 'volume'] = 1000000
        
        # Mark this region as ranging market regime
        self.df.loc[self.df.index[range_bound_indices], 'market_regime'] = 'ranging'
        
        # Now create a VERY clear upside breakout
        breakout_up_idx = 55
        self.df.loc[self.df.index[breakout_up_idx], 'open'] = range_center + range_half_width
        self.df.loc[self.df.index[breakout_up_idx], 'close'] = range_center + (range_half_width * 3)  # Big breakout
        self.df.loc[self.df.index[breakout_up_idx], 'high'] = range_center + (range_half_width * 3.5)
        self.df.loc[self.df.index[breakout_up_idx], 'low'] = range_center + range_half_width
        self.df.loc[self.df.index[breakout_up_idx], 'volume'] = 2000000  # Volume surge
        
        # Continue upward after breakout
        self.df.loc[self.df.index[breakout_up_idx+1], 'open'] = range_center + (range_half_width * 3)
        self.df.loc[self.df.index[breakout_up_idx+1], 'close'] = range_center + (range_half_width * 4)
        self.df.loc[self.df.index[breakout_up_idx+1], 'high'] = range_center + (range_half_width * 4.5)
        self.df.loc[self.df.index[breakout_up_idx+1], 'low'] = range_center + (range_half_width * 2.8)
        
        # Create a VERY clear downside breakout
        breakout_down_idx = 57
        self.df.loc[self.df.index[breakout_down_idx], 'open'] = range_center - range_half_width
        self.df.loc[self.df.index[breakout_down_idx], 'close'] = range_center - (range_half_width * 3)  # Big breakout
        self.df.loc[self.df.index[breakout_down_idx], 'high'] = range_center - range_half_width
        self.df.loc[self.df.index[breakout_down_idx], 'low'] = range_center - (range_half_width * 3.5)
        self.df.loc[self.df.index[breakout_down_idx], 'volume'] = 2000000  # Volume surge
        
        # Continue downward after breakout
        self.df.loc[self.df.index[breakout_down_idx+1], 'open'] = range_center - (range_half_width * 3)
        self.df.loc[self.df.index[breakout_down_idx+1], 'close'] = range_center - (range_half_width * 4)
        self.df.loc[self.df.index[breakout_down_idx+1], 'high'] = range_center - (range_half_width * 2.8)
        self.df.loc[self.df.index[breakout_down_idx+1], 'low'] = range_center - (range_half_width * 4.5)
        
        # Support/resistance data bölümü aynı kalabilir
        self.df['nearest_support'] = 95
        self.df['nearest_resistance'] = 105
        self.df['in_support_zone'] = False
        self.df['in_resistance_zone'] = False
        self.df['broke_support'] = False
        self.df['broke_resistance'] = False
        
        support_resistance_indices = list(range(70, 80))
        for i in support_resistance_indices:
            if i == 72:
                self.df.loc[self.df.index[i], 'close'] = 107
                self.df.loc[self.df.index[i], 'broke_resistance'] = True
                self.df.loc[self.df.index[i], 'volume'] = 2000000
            elif i == 73:
                self.df.loc[self.df.index[i], 'close'] = 108
            elif i == 77:
                self.df.loc[self.df.index[i], 'close'] = 93
                self.df.loc[self.df.index[i], 'broke_support'] = True
                self.df.loc[self.df.index[i], 'volume'] = 2000000
            elif i == 78:
                self.df.loc[self.df.index[i], 'close'] = 92
        
        # Keltner Channel data
        self.df['keltner_upper'] = self.df['bollinger_upper'] + 2
        self.df['keltner_lower'] = self.df['bollinger_lower'] - 2
        
        # Market regime info
        self.df.loc[self.df.index[20:30], 'volatility_regime'] = 'high'
        self.df.loc[self.df.index[40:60], 'volatility_regime'] = 'low'
        
        # Add bollinger_width - make it very tight during ranging period
        self.df['bollinger_width'] = (self.df['bollinger_upper'] - self.df['bollinger_lower']) / self.df['close']
        tight_band_indices = list(range(40, 60))
        self.df.loc[self.df.index[tight_band_indices], 'bollinger_width'] = 0.01  # Very tight bands

    def test_volatility_breakout_strategy_init(self):
        """Test VolatilityBreakoutStrategy initialization."""
        # Test with default parameters
        strategy = VolatilityBreakoutStrategy()
        self.assertEqual(strategy.name, "volatility_breakout")
        self.assertEqual(strategy.category, "breakout")
        self.assertEqual(strategy.params["atr_multiplier"], 2.0)
        self.assertEqual(strategy.params["lookback_period"], 14)
        
        # Test with custom parameters
        custom_params = {"atr_multiplier": 2.5, "lookback_period": 10}
        strategy = VolatilityBreakoutStrategy(custom_params)
        self.assertEqual(strategy.params["atr_multiplier"], 2.5)
        self.assertEqual(strategy.params["lookback_period"], 10)
    
    def test_volatility_breakout_strategy_signals(self):
        """Test VolatilityBreakoutStrategy signal generation."""
        strategy = VolatilityBreakoutStrategy()
        signals = strategy.generate_signals(self.df)
        
        # Check that signals is a Series with the same index as the DataFrame
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        
        # Check for upside breakout signals
        upside_indices = [i for i in range(20, 30) if i % 2 == 0]
        long_signals_count = sum(1 for i in upside_indices if signals.iloc[i] == 1)
        self.assertGreater(long_signals_count, 0, "Expected at least one long signal from volatility breakout")
        
        # Check for downside breakout signals
        downside_indices = [i for i in range(20, 30) if i % 2 == 1]
        short_signals_count = sum(1 for i in downside_indices if signals.iloc[i] == -1)
        self.assertGreater(short_signals_count, 0, "Expected at least one short signal from volatility breakout")
    
    def test_range_breakout_strategy_init(self):
        """Test RangeBreakoutStrategy initialization."""
        # Test with default parameters
        strategy = RangeBreakoutStrategy()
        self.assertEqual(strategy.name, "range_breakout")
        self.assertEqual(strategy.category, "breakout")
        self.assertEqual(strategy.params["range_period"], 20)
        self.assertEqual(strategy.params["range_threshold"], 0.03)
        
        # Test with custom parameters
        custom_params = {"range_period": 15, "range_threshold": 0.02}
        strategy = RangeBreakoutStrategy(custom_params)
        self.assertEqual(strategy.params["range_period"], 15)
        self.assertEqual(strategy.params["range_threshold"], 0.02)
    
    def test_range_breakout_strategy_signals(self):
        """Test RangeBreakoutStrategy signal generation."""
        # Create strategy with parameters that match our test data
        strategy = RangeBreakoutStrategy({
            "range_period": 15,          # Use a shorter period matching our test data
            "range_threshold": 0.03,     # Our range is about 3% of price
            "breakout_factor": 1.003     # Standard breakout factor
        })
        
        signals = strategy.generate_signals(self.df)
        
        # Print diagnostic information for debugging
        print("\nRange Breakout Strategy Test - Signal Information:")
        print(f"Upside breakout area (indices 55-56):")
        for i in range(55, 57):
            print(f"  Index {i}: Signal={signals.iloc[i]}, Close={self.df['close'].iloc[i]}")
        
        print(f"Downside breakout area (indices 57-58):")
        for i in range(57, 59):
            print(f"  Index {i}: Signal={signals.iloc[i]}, Close={self.df['close'].iloc[i]}")
        
        # Check for upside breakout signal - look at both potential signal locations
        upside_signal = signals.iloc[55] == 1 or signals.iloc[56] == 1
        self.assertTrue(upside_signal, "Expected long signal at either index 55 or 56 after upside range breakout")
        
        # Check for downside breakout signal - look at both potential signal locations
        downside_signal = signals.iloc[57] == -1 or signals.iloc[58] == -1
        self.assertTrue(downside_signal, "Expected short signal at either index 57 or 58 after downside range breakout")
        
        # Check that range breakouts are detected in general
        range_signals = sum(abs(signals.iloc[54:60]))
        self.assertGreater(range_signals, 0, "Expected at least one breakout signal in the range")
    
    def test_support_resistance_breakout_strategy_init(self):
        """Test SupportResistanceBreakoutStrategy initialization."""
        # Test with default parameters
        strategy = SupportResistanceBreakoutStrategy()
        self.assertEqual(strategy.name, "sr_breakout")
        self.assertEqual(strategy.category, "breakout")
        self.assertEqual(strategy.params["breakout_factor"], 1.003)
        self.assertEqual(strategy.params["level_strength_min"], 2)
        
        # Test with custom parameters
        custom_params = {"breakout_factor": 1.005, "level_strength_min": 3}
        strategy = SupportResistanceBreakoutStrategy(custom_params)
        self.assertEqual(strategy.params["breakout_factor"], 1.005)
        self.assertEqual(strategy.params["level_strength_min"], 3)

def test_support_resistance_breakout_strategy_signals(self):
    """Test SupportResistanceBreakoutStrategy signal generation."""
    strategy = SupportResistanceBreakoutStrategy()
    signals = strategy.generate_signals(self.df)
    
    # Check that signals is a Series with the same index as the DataFrame
    self.assertIsInstance(signals, pd.Series)
    self.assertEqual(len(signals), len(self.df))
    
    # Check for resistance breakout signals
    self.assertEqual(signals.iloc[72], 1, "Expected long signal after resistance breakout")
    
    # Check for support breakout signals
    self.assertEqual(signals.iloc[77], -1, "Expected short signal after support breakout")
    
    # Check volume confirmation
    # Since we've set high volume for breakout bars, these should be confirmed
    resistance_breakout_with_volume = signals.iloc[72] == 1
    support_breakout_with_volume = signals.iloc[77] == -1
    self.assertTrue(resistance_breakout_with_volume and support_breakout_with_volume, 
                   "Expected breakouts to be confirmed by volume")

def test_range_breakout_conditions(self):
    """Test RangeBreakoutStrategy condition generation directly."""
    strategy = RangeBreakoutStrategy({
        "range_period": 15,
        "range_threshold": 0.03,
        "breakout_factor": 1.003
    })
    
    # Check conditions at upside breakout
    upside_row = self.df.iloc[55]
    upside_conditions = strategy.generate_conditions(self.df, upside_row, 55)
    print(f"\nUpside conditions: {upside_conditions}")
    self.assertTrue(any(upside_conditions.get('long', [])), 
                    "Expected at least one long condition at upside breakout")
    
    # Check conditions at downside breakout
    downside_row = self.df.iloc[57]
    downside_conditions = strategy.generate_conditions(self.df, downside_row, 57)
    print(f"Downside conditions: {downside_conditions}")
    self.assertTrue(any(downside_conditions.get('short', [])), 
                    "Expected at least one short condition at downside breakout")
    
if __name__ == '__main__':
    unittest.main()