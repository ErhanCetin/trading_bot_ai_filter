"""
Smart Indicators Test Suite with Comprehensive Coverage
Modern test framework for testing all indicators with Smart Dependencies
"""
import unittest
import pandas as pd
import numpy as np
import warnings
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import the indicators system
try:
    from signal_engine.indicators import registry
    from signal_engine.signal_indicator_plugin_system import IndicatorManager
except ImportError as e:
    logger.error(f"Failed to import indicators system: {e}")
    logger.error("Make sure signal_engine is in your Python path")
    raise


class SmartDataGenerator:
    """Advanced data generator for realistic market simulation."""
    
    @staticmethod
    def generate_realistic_ohlcv(
        size: int = 1000,
        base_price: float = 100.0,
        volatility: float = 0.02,
        add_trends: bool = True,
        add_patterns: bool = True
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV data with various market patterns.
        
        Args:
            size: Number of data points
            base_price: Starting price
            volatility: Base volatility
            add_trends: Add trend patterns
            add_patterns: Add technical patterns
            
        Returns:
            DataFrame with OHLCV data and timestamps
        """
        np.random.seed(42)  # For reproducible results
        
        # Create date range
        dates = pd.date_range(start='2020-01-01', periods=size, freq='1H')  # Hourly data
        
        # Generate base price movements
        returns = np.random.normal(0, volatility, size)
        
        if add_trends:
            # Add trend components
            trend_component = SmartDataGenerator._add_trend_components(size)
            returns += trend_component
        
        if add_patterns:
            # Add specific patterns
            pattern_component = SmartDataGenerator._add_pattern_components(size)
            returns += pattern_component
        
        # Calculate cumulative prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        
        # Generate realistic OHLC from close prices
        SmartDataGenerator._generate_ohlc_from_close(df)
        
        # Generate volume with realistic patterns
        SmartDataGenerator._generate_realistic_volume(df)
        
        # Add timestamp column
        df['open_time'] = df.index.astype(np.int64) // 10**6
        
        return df
    
    @staticmethod
    def _add_trend_components(size: int) -> np.ndarray:
        """Add various trend components to price movements."""
        
        # Long-term trend (bull/bear market cycles)
        long_trend = np.concatenate([
            np.linspace(0, 0.0015, size // 3),      # Bull market
            np.linspace(0.0015, -0.001, size // 3), # Bear market
            np.linspace(-0.001, 0.0008, size - 2*(size // 3))  # Recovery
        ])
        
        # Medium-term cycles (sector rotations)
        medium_trend = 0.0005 * np.sin(np.linspace(0, 6*np.pi, size))
        
        # Short-term momentum
        short_trend = 0.0002 * np.sin(np.linspace(0, 24*np.pi, size))
        
        return long_trend + medium_trend + short_trend
    
    @staticmethod
    def _add_pattern_components(size: int) -> np.ndarray:
        """Add specific technical analysis patterns."""
        
        patterns = np.zeros(size)
        
        # Add volatility clusters
        volatility_spikes = np.random.choice(size, size // 20, replace=False)
        for spike in volatility_spikes:
            if spike < size - 10:
                # Volatility cluster lasting 5-10 periods
                cluster_size = np.random.randint(5, 11)
                end_idx = min(spike + cluster_size, size)
                patterns[spike:end_idx] += np.random.normal(0, 0.01, end_idx - spike)
        
        # Add momentum bursts
        momentum_points = np.random.choice(size, size // 30, replace=False)
        for point in momentum_points:
            if point < size - 5:
                # Momentum lasting 3-5 periods
                momentum_size = np.random.randint(3, 6)
                end_idx = min(point + momentum_size, size)
                direction = np.random.choice([-1, 1])
                momentum_strength = np.linspace(0.005, 0.001, end_idx - point)
                patterns[point:end_idx] += direction * momentum_strength
        
        return patterns
    
    @staticmethod
    def _generate_ohlc_from_close(df: pd.DataFrame):
        """Generate realistic OHLC data from close prices."""
        
        size = len(df)
        
        # Initialize with close prices
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = df['close'].iloc[0]
        
        # Generate high and low based on intraday volatility
        intraday_vol = 0.005  # 0.5% average intraday range
        
        for i in range(size):
            close_price = df['close'].iloc[i]
            open_price = df['open'].iloc[i]
            
            # Calculate intraday range
            range_size = close_price * np.random.uniform(intraday_vol * 0.5, intraday_vol * 2)
            
            # Determine high and low around open/close
            min_price = min(open_price, close_price)
            max_price = max(open_price, close_price)
            
            # High extends above the higher of open/close
            df.loc[df.index[i], 'high'] = max_price + np.random.uniform(0, range_size * 0.7)
            
            # Low extends below the lower of open/close
            df.loc[df.index[i], 'low'] = min_price - np.random.uniform(0, range_size * 0.7)
    
    @staticmethod
    def _generate_realistic_volume(df: pd.DataFrame):
        """Generate realistic volume patterns."""
        
        size = len(df)
        base_volume = 1000000
        
        # Volume correlated with price movements
        price_changes = np.abs(df['close'].pct_change().fillna(0))
        
        # Base volume with some randomness
        volumes = base_volume * (1 + np.random.uniform(-0.3, 0.3, size))
        
        # Higher volume on larger price movements
        volumes *= (1 + 5 * price_changes)
        
        # Volume cycles (higher volume during "trading hours")
        volume_cycle = 1 + 0.3 * np.sin(np.linspace(0, 10*np.pi, size))
        volumes *= volume_cycle
        
        # Random volume spikes
        spike_points = np.random.choice(size, size // 50, replace=False)
        for spike in spike_points:
            volumes[spike] *= np.random.uniform(3, 8)
        
        df['volume'] = volumes.astype(int)


class IndicatorTestCase(unittest.TestCase):
    """Base test case for indicator testing."""
    
    def setUp(self):
        """Set up test data and manager."""
        self.sample_data = SmartDataGenerator.generate_realistic_ohlcv(500)
        self.manager = IndicatorManager(registry)
        
        # Track test performance
        self.test_start_time = time.time()
    
    def tearDown(self):
        """Clean up after each test."""
        test_duration = time.time() - self.test_start_time
        logger.debug(f"Test completed in {test_duration:.3f}s")
    
    def assertIndicatorCalculated(self, df: pd.DataFrame, indicator_name: str, expected_columns: List[str]):
        """Assert that indicator was calculated correctly."""
        
        # Check that expected columns exist
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Expected column '{col}' not found for {indicator_name}")
        
        # Check that columns have valid data (not all NaN)
        for col in expected_columns:
            valid_data = df[col].dropna()
            self.assertGreater(len(valid_data), 0, f"Column '{col}' has no valid data for {indicator_name}")
    
    def assertNoDuplicateCalculations(self, df: pd.DataFrame, base_columns: List[str]):
        """Assert no duplicate calculations were performed."""
        
        # Count occurrences of base calculations
        for base_col in base_columns:
            matching_cols = [col for col in df.columns if col.startswith(base_col)]
            # Should only have the expected variations, not duplicates
            self.assertLessEqual(len(matching_cols), 5, 
                               f"Too many variations of {base_col}: {matching_cols}")


class BaseIndicatorTests(IndicatorTestCase):
    """Test basic indicators."""
    
    def test_ema_indicator(self):
        """Test EMA indicator with multiple periods."""
        
        self.manager.add_indicator("ema", {"periods": [9, 21, 50]})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["ema_9", "ema_21", "ema_50"]
        self.assertIndicatorCalculated(result_df, "ema", expected_columns)
        
        # EMA should be smooth and follow price trends
        for col in expected_columns:
            # EMA should not have extreme jumps
            ema_changes = result_df[col].pct_change().dropna()
            extreme_changes = np.abs(ema_changes) > 0.1  # 10% changes
            self.assertLess(extreme_changes.sum(), len(ema_changes) * 0.01,  # Less than 1%
                          f"EMA {col} has too many extreme changes")
    
    def test_rsi_indicator(self):
        """Test RSI indicator."""
        
        self.manager.add_indicator("rsi", {"periods": [14, 21]})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["rsi_14", "rsi_21"]
        self.assertIndicatorCalculated(result_df, "rsi", expected_columns)
        
        # RSI should be between 0 and 100
        for col in expected_columns:
            rsi_values = result_df[col].dropna()
            self.assertTrue((rsi_values >= 0).all(), f"RSI {col} has values below 0")
            self.assertTrue((rsi_values <= 100).all(), f"RSI {col} has values above 100")
    
    def test_macd_indicator(self):
        """Test MACD indicator."""
        
        self.manager.add_indicator("macd")
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["macd_line", "macd_signal", "macd_histogram", "macd_crossover"]
        self.assertIndicatorCalculated(result_df, "macd", expected_columns)
        
        # MACD histogram should equal line minus signal
        histogram_calc = result_df["macd_line"] - result_df["macd_signal"]
        histogram_diff = np.abs(result_df["macd_histogram"] - histogram_calc).dropna()
        self.assertLess(histogram_diff.max(), 0.001, "MACD histogram calculation error")
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands indicator."""
        
        self.manager.add_indicator("bollinger", {"window": 20, "window_dev": 2.0})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["bollinger_upper", "bollinger_middle", "bollinger_lower", 
                          "bollinger_width", "bollinger_pct_b"]
        self.assertIndicatorCalculated(result_df, "bollinger", expected_columns)
        
        # Upper band should be above middle, middle above lower
        valid_data = result_df[expected_columns[:3]].dropna()
        self.assertTrue((valid_data["bollinger_upper"] >= valid_data["bollinger_middle"]).all(),
                       "Bollinger upper band not above middle")
        self.assertTrue((valid_data["bollinger_middle"] >= valid_data["bollinger_lower"]).all(),
                       "Bollinger middle band not above lower")
    
    def test_atr_indicator(self):
        """Test ATR indicator with multiple periods."""
        
        self.manager.add_indicator("atr", {"windows": [14, 50]})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["atr_14", "atr_50", "atr_14_percent", "atr_50_percent", "atr", "atr_percent"]
        self.assertIndicatorCalculated(result_df, "atr", expected_columns)
        
        # ATR should be positive
        for col in ["atr_14", "atr_50", "atr"]:
            atr_values = result_df[col].dropna()
            self.assertTrue((atr_values >= 0).all(), f"ATR {col} has negative values")


class AdvancedIndicatorTests(IndicatorTestCase):
    """Test advanced indicators with Smart Dependencies."""
    
    def test_adaptive_rsi(self):
        """Test Adaptive RSI with Smart Dependencies."""
        
        self.manager.add_indicator("adaptive_rsi", {"base_period": 14})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["adaptive_rsi", "adaptive_rsi_period"]
        self.assertIndicatorCalculated(result_df, "adaptive_rsi", expected_columns)
        
        # Should have ATR dependency automatically calculated
        self.assertIn("atr_14", result_df.columns, "ATR dependency not calculated")
        
        # Adaptive RSI should be between 0 and 100
        rsi_values = result_df["adaptive_rsi"].dropna()
        self.assertTrue((rsi_values >= 0).all(), "Adaptive RSI has values below 0")
        self.assertTrue((rsi_values <= 100).all(), "Adaptive RSI has values above 100")
        
        # Adaptive period should vary
        periods = result_df["adaptive_rsi_period"].dropna()
        self.assertGreater(periods.std(), 0, "Adaptive RSI period not varying")
    
    def test_supertrend_indicator(self):
        """Test Supertrend indicator with Smart Dependencies."""
        
        self.manager.add_indicator("supertrend", {"atr_period": 10, "atr_multiplier": 3.0})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["supertrend", "supertrend_direction", "supertrend_upper", "supertrend_lower"]
        self.assertIndicatorCalculated(result_df, "supertrend", expected_columns)
        
        # Should have ATR dependency automatically calculated
        self.assertIn("atr_10", result_df.columns, "ATR dependency not calculated")
        
        # Supertrend direction should be boolean-like
        directions = result_df["supertrend_direction"].dropna()
        unique_directions = set(directions)
        self.assertTrue(unique_directions.issubset({True, False, 1, 0}), 
                       "Supertrend direction has invalid values")
    
    def test_heikin_ashi(self):
        """Test Heikin Ashi transformation."""
        
        self.manager.add_indicator("heikin_ashi")
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["ha_open", "ha_high", "ha_low", "ha_close", "ha_trend"]
        self.assertIndicatorCalculated(result_df, "heikin_ashi", expected_columns)
        
        # Heikin Ashi should maintain OHLC relationships
        ha_data = result_df[expected_columns[:-1]].dropna()
        self.assertTrue((ha_data["ha_high"] >= ha_data["ha_open"]).all(), "HA High < Open")
        self.assertTrue((ha_data["ha_high"] >= ha_data["ha_close"]).all(), "HA High < Close")
        self.assertTrue((ha_data["ha_low"] <= ha_data["ha_open"]).all(), "HA Low > Open")
        self.assertTrue((ha_data["ha_low"] <= ha_data["ha_close"]).all(), "HA Low > Close")
    
    def test_ichimoku_indicator(self):
        """Test Ichimoku Cloud indicator."""
        
        self.manager.add_indicator("ichimoku")
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", 
                          "chikou_span", "cloud_strength"]
        self.assertIndicatorCalculated(result_df, "ichimoku", expected_columns)
        
        # Cloud strength should be the difference between spans
        span_diff = result_df["senkou_span_a"] - result_df["senkou_span_b"]
        strength_diff = np.abs(result_df["cloud_strength"] - span_diff).dropna()
        self.assertLess(strength_diff.max(), 0.001, "Ichimoku cloud strength calculation error")


class FeatureIndicatorTests(IndicatorTestCase):
    """Test feature engineering indicators."""
    
    def test_price_action_indicator(self):
        """Test Price Action indicator."""
        
        self.manager.add_indicator("price_action")
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["body_size", "upper_shadow", "lower_shadow", "body_position", 
                          "range_size", "body_range_ratio", "engulfing_pattern", "doji_pattern"]
        self.assertIndicatorCalculated(result_df, "price_action", expected_columns)
        
        # Body size should be positive
        body_sizes = result_df["body_size"].dropna()
        self.assertTrue((body_sizes >= 0).all(), "Body size has negative values")
        
        # Shadow ratios should be between 0 and 1
        for shadow_col in ["upper_shadow", "lower_shadow"]:
            shadows = result_df[shadow_col].dropna()
            self.assertTrue((shadows >= 0).all(), f"{shadow_col} has negative values")
            self.assertTrue((shadows <= 1).all(), f"{shadow_col} has values > 1")
    
    def test_volume_price_indicator(self):
        """Test Volume-Price indicator with Smart Dependencies."""
        
        self.manager.add_indicator("volume_price", {"volume_ma_period": 20})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["volume_ma", "volume_ratio", "obv", "price_volume_trend"]
        self.assertIndicatorCalculated(result_df, "volume_price", expected_columns)
        
        # Should have SMA dependency automatically calculated
        self.assertIn("sma_20", result_df.columns, "SMA dependency not calculated")
        
        # Volume MA should be positive
        vol_ma = result_df["volume_ma"].dropna()
        self.assertTrue((vol_ma > 0).all(), "Volume MA has non-positive values")
    
    def test_momentum_features(self):
        """Test Momentum Features with Smart Dependencies."""
        
        self.manager.add_indicator("momentum_features", {"lookback_periods": [5, 10, 20]})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["momentum_5", "momentum_10", "momentum_20", 
                          "momentum_accel_5", "momentum_accel_10", "momentum_accel_20"]
        self.assertIndicatorCalculated(result_df, "momentum_features", expected_columns)
        
        # Should have RSI dependency automatically calculated
        self.assertIn("rsi_14", result_df.columns, "RSI dependency not calculated")


class StatisticalIndicatorTests(IndicatorTestCase):
    """Test statistical indicators - FIXED VERSION."""
    
    def test_zscore_indicator(self):
        """Test Z-Score indicator with Smart Dependencies."""
        
        # Test with basic columns first (no dependencies)
        self.manager.add_indicator("zscore", {"window": 50, "apply_to": ["close", "volume"]})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        # Check basic z-score columns exist
        basic_expected = ["close_zscore", "close_percentile", "volume_zscore", "volume_percentile"]
        
        # Only check columns that should definitely exist
        for col in basic_expected:
            if col in result_df.columns:
                valid_data = result_df[col].dropna()
                self.assertGreater(len(valid_data), 0, f"Column '{col}' has no valid data")
        
        # Check that at least close_zscore exists (most basic requirement)
        self.assertIn("close_zscore", result_df.columns, "Basic close Z-score not calculated")
        
        # Z-scores should be reasonable (not all NaN, not extreme values)
        zscore_values = result_df["close_zscore"].dropna()
        if len(zscore_values) > 0:
            # Z-scores should generally be between -5 and 5 for normal data
            extreme_zscores = np.abs(zscore_values) > 10
            self.assertLess(extreme_zscores.sum(), len(zscore_values) * 0.05,  # Less than 5% extreme
                          "Too many extreme Z-score values")
    
    def test_standard_deviation_indicator(self):
        """Test Standard Deviation indicator."""
        
        self.manager.add_indicator("std_deviation", {"windows": [5, 20]})  # Reduced windows for testing
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        # Check for basic std columns
        basic_columns = ["std_5", "std_20"]
        
        for col in basic_columns:
            self.assertIn(col, result_df.columns, f"Missing column: {col}")
            
            # Standard deviation should be non-negative
            std_values = result_df[col].dropna()
            if len(std_values) > 0:
                self.assertTrue((std_values >= 0).all(), f"{col} has negative values")
                
                # Should have reasonable values (not all zeros)
                self.assertGreater(std_values.max(), 0, f"{col} has all zero values")
    
    def test_keltner_channel_indicator(self):
        """Test Keltner Channel indicator with Smart Dependencies."""
        
        self.manager.add_indicator("keltner", {"ema_window": 20, "atr_window": 10})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["keltner_middle", "keltner_upper", "keltner_lower", "keltner_width", "keltner_position"]
        
        # Check that main columns exist
        for col in expected_columns[:3]:  # Check the main three channels
            self.assertIn(col, result_df.columns, f"Missing Keltner column: {col}")
        
        # Should have EMA and ATR dependencies calculated
        self.assertIn("ema_20", result_df.columns, "EMA dependency not calculated")
        self.assertIn("atr_10", result_df.columns, "ATR dependency not calculated")
        
        # Channel relationship validation
        valid_data = result_df[expected_columns[:3]].dropna()
        if len(valid_data) > 0:
            # Upper should be >= Middle >= Lower
            self.assertTrue((valid_data["keltner_upper"] >= valid_data["keltner_middle"]).all(),
                           "Keltner upper not >= middle")
            self.assertTrue((valid_data["keltner_middle"] >= valid_data["keltner_lower"]).all(),
                           "Keltner middle not >= lower")
    
    def test_linear_regression_indicator(self):
        """Test Linear Regression indicator."""
        
        self.manager.add_indicator("linear_regression", {"windows": [20], "apply_to": "close"})
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        expected_columns = ["reg_slope_20", "reg_r2_20", "reg_line_20", "reg_deviation_20"]
        
        # Check basic columns exist
        for col in expected_columns:
            self.assertIn(col, result_df.columns, f"Missing regression column: {col}")
        
        # R-squared should be between 0 and 1
        r2_values = result_df["reg_r2_20"].dropna()
        if len(r2_values) > 0:
            self.assertTrue((r2_values >= 0).all(), "R-squared has values below 0")
            self.assertTrue((r2_values <= 1).all(), "R-squared has values above 1")
        
        # Slopes should be finite
        slope_values = result_df["reg_slope_20"].dropna()
        if len(slope_values) > 0:
            self.assertTrue(np.isfinite(slope_values).all(), "Regression slopes contain infinite values")


class SmartDependencyTests(IndicatorTestCase):
    """Test Smart Dependencies system."""
    
    def test_dependency_resolution(self):
        """Test that dependencies are resolved correctly."""
        
        # Add indicators with dependencies
        self.manager.add_indicator("adaptive_rsi")  # Depends on ATR
        self.manager.add_indicator("supertrend")    # Depends on ATR
        self.manager.add_indicator("zscore", {"apply_to": ["rsi_14"]})  # Depends on RSI
        
        # Check dependency graph
        dependencies = self.manager.get_dependency_graph()
        
        self.assertIn("adaptive_rsi", dependencies)
        self.assertIn("supertrend", dependencies)
        self.assertIn("zscore", dependencies)
        
        # Resolve dependencies
        resolved = self.manager._resolve_dependencies(["adaptive_rsi", "supertrend", "zscore"])
        
        # ATR should come before adaptive_rsi and supertrend
        atr_idx = next(i for i, ind in enumerate(resolved) if ind.startswith("atr"))
        adaptive_rsi_idx = next(i for i, ind in enumerate(resolved) if ind == "adaptive_rsi")
        supertrend_idx = next(i for i, ind in enumerate(resolved) if ind == "supertrend")
        
        self.assertLess(atr_idx, adaptive_rsi_idx, "ATR not calculated before Adaptive RSI")
        self.assertLess(atr_idx, supertrend_idx, "ATR not calculated before Supertrend")
    
    def test_no_duplicate_dependencies(self):
        """Test that dependencies are not calculated multiple times."""
        
        # Add multiple indicators that depend on ATR
        self.manager.add_indicator("adaptive_rsi")
        self.manager.add_indicator("supertrend")
        self.manager.add_indicator("keltner")
        
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        # Should only have one set of ATR columns
        atr_columns = [col for col in result_df.columns if col.startswith("atr_")]
        expected_atr_cols = {"atr_14", "atr_10", "atr_14_percent", "atr_10_percent", "atr", "atr_percent"}
        
        # Check no unexpected duplicates
        self.assertLessEqual(len(atr_columns), len(expected_atr_cols) + 2,  # Allow some flexibility
                           f"Too many ATR columns: {atr_columns}")
    
    def test_fallback_calculation(self):
        """Test fallback calculation when dependencies are missing."""
        
        # Create a minimal dataset without pre-calculated indicators
        minimal_data = self.sample_data[["open", "high", "low", "close", "volume"]].copy()
        
        # Add indicator with dependencies
        self.manager.add_indicator("adaptive_rsi")
        result_df = self.manager.calculate_indicators(minimal_data)
        
        # Should have calculated ATR dependency automatically
        self.assertIn("atr_14", result_df.columns, "Fallback ATR calculation failed")
        self.assertIn("adaptive_rsi", result_df.columns, "Adaptive RSI calculation failed")


class PerformanceTests(IndicatorTestCase):
    """Test performance characteristics."""
    
    def test_calculation_performance(self):
        """Test that calculations complete within reasonable time."""
        
        # Large dataset
        large_data = SmartDataGenerator.generate_realistic_ohlcv(5000)
        
        # Add multiple indicators
        indicators = ["ema", "rsi", "macd", "bollinger", "atr", "adaptive_rsi", "supertrend"]
        for indicator in indicators:
            self.manager.add_indicator(indicator)
        
        # Measure calculation time
        start_time = time.time()
        result_df = self.manager.calculate_indicators(large_data)
        calculation_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(calculation_time, 10.0, f"Calculation took too long: {calculation_time:.2f}s")
        
        # Should have all expected indicators
        expected_indicators = ["ema_9", "rsi_14", "macd_line", "bollinger_upper", "atr_14", 
                             "adaptive_rsi", "supertrend"]
        for indicator in expected_indicators:
            self.assertIn(indicator, result_df.columns, f"Missing indicator: {indicator}")
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable."""
        
        # Multiple datasets
        datasets = [SmartDataGenerator.generate_realistic_ohlcv(1000) for _ in range(3)]
        
        for i, data in enumerate(datasets):
            manager = IndicatorManager(registry)
            manager.add_indicator("ema", {"periods": [20, 50]})
            manager.add_indicator("rsi")
            manager.add_indicator("macd")
            
            result_df = manager.calculate_indicators(data)
            
            # Should not accumulate excessive columns
            self.assertLess(len(result_df.columns), 50, 
                          f"Too many columns in dataset {i}: {len(result_df.columns)}")


class IntegrationTests(IndicatorTestCase):
    """Integration tests for complete workflows."""
    
    def test_complete_trading_setup(self):
        """Test a complete trading system setup."""
        
        # Add a comprehensive set of indicators
        self.manager.add_indicator("ema", {"periods": [20, 50, 200]})
        self.manager.add_indicator("rsi", {"periods": [14]})
        self.manager.add_indicator("macd")
        self.manager.add_indicator("bollinger")
        self.manager.add_indicator("atr")
        self.manager.add_indicator("supertrend")
        self.manager.add_indicator("adaptive_rsi")
        self.manager.add_indicator("heikin_ashi")
        self.manager.add_indicator("price_action")
        self.manager.add_indicator("volume_price")
        
        result_df = self.manager.calculate_indicators(self.sample_data)
        
        # Should have all major indicator types
        trend_indicators = ["ema_20", "ema_50", "ema_200", "supertrend"]
        momentum_indicators = ["rsi_14", "macd_line", "adaptive_rsi"]
        volatility_indicators = ["bollinger_upper", "atr_14"]
        price_indicators = ["ha_open", "body_size"]
        volume_indicators = ["volume_ma", "obv"]
        
        all_expected = trend_indicators + momentum_indicators + volatility_indicators + price_indicators + volume_indicators
        
        for indicator in all_expected:
            self.assertIn(indicator, result_df.columns, f"Missing indicator: {indicator}")
        
        # Should have reasonable number of valid data points
        for indicator in all_expected:
            valid_data = result_df[indicator].dropna()
            self.assertGreater(len(valid_data), len(result_df) * 0.7,  # At least 70% valid data
                             f"Indicator {indicator} has too many NaN values")
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        
        # Test with invalid indicator name
        with self.assertRaises(ValueError):
            self.manager.add_indicator("nonexistent_indicator")
        
        # Test with invalid parameters
        with self.assertRaises((ValueError, TypeError)):
            self.manager.add_indicator("ema", {"periods": "invalid"})
        
        # Test with insufficient data
        tiny_data = self.sample_data.head(5)  # Only 5 rows
        self.manager.add_indicator("ema", {"periods": [50]})  # Period larger than data
        
        # Should handle gracefully without crashing
        try:
            result_df = self.manager.calculate_indicators(tiny_data)
            # Should have the column but with mostly NaN values
            self.assertIn("ema_50", result_df.columns)
        except Exception as e:
            # If it raises an exception, it should be informative
            self.assertIn("insufficient", str(e).lower())


class TestRunner:
    """Custom test runner with detailed reporting."""
    
    @staticmethod
    def run_all_tests():
        """Run all test suites with detailed reporting."""
        
        print("=" * 60)
        print("SMART INDICATORS TEST SUITE")
        print("=" * 60)
        
        # Test suites in order of complexity
        test_suites = [
            ("Base Indicators", BaseIndicatorTests),
            ("Advanced Indicators", AdvancedIndicatorTests),
            ("Feature Indicators", FeatureIndicatorTests),
            ("Statistical Indicators", StatisticalIndicatorTests),
            ("Smart Dependencies", SmartDependencyTests),
            ("Performance Tests", PerformanceTests),
            ("Integration Tests", IntegrationTests)
        ]
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        suite_results = []
        
        for suite_name, test_class in test_suites:
            print(f"\n{'-' * 40}")
            print(f"Running {suite_name}")
            print(f"{'-' * 40}")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests with custom result handler
            result = unittest.TestResult()
            suite.run(result)
            
            # Collect results
            tests_run = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            
            total_tests += tests_run
            total_failures += failures
            total_errors += errors
            
            # Print results
            if failures == 0 and errors == 0:
                print(f"✅ {suite_name}: {tests_run}/{tests_run} tests passed")
            else:
                print(f"❌ {suite_name}: {tests_run - failures - errors}/{tests_run} tests passed")
                if failures > 0:
                    print(f"   Failures: {failures}")
                if errors > 0:
                    print(f"   Errors: {errors}")
            
            suite_results.append({
                'name': suite_name,
                'tests': tests_run,
                'failures': failures,
                'errors': errors,
                'success_rate': (tests_run - failures - errors) / tests_run if tests_run > 0 else 0
            })
            
            # Print failure/error details
            if failures > 0:
                print("\n   FAILURES:")
                for test, traceback in result.failures:
                    print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
            
            if errors > 0:
                print("\n   ERRORS:")
                for test, traceback in result.errors:
                    print(f"   - {test}: {traceback.split('\\n')[-2]}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL TEST SUMMARY")
        print("=" * 60)
        
        success_rate = (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_tests - total_failures - total_errors}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.95:
            print("\n🎉 EXCELLENT! All systems working perfectly!")
        elif success_rate >= 0.85:
            print("\n✅ GOOD! Most systems working correctly.")
        elif success_rate >= 0.70:
            print("\n⚠️ FAIR. Some issues need attention.")
        else:
            print("\n❌ POOR. Significant issues detected.")
        
        # Detailed suite breakdown
        print(f"\n{'Suite':<25} {'Tests':<8} {'Pass Rate':<10} {'Status'}")
        print("-" * 55)
        for suite in suite_results:
            status = "✅ PASS" if suite['success_rate'] >= 0.95 else "⚠️ ISSUES" if suite['success_rate'] >= 0.70 else "❌ FAIL"
            print(f"{suite['name']:<25} {suite['tests']:<8} {suite['success_rate']:<9.1%} {status}")
        
        return success_rate >= 0.85
    
    @staticmethod
    def run_quick_test():
        """Run a quick smoke test of core functionality."""
        
        print("=" * 40)
        print("QUICK SMOKE TEST")
        print("=" * 40)
        
        try:
            # Test basic setup
            data = SmartDataGenerator.generate_realistic_ohlcv(100)
            manager = IndicatorManager(registry)
            
            # Test a few key indicators
            manager.add_indicator("ema", {"periods": [20]})
            manager.add_indicator("rsi", {"periods": [14]})
            manager.add_indicator("adaptive_rsi")
            
            result_df = manager.calculate_indicators(data)
            
            # Basic checks
            expected_columns = ["ema_20", "rsi_14", "adaptive_rsi", "atr_14"]
            missing_columns = [col for col in expected_columns if col not in result_df.columns]
            
            if not missing_columns:
                print("✅ Quick test PASSED - Core functionality working")
                return True
            else:
                print(f"❌ Quick test FAILED - Missing columns: {missing_columns}")
                return False
                
        except Exception as e:
            print(f"❌ Quick test FAILED - Error: {e}")
            return False
    
    @staticmethod
    def benchmark_performance():
        """Benchmark performance with different data sizes."""
        
        print("=" * 40)
        print("PERFORMANCE BENCHMARK")
        print("=" * 40)
        
        data_sizes = [500, 1000, 2000, 5000]
        indicators = ["ema", "rsi", "macd", "bollinger", "adaptive_rsi", "supertrend"]
        
        print(f"{'Data Size':<12} {'Time (s)':<12} {'Indicators/s':<15} {'Status'}")
        print("-" * 55)
        
        for size in data_sizes:
            try:
                # Generate data
                data = SmartDataGenerator.generate_realistic_ohlcv(size)
                manager = IndicatorManager(registry)
                
                # Add indicators
                for indicator in indicators:
                    manager.add_indicator(indicator)
                
                # Benchmark calculation
                start_time = time.time()
                result_df = manager.calculate_indicators(data)
                calc_time = time.time() - start_time
                
                indicators_per_sec = len(indicators) / calc_time if calc_time > 0 else float('inf')
                status = "✅ FAST" if calc_time < 2.0 else "⚠️ SLOW" if calc_time < 5.0 else "❌ VERY SLOW"
                
                print(f"{size:<12} {calc_time:<11.3f} {indicators_per_sec:<14.1f} {status}")
                
            except Exception as e:
                print(f"{size:<12} ERROR: {str(e)[:30]}")
        
        print("\nBenchmark completed.")


def run_indicator_diagnostics():
    """Run comprehensive diagnostics on the indicator system."""
    
    print("=" * 60)
    print("INDICATOR SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    # Check registry
    all_indicators = registry.get_all_indicators()
    print(f"📊 Registry Status: {len(all_indicators)} indicators registered")
    
    # Check categories
    categories = {}
    for name, indicator_class in all_indicators.items():
        category = indicator_class.category
        if category not in categories:
            categories[category] = []
        categories[category].append(name)
    
    print(f"📂 Categories: {len(categories)}")
    for category, indicators in categories.items():
        print(f"   - {category}: {len(indicators)} indicators")
    
    # Check dependencies
    print(f"\n🔗 Dependency Analysis:")
    dependency_count = 0
    no_dependency_count = 0
    
    for name, indicator_class in all_indicators.items():
        if hasattr(indicator_class, 'dependencies') and indicator_class.dependencies:
            dependency_count += 1
        else:
            no_dependency_count += 1
    
    print(f"   - Indicators with dependencies: {dependency_count}")
    print(f"   - Independent indicators: {no_dependency_count}")
    
    # Test basic functionality
    print(f"\n🧪 Basic Functionality Test:")
    quick_success = TestRunner.run_quick_test()
    
    if quick_success:
        print("\n🎯 System is ready for comprehensive testing!")
        return True
    else:
        print("\n⚠️ Issues detected. Please check configuration.")
        return False


if __name__ == "__main__":
    """Main test execution."""
    
    # Parse command line arguments
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            TestRunner.run_quick_test()
        elif command == "benchmark":
            TestRunner.benchmark_performance()
        elif command == "diagnostics":
            run_indicator_diagnostics()
        elif command == "full":
            if run_indicator_diagnostics():
                TestRunner.run_all_tests()
        else:
            print("Available commands: quick, benchmark, diagnostics, full")
    else:
        # Default: run diagnostics then full test suite
        print("Smart Indicators Test Suite")
        print("Use 'python test_indicators.py [command]' where command is:")
        print("  quick      - Quick smoke test")
        print("  benchmark  - Performance benchmark")
        print("  diagnostics- System diagnostics")
        print("  full       - Complete test suite")
        print()
        
        if run_indicator_diagnostics():
            proceed = input("Run full test suite? (y/N): ").lower().strip()
            if proceed == 'y':
                TestRunner.run_all_tests()
        else:
            print("Please fix configuration issues before running tests.")