"""
Data Quality Validation Module for Testing Framework
Comprehensive data quality checks and validation logic
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Dict, Any, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data quality validator for testing framework.
    Ensures data meets quality standards before indicator testing.
    """
    
    def __init__(self, quality_thresholds: Dict[str, Any] = None):
        """
        Initialize data validator with quality thresholds.
        
        Args:
            quality_thresholds: Dictionary with data quality thresholds
        """
        # Default quality thresholds
        self.default_thresholds = {
            'max_missing_data_pct': 5.0,
            'min_volume_threshold': 1000,
            'min_data_quality_score': 75.0,
            'max_extreme_price_moves_pct': 1.0,  # Max 1% of data can have >10% moves
            'max_time_gaps_pct': 2.0  # Max 2% of data can have large time gaps
        }
        
        # Use provided thresholds or defaults
        if quality_thresholds:
            self.thresholds = {**self.default_thresholds, **quality_thresholds}
        else:
            self.thresholds = self.default_thresholds
        
        logger.debug("🔧 DataValidator initialized")
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str, interval: str, 
                            min_data_points: int) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            symbol: Trading symbol (for logging)
            interval: Time interval (for logging)
            min_data_points: Minimum required data points
            
        Returns:
            Dictionary with validation results and quality score
            
        Raises:
            ValueError: If data quality is unacceptable
        """
        logger.info(f"🔍 Validating data quality for {symbol} {interval}")
        
        validation_results = {
            'symbol': symbol,
            'interval': interval,
            'total_rows': len(df),
            'min_required_rows': min_data_points,
            'checks': {},
            'warnings': [],
            'errors': [],
            'quality_score': 0.0,
            'is_valid': False
        }
        
        try:
            # 1. Check minimum data points
            self._check_minimum_data_points(df, min_data_points, validation_results)
            
            # 2. Check missing data percentage
            self._check_missing_data(df, validation_results)
            
            # 3. Validate OHLC data consistency
            self._check_ohlc_consistency(df, validation_results)
            
            # 4. Check volume data quality
            self._check_volume_quality(df, validation_results)
            
            # 5. Analyze time gaps
            self._check_time_gaps(df, interval, validation_results)
            
            # 6. Check for extreme price movements
            self._check_extreme_price_movements(df, validation_results)
            
            # 7. Calculate overall quality score
            quality_score = self._calculate_quality_score(validation_results)
            validation_results['quality_score'] = quality_score
            
            # 8. Determine if data is acceptable
            is_valid = self._determine_validity(validation_results)
            validation_results['is_valid'] = is_valid
            
            # Log summary
            self._log_validation_summary(validation_results)
            
            # Raise exception if data is not valid
            if not is_valid:
                raise ValueError(f"Data quality unacceptable for {symbol} {interval}")
            
            return validation_results
            
        except Exception as e:
            validation_results['errors'].append(str(e))
            logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _check_minimum_data_points(self, df: pd.DataFrame, min_required: int, 
                                 results: Dict[str, Any]) -> None:
        """Check if we have minimum required data points."""
        actual_count = len(df)
        is_sufficient = actual_count >= min_required
        
        results['checks']['minimum_data_points'] = {
            'required': min_required,
            'actual': actual_count,
            'passed': is_sufficient,
            'score': 100.0 if is_sufficient else 0.0
        }
        
        if not is_sufficient:
            error_msg = f"Insufficient data: {actual_count} < {min_required} required"
            results['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
    
    def _check_missing_data(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Check percentage of missing data."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        max_allowed = self.thresholds['max_missing_data_pct']
        is_acceptable = missing_pct <= max_allowed
        
        results['checks']['missing_data'] = {
            'missing_percentage': round(missing_pct, 2),
            'max_allowed': max_allowed,
            'passed': is_acceptable,
            'score': max(0, 100 - (missing_pct / max_allowed * 100))
        }
        
        if not is_acceptable:
            warning_msg = f"High missing data: {missing_pct:.2f}% > {max_allowed}% allowed"
            results['warnings'].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
    
    def _check_ohlc_consistency(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate OHLC data consistency (High >= Low, etc.)."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            results['checks']['ohlc_consistency'] = {
                'passed': False,
                'score': 0.0,
                'error': 'Missing OHLC columns'
            }
            return
        
        # Check OHLC relationships
        invalid_conditions = [
            df['high'] < df['low'],      # High < Low
            df['high'] < df['open'],     # High < Open
            df['high'] < df['close'],    # High < Close
            df['low'] > df['open'],      # Low > Open
            df['low'] > df['close'],     # Low > Close
        ]
        
        # Count invalid rows
        invalid_mask = pd.concat(invalid_conditions, axis=1).any(axis=1)
        invalid_count = invalid_mask.sum()
        invalid_pct = (invalid_count / len(df)) * 100 if len(df) > 0 else 0
        
        is_acceptable = invalid_pct <= 1.0  # Allow max 1% invalid OHLC
        
        results['checks']['ohlc_consistency'] = {
            'invalid_rows': invalid_count,
            'invalid_percentage': round(invalid_pct, 2),
            'passed': is_acceptable,
            'score': max(0, 100 - invalid_pct * 10)  # Penalize heavily
        }
        
        if not is_acceptable:
            warning_msg = f"OHLC inconsistency: {invalid_count} rows ({invalid_pct:.2f}%) have invalid relationships"
            results['warnings'].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
    
    def _check_volume_quality(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Check volume data quality."""
        if 'volume' not in df.columns:
            results['checks']['volume_quality'] = {
                'passed': False,
                'score': 0.0,
                'error': 'Missing volume column'
            }
            return
        
        # Calculate volume statistics
        volume_data = df['volume'].dropna()
        
        if len(volume_data) == 0:
            results['checks']['volume_quality'] = {
                'passed': False,
                'score': 0.0,
                'error': 'No valid volume data'
            }
            return
        
        avg_volume = volume_data.mean()
        min_threshold = self.thresholds['min_volume_threshold']
        zero_volume_count = (volume_data == 0).sum()
        zero_volume_pct = (zero_volume_count / len(volume_data)) * 100
        
        # Volume quality checks
        sufficient_avg_volume = avg_volume >= min_threshold
        acceptable_zero_volume = zero_volume_pct <= 5.0  # Max 5% zero volume
        
        overall_passed = sufficient_avg_volume and acceptable_zero_volume
        
        # Calculate score
        volume_score = 50 if sufficient_avg_volume else 0
        zero_volume_penalty = max(0, 50 - (zero_volume_pct * 10))
        total_score = volume_score + zero_volume_penalty
        
        results['checks']['volume_quality'] = {
            'avg_volume': round(avg_volume, 2),
            'min_threshold': min_threshold,
            'zero_volume_percentage': round(zero_volume_pct, 2),
            'passed': overall_passed,
            'score': total_score
        }
        
        if not sufficient_avg_volume:
            warning_msg = f"Low average volume: {avg_volume:.2f} < {min_threshold} threshold"
            results['warnings'].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
        
        if not acceptable_zero_volume:
            warning_msg = f"High zero volume percentage: {zero_volume_pct:.2f}%"
            results['warnings'].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
    
    def _check_time_gaps(self, df: pd.DataFrame, interval: str, results: Dict[str, Any]) -> None:
        """Check for large time gaps in data."""
        if 'open_time' not in df.columns:
            results['checks']['time_gaps'] = {
                'passed': False,
                'score': 0.0,
                'error': 'Missing open_time column'
            }
            return
        
        # Get expected time gap for interval
        expected_gap = self._get_expected_time_gap(interval)
        
        # Calculate time differences
        time_diffs = df['open_time'].diff().dropna()
        
        if len(time_diffs) == 0:
            results['checks']['time_gaps'] = {
                'passed': True,
                'score': 100.0,
                'note': 'Single data point, no gaps to check'
            }
            return
        
        # Find large gaps (more than 2x expected gap)
        large_gaps = time_diffs > (expected_gap * 2)
        large_gap_count = large_gaps.sum()
        large_gap_pct = (large_gap_count / len(time_diffs)) * 100
        
        max_allowed_gap_pct = self.thresholds['max_time_gaps_pct']
        is_acceptable = large_gap_pct <= max_allowed_gap_pct
        
        results['checks']['time_gaps'] = {
            'expected_gap_minutes': expected_gap.total_seconds() / 60,
            'large_gaps_count': large_gap_count,
            'large_gaps_percentage': round(large_gap_pct, 2),
            'max_allowed_percentage': max_allowed_gap_pct,
            'passed': is_acceptable,
            'score': max(0, 100 - (large_gap_pct / max_allowed_gap_pct * 50))
        }
        
        if not is_acceptable:
            warning_msg = f"Large time gaps: {large_gap_count} gaps ({large_gap_pct:.2f}%)"
            results['warnings'].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
    
    def _check_extreme_price_movements(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Check for extreme price movements that might indicate data errors."""
        if 'close' not in df.columns:
            results['checks']['extreme_movements'] = {
                'passed': False,
                'score': 0.0,
                'error': 'Missing close price column'
            }
            return
        
        # Calculate price change percentages
        price_changes = df['close'].pct_change(fill_method=None).abs().dropna()
        
        if len(price_changes) == 0:
            results['checks']['extreme_movements'] = {
                'passed': True,
                'score': 100.0,
                'note': 'No price changes to analyze'
            }
            return
        
        # Find extreme movements (>10% price change)
        extreme_threshold = 0.10  # 10%
        extreme_moves = price_changes > extreme_threshold
        extreme_count = extreme_moves.sum()
        extreme_pct = (extreme_count / len(price_changes)) * 100
        
        max_allowed_extreme_pct = self.thresholds['max_extreme_price_moves_pct']
        is_acceptable = extreme_pct <= max_allowed_extreme_pct
        
        results['checks']['extreme_movements'] = {
            'extreme_threshold_pct': extreme_threshold * 100,
            'extreme_moves_count': extreme_count,
            'extreme_moves_percentage': round(extreme_pct, 2),
            'max_allowed_percentage': max_allowed_extreme_pct,
            'max_price_change_pct': round(price_changes.max() * 100, 2),
            'passed': is_acceptable,
            'score': max(0, 100 - (extreme_pct / max_allowed_extreme_pct * 30))
        }
        
        if not is_acceptable:
            warning_msg = f"Extreme price movements: {extreme_count} moves ({extreme_pct:.2f}%) > {extreme_threshold*100}%"
            results['warnings'].append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
    
    def _get_expected_time_gap(self, interval: str) -> timedelta:
        """Get expected time gap between consecutive data points."""
        interval_map = {
            '1m': timedelta(minutes=1),
            '3m': timedelta(minutes=3),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
        }
        
        return interval_map.get(interval, timedelta(minutes=5))  # Default to 5m
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        checks = results['checks']
        
        if not checks:
            return 0.0
        
        # Weight different checks by importance
        weights = {
            'minimum_data_points': 0.30,  # Critical
            'missing_data': 0.20,
            'ohlc_consistency': 0.25,     # Critical for trading data
            'volume_quality': 0.15,
            'time_gaps': 0.05,
            'extreme_movements': 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_name, weight in weights.items():
            if check_name in checks and 'score' in checks[check_name]:
                score = checks[check_name]['score']
                total_score += score * weight
                total_weight += weight
        
        # Normalize score
        final_score = (total_score / total_weight) if total_weight > 0 else 0.0
        
        return round(final_score, 1)
    
    def _determine_validity(self, results: Dict[str, Any]) -> bool:
        """Determine if data quality is acceptable for testing."""
        quality_score = results['quality_score']
        min_required_score = self.thresholds['min_data_quality_score']
        
        # Must have minimum quality score
        if quality_score < min_required_score:
            return False
        
        # Critical checks must pass
        critical_checks = ['minimum_data_points', 'ohlc_consistency']
        
        for check_name in critical_checks:
            if check_name in results['checks']:
                if not results['checks'][check_name].get('passed', False):
                    return False
        
        # Must have no critical errors
        if results['errors']:
            return False
        
        return True
    
    def _log_validation_summary(self, results: Dict[str, Any]) -> None:
        """Log validation summary."""
        symbol = results['symbol']
        interval = results['interval']
        quality_score = results['quality_score']
        is_valid = results['is_valid']
        
        status_emoji = "✅" if is_valid else "❌"
        status_text = "PASSED" if is_valid else "FAILED"
        
        logger.info(f"{status_emoji} Data validation {status_text} for {symbol} {interval}")
        logger.info(f"📊 Quality score: {quality_score}/100")
        logger.info(f"📋 Total rows: {results['total_rows']}")
        
        if results['warnings']:
            logger.info(f"⚠️ Warnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                logger.info(f"   • {warning}")
        
        if results['errors']:
            logger.info(f"❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                logger.info(f"   • {error}")


if __name__ == "__main__":
    """Test the data validator module."""
    
    print("🧪 Testing DataValidator...")
    
    # Create sample data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate test data
    start_time = datetime.now() - timedelta(days=7)
    time_range = pd.date_range(start=start_time, periods=2000, freq='5min')
    
    # Create realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.01, len(time_range))
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    test_df = pd.DataFrame({
        'open_time': time_range,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, len(prices))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(prices))),
        'close': prices * (1 + np.random.normal(0, 0.005, len(prices))),
        'volume': np.random.uniform(1000, 10000, len(prices))
    })
    
    # Ensure OHLC consistency
    test_df['high'] = test_df[['open', 'high', 'close']].max(axis=1)
    test_df['low'] = test_df[['open', 'low', 'close']].min(axis=1)
    
    # Test validator
    try:
        validator = DataValidator()
        
        # Test with good data
        print("🔍 Testing with good quality data...")
        results = validator.validate_data_quality(test_df, "ETHUSDT", "5m", 1000)
        print(f"✅ Validation passed: Quality score {results['quality_score']}/100")
        
        # Test with problematic data
        print("🔍 Testing with problematic data...")
        bad_df = test_df.copy()
        
        # Introduce data quality issues
        bad_df.loc[100:200, 'close'] = np.nan  # Missing data
        bad_df.loc[300, 'high'] = bad_df.loc[300, 'low'] - 10  # Invalid OHLC
        bad_df.loc[400:420, 'volume'] = 0  # Zero volume
        
        try:
            bad_results = validator.validate_data_quality(bad_df, "TESTCOIN", "5m", 1000)
            print(f"⚠️ Validation completed with warnings: Quality score {bad_results['quality_score']}/100")
        except ValueError as e:
            print(f"❌ Validation failed as expected: {e}")
        
    except Exception as e:
        print(f"❌ DataValidator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 DataValidator test completed!")