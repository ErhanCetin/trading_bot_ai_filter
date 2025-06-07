"""
Failed Tests Debug Script
Analyzes why 9 tests failed and provides solutions
"""
import sys
import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from testing.shared.data_pipeline import TestDataPipeline
from testing.shared.database import db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailedTestsDebugger:
    """Debug failed tests and identify root causes."""
    
    def __init__(self):
        self.data_pipeline = TestDataPipeline()
        self.failed_configs = [
            "base_conditions:macd_crossover",
            "base_conditions:ema_crossover", 
            "base_conditions:adx_trend_strength",
            "advanced_conditions:heikin_ashi_trend",
            "feature_conditions:momentum_divergence",
            "feature_conditions:support_resistance_breakout",
            "regime_conditions:market_regime_trend",
            "regime_conditions:volatility_regime_low",
            "regime_conditions:statistical_outlier"
        ]
    
    def debug_all_failed_tests(self):
        """Debug all failed test configurations."""
        logger.info("🔍 Starting Failed Tests Debug Analysis...")
        
        results = {}
        
        for config_name in self.failed_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Debugging: {config_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = self._debug_single_config(config_name)
                results[config_name] = result
                
            except Exception as e:
                logger.error(f"❌ Debug failed for {config_name}: {e}")
                results[config_name] = {
                    'status': 'debug_failed',
                    'error': str(e)
                }
        
        # Summary analysis
        self._analyze_failure_patterns(results)
        return results
    
    def _debug_single_config(self, config_name: str) -> dict:
        """Debug a single failed configuration."""
        # Parse config name
        if ':' in config_name:
            phase_name, actual_config_name = config_name.split(':', 1)
        else:
            phase_name = None
            actual_config_name = config_name
        
        # Load config
        try:
            if phase_name:
                config = self.data_pipeline.config_manager.load_specific_config(
                    actual_config_name, phase_name
                )
            else:
                config = self.data_pipeline.load_indicator_config(actual_config_name)
                
            logger.info(f"✅ Config loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Config loading failed: {e}")
            return {
                'status': 'config_load_failed',
                'error': str(e),
                'symbol': 'unknown',
                'interval': 'unknown'
            }
        
        symbol = config.get('symbol', 'unknown')
        interval = config.get('interval', 'unknown')
        
        logger.info(f"📊 Symbol: {symbol}, Interval: {interval}")
        
        # Check data availability
        data_status = self._check_data_availability(symbol, interval)
        
        # Check data quality if data exists
        if data_status['has_data']:
            quality_status = self._check_data_quality(symbol, interval)
        else:
            quality_status = {'quality_acceptable': False, 'reason': 'no_data'}
        
        return {
            'status': 'analyzed',
            'symbol': symbol,
            'interval': interval,
            'data_availability': data_status,
            'data_quality': quality_status,
            'recommended_action': self._get_recommended_action(data_status, quality_status)
        }
    
    def _check_data_availability(self, symbol: str, interval: str) -> dict:
        """Check if data is available for symbol/interval."""
        try:
            # Try to fetch data
            data = self.data_pipeline.fetch_test_data(
                symbol=symbol,
                interval=interval,
                force_refresh=False
            )
            
            has_data = len(data) > 0
            row_count = len(data)
            
            if has_data:
                date_range = {
                    'start': data['open_time'].min() if 'open_time' in data.columns else 'unknown',
                    'end': data['open_time'].max() if 'open_time' in data.columns else 'unknown'
                }
            else:
                date_range = None
            
            logger.info(f"📈 Data rows: {row_count}")
            
            return {
                'has_data': has_data,
                'row_count': row_count,
                'date_range': date_range,
                'columns': list(data.columns) if has_data else []
            }
            
        except Exception as e:
            logger.error(f"❌ Data availability check failed: {e}")
            return {
                'has_data': False,
                'error': str(e),
                'row_count': 0
            }
    
    def _check_data_quality(self, symbol: str, interval: str) -> dict:
        """Check data quality for symbol/interval."""
        try:
            # Get data quality metrics
            validation = self.data_pipeline.data_validator.validate_data_quality(
                symbol, interval
            )
            
            logger.info(f"📊 Data quality score: {validation.get('quality_score', 0)}")
            logger.info(f"📊 Quality acceptable: {validation.get('quality_acceptable', False)}")
            
            if not validation.get('quality_acceptable', False):
                issues = validation.get('issues', [])
                logger.warning(f"⚠️ Quality issues: {issues}")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Data quality check failed: {e}")
            return {
                'quality_acceptable': False,
                'error': str(e),
                'quality_score': 0
            }
    
    def _get_recommended_action(self, data_status: dict, quality_status: dict) -> str:
        """Get recommended action based on debug results."""
        if not data_status.get('has_data', False):
            return "FETCH_DATA: Symbol/interval data not available in database"
        
        if data_status.get('row_count', 0) < 100:
            return "INSUFFICIENT_DATA: Less than 100 rows, fetch more historical data"
        
        if not quality_status.get('quality_acceptable', False):
            issues = quality_status.get('issues', [])
            if 'missing_values' in str(issues):
                return "FIX_MISSING_VALUES: Clean missing values in dataset"
            elif 'zero_variance' in str(issues):
                return "FIX_VARIANCE: Price data has zero variance"
            else:
                return f"FIX_QUALITY: Data quality issues - {issues}"
        
        return "UNKNOWN: Data seems OK, investigate test logic"
    
    def _analyze_failure_patterns(self, results: dict):
        """Analyze patterns in failed tests."""
        logger.info(f"\n{'='*80}")
        logger.info("📊 FAILURE PATTERN ANALYSIS")
        logger.info(f"{'='*80}")
        
        # Group by failure type
        failure_types = {}
        symbols = {}
        intervals = {}
        
        for config_name, result in results.items():
            if result.get('status') == 'analyzed':
                # Group by recommended action
                action = result.get('recommended_action', 'unknown')
                if action not in failure_types:
                    failure_types[action] = []
                failure_types[action].append(config_name)
                
                # Count symbols
                symbol = result.get('symbol', 'unknown')
                symbols[symbol] = symbols.get(symbol, 0) + 1
                
                # Count intervals
                interval = result.get('interval', 'unknown')
                intervals[interval] = intervals.get(interval, 0) + 1
        
        # Print analysis
        logger.info("🔍 Failure Types:")
        for failure_type, configs in failure_types.items():
            logger.info(f"   {failure_type}: {len(configs)} tests")
            for config in configs:
                logger.info(f"     - {config}")
        
        logger.info(f"\n📊 Most Problematic Symbols:")
        for symbol, count in sorted(symbols.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {symbol}: {count} failures")
        
        logger.info(f"\n⏰ Most Problematic Intervals:")
        for interval, count in sorted(intervals.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {interval}: {count} failures")
    
    def fix_data_issues(self, symbol: str, interval: str):
        """Fix data issues for specific symbol/interval."""
        logger.info(f"🔧 Attempting to fix data issues for {symbol} {interval}")
        
        try:
            # Force refresh data
            logger.info("📥 Force refreshing data from Binance...")
            data = self.data_pipeline.fetch_test_data(
                symbol=symbol,
                interval=interval,
                force_refresh=True
            )
            
            logger.info(f"✅ Fetched {len(data)} rows")
            
            # Re-validate quality
            validation = self.data_pipeline.data_validator.validate_data_quality(
                symbol, interval
            )
            
            logger.info(f"📊 New quality score: {validation.get('quality_score', 0)}")
            
            if validation.get('quality_acceptable', False):
                logger.info(f"✅ Data quality fixed for {symbol} {interval}")
                return True
            else:
                logger.warning(f"⚠️ Data quality still poor: {validation.get('issues', [])}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fix failed for {symbol} {interval}: {e}")
            return False
    
    def run_comprehensive_debug(self):
        """Run comprehensive debug and attempt fixes."""
        logger.info("🚀 Starting Comprehensive Failed Tests Debug...")
        
        # Debug all failed tests
        results = self.debug_all_failed_tests()
        
        # Attempt fixes for data issues
        logger.info(f"\n{'='*80}")
        logger.info("🔧 ATTEMPTING FIXES")
        logger.info(f"{'='*80}")
        
        fixed_count = 0
        for config_name, result in results.items():
            if result.get('status') == 'analyzed':
                action = result.get('recommended_action', '')
                
                if action.startswith('FETCH_DATA') or action.startswith('FIX_'):
                    symbol = result.get('symbol')
                    interval = result.get('interval')
                    
                    if symbol and interval and symbol != 'unknown':
                        logger.info(f"\n🔧 Fixing {config_name} ({symbol} {interval})")
                        
                        if self.fix_data_issues(symbol, interval):
                            fixed_count += 1
        
        logger.info(f"\n📊 SUMMARY: Fixed {fixed_count} out of {len(results)} failed tests")
        
        return results


def main():
    """Main debug function."""
    debugger = FailedTestsDebugger()
    
    # Run comprehensive debug
    results = debugger.run_comprehensive_debug()
    
    print(f"\n🎯 Debug completed. Check logs for detailed analysis.")
    return results


if __name__ == "__main__":
    main()