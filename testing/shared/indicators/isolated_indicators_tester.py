"""
Indicators Isolated Tester - UPDATED MODULAR VERSION
Lightweight coordinator using modular components for comprehensive testing
"""
import sys
import os
import pandas as pd
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import testing framework components
from testing.shared.data_pipeline import TestDataPipeline
from signal_engine.indicators import registry
from signal_engine.signal_indicator_plugin_system import IndicatorManager

# Import modular components
from testing.shared.database import db_connection, schema_manager, result_writer
from testing.shared.utils import (
    test_id_manager, generate_execution_id, generate_test_id, 
    PerformanceAnalyzer, analyze_performance
)
from testing.shared.export import CSVExporter, JSONExporter

# Import modular signal conditions handler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from signal_conditions import SignalConditionsManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModularIndicatorTester:
    """
    Modular indicator testing coordinator.
    Uses shared components for clean separation of concerns.
    """
    
    def __init__(self):
        """Initialize modular tester with all components."""
        logger.info("🔧 Initializing Modular Indicators Tester...")
        
        # Core components
        self.data_pipeline = TestDataPipeline()
        self.indicator_manager = IndicatorManager(registry)
        self.signal_conditions = SignalConditionsManager()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Export components
        self.csv_exporter = CSVExporter()
        self.json_exporter = JSONExporter()
        
        # Database setup
        self._ensure_database_ready()
        
        # Test tracking
        self.current_execution_id = None
        self.test_results = []
        self.performance_data = {}
        
        # Statistics
        self.start_time = None
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        
        logger.info("✅ Modular tester initialized successfully")
    
    def run_comprehensive_test(self, test_type: str = "indicators") -> Dict[str, Any]:
        """
        Run comprehensive isolated tests across all available configurations.
        
        Args:
            test_type: Type of tests to run
            
        Returns:
            Comprehensive test results
        """
        logger.info(f"🚀 Starting comprehensive {test_type} testing...")
        
        # Generate execution ID
        self.current_execution_id = generate_execution_id(test_type)
        self.start_time = time.time()
        
        # Initialize execution in database
        execution_data = {
            'execution_id': self.current_execution_id,
            'execution_type': test_type,
            'start_time': datetime.now(),
            'status': 'running'
        }
        result_writer.write_test_execution(execution_data)
        
        try:
            # Get available configurations
            available_configs = self.data_pipeline.get_available_configs()
            logger.info(f"📋 Found {len(available_configs)} configurations to test")
            
            if not available_configs:
                raise ValueError("No test configurations found")
            
            # Run tests for each configuration
            for config_name in available_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing: {config_name}")
                logger.info(f"{'='*60}")
                
                try:
                    result = self._run_single_test(config_name)
                    self.test_results.append(result)
                    
                    if result.get('status') == 'success':
                        self.successful_tests += 1
                        logger.info(f"✅ {config_name}: {result.get('accuracy_pct', 0):.2f}% accuracy")
                    else:
                        self.failed_tests += 1
                        logger.error(f"❌ {config_name}: {result.get('error_message', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"❌ Critical error testing {config_name}: {e}")
                    self.failed_tests += 1
                
                self.total_tests += 1
            
            # Finalize execution
            return self._finalize_execution()
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return self._handle_execution_error(str(e))
    
    def run_single_config_test(self, config_name: str) -> Dict[str, Any]:
        """
        Run test for a single configuration.
        
        Args:
            config_name: Name of configuration to test
            
        Returns:
            Test result dictionary
        """
        logger.info(f"🎯 Running single test: {config_name}")
        
        # Generate execution ID for single test
        if not self.current_execution_id:
            self.current_execution_id = generate_execution_id("single_test")
            self.start_time = time.time()
        
        try:
            result = self._run_single_test(config_name)
            
            # Export results immediately for single test
            if result.get('status') == 'success':
                self._export_single_test_results(result, config_name)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Single test failed for {config_name}: {e}")
            return self._create_error_result(config_name, str(e))
    
    def _run_single_test(self, config_name: str) -> Dict[str, Any]:
        """Run isolated test for a single configuration."""
        test_id = generate_test_id(config_name)
        
        try:
            # Parse config name if it contains phase prefix
            actual_config_name = config_name
            phase_name = None
            
            if ':' in config_name:
                phase_name, actual_config_name = config_name.split(':', 1)
                logger.info(f"🔍 Parsed config: phase='{phase_name}', config='{actual_config_name}'")
            
            # Load configuration with proper phase handling
            try:
                if phase_name:
                    config = self.data_pipeline.config_manager.load_specific_config(actual_config_name, phase_name)
                else:
                    config = self.data_pipeline.load_indicator_config(actual_config_name)
            except ValueError as e:
                logger.error(f"❌ Config loading failed: {e}")
                return self._create_error_result(config_name, f"Config not found: {e}")
            
            # Fetch and prepare test data
            test_data = self.data_pipeline.fetch_test_data(
                symbol=config['symbol'],
                interval=config['interval'],
                force_refresh=False
            )
            
            # Calculate indicator
            indicator_data = self._calculate_indicator(test_data, config)
            
            # Generate signals
            signals_df = self._generate_signals(indicator_data, config)
            
            # Analyze performance
            performance = self.performance_analyzer.analyze_signals_performance(
                signals_df, indicator_data, config
            )
            
            # Store performance data
            self.performance_data[config_name] = performance
            
            # Create test result
            result = self._create_success_result(
                test_id, config_name, config, performance, 
                len(test_data), len(signals_df)
            )
            
            # Write to database
            if db_connection.is_available:
                result_dict = {**result, 'test_execution_id': self.current_execution_id}
                result_writer.write_indicator_results([result_dict])
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Test execution failed for {config_name}: {e}")
            return self._create_error_result(config_name, str(e))
    
    def _calculate_indicator(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calculate indicator using IndicatorManager."""
        indicator_name = config['indicator_name']
        parameters = config.get('parameters', {})
        
        logger.info(f"🔄 Calculating indicator: {indicator_name}")
        
        # Clear existing indicators
        self.indicator_manager._indicators_to_calculate = []
        self.indicator_manager._indicator_params = {}
        
        # Add specific indicator
        self.indicator_manager.add_indicator(indicator_name, parameters)
        
        # Calculate indicators
        result_data = self.indicator_manager.calculate_indicators(data)
        
        logger.info(f"✅ Indicator calculated successfully")
        return result_data
    
    def _generate_signals(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate trading signals using modular signal conditions."""
        logger.info("🎯 Generating trading signals...")
        
        signal_generation = config.get('signal_generation', {})
        buy_conditions = signal_generation.get('buy_conditions', [])
        sell_conditions = signal_generation.get('sell_conditions', [])
        
        signals = []
        
        for i in range(len(data)):
            current_row = data.iloc[i]
            
            # Check buy conditions
            if self.signal_conditions.check_conditions(current_row, buy_conditions):
                signal = {
                    'timestamp': current_row.get('open_time', i),
                    'index': i,
                    'signal_type': 'buy',
                    'price': current_row['close'],
                    'signal_strength': self.signal_conditions.calculate_signal_strength(
                        current_row, buy_conditions
                    )
                }
                signals.append(signal)
            
            # Check sell conditions
            elif self.signal_conditions.check_conditions(current_row, sell_conditions):
                signal = {
                    'timestamp': current_row.get('open_time', i),
                    'index': i,
                    'signal_type': 'sell',
                    'price': current_row['close'],
                    'signal_strength': self.signal_conditions.calculate_signal_strength(
                        current_row, sell_conditions
                    )
                }
                signals.append(signal)
        
        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        logger.info(f"✅ Generated {len(signals_df)} signals")
        
        return signals_df
    
    def _create_success_result(self, test_id: str, config_name: str, config: Dict[str, Any],
                             performance: Dict[str, Any], data_rows: int, signal_count: int) -> Dict[str, Any]:
        """Create standardized success result."""
        basic_metrics = performance.get('basic_metrics', {})
        risk_metrics = performance.get('risk_metrics', {})
        advanced_metrics = performance.get('advanced_metrics', {})
        summary_scores = performance.get('summary_scores', {})
        
        return {
            'test_id': test_id,
            'test_date': datetime.now().isoformat(),
            'config_name': config_name,
            'indicator_name': config['indicator_name'],
            'symbol': config['symbol'],
            'interval': config['interval'],
            'status': 'success',
            'signal_count': signal_count,
            'accuracy_pct': round(summary_scores.get('accuracy_score', 0), 2),
            'sharpe_ratio': round(risk_metrics.get('sharpe_ratio', 0), 3),
            'max_drawdown': round(advanced_metrics.get('max_drawdown_pct', 0), 2),
            'win_rate': round(basic_metrics.get('win_rate', 0), 2),
            'profit_factor': round(basic_metrics.get('profit_factor', 0), 3),
            'total_trades': basic_metrics.get('total_trades', 0),
            'buy_signals': len([s for s in [] if s.get('signal_type') == 'buy']),  # Would need signals data
            'sell_signals': len([s for s in [] if s.get('signal_type') == 'sell']),  # Would need signals data
            'total_periods': data_rows,
            'data_quality_score': 100.0,  # From data pipeline
            'execution_time_ms': 0,  # Would implement timing
            'test_duration_days': config.get('test_settings', {}).get('test_duration_days', 7),
            'parameters': str(config.get('parameters', {})),
            'error_message': None,
            'performance_grade': summary_scores.get('grade', 'F'),
            'overall_score': round(summary_scores.get('overall_score', 0), 1)
        }
    
    def _create_error_result(self, config_name: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        test_id = generate_test_id(config_name)
        
        return {
            'test_id': test_id,
            'test_date': datetime.now().isoformat(),
            'config_name': config_name,
            'status': 'failed',
            'error_message': error_message,
            'signal_count': 0,
            'accuracy_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'total_periods': 0,
            'execution_time_ms': 0,
            'performance_grade': 'F',
            'overall_score': 0.0
        }
    
    def _finalize_execution(self) -> Dict[str, Any]:
        """Finalize test execution and export results."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("FINALIZING TEST EXECUTION")
        logger.info(f"{'='*80}")
        
        # Calculate execution summary
        successful_results = [r for r in self.test_results if r.get('status') == 'success']
        
        execution_summary = {
            'execution_id': self.current_execution_id,
            'end_time': datetime.now(),
            'duration_seconds': round(duration, 2),
            'total_tests': self.total_tests,
            'successful_tests': self.successful_tests,
            'failed_tests': self.failed_tests,
            'overall_success_rate': round((self.successful_tests / self.total_tests * 100), 2) if self.total_tests > 0 else 0,
            'status': 'completed'
        }
        
        if successful_results:
            accuracies = [r.get('accuracy_pct', 0) for r in successful_results]
            sharpe_ratios = [r.get('sharpe_ratio', 0) for r in successful_results]
            
            best_performer = max(successful_results, key=lambda x: x.get('accuracy_pct', 0))
            
            execution_summary.update({
                'best_performer_config': best_performer.get('config_name'),
                'best_performer_accuracy': best_performer.get('accuracy_pct', 0),
                'average_accuracy': round(sum(accuracies) / len(accuracies), 2),
                'average_sharpe': round(sum(sharpe_ratios) / len(sharpe_ratios), 3)
            })
        
        # Update database
        if db_connection.is_available:
            result_writer.write_test_execution(execution_summary)
        
        # Export results
        exported_files = self._export_all_results(execution_summary)
        
        # Log final summary
        self._log_final_summary(execution_summary)
        
        return {
            'execution_summary': execution_summary,
            'test_results': self.test_results,
            'exported_files': exported_files,
            'performance_data': self.performance_data
        }
    
    def _handle_execution_error(self, error_message: str) -> Dict[str, Any]:
        """Handle execution error and cleanup."""
        execution_summary = {
            'execution_id': self.current_execution_id,
            'end_time': datetime.now(),
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'total_tests': self.total_tests,
            'successful_tests': self.successful_tests,
            'failed_tests': self.failed_tests,
            'status': 'failed',
            'error_message': error_message
        }
        
        # Update database
        if db_connection.is_available:
            result_writer.write_test_execution(execution_summary)
        
        logger.error(f"❌ Execution failed: {error_message}")
        
        return {
            'execution_summary': execution_summary,
            'test_results': self.test_results,
            'error': error_message
        }
    
    def _export_all_results(self, execution_summary: Dict[str, Any]) -> List[str]:
        """Export all results in multiple formats."""
        exported_files = []
        
        try:
            logger.info("💾 Exporting test results...")
            
            # Export CSV results
            if self.test_results:
                csv_file = self.csv_exporter.export_test_results(
                    self.test_results, self.current_execution_id
                )
                if csv_file:
                    exported_files.append(csv_file)
                    logger.info(f"📊 CSV results exported: {csv_file}")
            
            # Export JSON dashboard data
            if self.test_results:
                json_file = self.json_exporter.export_web_dashboard_data(
                    self.test_results, self.current_execution_id
                )
                if json_file:
                    exported_files.append(json_file)
                    logger.info(f"🎯 Dashboard JSON exported: {json_file}")
            
            # Export execution summary
            summary_csv = self.csv_exporter.export_execution_summary(execution_summary)
            if summary_csv:
                exported_files.append(summary_csv)
                logger.info(f"📋 Summary CSV exported: {summary_csv}")
            
            # Export performance analyses
            for config_name, performance in self.performance_data.items():
                perf_csv = self.csv_exporter.export_performance_analysis(
                    performance, config_name
                )
                if perf_csv:
                    exported_files.append(perf_csv)
                
                perf_json = self.json_exporter.export_performance_analysis_json(
                    performance, config_name
                )
                if perf_json:
                    exported_files.append(perf_json)
            
            # Create API manifest
            if exported_files:
                manifest_file = self.json_exporter.create_web_api_manifest(exported_files)
                if manifest_file:
                    exported_files.append(manifest_file)
            
            logger.info(f"✅ Export completed: {len(exported_files)} files")
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
        
        return exported_files
    
    def _export_single_test_results(self, result: Dict[str, Any], config_name: str):
        """Export results for single test."""
        try:
            # Export basic CSV
            csv_file = self.csv_exporter.export_test_results([result], f"single_{config_name}")
            
            # Export JSON for dashboard
            json_file = self.json_exporter.export_web_dashboard_data([result], f"single_{config_name}")
            
            logger.info(f"💾 Single test results exported: CSV={csv_file}, JSON={json_file}")
            
        except Exception as e:
            logger.error(f"❌ Single test export failed: {e}")
    
    def _ensure_database_ready(self):
        """Ensure database is ready for testing."""
        if db_connection.is_available:
            # Validate and create schema if needed
            validation = schema_manager.validate_schema()
            
            if not validation['valid']:
                logger.info("🏗️ Creating database schema...")
                schema_success = schema_manager.create_all_tables()
                
                if schema_success:
                    logger.info("✅ Database schema ready")
                else:
                    logger.warning("⚠️ Database schema creation failed")
            else:
                logger.info("✅ Database schema validated")
        else:
            logger.warning("⚠️ Database not available - results will only be exported to files")
    
    def _log_final_summary(self, execution_summary: Dict[str, Any]):
        """Log comprehensive final summary."""
        logger.info(f"\n{'='*80}")
        logger.info("EXECUTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"🆔 Execution ID: {execution_summary['execution_id']}")
        logger.info(f"⏱️ Duration: {execution_summary['duration_seconds']:.1f} seconds")
        logger.info(f"📊 Total Tests: {execution_summary['total_tests']}")
        logger.info(f"✅ Successful: {execution_summary['successful_tests']}")
        logger.info(f"❌ Failed: {execution_summary['failed_tests']}")
        logger.info(f"🎯 Success Rate: {execution_summary['overall_success_rate']:.1f}%")
        
        if execution_summary.get('best_performer_config'):
            logger.info(f"🏆 Best Performer: {execution_summary['best_performer_config']}")
            logger.info(f"   📈 Accuracy: {execution_summary['best_performer_accuracy']:.2f}%")
        
        if execution_summary.get('average_accuracy'):
            logger.info(f"📊 Average Accuracy: {execution_summary['average_accuracy']:.2f}%")
            logger.info(f"📊 Average Sharpe: {execution_summary['average_sharpe']:.3f}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get current execution statistics."""
        return {
            'execution_id': self.current_execution_id,
            'total_tests': self.total_tests,
            'successful_tests': self.successful_tests,
            'failed_tests': self.failed_tests,
            'success_rate': round((self.successful_tests / self.total_tests * 100), 2) if self.total_tests > 0 else 0,
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'results_count': len(self.test_results),
            'performance_analyses': len(self.performance_data)
        }


# Convenience functions for easy usage
def run_comprehensive_test() -> Dict[str, Any]:
    """
    Convenience function to run comprehensive indicator tests.
    
    Returns:
        Comprehensive test results
    """
    tester = ModularIndicatorTester()
    return tester.run_comprehensive_test()


def run_single_test(config_name: str) -> Dict[str, Any]:
    """
    Convenience function to run single configuration test.
    
    Args:
        config_name: Configuration name to test
        
    Returns:
        Single test result
    """
    tester = ModularIndicatorTester()
    return tester.run_single_config_test(config_name)


if __name__ == "__main__":
    """Run modular testing when executed directly."""
    
    print("🚀 Starting Modular Indicators Testing...")
    
    try:
        # Create modular tester
        tester = ModularIndicatorTester()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Print summary
        summary = results['execution_summary']
        print(f"\n📊 TEST EXECUTION COMPLETED")
        print(f"   🆔 Execution ID: {summary['execution_id']}")
        print(f"   ⏱️ Duration: {summary['duration_seconds']:.1f}s")
        print(f"   🎯 Success Rate: {summary['overall_success_rate']:.1f}%")
        
        if summary.get('best_performer_config'):
            print(f"   🏆 Best: {summary['best_performer_config']} ({summary['best_performer_accuracy']:.2f}%)")
        
        exported_files = results.get('exported_files', [])
        print(f"   💾 Exported: {len(exported_files)} files")
        
    except Exception as e:
        print(f"❌ Modular testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Modular testing completed!")