"""
CSV Export Operations
Handles comprehensive CSV export functionality for test results
"""
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CSVExporter:
    """
    Comprehensive CSV export functionality for testing framework.
    Handles multiple export formats and data transformations.
    """
    
    def __init__(self, base_export_path: str = "testing/results/csv/"):
        """
        Initialize CSV exporter.
        
        Args:
            base_export_path: Base directory for CSV exports
        """
        self.base_export_path = Path(base_export_path)
        self.base_export_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"🔧 CSVExporter initialized: {self.base_export_path}")
    
    def export_test_results(self, results: List[Dict[str, Any]], 
                           execution_id: str, filename: Optional[str] = None) -> str:
        """
        Export test results to CSV file.
        
        Args:
            results: List of test result dictionaries
            execution_id: Test execution ID
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        if not results:
            logger.warning("⚠️ No results to export")
            return ""
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_results_{execution_id}_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Clean and format data
            df = self._clean_dataframe_for_csv(df)
            
            # Sort by accuracy descending
            if 'accuracy_pct' in df.columns:
                df = df.sort_values('accuracy_pct', ascending=False)
            
            # Save to CSV
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Test results exported to: {file_path}")
            logger.info(f"📊 Exported {len(results)} test results")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            return ""
    
    def export_performance_analysis(self, performance_data: Dict[str, Any], 
                                  config_name: str, filename: Optional[str] = None) -> str:
        """
        Export detailed performance analysis to CSV.
        
        Args:
            performance_data: Performance analysis dictionary
            config_name: Configuration name
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_config = self._clean_name_for_filename(config_name)
                filename = f"performance_analysis_{clean_config}_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Flatten performance data into rows
            rows = []
            
            # Basic metrics
            basic = performance_data.get('basic_metrics', {})
            for metric, value in basic.items():
                rows.append({
                    'category': 'basic_metrics',
                    'metric': metric,
                    'value': value,
                    'description': self._get_metric_description(metric)
                })
            
            # Risk metrics
            risk = performance_data.get('risk_metrics', {})
            for metric, value in risk.items():
                rows.append({
                    'category': 'risk_metrics',
                    'metric': metric,
                    'value': value,
                    'description': self._get_metric_description(metric)
                })
            
            # Advanced metrics
            advanced = performance_data.get('advanced_metrics', {})
            for metric, value in advanced.items():
                rows.append({
                    'category': 'advanced_metrics',
                    'metric': metric,
                    'value': value,
                    'description': self._get_metric_description(metric)
                })
            
            # Summary scores
            scores = performance_data.get('summary_scores', {})
            for metric, value in scores.items():
                rows.append({
                    'category': 'summary_scores',
                    'metric': metric,
                    'value': value,
                    'description': self._get_metric_description(metric)
                })
            
            # Convert to DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Performance analysis exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Performance analysis CSV export failed: {e}")
            return ""
    
    def export_phase_summary(self, phase_summaries: Dict[str, Dict[str, Any]], 
                           execution_id: str, filename: Optional[str] = None) -> str:
        """
        Export phase summaries to CSV.
        
        Args:
            phase_summaries: Dictionary of phase summaries
            execution_id: Test execution ID
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        if not phase_summaries:
            logger.warning("⚠️ No phase summaries to export")
            return ""
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"phase_summary_{execution_id}_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Convert phase summaries to rows
            rows = []
            for phase_name, summary in phase_summaries.items():
                row = {
                    'execution_id': execution_id,
                    'phase_name': phase_name,
                    **summary
                }
                rows.append(row)
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            df = self._clean_dataframe_for_csv(df)
            
            # Sort by success rate descending
            if 'success_rate' in df.columns:
                df = df.sort_values('success_rate', ascending=False)
            
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Phase summary exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Phase summary CSV export failed: {e}")
            return ""
    
    def export_signals_data(self, signals_df: pd.DataFrame, price_data: pd.DataFrame,
                          config_name: str, filename: Optional[str] = None) -> str:
        """
        Export signals and price data to CSV for analysis.
        
        Args:
            signals_df: DataFrame with signals
            price_data: DataFrame with price data
            config_name: Configuration name
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_config = self._clean_name_for_filename(config_name)
                filename = f"signals_data_{clean_config}_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Merge signals with price data
            if 'index' in signals_df.columns:
                # Use index column to merge
                merged_df = price_data.copy()
                merged_df['has_signal'] = False
                merged_df['signal_type'] = 'none'
                merged_df['signal_strength'] = 0.0
                
                for _, signal in signals_df.iterrows():
                    idx = signal['index']
                    if idx < len(merged_df):
                        merged_df.loc[idx, 'has_signal'] = True
                        merged_df.loc[idx, 'signal_type'] = signal.get('signal_type', 'none')
                        merged_df.loc[idx, 'signal_strength'] = signal.get('signal_strength', 0.0)
            else:
                # Just concatenate if no clear merge strategy
                merged_df = price_data.copy()
                
                # Add signal columns
                merged_df['has_signal'] = False
                merged_df['signal_type'] = 'none'
                merged_df['signal_strength'] = 0.0
            
            # Clean and save
            merged_df = self._clean_dataframe_for_csv(merged_df)
            merged_df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Signals data exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Signals data CSV export failed: {e}")
            return ""
    
    def export_trades_analysis(self, trades: List[Dict[str, Any]], 
                             config_name: str, filename: Optional[str] = None) -> str:
        """
        Export individual trades analysis to CSV.
        
        Args:
            trades: List of trade dictionaries
            config_name: Configuration name
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        if not trades:
            logger.warning("⚠️ No trades to export")
            return ""
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_config = self._clean_name_for_filename(config_name)
                filename = f"trades_analysis_{clean_config}_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Convert trades to DataFrame
            df = pd.DataFrame(trades)
            
            # Add calculated columns
            if 'pnl_pct' in df.columns:
                df['cumulative_pnl_pct'] = df['pnl_pct'].cumsum()
            
            if 'exit_capital' in df.columns:
                df['capital_growth_pct'] = ((df['exit_capital'] / df['exit_capital'].iloc[0]) - 1) * 100
            
            # Clean and save
            df = self._clean_dataframe_for_csv(df)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Trades analysis exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Trades analysis CSV export failed: {e}")
            return ""
    
    def export_comparative_analysis(self, results_by_config: Dict[str, Dict[str, Any]], 
                                  filename: Optional[str] = None) -> str:
        """
        Export comparative analysis across multiple configurations.
        
        Args:
            results_by_config: Dictionary mapping config names to results
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        if not results_by_config:
            logger.warning("⚠️ No comparative data to export")
            return ""
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparative_analysis_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Extract key metrics for comparison
            comparison_rows = []
            
            for config_name, result in results_by_config.items():
                row = {
                    'config_name': config_name,
                    'indicator_name': result.get('indicator_name', 'unknown'),
                    'symbol': result.get('symbol', 'unknown'),
                    'interval': result.get('interval', 'unknown'),
                    'status': result.get('status', 'unknown'),
                    'accuracy_pct': result.get('accuracy_pct', 0.0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0.0),
                    'max_drawdown': result.get('max_drawdown', 0.0),
                    'win_rate': result.get('win_rate', 0.0),
                    'profit_factor': result.get('profit_factor', 0.0),
                    'total_trades': result.get('total_trades', 0),
                    'signal_count': result.get('signal_count', 0),
                    'execution_time_ms': result.get('execution_time_ms', 0),
                    'error_message': result.get('error_message', '')
                }
                comparison_rows.append(row)
            
            # Convert to DataFrame and sort by accuracy
            df = pd.DataFrame(comparison_rows)
            df = self._clean_dataframe_for_csv(df)
            
            if 'accuracy_pct' in df.columns:
                df = df.sort_values('accuracy_pct', ascending=False)
            
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Comparative analysis exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Comparative analysis CSV export failed: {e}")
            return ""
    
    def _clean_dataframe_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for CSV export."""
        # Make a copy to avoid modifying original
        clean_df = df.copy()
        
        # Handle NaN values
        clean_df = clean_df.fillna('')
        
        # Round numeric columns to reasonable precision
        numeric_columns = clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col.endswith('_pct') or col.endswith('_ratio'):
                clean_df[col] = clean_df[col].round(3)
            elif col.endswith('_ms') or col.endswith('_count'):
                clean_df[col] = clean_df[col].round(0).astype(int)
            else:
                clean_df[col] = clean_df[col].round(6)
        
        # Convert datetime columns to string
        datetime_columns = clean_df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_columns:
            clean_df[col] = clean_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle list/dict columns by converting to string
        for col in clean_df.columns:
            if clean_df[col].dtype == 'object':
                clean_df[col] = clean_df[col].astype(str)
        
        return clean_df
    
    def _clean_name_for_filename(self, name: str) -> str:
        """Clean name to be safe for filename."""
        # Remove special characters and limit length
        clean = ''.join(c for c in name if c.isalnum() or c in '_-')
        return clean[:50]  # Limit length
    
    def _get_metric_description(self, metric: str) -> str:
        """Get description for a metric."""
        descriptions = {
            'total_trades': 'Total number of completed trades',
            'winning_trades': 'Number of profitable trades',
            'losing_trades': 'Number of losing trades', 
            'win_rate': 'Percentage of winning trades',
            'profit_factor': 'Gross profit divided by gross loss',
            'sharpe_ratio': 'Risk-adjusted return measure',
            'max_drawdown_pct': 'Maximum peak-to-trough decline',
            'total_return_pct': 'Total percentage return',
            'avg_trade_pnl': 'Average profit/loss per trade',
            'volatility': 'Standard deviation of returns',
            'sortino_ratio': 'Return relative to downside deviation',
            'calmar_ratio': 'Annual return over maximum drawdown',
            'recovery_factor': 'Net profit over maximum drawdown',
            'overall_score': 'Composite performance score (0-100)',
            'grade': 'Letter grade based on overall score',
            'signal_accuracy_pct': 'Percentage of accurate signals'
        }
        
        return descriptions.get(metric, 'Performance metric')
    
    def create_export_summary(self, exported_files: List[str]) -> Dict[str, Any]:
        """
        Create summary of exported files.
        
        Args:
            exported_files: List of exported file paths
            
        Returns:
            Export summary dictionary
        """
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_files_exported': len(exported_files),
            'export_directory': str(self.base_export_path),
            'exported_files': [],
            'total_size_bytes': 0
        }
        
        for file_path in exported_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                summary['exported_files'].append({
                    'filename': os.path.basename(file_path),
                    'full_path': file_path,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / 1024 / 1024, 2)
                })
                summary['total_size_bytes'] += file_size
        
        summary['total_size_mb'] = round(summary['total_size_bytes'] / 1024 / 1024, 2)
        
        return summary
    
    def export_execution_summary(self, execution_data: Dict[str, Any], 
                               filename: Optional[str] = None) -> str:
        """
        Export execution summary to CSV.
        
        Args:
            execution_data: Execution summary data
            filename: Optional custom filename
            
        Returns:
            Path to exported CSV file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                execution_id = execution_data.get('execution_id', 'unknown')
                filename = f"execution_summary_{execution_id}_{timestamp}.csv"
            
            file_path = self.base_export_path / filename
            
            # Convert execution data to rows
            rows = []
            
            # Flatten execution data
            for key, value in execution_data.items():
                if isinstance(value, (dict, list)):
                    value = str(value)
                
                rows.append({
                    'property': key,
                    'value': value,
                    'data_type': type(value).__name__
                })
            
            # Convert to DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            logger.info(f"💾 Execution summary exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Execution summary CSV export failed: {e}")
            return ""
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about CSV exports."""
        try:
            csv_files = list(self.base_export_path.glob("*.csv"))
            
            if not csv_files:
                return {
                    'export_directory': str(self.base_export_path),
                    'total_files': 0,
                    'total_size_mb': 0,
                    'files': []
                }
            
            total_size = sum(f.stat().st_size for f in csv_files)
            
            file_info = []
            for f in csv_files:
                stat = f.stat()
                file_info.append({
                    'filename': f.name,
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # Sort by modification time, newest first
            file_info.sort(key=lambda x: x['modified'], reverse=True)
            
            return {
                'export_directory': str(self.base_export_path),
                'total_files': len(csv_files),
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'files': file_info[:10]  # Show only latest 10 files
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting export statistics: {e}")
            return {'error': str(e)}


# Convenience functions for easy importing
def export_test_results(results: List[Dict[str, Any]], execution_id: str, 
                       export_path: str = "testing/results/csv/") -> str:
    """
    Convenience function to export test results.
    
    Args:
        results: List of test result dictionaries
        execution_id: Test execution ID
        export_path: Export directory path
        
    Returns:
        Path to exported CSV file
    """
    exporter = CSVExporter(export_path)
    return exporter.export_test_results(results, execution_id)


def export_performance_analysis(performance_data: Dict[str, Any], config_name: str,
                               export_path: str = "testing/results/csv/") -> str:
    """
    Convenience function to export performance analysis.
    
    Args:
        performance_data: Performance analysis dictionary
        config_name: Configuration name
        export_path: Export directory path
        
    Returns:
        Path to exported CSV file
    """
    exporter = CSVExporter(export_path)
    return exporter.export_performance_analysis(performance_data, config_name)


if __name__ == "__main__":
    """Test the CSV exporter."""
    
    print("🧪 Testing CSVExporter...")
    
    # Create test data
    test_results = [
        {
            'test_id': 'TEST_001',
            'config_name': 'rsi_14_oversold',
            'indicator_name': 'rsi',
            'symbol': 'ETHUSDT',
            'interval': '5m',
            'status': 'success',
            'accuracy_pct': 72.5,
            'sharpe_ratio': 1.45,
            'max_drawdown': 8.2,
            'win_rate': 68.0,
            'profit_factor': 1.85,
            'total_trades': 25,
            'signal_count': 30
        },
        {
            'test_id': 'TEST_002',
            'config_name': 'macd_crossover',
            'indicator_name': 'macd',
            'symbol': 'BTCUSDT',
            'interval': '15m',
            'status': 'success',
            'accuracy_pct': 65.3,
            'sharpe_ratio': 1.12,
            'max_drawdown': 12.1,
            'win_rate': 61.0,
            'profit_factor': 1.52,
            'total_trades': 18,
            'signal_count': 22
        }
    ]
    
    performance_data = {
        'basic_metrics': {
            'total_trades': 25,
            'win_rate': 68.0,
            'profit_factor': 1.85
        },
        'risk_metrics': {
            'sharpe_ratio': 1.45,
            'volatility': 0.023
        },
        'advanced_metrics': {
            'max_drawdown_pct': 8.2,
            'total_return_pct': 15.7
        },
        'summary_scores': {
            'overall_score': 78.5,
            'grade': 'B+'
        }
    }
    
    try:
        # Test CSV exporter
        exporter = CSVExporter("testing/test_exports/")
        
        # Test results export
        print("📊 Testing test results export...")
        results_file = exporter.export_test_results(test_results, "TEST_EXEC_001")
        print(f"✅ Test results exported: {results_file}")
        
        # Test performance analysis export
        print("📈 Testing performance analysis export...")
        perf_file = exporter.export_performance_analysis(performance_data, "rsi_test")
        print(f"✅ Performance analysis exported: {perf_file}")
        
        # Test comparative analysis
        print("🔍 Testing comparative analysis...")
        comparison_data = {
            'rsi_14_oversold': test_results[0],
            'macd_crossover': test_results[1]
        }
        comp_file = exporter.export_comparative_analysis(comparison_data)
        print(f"✅ Comparative analysis exported: {comp_file}")
        
        # Test export statistics
        print("📊 Testing export statistics...")
        stats = exporter.get_export_statistics()
        print(f"✅ Export stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
        
        # Test convenience functions
        print("🔧 Testing convenience functions...")
        conv_file = export_test_results(test_results, "CONV_TEST")
        print(f"✅ Convenience export: {conv_file}")
        
    except Exception as e:
        print(f"❌ CSV exporter test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 CSVExporter test completed!")