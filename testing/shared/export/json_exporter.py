"""
JSON Export Operations - Web UI & API Integration
Handles JSON export functionality optimized for web dashboards and API consumption
"""
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class JSONExporter:
    """
    JSON export functionality optimized for web UI and API integration.
    Handles structured data export with proper formatting for dashboard consumption.
    """
    
    def __init__(self, base_export_path: str = "testing/results/json/"):
        """
        Initialize JSON exporter.
        
        Args:
            base_export_path: Base directory for JSON exports
        """
        self.base_export_path = Path(base_export_path)
        self.base_export_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"🔧 JSONExporter initialized: {self.base_export_path}")
    
    def export_web_dashboard_data(self, results: List[Dict[str, Any]], 
                                execution_id: str, filename: Optional[str] = None) -> str:
        """
        Export data optimized for web dashboard consumption.
        
        Args:
            results: List of test result dictionaries
            execution_id: Test execution ID
            filename: Optional custom filename
            
        Returns:
            Path to exported JSON file
        """
        if not results:
            logger.warning("⚠️ No results to export for web dashboard")
            return ""
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dashboard_data_{execution_id}_{timestamp}.json"
            
            file_path = self.base_export_path / filename
            
            # Structure data for web dashboard
            dashboard_data = {
                'metadata': {
                    'execution_id': execution_id,
                    'export_timestamp': datetime.now().isoformat(),
                    'total_tests': len(results),
                    'data_version': '1.0.0'
                },
                'summary': self._create_execution_summary(results),
                'test_results': self._format_results_for_dashboard(results),
                'charts': self._generate_chart_data(results),
                'filters': self._generate_filter_options(results),
                'performance_rankings': self._create_performance_rankings(results)
            }
            
            # Write JSON with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False, 
                         default=self._json_serializer)
            
            logger.info(f"💾 Web dashboard data exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Web dashboard JSON export failed: {e}")
            return ""
    
    def export_api_response_format(self, data: Dict[str, Any], 
                                 endpoint_name: str, filename: Optional[str] = None) -> str:
        """
        Export data in API response format for backend integration.
        
        Args:
            data: Data to export
            endpoint_name: API endpoint name for context
            filename: Optional custom filename
            
        Returns:
            Path to exported JSON file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"api_response_{endpoint_name}_{timestamp}.json"
            
            file_path = self.base_export_path / filename
            
            # Structure as API response
            api_response = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint_name,
                'data': data,
                'meta': {
                    'total_count': len(data) if isinstance(data, list) else 1,
                    'response_time_ms': 0,  # Placeholder
                    'version': '1.0.0'
                }
            }
            
            # Write JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(api_response, f, indent=2, ensure_ascii=False,
                         default=self._json_serializer)
            
            logger.info(f"💾 API response format exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ API format JSON export failed: {e}")
            return ""
    
    def export_performance_analysis_json(self, performance_data: Dict[str, Any], 
                                       config_name: str, filename: Optional[str] = None) -> str:
        """
        Export performance analysis in structured JSON format.
        
        Args:
            performance_data: Performance analysis dictionary
            config_name: Configuration name
            filename: Optional custom filename
            
        Returns:
            Path to exported JSON file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_config = self._clean_name_for_filename(config_name)
                filename = f"performance_{clean_config}_{timestamp}.json"
            
            file_path = self.base_export_path / filename
            
            # Structure performance data with metadata
            structured_data = {
                'config_info': {
                    'config_name': config_name,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data_version': '1.0.0'
                },
                'performance_analysis': performance_data,
                'summary': self._extract_key_performance_metrics(performance_data),
                'visualizations': self._generate_performance_chart_configs(performance_data)
            }
            
            # Write JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False,
                         default=self._json_serializer)
            
            logger.info(f"💾 Performance analysis JSON exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Performance analysis JSON export failed: {e}")
            return ""
    
    def export_chart_data(self, chart_configs: List[Dict[str, Any]], 
                        execution_id: str, filename: Optional[str] = None) -> str:
        """
        Export chart configuration data for web dashboard.
        
        Args:
            chart_configs: List of chart configuration dictionaries
            execution_id: Test execution ID
            filename: Optional custom filename
            
        Returns:
            Path to exported JSON file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chart_data_{execution_id}_{timestamp}.json"
            
            file_path = self.base_export_path / filename
            
            # Structure chart data
            chart_data = {
                'metadata': {
                    'execution_id': execution_id,
                    'export_timestamp': datetime.now().isoformat(),
                    'total_charts': len(chart_configs),
                    'chart_library': 'plotly'  # Default recommendation
                },
                'charts': chart_configs,
                'dashboard_layout': self._generate_dashboard_layout(chart_configs)
            }
            
            # Write JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chart_data, f, indent=2, ensure_ascii=False,
                         default=self._json_serializer)
            
            logger.info(f"💾 Chart data exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Chart data JSON export failed: {e}")
            return ""
    
    def export_real_time_data(self, data: Dict[str, Any], data_type: str) -> str:
        """
        Export real-time data for live dashboard updates.
        
        Args:
            data: Real-time data to export
            data_type: Type of data (e.g., 'test_progress', 'live_metrics')
            
        Returns:
            Path to exported JSON file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"realtime_{data_type}_{timestamp}.json"
            file_path = self.base_export_path / filename
            
            # Structure real-time data
            realtime_data = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'data': data,
                'ttl_seconds': 60  # Time to live for cache
            }
            
            # Write JSON (compact format for real-time)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(realtime_data, f, separators=(',', ':'), 
                         default=self._json_serializer)
            
            logger.debug(f"📡 Real-time data exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Real-time data JSON export failed: {e}")
            return ""
    
    def _create_execution_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create execution summary from results."""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'total_tests': len(results),
                'successful_tests': 0,
                'success_rate': 0.0,
                'average_accuracy': 0.0,
                'best_performer': None
            }
        
        accuracies = [r.get('accuracy_pct', 0) for r in successful_results if r.get('accuracy_pct')]
        
        # Find best performer
        best_performer = max(successful_results, key=lambda x: x.get('accuracy_pct', 0))
        
        return {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'failed_tests': len(results) - len(successful_results),
            'success_rate': round((len(successful_results) / len(results)) * 100, 2),
            'average_accuracy': round(sum(accuracies) / len(accuracies), 2) if accuracies else 0,
            'best_accuracy': round(max(accuracies), 2) if accuracies else 0,
            'worst_accuracy': round(min(accuracies), 2) if accuracies else 0,
            'best_performer': {
                'config_name': best_performer.get('config_name', 'unknown'),
                'accuracy_pct': best_performer.get('accuracy_pct', 0),
                'sharpe_ratio': best_performer.get('sharpe_ratio', 0)
            }
        }
    
    def _format_results_for_dashboard(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format results optimized for dashboard display."""
        formatted_results = []
        
        for result in results:
            formatted_result = {
                'id': result.get('test_id', ''),
                'config_name': result.get('config_name', ''),
                'indicator': result.get('indicator_name', ''),
                'symbol': result.get('symbol', ''),
                'interval': result.get('interval', ''),
                'status': result.get('status', 'unknown'),
                'metrics': {
                    'accuracy': result.get('accuracy_pct', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'profit_factor': result.get('profit_factor', 0),
                    'total_trades': result.get('total_trades', 0)
                },
                'performance_grade': self._calculate_performance_grade(result),
                'test_date': result.get('test_date', datetime.now().isoformat()),
                'execution_time': result.get('execution_time_ms', 0)
            }
            
            # Add error information if failed
            if result.get('status') == 'failed':
                formatted_result['error'] = result.get('error_message', 'Unknown error')
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _generate_chart_data(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate chart data configurations for dashboard."""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {}
        
        charts = {
            'accuracy_distribution': {
                'type': 'histogram',
                'title': 'Accuracy Distribution',
                'data': {
                    'values': [r.get('accuracy_pct', 0) for r in successful_results],
                    'bins': 10
                },
                'layout': {
                    'xaxis': {'title': 'Accuracy (%)'},
                    'yaxis': {'title': 'Frequency'}
                }
            },
            'performance_scatter': {
                'type': 'scatter',
                'title': 'Risk vs Return',
                'data': {
                    'x': [r.get('max_drawdown', 0) for r in successful_results],
                    'y': [r.get('accuracy_pct', 0) for r in successful_results],
                    'text': [r.get('config_name', '') for r in successful_results],
                    'mode': 'markers+text'
                },
                'layout': {
                    'xaxis': {'title': 'Max Drawdown (%)'},
                    'yaxis': {'title': 'Accuracy (%)'}
                }
            },
            'top_performers': {
                'type': 'bar',
                'title': 'Top 10 Performers',
                'data': self._get_top_performers_chart_data(successful_results)
            }
        }
        
        return charts
    
    def _generate_filter_options(self, results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate filter options for dashboard."""
        return {
            'indicators': list(set(r.get('indicator_name', '') for r in results)),
            'symbols': list(set(r.get('symbol', '') for r in results)),
            'intervals': list(set(r.get('interval', '') for r in results)),
            'statuses': list(set(r.get('status', '') for r in results)),
            'phases': list(set(r.get('phase_name', '') for r in results if r.get('phase_name')))
        }
    
    def _create_performance_rankings(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create performance rankings for different metrics."""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {}
        
        rankings = {}
        
        # Top performers by accuracy
        rankings['by_accuracy'] = sorted(
            successful_results, 
            key=lambda x: x.get('accuracy_pct', 0), 
            reverse=True
        )[:10]
        
        # Top performers by Sharpe ratio
        rankings['by_sharpe'] = sorted(
            successful_results, 
            key=lambda x: x.get('sharpe_ratio', 0), 
            reverse=True
        )[:10]
        
        # Best risk-adjusted (low drawdown)
        rankings['by_low_drawdown'] = sorted(
            successful_results, 
            key=lambda x: x.get('max_drawdown', 100)
        )[:10]
        
        # Most consistent (high win rate)
        rankings['by_win_rate'] = sorted(
            successful_results, 
            key=lambda x: x.get('win_rate', 0), 
            reverse=True
        )[:10]
        
        return rankings
    
    def _get_top_performers_chart_data(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get chart data for top performers."""
        top_10 = sorted(results, key=lambda x: x.get('accuracy_pct', 0), reverse=True)[:10]
        
        return {
            'x': [r.get('config_name', '')[:20] + '...' if len(r.get('config_name', '')) > 20 
                  else r.get('config_name', '') for r in top_10],
            'y': [r.get('accuracy_pct', 0) for r in top_10],
            'type': 'bar',
            'marker': {'color': 'rgba(55, 128, 191, 0.7)'}
        }
    
    def _calculate_performance_grade(self, result: Dict[str, Any]) -> str:
        """Calculate performance grade for a result."""
        if result.get('status') != 'success':
            return 'F'
        
        accuracy = result.get('accuracy_pct', 0)
        sharpe = result.get('sharpe_ratio', 0)
        drawdown = result.get('max_drawdown', 100)
        
        # Simple scoring algorithm
        score = (accuracy * 0.4) + (sharpe * 20 * 0.3) + (max(0, 20 - drawdown) * 0.3)
        
        if score >= 85:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 75:
            return 'A-'
        elif score >= 70:
            return 'B+'
        elif score >= 65:
            return 'B'
        elif score >= 60:
            return 'B-'
        elif score >= 55:
            return 'C+'
        elif score >= 50:
            return 'C'
        elif score >= 45:
            return 'C-'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def _extract_key_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from performance analysis."""
        basic = performance_data.get('basic_metrics', {})
        risk = performance_data.get('risk_metrics', {})
        advanced = performance_data.get('advanced_metrics', {})
        scores = performance_data.get('summary_scores', {})
        
        return {
            'total_trades': basic.get('total_trades', 0),
            'win_rate': basic.get('win_rate', 0),
            'profit_factor': basic.get('profit_factor', 0),
            'sharpe_ratio': risk.get('sharpe_ratio', 0),
            'max_drawdown': advanced.get('max_drawdown_pct', 0),
            'total_return': advanced.get('total_return_pct', 0),
            'overall_score': scores.get('overall_score', 0),
            'grade': scores.get('grade', 'F')
        }
    
    def _generate_performance_chart_configs(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart configurations for performance visualization."""
        return {
            'metrics_radar': {
                'type': 'radar',
                'title': 'Performance Metrics Radar',
                'data': self._create_radar_chart_data(performance_data)
            },
            'equity_curve': {
                'type': 'line',
                'title': 'Equity Curve',
                'data': self._create_equity_curve_data(performance_data)
            },
            'monthly_returns': {
                'type': 'bar',
                'title': 'Monthly Returns',
                'data': self._create_monthly_returns_data(performance_data)
            }
        }
    
    def _create_radar_chart_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create radar chart data for performance metrics."""
        scores = performance_data.get('summary_scores', {})
        
        return {
            'theta': ['Accuracy', 'Risk Management', 'Profitability', 'Consistency', 'Overall'],
            'r': [
                scores.get('accuracy_score', 0),
                scores.get('risk_adjusted_score', 0),
                scores.get('profitability_score', 0),
                min(100, performance_data.get('basic_metrics', {}).get('win_rate', 0) * 1.2),
                scores.get('overall_score', 0)
            ],
            'fill': 'toself',
            'name': 'Performance Metrics'
        }
    
    def _create_equity_curve_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create equity curve data (placeholder - would need trade data)."""
        # This would typically require individual trade data
        # For now, return a simple placeholder
        return {
            'x': ['Start', 'End'],
            'y': [100, 100 + performance_data.get('advanced_metrics', {}).get('total_return_pct', 0)],
            'mode': 'lines+markers',
            'name': 'Equity Curve'
        }
    
    def _create_monthly_returns_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create monthly returns data (placeholder)."""
        # Placeholder data - would need actual monthly breakdown
        return {
            'x': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'y': [2.5, -1.2, 3.8, 1.9, -0.5, 4.2],
            'type': 'bar',
            'name': 'Monthly Returns (%)'
        }
    
    def _generate_dashboard_layout(self, chart_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate dashboard layout configuration."""
        return {
            'grid_layout': {
                'columns': 2,
                'rows': len(chart_configs) // 2 + 1,
                'gap': 20
            },
            'chart_positions': [
                {
                    'chart_id': i,
                    'row': i // 2,
                    'col': i % 2,
                    'width': 6,
                    'height': 400
                }
                for i in range(len(chart_configs))
            ],
            'responsive_breakpoints': {
                'mobile': 768,
                'tablet': 1024,
                'desktop': 1200
            }
        }
    
    def _clean_name_for_filename(self, name: str) -> str:
        """Clean name to be safe for filename."""
        clean = ''.join(c for c in name if c.isalnum() or c in '_-')
        return clean[:50]
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and pandas objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def create_web_api_manifest(self, exported_files: List[str]) -> str:
        """
        Create a manifest file for web API integration.
        
        Args:
            exported_files: List of exported JSON files
            
        Returns:
            Path to manifest file
        """
        try:
            manifest_file = self.base_export_path / "api_manifest.json"
            
            manifest = {
                'manifest_version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'base_path': str(self.base_export_path),
                'available_endpoints': [],
                'total_files': len(exported_files)
            }
            
            for file_path in exported_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path)
                    
                    # Determine endpoint type from filename
                    endpoint_type = 'unknown'
                    if 'dashboard' in filename:
                        endpoint_type = 'dashboard_data'
                    elif 'performance' in filename:
                        endpoint_type = 'performance_analysis'
                    elif 'chart' in filename:
                        endpoint_type = 'chart_data'
                    elif 'realtime' in filename:
                        endpoint_type = 'realtime_data'
                    
                    manifest['available_endpoints'].append({
                        'filename': filename,
                        'endpoint_type': endpoint_type,
                        'file_path': file_path,
                        'size_bytes': file_size,
                        'content_type': 'application/json'
                    })
            
            # Write manifest
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 API manifest created: {manifest_file}")
            return str(manifest_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to create API manifest: {e}")
            return ""
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about JSON exports."""
        try:
            json_files = list(self.base_export_path.glob("*.json"))
            
            if not json_files:
                return {
                    'export_directory': str(self.base_export_path),
                    'total_files': 0,
                    'total_size_mb': 0,
                    'files_by_type': {}
                }
            
            total_size = sum(f.stat().st_size for f in json_files)
            
            # Categorize files by type
            files_by_type = {}
            for f in json_files:
                if 'dashboard' in f.name:
                    file_type = 'dashboard'
                elif 'performance' in f.name:
                    file_type = 'performance'
                elif 'chart' in f.name:
                    file_type = 'chart'
                elif 'realtime' in f.name:
                    file_type = 'realtime'
                elif 'api_response' in f.name:
                    file_type = 'api_response'
                else:
                    file_type = 'other'
                
                if file_type not in files_by_type:
                    files_by_type[file_type] = []
                
                files_by_type[file_type].append({
                    'filename': f.name,
                    'size_mb': round(f.stat().st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
            
            return {
                'export_directory': str(self.base_export_path),
                'total_files': len(json_files),
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'files_by_type': files_by_type
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting JSON export statistics: {e}")
            return {'error': str(e)}


# Convenience functions for easy importing
def export_dashboard_data(results: List[Dict[str, Any]], execution_id: str,
                         export_path: str = "testing/results/json/") -> str:
    """
    Convenience function to export dashboard data.
    
    Args:
        results: List of test result dictionaries
        execution_id: Test execution ID
        export_path: Export directory path
        
    Returns:
        Path to exported JSON file
    """
    exporter = JSONExporter(export_path)
    return exporter.export_web_dashboard_data(results, execution_id)


def export_api_format(data: Dict[str, Any], endpoint_name: str,
                     export_path: str = "testing/results/json/") -> str:
    """
    Convenience function to export API format data.
    
    Args:
        data: Data to export
        endpoint_name: API endpoint name
        export_path: Export directory path
        
    Returns:
        Path to exported JSON file
    """
    exporter = JSONExporter(export_path)
    return exporter.export_api_response_format(data, endpoint_name)


if __name__ == "__main__":
    """Test the JSON exporter."""
    
    print("🧪 Testing JSONExporter...")
    
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
            'test_date': datetime.now().isoformat()
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
            'test_date': datetime.now().isoformat()
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
            'grade': 'B+',
            'accuracy_score': 85.0,
            'risk_adjusted_score': 72.0,
            'profitability_score': 79.0
        }
    }
    
    try:
        # Test JSON exporter
        exporter = JSONExporter("testing/test_exports/json/")
        
        # Test dashboard data export
        print("🎯 Testing dashboard data export...")
        dashboard_file = exporter.export_web_dashboard_data(test_results, "TEST_EXEC_001")
        print(f"✅ Dashboard data exported: {dashboard_file}")
        
        # Test performance analysis export
        print("📊 Testing performance analysis export...")
        perf_file = exporter.export_performance_analysis_json(performance_data, "rsi_test")
        print(f"✅ Performance analysis exported: {perf_file}")
        
        # Test API response format
        print("🔌 Testing API response format...")
        api_file = exporter.export_api_response_format(test_results, "test_results")
        print(f"✅ API format exported: {api_file}")
        
        # Test real-time data
        print("📡 Testing real-time data export...")
        realtime_data = {'current_tests': 5, 'completed': 3, 'progress': 60}
        realtime_file = exporter.export_real_time_data(realtime_data, "test_progress")
        print(f"✅ Real-time data exported: {realtime_file}")
        
        # Test manifest creation
        print("📋 Testing API manifest creation...")
        exported_files = [dashboard_file, perf_file, api_file, realtime_file]
        manifest_file = exporter.create_web_api_manifest(exported_files)
        print(f"✅ API manifest created: {manifest_file}")
        
        # Test export statistics
        print("📊 Testing export statistics...")
        stats = exporter.get_export_statistics()
        print(f"✅ Export stats: {stats['total_files']} files, {stats['total_size_mb']} MB")
        
        # Test convenience functions
        print("🔧 Testing convenience functions...")
        conv_file = export_dashboard_data(test_results, "CONV_TEST")
        print(f"✅ Convenience export: {conv_file}")
        
    except Exception as e:
        print(f"❌ JSON exporter test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 JSONExporter test completed!")