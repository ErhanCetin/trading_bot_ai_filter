"""
Unified Results Exporter
Creates comprehensive single-file exports for analysis
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def export_unified_analysis_report(execution_id: str, output_format: str = 'json'):
    """
    Export comprehensive unified analysis report.
    
    Args:
        execution_id: Test execution ID to analyze
        output_format: 'json' or 'csv'
    """
    
    # Search for result files
    results_dir = Path("testing/results")
    json_dir = results_dir / "json"
    csv_dir = results_dir / "csv"
    
    print(f"🔍 Searching for execution: {execution_id}")
    
    # Find dashboard JSON file
    dashboard_files = list(json_dir.glob(f"dashboard_data_{execution_id}_*.json"))
    
    if not dashboard_files:
        print(f"❌ Dashboard data not found for {execution_id}")
        return
    
    dashboard_file = dashboard_files[0]
    print(f"📊 Found dashboard data: {dashboard_file}")
    
    # Load dashboard data
    with open(dashboard_file, 'r') as f:
        dashboard_data = json.load(f)
    
    # Create unified report
    unified_report = {
        'analysis_metadata': {
            'execution_id': execution_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_tests': dashboard_data['metadata']['total_tests'],
            'success_rate': dashboard_data['summary'].get('success_rate', 0),
            'best_performer': dashboard_data['summary'].get('best_performer', {}),
            'duration_info': f"Analysis generated from {dashboard_data['metadata']['export_timestamp']}"
        },
        
        'performance_summary': {
            'top_performers': dashboard_data.get('performance_rankings', {}).get('by_accuracy', [])[:5],
            'worst_performers': sorted(
                dashboard_data.get('test_results', []), 
                key=lambda x: x.get('metrics', {}).get('accuracy', 0)
            )[:5],
            'risk_analysis': dashboard_data.get('performance_rankings', {}).get('by_low_drawdown', [])[:5],
            'consistency_analysis': dashboard_data.get('performance_rankings', {}).get('by_win_rate', [])[:5]
        },
        
        'detailed_results': dashboard_data.get('test_results', []),
        
        'statistical_analysis': {
            'accuracy_stats': _calculate_accuracy_stats(dashboard_data.get('test_results', [])),
            'risk_stats': _calculate_risk_stats(dashboard_data.get('test_results', [])),
            'indicator_breakdown': _analyze_by_indicator(dashboard_data.get('test_results', [])),
            'symbol_breakdown': _analyze_by_symbol(dashboard_data.get('test_results', [])),
            'interval_breakdown': _analyze_by_interval(dashboard_data.get('test_results', []))
        },
        
        'recommendations': _generate_recommendations(dashboard_data.get('test_results', [])),
        
        'charts_data': dashboard_data.get('charts', {}),
        'filters': dashboard_data.get('filters', {})
    }
    
    # Export unified report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format.lower() == 'json':
        output_file = results_dir / f"unified_analysis_{execution_id}_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unified_report, f, indent=2, ensure_ascii=False)
        print(f"📄 Unified JSON report: {output_file}")
    
    else:  # CSV format
        output_file = results_dir / f"unified_analysis_{execution_id}_{timestamp}.csv"
        
        # Flatten for CSV
        csv_data = []
        for result in unified_report['detailed_results']:
            row = {
                'config_name': result.get('config_name', ''),
                'indicator': result.get('indicator', ''),
                'symbol': result.get('symbol', ''),
                'interval': result.get('interval', ''),
                'status': result.get('status', ''),
                'accuracy': result.get('metrics', {}).get('accuracy', 0),
                'sharpe_ratio': result.get('metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': result.get('metrics', {}).get('max_drawdown', 0),
                'win_rate': result.get('metrics', {}).get('win_rate', 0),
                'profit_factor': result.get('metrics', {}).get('profit_factor', 0),
                'total_trades': result.get('metrics', {}).get('total_trades', 0),
                'performance_grade': result.get('performance_grade', 'F'),
                'execution_time': result.get('execution_time', 0)
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"📄 Unified CSV report: {output_file}")
    
    return str(output_file)

def _calculate_accuracy_stats(results):
    """Calculate accuracy statistics."""
    successful_results = [r for r in results if r.get('status') == 'success']
    if not successful_results:
        return {}
    
    accuracies = [r.get('metrics', {}).get('accuracy', 0) for r in successful_results]
    
    return {
        'mean': round(sum(accuracies) / len(accuracies), 2),
        'median': round(sorted(accuracies)[len(accuracies)//2], 2),
        'std': round(pd.Series(accuracies).std(), 2),
        'min': round(min(accuracies), 2),
        'max': round(max(accuracies), 2),
        'count': len(accuracies)
    }

def _calculate_risk_stats(results):
    """Calculate risk statistics."""
    successful_results = [r for r in results if r.get('status') == 'success']
    if not successful_results:
        return {}
    
    sharpe_ratios = [r.get('metrics', {}).get('sharpe_ratio', 0) for r in successful_results]
    drawdowns = [r.get('metrics', {}).get('max_drawdown', 0) for r in successful_results]
    
    return {
        'avg_sharpe': round(sum(sharpe_ratios) / len(sharpe_ratios), 3),
        'avg_drawdown': round(sum(drawdowns) / len(drawdowns), 2),
        'best_sharpe': round(max(sharpe_ratios), 3),
        'worst_drawdown': round(max(drawdowns), 2)
    }

def _analyze_by_indicator(results):
    """Analyze results by indicator type."""
    indicator_stats = {}
    
    for result in results:
        if result.get('status') != 'success':
            continue
        
        indicator = result.get('indicator', 'unknown')
        if indicator not in indicator_stats:
            indicator_stats[indicator] = {
                'count': 0,
                'total_accuracy': 0,
                'best_accuracy': 0,
                'configs': []
            }
        
        accuracy = result.get('metrics', {}).get('accuracy', 0)
        indicator_stats[indicator]['count'] += 1
        indicator_stats[indicator]['total_accuracy'] += accuracy
        indicator_stats[indicator]['best_accuracy'] = max(
            indicator_stats[indicator]['best_accuracy'], accuracy
        )
        indicator_stats[indicator]['configs'].append(result.get('config_name', ''))
    
    # Calculate averages
    for indicator, stats in indicator_stats.items():
        stats['avg_accuracy'] = round(stats['total_accuracy'] / stats['count'], 2)
    
    return indicator_stats

def _analyze_by_symbol(results):
    """Analyze results by trading symbol."""
    symbol_stats = {}
    
    for result in results:
        if result.get('status') != 'success':
            continue
        
        symbol = result.get('symbol', 'unknown')
        if symbol not in symbol_stats:
            symbol_stats[symbol] = {'count': 0, 'total_accuracy': 0}
        
        symbol_stats[symbol]['count'] += 1
        symbol_stats[symbol]['total_accuracy'] += result.get('metrics', {}).get('accuracy', 0)
    
    for symbol, stats in symbol_stats.items():
        stats['avg_accuracy'] = round(stats['total_accuracy'] / stats['count'], 2)
    
    return symbol_stats

def _analyze_by_interval(results):
    """Analyze results by time interval."""
    interval_stats = {}
    
    for result in results:
        if result.get('status') != 'success':
            continue
        
        interval = result.get('interval', 'unknown')
        if interval not in interval_stats:
            interval_stats[interval] = {'count': 0, 'total_accuracy': 0}
        
        interval_stats[interval]['count'] += 1
        interval_stats[interval]['total_accuracy'] += result.get('metrics', {}).get('accuracy', 0)
    
    for interval, stats in interval_stats.items():
        stats['avg_accuracy'] = round(stats['total_accuracy'] / stats['count'], 2)
    
    return interval_stats

def _generate_recommendations(results):
    """Generate optimization recommendations."""
    successful_results = [r for r in results if r.get('status') == 'success']
    
    if not successful_results:
        return ["No successful tests to analyze"]
    
    recommendations = []
    
    # Best performers
    best_configs = sorted(successful_results, key=lambda x: x.get('metrics', {}).get('accuracy', 0), reverse=True)[:3]
    recommendations.append(f"🏆 Top performers: {', '.join([c.get('config_name', '') for c in best_configs])}")
    
    # Risk analysis
    low_risk_configs = [r for r in successful_results if r.get('metrics', {}).get('max_drawdown', 100) < 5]
    if low_risk_configs:
        recommendations.append(f"🛡️ Low risk options: {len(low_risk_configs)} configs with <5% drawdown")
    
    # Consistency analysis
    consistent_configs = [r for r in successful_results if r.get('metrics', {}).get('win_rate', 0) > 60]
    if consistent_configs:
        recommendations.append(f"🎯 Consistent performers: {len(consistent_configs)} configs with >60% win rate")
    
    return recommendations

if __name__ == "__main__":
    # Test with latest execution
    execution_id = "INDICATORS_EXEC_20250607_110951_E74C00CE_0001"
    
    print("🚀 Creating unified analysis report...")
    
    # Export both formats
    json_file = export_unified_analysis_report(execution_id, 'json')
    csv_file = export_unified_analysis_report(execution_id, 'csv')
    
    print("✅ Unified reports created!")