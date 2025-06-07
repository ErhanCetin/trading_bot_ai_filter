"""
Indicator System Column Analyzer
Analyzes all indicators and their output columns
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project paths (adjust as needed for your project structure)
sys.path.append('signal_engine')
sys.path.append('.')

def generate_sample_data(size=200):
    """Generate realistic sample OHLCV data for testing."""
    
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=size, freq='5min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.01, size)  # 1% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'close': prices * (1 + np.random.normal(0, 0.005, size)),
        'high': prices * (1 + np.random.uniform(0, 0.02, size)),
        'low': prices * (1 - np.random.uniform(0, 0.02, size)),
        'volume': np.random.uniform(1000, 10000, size)
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def analyze_all_indicators():
    """Analyze all indicators and their output columns."""
    
    try:
        # Import the indicator system
        from signal_engine.indicators import registry
        from signal_engine.signal_indicator_plugin_system import IndicatorManager
        
        print("=" * 80)
        print("INDICATOR SYSTEM COLUMN ANALYSIS")
        print("=" * 80)
        
        # Create sample data
        print("📊 Generating sample data...")
        sample_data = generate_sample_data(200)
        print(f"✅ Sample data created: {len(sample_data)} rows")
        
        # Get all indicators
        all_indicators = registry.get_all_indicators()
        print(f"📋 Found {len(all_indicators)} indicators")
        
        # Create manager
        manager = IndicatorManager(registry)
        
        # Store results
        indicator_analysis = {}
        
        print("\n" + "=" * 80)
        print("ANALYZING EACH INDICATOR")
        print("=" * 80)
        
        # Test each indicator individually
        for indicator_name, indicator_class in all_indicators.items():
            print(f"\n🔍 Testing: {indicator_name}")
            print(f"   Category: {indicator_class.category}")
            print(f"   Description: {indicator_class.description}")
            
            try:
                # Create fresh manager for each indicator
                test_manager = IndicatorManager(registry)
                
                # Add the indicator with default parameters
                test_manager.add_indicator(indicator_name)
                
                # Get original columns
                original_columns = set(sample_data.columns)
                
                # Calculate the indicator
                result_df = test_manager.calculate_indicators(sample_data.copy())
                
                # Find new columns added by this indicator
                new_columns = set(result_df.columns) - original_columns
                new_columns = sorted(list(new_columns))
                
                # Store analysis
                indicator_analysis[indicator_name] = {
                    'class_name': indicator_class.__name__,
                    'category': indicator_class.category,
                    'description': indicator_class.description,
                    'declared_output_columns': getattr(indicator_class, 'output_columns', []),
                    'actual_output_columns': new_columns,
                    'dependencies': getattr(indicator_class, 'dependencies', []),
                    'requires_columns': getattr(indicator_class, 'requires_columns', []),
                    'default_params': getattr(indicator_class, 'default_params', {}),
                    'status': 'success',
                    'column_count': len(new_columns)
                }
                
                print(f"   ✅ Success: {len(new_columns)} columns created")
                for col in new_columns:
                    print(f"      - {col}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                indicator_analysis[indicator_name] = {
                    'class_name': indicator_class.__name__,
                    'category': indicator_class.category,
                    'description': indicator_class.description,
                    'declared_output_columns': getattr(indicator_class, 'output_columns', []),
                    'actual_output_columns': [],
                    'dependencies': getattr(indicator_class, 'dependencies', []),
                    'requires_columns': getattr(indicator_class, 'requires_columns', []),
                    'default_params': getattr(indicator_class, 'default_params', {}),
                    'status': 'error',
                    'error_message': str(e),
                    'column_count': 0
                }
        
        # Generate comprehensive report
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        # Group by category
        categories = {}
        for indicator_name, analysis in indicator_analysis.items():
            category = analysis['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((indicator_name, analysis))
        
        # Print category-wise analysis
        for category, indicators in categories.items():
            print(f"\n📂 CATEGORY: {category.upper()}")
            print("-" * 60)
            
            for indicator_name, analysis in indicators:
                status_icon = "✅" if analysis['status'] == 'success' else "❌"
                print(f"{status_icon} {indicator_name} ({analysis['class_name']})")
                print(f"   📝 Description: {analysis['description']}")
                print(f"   🔧 Dependencies: {analysis['dependencies']}")
                print(f"   📊 Requires: {analysis['requires_columns']}")
                print(f"   📈 Declared Outputs: {analysis['declared_output_columns']}")
                print(f"   🎯 Actual Outputs ({analysis['column_count']}): {analysis['actual_output_columns']}")
                
                if analysis['status'] == 'error':
                    print(f"   ⚠️ Error: {analysis.get('error_message', 'Unknown error')}")
                
                # Check for discrepancies
                declared = set(analysis['declared_output_columns'])
                actual = set(analysis['actual_output_columns'])
                
                if declared != actual:
                    missing = declared - actual
                    extra = actual - declared
                    if missing:
                        print(f"   ⚠️ Missing declared columns: {list(missing)}")
                    if extra:
                        print(f"   ℹ️ Extra columns created: {list(extra)}")
                
                print()
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        total_indicators = len(indicator_analysis)
        successful_indicators = sum(1 for a in indicator_analysis.values() if a['status'] == 'success')
        total_columns = sum(a['column_count'] for a in indicator_analysis.values())
        
        print(f"📊 Total Indicators: {total_indicators}")
        print(f"✅ Successful: {successful_indicators}")
        print(f"❌ Failed: {total_indicators - successful_indicators}")
        print(f"📈 Total Columns Created: {total_columns}")
        print(f"🎯 Success Rate: {successful_indicators/total_indicators*100:.1f}%")
        
        # Category breakdown
        print(f"\n📂 Indicators by Category:")
        category_counts = {}
        for analysis in indicator_analysis.values():
            category = analysis['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"   - {category}: {count} indicators")
        
        # Export detailed analysis
        print("\n" + "=" * 80)
        print("DETAILED COLUMN MAPPING")
        print("=" * 80)
        
        print("\n# EXACT COLUMN NAMES FOR JSON CONFIGS:")
        print("# Copy-paste ready for accurate config creation\n")
        
        for category, indicators in categories.items():
            print(f"## {category.upper()} INDICATORS")
            for indicator_name, analysis in indicators:
                if analysis['status'] == 'success' and analysis['actual_output_columns']:
                    print(f"# {indicator_name}:")
                    for col in analysis['actual_output_columns']:
                        print(f'"{col}",')
                    print()
        
        return indicator_analysis
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure the signal_engine.indicators module is in your Python path")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_analysis_to_file(analysis, filename="signal_engine/indicators/indicator_analysis.txt"):
    """Export analysis to a text file."""
    if not analysis:
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("INDICATOR SYSTEM COLUMN ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by category for file output
            categories = {}
            for indicator_name, analysis in analysis.items():
                category = analysis['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append((indicator_name, analysis))
            
            # Write category-wise analysis
            for category, indicators in categories.items():
                f.write(f"\nCATEGORY: {category.upper()}\n")
                f.write("-" * 60 + "\n")
                
                for indicator_name, analysis in indicators:
                    f.write(f"\nIndicator: {indicator_name}\n")
                    f.write(f"Class: {analysis['class_name']}\n")
                    f.write(f"Description: {analysis['description']}\n")
                    f.write(f"Dependencies: {analysis['dependencies']}\n")
                    f.write(f"Output Columns: {analysis['actual_output_columns']}\n")
                    f.write(f"Status: {analysis['status']}\n")
                    if analysis['status'] == 'error':
                        f.write(f"Error: {analysis.get('error_message', '')}\n")
                    f.write("\n")
        
        print(f"📁 Analysis exported to: {filename}")
        
    except Exception as e:
        print(f"❌ Error exporting analysis: {e}")

if __name__ == "__main__":
    """Run the indicator analysis."""
    
    print("🚀 Starting Indicator System Analysis...")
    
    # Run analysis
    analysis_results = analyze_all_indicators()
    
    if analysis_results:
        # Export to file
        export_analysis_to_file(analysis_results)
        print("\n🎉 Analysis completed successfully!")
        print("📋 Use the DETAILED COLUMN MAPPING section above for accurate JSON configs")
    else:
        print("\n❌ Analysis failed!")
        print("Please check your indicator system setup and try again.")