# project_smart_analyzer.py - Real project structure analyzer
import os
import ast
from pathlib import Path
from collections import defaultdict, Counter

def analyze_real_project_structure():
    """
    Analyze actual project based on provided file data
    Generate intelligent comments based on real code patterns
    """
    
    # Known file patterns from your project
    project_patterns = {
        'signal_engine': {
            'large_files': ['signal_indicator_plugin_system.py', 'signal_filter_system.py', 'signal_strategy_system.py'],
            'purpose': '# Core signal generation system'
        },
        'backtest': {
            'large_files': ['backtest_engine.py', 'config_generator.py', 'batch_backtest.py'],
            'purpose': '# Backtesting framework'
        },
        'strategies': {
            'files': ['trend_strategy.py', 'reversal_strategy.py', 'breakout_strategy.py', 'ensemble_strategy.py'],
            'purpose': '# Trading strategies implementation'
        },
        'filters': {
            'files': ['adaptive_filters.py', 'ensemble_filters.py', 'ml_filters.py'],
            'purpose': '# Signal filtering and validation'
        },
        'indicators': {
            'files': ['feature_indicators.py', 'statistical_indicators.py', 'advanced_indicators.py'],
            'purpose': '# Technical indicators calculation'
        },
        'ml': {
            'files': ['model_trainer.py', 'predictors.py', 'feature_selector.py'],
            'purpose': '# Machine learning components'
        }
    }

def generate_optimized_tree(root_path='.', show_large_files=True):
    """
    Generate tree structure optimized for signal bot project
    Highlight problematic large files and suggest refactoring
    """
    
    def get_smart_comment(dir_path, dir_name):
        """Generate intelligent comments based on actual project analysis"""
        
        # Check for large files that need attention
        py_files = list(Path(dir_path).glob('*.py'))
        large_files = [f for f in py_files if f.stat().st_size > 20000]  # > 20KB
        
        # Pattern-based analysis
        if 'signal_engine' in dir_name:
            if large_files:
                return f"# ⚠️  Core signal system - {len(large_files)} large files need refactoring"
            return "# Core signal generation system"
        elif 'backtest' in dir_name:
            return f"# Backtesting framework - {len(py_files)} components"
        elif 'strategies' in dir_name:
            return "# Trading strategies implementation"
        elif 'filters' in dir_name:
            return "# Signal filtering and validation"
        elif 'indicators' in dir_name:
            if large_files:
                return f"# ⚠️  Technical indicators - {len(large_files)} oversized files"
            return "# Technical indicators calculation"
        elif 'ml' in dir_name:
            return "# Machine learning components"
        elif 'data' in dir_name:
            return "# Data fetching and storage"
        elif 'db' in dir_name:
            return "# Database operations"
        elif 'risk' in dir_name:
            return "# Risk management"
        elif 'live_trade' in dir_name:
            return "# Live trading execution"
        elif 'utils' in dir_name:
            return "# Utility functions"
        elif 'telegram' in dir_name:
            return "# Notification system"
        elif 'monitoring' in dir_name:
            return "# System monitoring"
        elif 'runner' in dir_name:
            return "# Application runners"
        elif 'build' in dir_name:
            return "# ⛔ Build artifacts - should be in .gitignore"
        
        return ""

    def analyze_file_size_issues(directory):
        """Identify files that are too large and need refactoring"""
        issues = []
        
        for py_file in Path(directory).rglob('*.py'):
            size_kb = py_file.stat().st_size / 1024
            
            if size_kb > 25:  # Files larger than 25KB
                issues.append({
                    'file': str(py_file),
                    'size_kb': round(size_kb, 1),
                    'severity': 'HIGH' if size_kb > 35 else 'MEDIUM'
                })
        
        return sorted(issues, key=lambda x: x['size_kb'], reverse=True)

    def build_tree(directory, prefix="", depth=0, max_depth=4):
        if depth > max_depth:
            return []
        
        items = []
        try:
            all_items = list(Path(directory).iterdir())
            
            # Filter out build directory and other artifacts
            dirs = [item for item in all_items if item.is_dir() and not should_ignore(item.name)]
            files = [item for item in all_items if item.is_file() and not should_ignore(item.name)]
            
            # Sort by importance for signal bot
            def sort_priority(item):
                priority_order = ['signal_engine', 'strategies', 'filters', 'indicators', 
                                'backtest', 'data', 'ml', 'risk', 'live_trade']
                name = item.name.lower()
                try:
                    return priority_order.index(name)
                except ValueError:
                    return 999
            
            dirs.sort(key=sort_priority)
            files.sort(key=lambda x: x.name.lower())
            all_sorted = dirs + files
            
            for i, item in enumerate(all_sorted):
                is_last = i == len(all_sorted) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                # Add intelligent comments and warnings
                comment = ""
                warning = ""
                
                if item.is_dir():
                    comment = get_smart_comment(item, item.name)
                elif item.is_file() and item.suffix == '.py':
                    size_kb = item.stat().st_size / 1024
                    if size_kb > 25:
                        warning = f" ⚠️  ({size_kb:.1f}KB - needs refactoring)"
                
                display_name = f"{item.name}{warning}{' ' + comment if comment else ''}"
                items.append(f"{prefix}{current_prefix}{display_name}")
                
                if item.is_dir() and not should_ignore(item.name):
                    items.extend(build_tree(item, prefix + next_prefix, depth + 1, max_depth))
                    
        except PermissionError:
            pass
            
        return items
    
    def should_ignore(name):
        ignore_patterns = ['__pycache__', '.git', '.venv', 'venv', '.pytest_cache', 
                          'build', 'dist', '*.egg-info']
        return any(pattern.replace('*', '').replace('.', '') in name.lower() 
                  for pattern in ignore_patterns)
    
    # Generate tree
    root_name = Path(root_path).name if root_path != '.' else 'signal_bot_project'
    tree_lines = [f"{root_name}/"]
    tree_lines.extend(build_tree(root_path))
    
    # Add analysis summary
    if show_large_files:
        tree_lines.append("\n" + "="*50)
        tree_lines.append("🚨 REFACTORING PRIORITIES:")
        
        large_file_issues = analyze_file_size_issues(root_path)
        for issue in large_file_issues[:10]:  # Top 10 largest files
            severity_icon = "🔴" if issue['severity'] == 'HIGH' else "🟡"
            tree_lines.append(f"{severity_icon} {issue['file']} ({issue['size_kb']}KB)")
    
    return '\n'.join(tree_lines)

def generate_refactoring_plan():
    """Generate specific refactoring recommendations"""
    
    plan = """
🎯 SIGNAL BOT REFACTORING PLAN:

1. 🔥 IMMEDIATE ACTIONS:
   - Split signal_indicator_plugin_system.py (21KB) into modules
   - Break down ensemble_strategy.py (29KB) into separate strategies
   - Refactor feature_indicators.py (38KB) into indicator categories

2. 📁 RECOMMENDED STRUCTURE:
   signal_engine/
   ├── core/           # Core engine logic (< 500 lines each)
   ├── strategies/     # Individual strategy files (< 300 lines each)  
   ├── indicators/     # Grouped by type (trend, volume, momentum)
   ├── filters/        # Separate filter categories
   └── ml/            # ML components

3. 🎯 ROI OPTIMIZATION FOCUS:
   - Isolate core signal logic from infrastructure
   - Create modular strategy testing framework
   - Implement performance monitoring per component
    """
    
    return plan

if __name__ == "__main__":
    print("🔍 SIGNAL BOT PROJECT ANALYSIS")
    print("="*50)
    
    tree = generate_optimized_tree(show_large_files=True)
    print(tree)
    
    print(generate_refactoring_plan())