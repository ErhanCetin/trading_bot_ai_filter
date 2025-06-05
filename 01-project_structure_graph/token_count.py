# token_calculator.py - Calculate API costs for trading AI project
import os
import re
from pathlib import Path
from collections import defaultdict

def estimate_tokens(text):
    """
    Estimate token count for text
    Rough approximation: 1 token ≈ 4 characters or 0.75 words
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Character-based estimation (more accurate for code)
    char_count = len(text)
    tokens_by_chars = char_count / 4
    
    # Word-based estimation
    word_count = len(text.split())
    tokens_by_words = word_count / 0.75
    
    # Use average of both methods
    estimated_tokens = int((tokens_by_chars + tokens_by_words) / 2)
    
    return estimated_tokens

def analyze_project_tokens(root_path='.'):
    """
    Analyze entire project and calculate token usage
    """
    
    total_tokens = 0
    file_tokens = {}
    category_tokens = defaultdict(int)
    
    # File categories for trading bot
    categories = {
        'signal_engine': ['signal_engine'],
        'strategies': ['strategies', 'strategy'],
        'indicators': ['indicators', 'indicator'],
        'filters': ['filters', 'filter'],
        'backtest': ['backtest', 'test'],
        'ml': ['ml', 'model', 'predict'],
        'data': ['data', 'fetch'],
        'risk': ['risk'],
        'live_trade': ['live', 'trade'],
        'utils': ['utils', 'helper'],
        'config': ['config', 'settings'],
        'other': []
    }
    
    def categorize_file(file_path):
        path_str = str(file_path).lower()
        for category, keywords in categories.items():
            if category == 'other':
                continue
            if any(keyword in path_str for keyword in keywords):
                return category
        return 'other'
    
    print(f"🔍 Analyzing tokens in: {root_path}")
    print("=" * 60)
    
    # Process all Python files
    python_files = list(Path(root_path).rglob('*.py'))
    
    for py_file in python_files:
        # Skip common ignore patterns
        if any(ignore in str(py_file) for ignore in ['__pycache__', '.venv', 'venv', '.git']):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            tokens = estimate_tokens(content)
            file_tokens[str(py_file)] = {
                'tokens': tokens,
                'size_kb': py_file.stat().st_size / 1024,
                'lines': content.count('\n') + 1
            }
            
            total_tokens += tokens
            
            # Categorize
            category = categorize_file(py_file)
            category_tokens[category] += tokens
            
        except Exception as e:
            print(f"⚠️  Error reading {py_file}: {e}")
    
    return total_tokens, file_tokens, dict(category_tokens)

def calculate_api_costs(total_tokens, input_ratio=0.4):
    """
    Calculate Claude API costs
    input_ratio: What percentage of tokens are input vs output
    """
    
    # Claude Sonnet 4 pricing (per million tokens)
    INPUT_COST = 3.0   # $3 per 1M input tokens
    OUTPUT_COST = 15.0 # $15 per 1M output tokens
    
    input_tokens = int(total_tokens * input_ratio)
    output_tokens = int(total_tokens * (1 - input_ratio))
    
    input_cost = (input_tokens / 1_000_000) * INPUT_COST
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST
    total_cost = input_cost + output_cost
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }

def generate_cost_report(root_path='.'):
    """
    Generate comprehensive cost analysis report
    """
    
    print("💰 CLAUDE API COST ANALYSIS FOR TRADING AI PROJECT")
    print("=" * 60)
    
    # Analyze project
    total_tokens, file_tokens, category_tokens = analyze_project_tokens(root_path)
    
    # Calculate costs for different scenarios
    scenarios = {
        'Current Project Only (1x)': 1.0,
        'With Discussions (3x)': 3.0,
        'Heavy Development (5x)': 5.0,
        'Full Month Usage (10x)': 10.0
    }
    
    print(f"\n📊 PROJECT TOKEN ANALYSIS:")
    print(f"Total Python files analyzed: {len(file_tokens)}")
    print(f"Total estimated tokens: {total_tokens:,}")
    print(f"Average tokens per file: {total_tokens // len(file_tokens):,}")
    
    # Category breakdown
    print(f"\n📁 TOKENS BY CATEGORY:")
    sorted_categories = sorted(category_tokens.items(), key=lambda x: x[1], reverse=True)
    for category, tokens in sorted_categories:
        percentage = (tokens / total_tokens) * 100
        print(f"  {category:15} {tokens:8,} tokens ({percentage:5.1f}%)")
    
    # Largest files
    print(f"\n🔥 TOP 10 LARGEST FILES BY TOKENS:")
    sorted_files = sorted(file_tokens.items(), key=lambda x: x[1]['tokens'], reverse=True)
    for file_path, info in sorted_files[:10]:
        file_name = Path(file_path).name
        print(f"  {file_name:30} {info['tokens']:6,} tokens ({info['size_kb']:5.1f}KB)")
    
    # Cost scenarios
    print(f"\n💸 CLAUDE API COST SCENARIOS:")
    print(f"{'Scenario':<25} {'Tokens':<12} {'Input Cost':<12} {'Output Cost':<12} {'Total Cost':<12}")
    print("-" * 75)
    
    for scenario_name, multiplier in scenarios.items():
        scenario_tokens = int(total_tokens * multiplier)
        costs = calculate_api_costs(scenario_tokens)
        
        print(f"{scenario_name:<25} {costs['total_tokens']:>10,} "
              f"${costs['input_cost']:>10.2f} ${costs['output_cost']:>10.2f} "
              f"${costs['total_cost']:>10.2f}")
    
    # Recommendations
    print(f"\n💡 COST OPTIMIZATION RECOMMENDATIONS:")
    
    if total_tokens > 100000:
        print("  🔴 High token usage detected:")
        print("    - Consider breaking large files into smaller modules")
        print("    - Use more focused queries to Claude")
        print("    - Cache common responses")
    elif total_tokens > 50000:
        print("  🟡 Moderate token usage:")
        print("    - Monitor API usage closely")
        print("    - Consider using cheaper models for simple tasks")
    else:
        print("  🟢 Reasonable token usage")
        print("    - API costs should be manageable")
    
    # Pro vs API comparison
    monthly_api_cost = calculate_api_costs(total_tokens * 10)['total_cost']  # 10x usage
    print(f"\n⚖️  PRO vs API COMPARISON:")
    print(f"  Claude Pro: $20/month (fixed)")
    print(f"  Claude API: ${monthly_api_cost:.2f}/month (estimated heavy usage)")
    
    if monthly_api_cost < 20:
        print("  💡 Recommendation: API likely more cost-effective")
    else:
        print("  💡 Recommendation: Pro subscription better value")
    
    return {
        'total_tokens': total_tokens,
        'monthly_api_estimate': monthly_api_cost,
        'file_count': len(file_tokens),
        'largest_files': sorted_files[:5]
    }

def analyze_conversation_tokens(conversation_file=None):
    """
    Analyze tokens from actual conversation history
    Use this if you have exported chat history
    """
    
    if not conversation_file or not Path(conversation_file).exists():
        print("⚠️  No conversation file provided or file not found")
        print("To get accurate costs, export your chat history and analyze it")
        return None
    
    with open(conversation_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by user/assistant messages (rough estimation)
    messages = re.split(r'(Human:|Assistant:)', content)
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, message in enumerate(messages):
        if message.strip() in ['Human:', 'Assistant:']:
            continue
            
        tokens = estimate_tokens(message)
        
        # Determine if input (human) or output (assistant)
        if i > 0 and messages[i-1].strip() == 'Human:':
            total_input_tokens += tokens
        elif i > 0 and messages[i-1].strip() == 'Assistant:':
            total_output_tokens += tokens
    
    # Calculate actual costs
    input_cost = (total_input_tokens / 1_000_000) * 3.0
    output_cost = (total_output_tokens / 1_000_000) * 15.0
    total_cost = input_cost + output_cost
    
    print(f"\n💬 ACTUAL CONVERSATION ANALYSIS:")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")
    
    return total_cost

if __name__ == "__main__":
    # Analyze current directory
    report = generate_cost_report()
    
    print(f"\n🎯 SUMMARY:")
    print(f"Your trading AI project would cost approximately:")
    print(f"${report['monthly_api_estimate']:.2f}/month with heavy API usage")
    
    # Uncomment to analyze actual conversation
    # analyze_conversation_tokens('your_exported_chat.txt')