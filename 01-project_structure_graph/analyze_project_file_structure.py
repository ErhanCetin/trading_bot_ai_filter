# Fixed project_analyzer.py - Proper file handling
import os
import ast
from pathlib import Path

def analyze_project_structure():
    structure = {}
    
    for py_file in Path('.').rglob('*.py'):
        if '__pycache__' in str(py_file) or '.venv' in str(py_file):
            continue
            
        try:
            # Proper file handling with context manager
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines_count = len(content.splitlines())
                
            # Parse AST after file is read
            tree = ast.parse(content)
            
            classes = [node.name for node in ast.walk(tree) 
                      if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) 
                        if isinstance(node, ast.FunctionDef)]
            
            structure[str(py_file)] = {
                'classes': classes[:5],  # Limit output
                'functions': functions[:10],  # Limit output
                'lines': lines_count,
                'size_kb': round(py_file.stat().st_size / 1024, 2)
            }
            
        except (UnicodeDecodeError, SyntaxError) as e:
            structure[str(py_file)] = {'error': f'Parse error: {type(e).__name__}'}
        except Exception as e:
            structure[str(py_file)] = {'error': str(e)}
    
    return structure

def print_summary(structure):
    print(f"📊 Total Python files: {len(structure)}")
    print("\n🔍 File Overview:")
    
    for filepath, info in structure.items():
        if 'error' not in info:
            print(f"📄 {filepath} ({info['lines']} lines, {info['size_kb']}KB)")
            if info['classes']:
                print(f"   Classes: {', '.join(info['classes'])}")
            if info['functions']:
                print(f"   Functions: {', '.join(info['functions'][:3])}...")

if __name__ == "__main__":
    result = analyze_project_structure()
    print_summary(result)