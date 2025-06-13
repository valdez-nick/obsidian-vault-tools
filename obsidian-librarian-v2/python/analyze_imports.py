#!/usr/bin/env python3
"""
Comprehensive dependency and import analysis script.
"""

import ast
import sys
from pathlib import Path
import re

def extract_imports_from_file(file_path):
    """Extract all imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to get imports
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'level': 0,
                        'lineno': node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                level = getattr(node, 'level', 0)
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'level': level,
                        'lineno': node.lineno
                    })
        
        return imports, None
    except Exception as e:
        return [], str(e)

def analyze_file(file_path):
    """Analyze a single file for import issues."""
    imports, error = extract_imports_from_file(file_path)
    if error:
        return {'file': str(file_path), 'error': error}
    
    results = {
        'file': str(file_path),
        'imports': imports,
        'third_party': [],
        'local': [],
        'standard_library': []
    }
    
    # Standard library modules (partial list)
    stdlib_modules = {
        'asyncio', 'datetime', 'pathlib', 'typing', 'dataclasses', 'enum',
        'json', 'logging', 'sys', 'os', 'hashlib', 're', 'uuid', 'urllib',
        'concurrent', 'functools', 'collections', 'itertools', 'warnings',
        'sqlite3', 'xml', 'unittest', 'textwrap', 'copy', 'abc'
    }
    
    for imp in imports:
        module = imp['module']
        top_level = module.split('.')[0] if module else ''
        
        if imp.get('level', 0) > 0 or module.startswith('.'):
            # Relative import
            results['local'].append(imp)
        elif top_level in stdlib_modules:
            results['standard_library'].append(imp)
        elif top_level in {'obsidian_librarian'} or not module:
            results['local'].append(imp)
        else:
            results['third_party'].append(imp)
    
    return results

def main():
    # Get all Python files
    files = list(Path('.').rglob('*.py'))
    all_results = []

    for file_path in files:
        if 'venv' in str(file_path) or '__pycache__' in str(file_path):
            continue
        result = analyze_file(file_path)
        all_results.append(result)

    # Print comprehensive analysis
    print('=== DEPENDENCY AND IMPORT ANALYSIS ===')
    print()

    # Collect all third-party dependencies mentioned in code
    all_third_party = set()
    for result in all_results:
        if 'error' not in result:
            for imp in result['third_party']:
                module = imp['module'].split('.')[0]
                all_third_party.add(module)

    print('1. THIRD-PARTY DEPENDENCIES FOUND IN CODE:')
    for dep in sorted(all_third_party):
        print(f'   - {dep}')
    print()

    # Dependencies from pyproject.toml
    pyproject_deps = {
        'fastapi', 'uvicorn', 'httpx', 'typer', 'rich', 'langchain', 'openai',
        'anthropic', 'sentence_transformers', 'numpy', 'qdrant_client', 'duckdb',
        'sqlalchemy', 'alembic', 'redis', 'hiredis', 'aiohttp', 'beautifulsoup4',
        'html2text', 'trafilatura', 'pydantic', 'aiofiles',
        'frontmatter', 'dateutil', 'dotenv', 'click',
        'yaml', 'structlog', 'prometheus_client', 'watchdog', 'pytest', 'sklearn'
    }

    print('2. MISSING DEPENDENCIES (used in code but not in pyproject.toml):')
    missing_deps = all_third_party - pyproject_deps
    for dep in sorted(missing_deps):
        print(f'   - {dep}')
    print()

    print('3. POTENTIALLY UNUSED DEPENDENCIES (in pyproject.toml but not found in code):')
    unused_deps = pyproject_deps - all_third_party
    for dep in sorted(unused_deps):
        print(f'   - {dep}')
    print()

    print('4. FILES WITH IMPORT ERRORS:')
    error_count = 0
    for result in all_results:
        if 'error' in result:
            print(f'   {result["file"]}: {result["error"]}')
            error_count += 1
    if error_count == 0:
        print('   No import errors found!')
    print()

    print('5. OPTIONAL DEPENDENCY HANDLING:')
    # Check for try/except import patterns
    optional_imports = []
    for result in all_results:
        if 'error' not in result:
            file_path = result['file']
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'try:' in content and 'import' in content and 'except' in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'try:' in line:
                                # Look for import in next few lines
                                for j in range(i+1, min(i+5, len(lines))):
                                    if 'import' in lines[j] and 'except' in '\n'.join(lines[j:j+3]):
                                        optional_imports.append({
                                            'file': file_path,
                                            'line': j+1,
                                            'import': lines[j].strip()
                                        })
                                        break
            except:
                pass
    
    if optional_imports:
        print('   Found optional dependency patterns:')
        for opt in optional_imports:
            print(f'   {opt["file"]}:{opt["line"]} - {opt["import"]}')
    else:
        print('   No optional dependency patterns found')
    print()

    print('6. IMPORT STYLE ANALYSIS:')
    relative_imports = 0
    absolute_imports = 0
    
    for result in all_results:
        if 'error' not in result:
            for imp in result['local']:
                if imp.get('level', 0) > 0:
                    relative_imports += 1
                else:
                    absolute_imports += 1
    
    print(f'   Relative imports: {relative_imports}')
    print(f'   Absolute imports: {absolute_imports}')
    print()

    print('7. DETAILED FILE-BY-FILE BREAKDOWN:')
    for result in all_results:
        if 'error' not in result and result['third_party']:
            rel_path = result['file'].replace(str(Path.cwd()) + '/', '')
            print(f'   {rel_path}:')
            for imp in result['third_party']:
                if imp['type'] == 'import':
                    print(f'     import {imp["module"]}')
                else:
                    print(f'     from {imp["module"]} import {imp["name"]}')
    print()

if __name__ == '__main__':
    main()