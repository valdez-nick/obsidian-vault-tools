#!/usr/bin/env python3
"""Simple tag analyzer for Obsidian vault without dependencies."""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict
import json

def extract_tags_from_file(file_path):
    """Extract tags from a markdown file."""
    tags = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract frontmatter tags
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            
            # Array format: tags: [tag1, tag2]
            array_match = re.search(r'^tags:\s*\[(.*?)\]', frontmatter, re.MULTILINE | re.IGNORECASE)
            if array_match:
                tag_string = array_match.group(1)
                raw_tags = [tag.strip().strip('"\'') for tag in tag_string.split(',')]
                tags.update([tag for tag in raw_tags if tag])
            
            # List format:
            # tags:
            #   - tag1
            #   - tag2
            list_match = re.search(r'^tags:\s*\n((?:\s*-\s*.+\n?)+)', frontmatter, re.MULTILINE | re.IGNORECASE)
            if list_match:
                tag_lines = list_match.group(1).strip().split('\n')
                for line in tag_lines:
                    tag = line.strip().lstrip('-').strip()
                    if tag:
                        tags.add(tag)
        
        # Extract inline tags (#tag format)
        inline_tags = re.findall(r'(?:^|\s)#([a-zA-Z0-9/_-]+)(?:\s|$)', content, re.MULTILINE)
        tags.update(inline_tags)
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return list(tags)

def analyze_vault_tags(vault_path):
    """Analyze all tags in an Obsidian vault."""
    vault_path = Path(vault_path)
    
    # Tag statistics
    tag_counter = Counter()
    tag_to_files = defaultdict(list)
    file_to_tags = {}
    tag_hierarchies = defaultdict(set)
    
    # Find all markdown files
    markdown_files = []
    for root, dirs, files in os.walk(vault_path):
        # Skip hidden directories and obsidian config
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'obsidian-librarian-v2']
        
        for file in files:
            if file.endswith('.md'):
                file_path = Path(root) / file
                markdown_files.append(file_path)
    
    print(f"\nAnalyzing {len(markdown_files)} markdown files...")
    
    # Process each file
    for file_path in markdown_files:
        tags = extract_tags_from_file(file_path)
        relative_path = file_path.relative_to(vault_path)
        
        if tags:
            file_to_tags[str(relative_path)] = tags
            for tag in tags:
                tag_counter[tag] += 1
                tag_to_files[tag].append(str(relative_path))
                
                # Check for hierarchical tags
                if '/' in tag:
                    parts = tag.split('/')
                    for i in range(len(parts)):
                        parent = '/'.join(parts[:i+1])
                        if i < len(parts) - 1:
                            child = '/'.join(parts[:i+2])
                            tag_hierarchies[parent].add(child)
    
    # Generate report
    print("\n" + "="*60)
    print("TAG ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nTotal unique tags: {len(tag_counter)}")
    print(f"Total tag occurrences: {sum(tag_counter.values())}")
    print(f"Files with tags: {len(file_to_tags)}")
    print(f"Files without tags: {len(markdown_files) - len(file_to_tags)}")
    
    # Most common tags
    print("\n" + "-"*40)
    print("TOP 20 MOST USED TAGS:")
    print("-"*40)
    for tag, count in tag_counter.most_common(20):
        print(f"{tag:30} {count:4} occurrences")
    
    # Tag hierarchies
    if tag_hierarchies:
        print("\n" + "-"*40)
        print("TAG HIERARCHIES:")
        print("-"*40)
        for parent, children in sorted(tag_hierarchies.items()):
            if not any(parent in tag_hierarchies[p] for p in tag_hierarchies if p != parent):
                # This is a root tag
                print(f"\n{parent}")
                for child in sorted(children):
                    indent = "  " * (child.count('/') - parent.count('/'))
                    print(f"{indent}└── {child.split('/')[-1]}")
    
    # Tags by category (based on common prefixes)
    print("\n" + "-"*40)
    print("TAGS BY CATEGORY:")
    print("-"*40)
    categories = defaultdict(list)
    for tag in tag_counter:
        if '/' in tag:
            category = tag.split('/')[0]
            categories[category].append(tag)
        elif tag.startswith('todo'):
            categories['todo'].append(tag)
        elif tag.startswith('meeting'):
            categories['meeting'].append(tag)
        elif tag.startswith('initiative'):
            categories['initiative'].append(tag)
        else:
            categories['_other'].append(tag)
    
    for category, tags in sorted(categories.items()):
        if category != '_other':
            print(f"\n{category}: {len(tags)} tags")
            for tag in sorted(tags)[:10]:
                print(f"  - {tag} ({tag_counter[tag]} uses)")
            if len(tags) > 10:
                print(f"  ... and {len(tags) - 10} more")
    
    # Save detailed report
    output_file = vault_path / 'tag_analysis_report.json'
    report_data = {
        'summary': {
            'total_tags': len(tag_counter),
            'total_occurrences': sum(tag_counter.values()),
            'files_with_tags': len(file_to_tags),
            'files_without_tags': len(markdown_files) - len(file_to_tags),
            'analyzed_files': len(markdown_files)
        },
        'tag_frequencies': dict(tag_counter),
        'tag_to_files': dict(tag_to_files),
        'file_to_tags': file_to_tags,
        'hierarchies': {k: list(v) for k, v in tag_hierarchies.items()},
        'categories': {k: v for k, v in categories.items() if k != '_other'}
    }
    
    with open(output_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n\nDetailed report saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_tags_simple.py <vault_path>")
        sys.exit(1)
    
    vault_path = sys.argv[1]
    if not os.path.isdir(vault_path):
        print(f"Error: {vault_path} is not a valid directory")
        sys.exit(1)
    
    analyze_vault_tags(vault_path)