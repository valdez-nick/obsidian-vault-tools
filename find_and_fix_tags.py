#!/usr/bin/env python3
"""
Direct tag finder and fixer - finds all potential corrupted tags
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import json

def find_corrupted_tags(vault_path):
    """Find all potentially corrupted tags"""
    vault_path = Path(vault_path)
    corrupted_tags = defaultdict(list)
    
    # Known prefixes that should have hierarchical structure
    prefixes = ['initiative', 'todo', 'identity', 'payments', 'okr', 'project', 'personal', 'meeting', 'product']
    
    for md_file in vault_path.rglob("*.md"):
        if any(part.startswith('.') for part in md_file.parts):
            continue
            
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all tags
            tag_pattern = r'#([a-zA-Z0-9_/-]+)'
            tags = re.findall(tag_pattern, content)
            
            for tag in tags:
                # Check if tag starts with a known prefix but has no /
                for prefix in prefixes:
                    if tag.startswith(prefix) and '/' not in tag and len(tag) > len(prefix):
                        # This looks like a corrupted hierarchical tag
                        remaining = tag[len(prefix):]
                        # Check if the remaining part starts with a lowercase letter or dash
                        if remaining and (remaining[0].islower() or remaining[0] == '-'):
                            corrupted_tags[f"#{tag}"].append({
                                'file': str(md_file.relative_to(vault_path)),
                                'suggested_fix': f"#{prefix}/{remaining}",
                                'line': find_line_with_tag(md_file, f"#{tag}")
                            })
                        
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            
    return corrupted_tags

def find_line_with_tag(file_path, tag):
    """Find line number containing the tag"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if tag in line:
                    return i
    except:
        pass
    return None

def main():
    vault_path = "/Users/nvaldez/Documents/Obsidian Vault"
    
    print("Searching for corrupted tags...")
    corrupted = find_corrupted_tags(vault_path)
    
    print(f"\nFound {len(corrupted)} unique corrupted tags:")
    print("-" * 60)
    
    for tag, occurrences in sorted(corrupted.items()):
        print(f"\n{tag} → {occurrences[0]['suggested_fix']}")
        print(f"  Found in {len(occurrences)} file(s):")
        for occ in occurrences[:5]:
            print(f"    - {occ['file']} (line {occ['line']})")
        if len(occurrences) > 5:
            print(f"    ... and {len(occurrences) - 5} more")
            
    # Save detailed report
    report = {
        'corrupted_tags': dict(corrupted),
        'total_corrupted_tags': len(corrupted),
        'total_occurrences': sum(len(occs) for occs in corrupted.values())
    }
    
    with open('corrupted_tags_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDetailed report saved to: corrupted_tags_report.json")

if __name__ == '__main__':
    main()