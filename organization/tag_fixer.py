#!/usr/bin/env python3
"""
Fix Obsidian Vault Tags - Automated tag cleanup script
Based on the tag analysis report findings
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import argparse
from obsidian_vault_tools.memory import track_tool_usage, track_action

class TagFixer:
    def __init__(self, vault_path, dry_run=True, operations=None):
        self.vault_path = Path(vault_path)
        self.dry_run = dry_run
        self.changes_made = defaultdict(list)
        # Operations to perform - if None, do all operations
        self.operations = operations or ['quoted', 'similar', 'generic', 'hierarchy']
        
    def log_change(self, file_path, change_type, old_value, new_value):
        """Log a change for reporting"""
        self.changes_made[change_type].append({
            'file': str(file_path),
            'old': old_value,
            'new': new_value
        })
        
    def fix_quoted_tags(self, content, file_path):
        """Fix tags with quotes like "#todo" to #todo"""
        pattern = r'"#([a-zA-Z0-9/_-]+)"'
        new_content = content
        
        for match in re.finditer(pattern, content):
            old_tag = match.group(0)
            new_tag = f"#{match.group(1)}"
            new_content = new_content.replace(old_tag, new_tag)
            self.log_change(file_path, 'quoted_tags', old_tag, new_tag)
            
        return new_content
        
    def standardize_similar_tags(self, content, file_path):
        """Standardize similar tags to consistent format"""
        tag_mappings = {
            # Standardize to singular forms
            '#daily-notes': '#daily-note',
            '#1on1-notes': '#1on1',
            '"#1on1-notes"': '#1on1',
            
            # Fix incomplete tags
            '#todo/': '#todo',
            '#initiative/': '#initiative',
            
            # Standardize payment tags
            '#payment-q1-okr': '#payments/okr/q1',
            '#payment-q2-okr': '#payments/okr/q2',
            '#okr-q2-payment': '#payments/okr/q2',
            
            # Fix identity tags
            '#identity-': '#identity/',
            '#device-intellignece': '#identity/device-intelligence',
            '#device_fingerprinting': '#identity/device-fingerprinting',
            '#device-fingerprint': '#identity/device-fingerprinting',
            
            # Standardize various tags
            '#braindump': '#brain-dump',
            '#productresearch': '#product-research',
            '#customerfeedback': '#customer-feedback',
            '#mlperformance': '#ml-performance',
            '#offlineLLM': '#offline-llm',
            '#doordashdash': '#doordash',
            '#todo-urgent': '#todo/urgent',
        }
        
        new_content = content
        for old_tag, new_tag in tag_mappings.items():
            if old_tag in content:
                new_content = new_content.replace(old_tag, new_tag)
                self.log_change(file_path, 'standardized_tags', old_tag, new_tag)
                
        return new_content
        
    def remove_generic_tags(self, content, file_path):
        """Remove overly generic tags"""
        generic_tags = ['#notes', '#1', '#2', '#42']
        
        new_content = content
        for tag in generic_tags:
            if tag in content:
                # Remove the tag but keep the rest of the line
                new_content = re.sub(rf'\s*{re.escape(tag)}\b', '', new_content)
                self.log_change(file_path, 'removed_generic', tag, 'REMOVED')
                
        return new_content
        
    def fix_hierarchy_tags(self, content, file_path):
        """Fix hierarchical tag issues"""
        # Fix okr hierarchy
        content = re.sub(r'#"#okr/([^"]+)"', r'#okr/\1', content)
        
        # Fix personal hierarchy  
        content = re.sub(r'#"#personal/([^"]+)"', r'#personal/\1', content)
        
        # Fix improve-sift
        content = content.replace('"#improve-sift"', '#improvements/sift')
        
        return content
        
    @track_action(action_type="tag_fix")
    def process_file(self, file_path):
        """Process a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # Apply fixes based on selected operations
            if 'quoted' in self.operations:
                content = self.fix_quoted_tags(content, file_path)
            if 'similar' in self.operations:
                content = self.standardize_similar_tags(content, file_path)
            if 'generic' in self.operations:
                content = self.remove_generic_tags(content, file_path)
            if 'hierarchy' in self.operations:
                content = self.fix_hierarchy_tags(content, file_path)
            
            # Only write if changes were made
            if content != original_content and not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    @track_tool_usage(category="organization")
    def process_vault(self):
        """Process all markdown files in the vault"""
        md_files = list(self.vault_path.rglob("*.md"))
        
        print(f"Processing {len(md_files)} markdown files...")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE CHANGES'}")
        print("-" * 50)
        
        for file_path in md_files:
            # Skip certain directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if 'node_modules' in file_path.parts:
                continue
                
            self.process_file(file_path)
            
        # Print summary
        self.print_summary()
        
    def print_summary(self):
        """Print a summary of changes"""
        print("\n" + "=" * 50)
        print("TAG FIX SUMMARY")
        print("=" * 50)
        
        total_changes = 0
        for change_type, changes in self.changes_made.items():
            print(f"\n{change_type.replace('_', ' ').title()}:")
            print(f"  Total: {len(changes)} changes")
            
            # Show first 5 examples
            for i, change in enumerate(changes[:5]):
                print(f"  - {change['old']} → {change['new']} in {Path(change['file']).name}")
                
            if len(changes) > 5:
                print(f"  ... and {len(changes) - 5} more")
                
            total_changes += len(changes)
            
        print(f"\nTotal changes: {total_changes}")
        
        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No changes were made")
            print("Run with --apply to make actual changes")
        else:
            print("\n✅ Changes applied successfully!")
            
        # Save detailed report
        report_path = self.vault_path / 'tag_fix_report.json'
        with open(report_path, 'w') as f:
            json.dump(dict(self.changes_made), f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Fix tags in Obsidian vault')
    parser.add_argument('vault_path', help='Path to Obsidian vault')
    
    # Mode arguments (mutually exclusive with apply for backward compatibility)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--apply', action='store_true', help='Apply changes (default is dry run)')
    mode_group.add_argument('--dry-run', action='store_true', help='Preview changes without applying (default)')
    
    # Operation-specific arguments
    parser.add_argument('--fix-quoted-only', action='store_true', 
                       help='Only fix quoted tags (e.g., "#tag" to #tag)')
    parser.add_argument('--merge-similar', action='store_true',
                       help='Only merge similar tags')
    parser.add_argument('--remove-generic', action='store_true',
                       help='Only remove generic tags')
    parser.add_argument('--fix-hierarchy', action='store_true',
                       help='Only fix hierarchical tag issues')
    
    args = parser.parse_args()
    
    # Determine dry run mode
    if args.apply:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        # Default to dry run if no mode specified
        dry_run = True
    
    # Determine which operations to perform
    operations = []
    if args.fix_quoted_only:
        operations = ['quoted']
    elif args.merge_similar:
        operations = ['similar']
    elif args.remove_generic:
        operations = ['generic']
    elif args.fix_hierarchy:
        operations = ['hierarchy']
    else:
        # Default: perform all operations
        operations = ['quoted', 'similar', 'generic', 'hierarchy']
    
    fixer = TagFixer(args.vault_path, dry_run=dry_run, operations=operations)
    fixer.process_vault()

if __name__ == '__main__':
    main()