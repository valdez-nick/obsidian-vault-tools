#!/usr/bin/env python3
"""
Emergency Tag Hierarchy Restoration Script
Fixes tags that were corrupted by removing the "/" separator
Example: #initiativedfp-revamp → #initiative/dfp-revamp
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import argparse
import json
from datetime import datetime
import shutil

class TagHierarchyRestorer:
    def __init__(self, vault_path, dry_run=True, backup=True):
        self.vault_path = Path(vault_path)
        self.dry_run = dry_run
        self.backup = backup
        self.changes_made = defaultdict(list)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Common hierarchical tag prefixes that were corrupted
        self.tag_prefixes = {
            'initiative': ['dfp-revamp', 'payments', 'sift-decisions', '100-grain-limit', 
                          'doordash-dfp-rfp', 'chargeback-uploader', 'fraud-rule-generator',
                          'cbautolabel', 'sift', 'refactor', 'optimization'],
            'todo': ['urgent', 'dfp', 'payments', 'initiative', 'review', 'followup',
                    'today', 'tomorrow', 'thisweek', 'nextweek'],
            'identity': ['device-fingerprinting', 'device-intelligence', 'fraud-detection',
                        'user-verification', 'authentication'],
            'payments': ['okr', 'fraud', 'chargeback', 'dispute', 'processor', 'gateway'],
            'okr': ['q1', 'q2', 'q3', 'q4', '2024', '2025'],
            'project': ['active', 'completed', 'planning', 'blocked', 'review'],
            'personal': ['health', 'finance', 'learning', 'goals', 'habits'],
            'meeting': ['1on1', 'team', 'standup', 'review', 'planning'],
            'product': ['research', 'design', 'development', 'launch', 'feedback']
        }
        
    def create_backup(self):
        """Create a backup of the vault before making changes"""
        if self.backup and not self.dry_run:
            backup_dir = self.vault_path.parent / f"vault_backup_{self.timestamp}"
            print(f"Creating backup at: {backup_dir}")
            shutil.copytree(self.vault_path, backup_dir, 
                          ignore=shutil.ignore_patterns('.git', '.obsidian', 'node_modules'))
            return backup_dir
        return None
        
    def generate_restoration_patterns(self):
        """Generate regex patterns for corrupted tags"""
        patterns = []
        
        # First, add specific known corrupted patterns
        for prefix, suffixes in self.tag_prefixes.items():
            for suffix in suffixes:
                # Pattern for tags that lost their hierarchy
                corrupted = f"#{prefix}{suffix}"
                fixed = f"#{prefix}/{suffix}"
                # Use word boundary to avoid partial matches
                pattern = (rf'\b{re.escape(corrupted)}\b', fixed)
                patterns.append(pattern)
                
        # Handle patterns where prefix runs into suffix without separator
        # This catches tags like #initiativedfp-revamp, #todoapi-platform, etc.
        for prefix in self.tag_prefixes.keys():
            # Match tags like #initiative[word-with-dashes]
            pattern = (rf'\b#{prefix}([a-z][a-zA-Z0-9-]+)\b', rf'#{prefix}/\1')
            patterns.append(pattern)
            
        # Special patterns for specific corruptions observed
        specific_patterns = [
            (r'\b#initiativedfp-revamp\b', '#initiative/dfp-revamp'),
            (r'\b#initiativepayments\b', '#initiative/payments'),
            (r'\b#initiativechargeback-uploader\b', '#initiative/chargeback-uploader'),
            (r'\b#initiativecbautolabel\b', '#initiative/cbautolabel'),
            (r'\b#initiativesift-decisions\b', '#initiative/sift-decisions'),
            (r'\b#initiativefraudrulegenerator\b', '#initiative/fraud-rule-generator'),
            (r'\b#initiative100-grain-limit\b', '#initiative/100-grain-limit'),
            (r'\b#initiativedoordash-dfp-rfp\b', '#initiative/doordash-dfp-rfp'),
            (r'\b#todoapi-platform\b', '#todo/api-platform'),
            (r'\b#todopayments\b', '#todo/payments'),
            (r'\b#tododfp\b', '#todo/dfp'),
            (r'\b#todourgent\b', '#todo/urgent'),
        ]
        patterns.extend(specific_patterns)
            
        return patterns
    
    def restore_tags_in_content(self, content, file_path):
        """Restore hierarchical tags in content"""
        original_content = content
        patterns = self.generate_restoration_patterns()
        
        # Process each pattern
        for pattern, replacement in patterns:
            # Use re.sub for all replacements to handle groups properly
            if '\\1' in replacement:
                # Pattern with capturing group
                new_content = re.sub(pattern, replacement, content)
            else:
                # Simple replacement pattern
                new_content = re.sub(pattern, replacement, content)
            
            # Check if any changes were made
            if new_content != content:
                # Find what changed
                import difflib
                diff = list(difflib.unified_diff(content.splitlines(), new_content.splitlines(), lineterm=''))
                for line in diff:
                    if line.startswith('-') and not line.startswith('---'):
                        # Extract the old tag
                        for match in re.finditer(pattern, line[1:]):
                            old_tag = match.group(0)
                            if '\\1' in replacement:
                                new_tag = re.sub(pattern, replacement, old_tag)
                            else:
                                new_tag = replacement
                            self.log_change(file_path, 'restored_hierarchy', old_tag, new_tag)
                
                content = new_content
        
        # Also check for any remaining suspicious patterns
        # Tags that look like they should be hierarchical but aren't
        suspicious_pattern = r'\b#([a-z]+)([A-Z][a-zA-Z0-9_-]+)\b'
        for match in re.finditer(suspicious_pattern, content):
            tag = match.group(0)
            prefix = match.group(1)
            suffix = match.group(2).lower()
            
            # Check if this looks like it should be hierarchical
            if prefix in self.tag_prefixes:
                new_tag = f"#{prefix}/{suffix}"
                content = content.replace(tag, new_tag)
                self.log_change(file_path, 'restored_suspicious', tag, new_tag)
                
        return content
        
    def log_change(self, file_path, change_type, old_value, new_value):
        """Log a change for reporting"""
        self.changes_made[change_type].append({
            'file': str(file_path.relative_to(self.vault_path)),
            'old': old_value,
            'new': new_value,
            'line': self.find_line_number(file_path, old_value)
        })
        
    def find_line_number(self, file_path, text):
        """Find the line number where the text appears"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if text in line:
                        return i
        except:
            pass
        return None
        
    def process_file(self, file_path):
        """Process a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            content = self.restore_tags_in_content(content, file_path)
            
            # Only write if changes were made and not in dry run
            if content != original_content and not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    def process_vault(self):
        """Process all markdown files in the vault"""
        # Create backup first
        backup_path = None
        if not self.dry_run:
            backup_path = self.create_backup()
            
        md_files = list(self.vault_path.rglob("*.md"))
        
        print(f"Processing {len(md_files)} markdown files...")
        print(f"Mode: {'DRY RUN - No changes will be made' if self.dry_run else 'LIVE - Changes will be applied'}")
        print("-" * 60)
        
        for file_path in md_files:
            # Skip hidden directories and common excluded paths
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if any(exclude in file_path.parts for exclude in ['node_modules', '.trash', '.obsidian']):
                continue
                
            self.process_file(file_path)
            
        # Print summary
        self.print_summary(backup_path)
        
    def print_summary(self, backup_path=None):
        """Print a summary of changes"""
        print("\n" + "=" * 60)
        print("TAG HIERARCHY RESTORATION SUMMARY")
        print("=" * 60)
        
        total_changes = 0
        for change_type, changes in self.changes_made.items():
            print(f"\n{change_type.replace('_', ' ').title()}:")
            print(f"  Total: {len(changes)} tags restored")
            
            # Group by tag pattern
            tag_groups = defaultdict(list)
            for change in changes:
                base_tag = change['new'].split('/')[0]
                tag_groups[base_tag].append(change)
            
            # Show examples grouped by base tag
            for base_tag, group_changes in sorted(tag_groups.items()):
                print(f"\n  {base_tag} tags ({len(group_changes)} restored):")
                for change in group_changes[:3]:  # Show first 3 examples
                    print(f"    - {change['old']} → {change['new']} in {Path(change['file']).name}")
                if len(group_changes) > 3:
                    print(f"    ... and {len(group_changes) - 3} more")
                    
            total_changes += len(changes)
            
        print(f"\nTotal tags restored: {total_changes}")
        
        if backup_path:
            print(f"\n✅ Backup created at: {backup_path}")
            
        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No changes were made")
            print("Run with --apply to restore your tags")
        else:
            print("\n✅ Tag hierarchy restoration completed!")
            
        # Save detailed report
        report_path = self.vault_path / f'tag_restoration_report_{self.timestamp}.json'
        report_data = {
            'timestamp': self.timestamp,
            'dry_run': self.dry_run,
            'total_files_processed': len(list(self.vault_path.rglob("*.md"))),
            'total_changes': total_changes,
            'backup_path': str(backup_path) if backup_path else None,
            'changes': dict(self.changes_made)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Also create a human-readable summary
        summary_path = self.vault_path / f'tag_restoration_summary_{self.timestamp}.md'
        with open(summary_path, 'w') as f:
            f.write(f"# Tag Hierarchy Restoration Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Mode:** {'Dry Run' if self.dry_run else 'Applied'}\n")
            f.write(f"**Total Tags Restored:** {total_changes}\n\n")
            
            if backup_path:
                f.write(f"**Backup Location:** `{backup_path}`\n\n")
                
            f.write("## Restored Tags by Category\n\n")
            for base_tag, group_changes in sorted(tag_groups.items()):
                f.write(f"### {base_tag} ({len(group_changes)} tags)\n\n")
                for change in group_changes:
                    f.write(f"- `{change['old']}` → `{change['new']}` in {change['file']}\n")
                f.write("\n")
                
        print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Restore hierarchical tag structure in Obsidian vault',
        epilog='Example: python restore_tag_hierarchy.py "/path/to/vault" --dry-run'
    )
    parser.add_argument('vault_path', help='Path to Obsidian vault')
    parser.add_argument('--apply', action='store_true', 
                       help='Apply changes (default is dry run)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup (not recommended)')
    
    args = parser.parse_args()
    
    # Confirm before applying changes
    if args.apply:
        print("\n⚠️  WARNING: This will modify your vault files!")
        if not args.no_backup:
            print("A backup will be created first.")
        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    restorer = TagHierarchyRestorer(
        vault_path=args.vault_path,
        dry_run=not args.apply,
        backup=not args.no_backup
    )
    
    restorer.process_vault()

if __name__ == '__main__':
    main()