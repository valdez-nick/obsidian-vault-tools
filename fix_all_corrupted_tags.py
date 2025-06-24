#!/usr/bin/env python3
"""
Fix all corrupted hierarchical tags based on the discovered patterns
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import shutil
import argparse

class ComprehensiveTagFixer:
    def __init__(self, vault_path, dry_run=True, backup=True):
        self.vault_path = Path(vault_path)
        self.dry_run = dry_run
        self.backup = backup
        self.changes_made = defaultdict(list)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # All the corrupted tags we found
        self.tag_fixes = {
            # Initiative tags
            '#initiativecbautolabel': '#initiative/cbautolabel',
            '#initiativechargback-uploader': '#initiative/chargeback-uploader', 
            '#initiativedfp-revamp': '#initiative/dfp-revamp',
            '#initiativedoordash-dfp-rfp': '#initiative/doordash-dfp-rfp',
            '#initiativefraudrulegenerator': '#initiative/fraud-rule-generator',
            '#initiativelearnable-chargebacks': '#initiative/learnable-chargebacks',
            '#initiativename': '#initiative/name',
            '#initiativeneural-network-exploration': '#initiative/neural-network-exploration',
            '#initiativepayments': '#initiative/payments',
            '#initiativesift-decisions': '#initiative/sift-decisions',
            
            # Todo tags
            '#todoapi-platform': '#todo/api-platform',
            '#todocbbackfill': '#todo/cbbackfill',
            '#tododaily': '#todo/daily',
            '#tododfp': '#todo/dfp',
            '#todofeature-relase': '#todo/feature-release',  # Also fixing typo
            '#todoidentity': '#todo/identity',
            '#todoopen-question': '#todo/open-question',
            '#todooverdue': '#todo/overdue',
            '#todopayment': '#todo/payment',
            '#todopayments': '#todo/payments',
            '#todorecurring': '#todo/recurring',
            '#todourgent': '#todo/urgent',
            
            # Product tags (these had wrong fixes in the finder)
            '#product-advice': '#product/advice',
            '#product-design': '#product/design',
            '#product-ideas': '#product/ideas',
            '#product-knowledge': '#product/knowledge',
            '#product-leadership': '#product/leadership',
            '#product-management': '#product/management',
            '#product-research': '#product/research',
            '#product-tag': '#product/tag',
            '#product-team': '#product/team',
            
            # OKR tags
            '#okr-updates': '#okr/updates',
            
            # Project tags
            '#projects': '#project/s',  # or maybe #projects (plural)?
        }
        
    def create_backup(self):
        """Create a backup of the vault before making changes"""
        if self.backup and not self.dry_run:
            backup_dir = self.vault_path.parent / f"vault_backup_tags_{self.timestamp}"
            print(f"Creating backup at: {backup_dir}")
            shutil.copytree(self.vault_path, backup_dir, 
                          ignore=shutil.ignore_patterns('.git', '.obsidian', 'node_modules'))
            return backup_dir
        return None
        
    def fix_tags_in_content(self, content, file_path):
        """Fix all corrupted tags in content"""
        original_content = content
        
        for corrupted_tag, fixed_tag in self.tag_fixes.items():
            # Use proper pattern that works with # character
            # Match the tag when followed by whitespace, punctuation, or end of line
            pattern = rf'{re.escape(corrupted_tag)}(?=\s|$|[,;.!?)\]}}>])'
            
            # Count occurrences before replacement
            occurrences = len(re.findall(pattern, content))
            
            if occurrences > 0:
                # Replace all occurrences
                content = re.sub(pattern, fixed_tag, content)
                
                # Log each occurrence
                for _ in range(occurrences):
                    self.log_change(file_path, 'fixed_tag', corrupted_tag, fixed_tag)
                    
        return content
        
    def log_change(self, file_path, change_type, old_value, new_value):
        """Log a change for reporting"""
        self.changes_made[change_type].append({
            'file': str(file_path.relative_to(self.vault_path)),
            'old': old_value,
            'new': new_value
        })
        
    def process_file(self, file_path):
        """Process a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            content = self.fix_tags_in_content(content, file_path)
            
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
        print(f"Looking for {len(self.tag_fixes)} corrupted tag patterns")
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
        print("TAG FIX SUMMARY")
        print("=" * 60)
        
        if not self.changes_made:
            print("\nNo corrupted tags found!")
            return
            
        # Group changes by tag
        tag_summary = defaultdict(list)
        for change in self.changes_made.get('fixed_tag', []):
            tag_key = f"{change['old']} → {change['new']}"
            tag_summary[tag_key].append(change['file'])
            
        print(f"\nTotal tags fixed: {len(self.changes_made.get('fixed_tag', []))}")
        print(f"Unique tag patterns fixed: {len(tag_summary)}")
        
        print("\nFixed tags by type:")
        
        # Group by tag prefix
        prefix_groups = defaultdict(list)
        for tag_change, files in tag_summary.items():
            old_tag = tag_change.split(' → ')[0]
            prefix = old_tag.split('#')[1].split('-')[0] if '-' in old_tag else old_tag.split('#')[1][:4]
            prefix_groups[prefix].append((tag_change, files))
            
        for prefix, changes in sorted(prefix_groups.items()):
            print(f"\n{prefix.upper()} tags:")
            for tag_change, files in sorted(changes):
                print(f"  {tag_change} ({len(files)} occurrences)")
                # Show first few files
                for file in files[:3]:
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
                    
        if backup_path:
            print(f"\n✅ Backup created at: {backup_path}")
            
        if self.dry_run:
            print("\n⚠️  DRY RUN MODE - No changes were made")
            print("Run with --apply to fix your tags")
        else:
            print("\n✅ All corrupted tags have been fixed!")
            
        # Save detailed report
        report_path = self.vault_path / f'tag_fix_report_{self.timestamp}.json'
        report_data = {
            'timestamp': self.timestamp,
            'dry_run': self.dry_run,
            'total_files_processed': len(list(self.vault_path.rglob("*.md"))),
            'total_fixes': len(self.changes_made.get('fixed_tag', [])),
            'unique_patterns_fixed': len(tag_summary),
            'backup_path': str(backup_path) if backup_path else None,
            'tag_fixes_applied': self.tag_fixes,
            'changes': dict(self.changes_made)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Fix all corrupted hierarchical tags in Obsidian vault'
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
        print("This script will fix all corrupted hierarchical tags.")
        if not args.no_backup:
            print("A backup will be created first.")
        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    fixer = ComprehensiveTagFixer(
        vault_path=args.vault_path,
        dry_run=not args.apply,
        backup=not args.no_backup
    )
    
    fixer.process_vault()

if __name__ == '__main__':
    main()