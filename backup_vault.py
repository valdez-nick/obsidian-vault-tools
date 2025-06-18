#!/usr/bin/env python3
"""
Backup Obsidian Vault - Create timestamped backups before major changes
"""

import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import zipfile
import json

def create_backup(vault_path, backup_type="manual"):
    """Create a timestamped backup of the vault"""
    vault_path = Path(vault_path)
    
    if not vault_path.exists():
        print(f"❌ Vault path not found: {vault_path}")
        return None
        
    # Create backup directory
    backup_dir = vault_path.parent / "obsidian_backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"vault_backup_{backup_type}_{timestamp}"
    
    # Paths for backup
    backup_path = backup_dir / backup_name
    zip_path = backup_dir / f"{backup_name}.zip"
    
    print(f"🔄 Creating backup of: {vault_path.name}")
    print(f"📁 Backup location: {backup_dir}")
    
    # Items to exclude from backup
    exclude_patterns = [
        '.git',
        '.obsidian/workspace*',
        '.obsidian/cache',
        '.obsidian/plugins/*/data.json',
        'node_modules',
        '__pycache__',
        '*.pyc',
        '.DS_Store'
    ]
    
    def should_exclude(path):
        """Check if path should be excluded"""
        for pattern in exclude_patterns:
            if pattern in str(path):
                return True
        return False
    
    # Create backup info
    backup_info = {
        "timestamp": timestamp,
        "source_vault": str(vault_path),
        "backup_type": backup_type,
        "creation_date": datetime.now().isoformat(),
        "vault_stats": {
            "total_files": 0,
            "markdown_files": 0,
            "attachments": 0,
            "total_size_mb": 0
        }
    }
    
    # Count files
    total_size = 0
    for item in vault_path.rglob("*"):
        if item.is_file() and not should_exclude(item):
            backup_info["vault_stats"]["total_files"] += 1
            total_size += item.stat().st_size
            
            if item.suffix == ".md":
                backup_info["vault_stats"]["markdown_files"] += 1
            elif item.suffix in [".png", ".jpg", ".jpeg", ".pdf", ".mp3", ".mp4"]:
                backup_info["vault_stats"]["attachments"] += 1
    
    backup_info["vault_stats"]["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    print(f"📊 Vault stats:")
    print(f"   - Markdown files: {backup_info['vault_stats']['markdown_files']}")
    print(f"   - Attachments: {backup_info['vault_stats']['attachments']}")
    print(f"   - Total size: {backup_info['vault_stats']['total_size_mb']} MB")
    
    # Create zip backup
    print(f"\n📦 Creating zip archive...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add backup info
        zipf.writestr("backup_info.json", json.dumps(backup_info, indent=2))
        
        # Add vault files
        files_backed_up = 0
        for item in vault_path.rglob("*"):
            if item.is_file() and not should_exclude(item):
                arcname = item.relative_to(vault_path.parent)
                zipf.write(item, arcname)
                files_backed_up += 1
                
                # Progress indicator
                if files_backed_up % 100 == 0:
                    print(f"   Processed {files_backed_up} files...")
    
    # Get final size
    zip_size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)
    
    print(f"\n✅ Backup completed!")
    print(f"📦 Archive: {zip_path}")
    print(f"📏 Size: {zip_size_mb} MB (compressed from {backup_info['vault_stats']['total_size_mb']} MB)")
    print(f"📄 Files backed up: {files_backed_up}")
    
    # Create restore script
    restore_script = backup_dir / f"restore_{backup_name}.sh"
    with open(restore_script, 'w') as f:
        f.write(f"""#!/bin/bash
# Restore script for backup: {backup_name}
# Created: {datetime.now().isoformat()}

echo "⚠️  This will restore vault from backup: {backup_name}"
echo "Target location: {vault_path}"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 1
fi

# Create restore point of current vault
echo "Creating restore point of current vault..."
mv "{vault_path}" "{vault_path}_before_restore_{timestamp}"

# Extract backup
echo "Extracting backup..."
unzip -q "{zip_path}" -d "{vault_path.parent}"

echo "✅ Restore completed!"
echo "Previous vault saved as: {vault_path}_before_restore_{timestamp}"
""")
    
    os.chmod(restore_script, 0o755)
    print(f"\n🔧 Restore script created: {restore_script}")
    
    return str(zip_path)

def list_backups(vault_path):
    """List all available backups"""
    backup_dir = Path(vault_path).parent / "obsidian_backups"
    
    if not backup_dir.exists():
        print("No backups found")
        return
        
    backups = list(backup_dir.glob("vault_backup_*.zip"))
    backups.sort(reverse=True)
    
    print(f"\n📚 Available backups ({len(backups)} total):")
    print("-" * 60)
    
    for backup in backups[:10]:  # Show last 10
        size_mb = round(backup.stat().st_size / (1024 * 1024), 2)
        # Extract date from filename
        parts = backup.stem.split('_')
        if len(parts) >= 4:
            date_str = parts[3]
            time_str = parts[4] if len(parts) > 4 else "000000"
            
            # Format date nicely
            date = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            print(f"  {date.strftime('%Y-%m-%d %H:%M:%S')} - {parts[2]} - {size_mb} MB")
    
    if len(backups) > 10:
        print(f"  ... and {len(backups) - 10} more")

def main():
    parser = argparse.ArgumentParser(description='Backup Obsidian vault')
    parser.add_argument('vault_path', help='Path to Obsidian vault')
    parser.add_argument('--type', default='manual', 
                       choices=['manual', 'pre-tags', 'pre-cleanup', 'scheduled'],
                       help='Type of backup (for naming)')
    parser.add_argument('--list', action='store_true', help='List existing backups')
    
    args = parser.parse_args()
    
    # Validate vault path exists
    if not os.path.exists(args.vault_path):
        print(f"❌ Error: Vault path does not exist: {args.vault_path}")
        sys.exit(1)
    
    if args.list:
        list_backups(args.vault_path)
    else:
        create_backup(args.vault_path, args.type)

if __name__ == '__main__':
    main()