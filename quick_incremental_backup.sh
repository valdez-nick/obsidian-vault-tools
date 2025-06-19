#!/bin/bash
# Quick incremental backup using rsync

# Check if vault directory is provided as argument
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide vault directory path"
    echo "Usage: $0 <vault_directory_path>"
    exit 1
fi

VAULT_DIR="$1"
# Create backup directory next to the vault or in user's home
if [ -w "$(dirname "$VAULT_DIR")" ]; then
    BACKUP_BASE="$(dirname "$VAULT_DIR")/obsidian_backups"
else
    BACKUP_BASE="$HOME/ObsidianBackups/$(basename "$VAULT_DIR")"
fi
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LATEST_LINK="$BACKUP_BASE/latest"
BACKUP_DIR="$BACKUP_BASE/backup_$TIMESTAMP"

echo "🔄 Creating incremental backup of Obsidian vault..."
echo "📁 Source: $VAULT_DIR"
echo "📁 Destination: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_BASE"

# Perform incremental backup using hard links to save space
if [ -L "$LATEST_LINK" ] && [ -d "$(readlink "$LATEST_LINK")" ]; then
    echo "📦 Using previous backup for hard links: $(readlink "$LATEST_LINK")"
    rsync -av --delete \
        --link-dest="$(readlink "$LATEST_LINK")" \
        --exclude='.git' \
        --exclude='.obsidian/workspace*' \
        --exclude='.obsidian/cache' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        --exclude='obsidian-librarian-v2/rust-core/target' \
        --exclude='obsidian-librarian-v2/python/target' \
        "$VAULT_DIR/" "$BACKUP_DIR/"
else
    echo "📦 Creating first backup (no previous backup found)..."
    rsync -av --delete \
        --exclude='.git' \
        --exclude='.obsidian/workspace*' \
        --exclude='.obsidian/cache' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        --exclude='obsidian-librarian-v2/rust-core/target' \
        --exclude='obsidian-librarian-v2/python/target' \
        "$VAULT_DIR/" "$BACKUP_DIR/"
fi

# Update latest symlink
rm -f "$LATEST_LINK"
ln -s "$BACKUP_DIR" "$LATEST_LINK"

# Calculate sizes
VAULT_SIZE=$(du -sh "$VAULT_DIR" | cut -f1)
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
TOTAL_BACKUPS=$(ls -1d "$BACKUP_BASE"/backup_* 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "✅ Backup completed!"
echo "📏 Vault size: $VAULT_SIZE"
echo "📏 This backup: $BACKUP_SIZE (using hard links for unchanged files)"
echo "🗂️  Total backups: $TOTAL_BACKUPS"
echo "📁 Backup location: $BACKUP_DIR"
echo ""
echo "To restore from this backup:"
echo "  rsync -av \"$BACKUP_DIR/\" \"$VAULT_DIR/\""