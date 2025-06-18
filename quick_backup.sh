#!/bin/bash
# Quick backup script for Obsidian vault

VAULT_DIR="/Users/nvaldez/Documents/repos/Obsidian"
BACKUP_DIR="/Users/nvaldez/Documents/repos/obsidian_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="vault_backup_pre-tags_${TIMESTAMP}"

echo "🔄 Creating quick backup of Obsidian vault..."
echo "📁 Source: $VAULT_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create tar backup (faster than zip for large vaults)
echo "📦 Creating backup archive..."
cd "$(dirname "$VAULT_DIR")"
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" \
    --exclude='.git' \
    --exclude='.obsidian/workspace*' \
    --exclude='.obsidian/cache' \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='obsidian-librarian-v2/rust-core/target' \
    --exclude='obsidian-librarian-v2/python/target' \
    "$(basename "$VAULT_DIR")"

# Check size
SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)

echo "✅ Backup completed!"
echo "📦 Archive: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo "📏 Size: $SIZE"
echo ""
echo "To restore, run:"
echo "  cd $(dirname "$VAULT_DIR")"
echo "  tar -xzf $BACKUP_DIR/${BACKUP_NAME}.tar.gz"