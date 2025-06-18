#!/bin/bash
# Setup Kopia backup for Obsidian vault

echo "🔧 Setting up Kopia backup for Obsidian vault..."
echo ""
echo "This script will:"
echo "1. Create a Kopia repository for backups"
echo "2. Configure optimal settings for Obsidian"
echo "3. Create your first backup snapshot"
echo ""

BACKUP_PATH="/Users/nvaldez/Documents/repos/obsidian_backups/kopia"
VAULT_PATH="/Users/nvaldez/Documents/repos/Obsidian"

# Check if repository already exists
if [ -f "$BACKUP_PATH/kopia.repository.f" ]; then
    echo "✅ Kopia repository already exists at: $BACKUP_PATH"
    echo "Connecting to existing repository..."
    kopia repository connect filesystem --path="$BACKUP_PATH"
else
    echo "Creating new Kopia repository at: $BACKUP_PATH"
    echo ""
    echo "⚠️  You'll need to set a password for your backup repository."
    echo "This password encrypts your backups. Store it safely!"
    echo ""
    
    # Create repository (will prompt for password)
    kopia repository create filesystem --path="$BACKUP_PATH"
fi

# Check if connected successfully
if kopia repository status &>/dev/null; then
    echo ""
    echo "✅ Repository connected successfully!"
    
    # Set up optimal policy for Obsidian
    echo ""
    echo "📋 Configuring backup policy for Obsidian..."
    
    kopia policy set --global \
        --compression=zstd-better-compression \
        --keep-latest=50 \
        --keep-hourly=24 \
        --keep-daily=30 \
        --keep-weekly=8 \
        --keep-monthly=12 \
        --keep-annual=3
    
    # Add ignore rules for Obsidian
    kopia policy set --global \
        --add-ignore=".obsidian/workspace*" \
        --add-ignore=".obsidian/cache" \
        --add-ignore=".obsidian/plugins/*/data.json" \
        --add-ignore="node_modules/" \
        --add-ignore="__pycache__/" \
        --add-ignore="*.pyc" \
        --add-ignore=".DS_Store" \
        --add-ignore="obsidian-librarian-v2/rust-core/target/" \
        --add-ignore="obsidian-librarian-v2/python/target/"
    
    echo "✅ Policy configured!"
    
    # Create first snapshot
    echo ""
    echo "📸 Creating first backup snapshot..."
    echo "This may take a few minutes for the initial backup..."
    
    kopia snapshot create "$VAULT_PATH"
    
    # Show snapshot info
    echo ""
    echo "📊 Backup Summary:"
    kopia snapshot list
    
    # Show repository stats
    echo ""
    echo "💾 Repository Statistics:"
    kopia repository status
    
    echo ""
    echo "✅ Backup setup complete!"
    echo ""
    echo "🔧 Useful commands:"
    echo "  - Create new snapshot:  kopia snapshot create $VAULT_PATH"
    echo "  - List snapshots:       kopia snapshot list"
    echo "  - Mount snapshot:       kopia mount all /tmp/kopia-mount"
    echo "  - Show differences:     kopia diff k<snapshot-id-1> k<snapshot-id-2>"
    echo ""
    echo "💡 To automate backups, add this to your crontab:"
    echo "  0 * * * * /opt/homebrew/bin/kopia snapshot create $VAULT_PATH"
    
else
    echo "❌ Failed to connect to repository. Please check your password and try again."
    exit 1
fi