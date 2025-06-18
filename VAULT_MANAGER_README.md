# Obsidian Vault Manager 🎮

An interactive, menu-driven interface for managing your Obsidian vault - inspired by classic text-based adventure games!

## Quick Start

Simply run:
```bash
./obsidian_manager
```

Or with Python directly:
```bash
python3 vault_manager.py
```

## Features

### 📊 Vault Analysis
- View tag statistics at a glance
- Find files without tags
- Generate detailed reports
- Export analysis data

### 🏷️ Tag Management
- **Preview Issues**: See tag problems before fixing
- **Auto-Fix All**: One-click fix for all tag issues
- **Selective Fixes**: Fix only specific types of issues
- **Smart Merging**: Intelligently merge similar tags

### 💾 Backup Solutions
- **Quick Backup**: Fast incremental backups using hard links
- **Full Backup**: Compressed .zip archives with metadata
- **Kopia Setup**: Advanced backup with deduplication
- **Restore**: Easy restoration from any backup

### 🎮 User Experience
- **Menu Navigation**: No commands to memorize!
- **Color-Coded Output**: Easy to read status and errors
- **Progress Indicators**: Know what's happening
- **Confirmation Prompts**: Never accidentally modify files
- **Persistent Settings**: Remembers your vault location

## Menu Structure

```
MAIN MENU
├── 📊 Analyze Vault
│   ├── View tag statistics
│   ├── Generate detailed report
│   ├── Analyze folder structure
│   ├── Find files without tags
│   └── Export analysis to JSON
│
├── 🏷️ Manage Tags
│   ├── Preview tag issues
│   ├── Fix all tag issues (auto)
│   ├── Fix quoted tags only
│   ├── Merge similar tags
│   ├── Remove generic tags
│   └── Auto-tag untagged files
│
├── 💾 Backup Vault
│   ├── Quick backup (incremental)
│   ├── Full backup (compressed)
│   ├── Setup Kopia (advanced)
│   ├── View backup history
│   └── Restore from backup
│
├── 🔧 Advanced Tools
├── 📚 Help & Documentation
└── ⚙️ Settings
```

## First Run

On first run, the manager will:
1. Welcome you with a friendly message
2. Ask for your vault location
3. Save your preference for next time
4. Show the main menu

## Tag Fixes Applied

The tag management system can fix:
- **Quoted tags**: `"#todo"` → `#todo`
- **Similar tags**: `daily-notes` → `daily-note`
- **Generic tags**: Removes `#1`, `#2`, `#notes`
- **Hierarchies**: `#todo/` → `#todo`

## Backup Strategies

### Quick Backup (Incremental)
- Uses rsync with hard links
- Very fast after first backup
- Space-efficient
- Perfect for daily use

### Full Backup (Compressed)
- Creates .zip archives
- Includes vault statistics
- Generates restore scripts
- Good for archival

### Kopia (Advanced)
- Deduplication across backups
- Compression and encryption
- Retention policies
- Best for large vaults

## Configuration

Settings are saved in:
```
~/.obsidian_vault_manager.json
```

## Tips

1. **Always backup first**: Before any tag operations
2. **Use preview mode**: Check changes before applying
3. **Regular backups**: Set up a backup routine
4. **Check for updates**: New features added regularly

## Requirements

- Python 3.6+
- Bash (for backup scripts)
- Optional: Kopia (for advanced backups)

## Troubleshooting

### Colors not showing?
- Make sure your terminal supports ANSI colors
- Try a different terminal emulator

### Scripts not found?
- Ensure all `.py` and `.sh` files are in the same directory
- Check file permissions (`chmod +x *.py *.sh`)

### Backup failing?
- Check disk space
- Ensure write permissions to backup location
- Try running with more verbose output

## Future Features

- [ ] Batch tag operations
- [ ] Scheduled backups
- [ ] Plugin management
- [ ] Vault statistics dashboard
- [ ] Export/import settings
- [ ] Theme selection

---

Enjoy managing your vault the fun way! 🎮📚