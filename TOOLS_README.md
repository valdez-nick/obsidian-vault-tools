# Obsidian Vault Management Tools - User Guide

A comprehensive suite of tools for managing, analyzing, and enhancing your Obsidian vault, featuring interactive menu interfaces and ASCII art integration.

## 🚀 Quick Start

### Interactive Vault Manager (Recommended)
```bash
# Standard version
./obsidian_manager

# Enhanced version with ASCII art
./obsidian_manager_enhanced
```

## 📋 Available Tools

### 1. Interactive Vault Manager
**File:** `vault_manager.py` / `obsidian_manager`

A menu-driven interface inspired by classic text-based adventure games. No commands to memorize!

**Features:**
- 📊 **Vault Analysis** - Tag statistics, folder structure, untagged files
- 🏷️ **Tag Management** - Fix quoted tags, merge similar tags, remove generic tags
- 💾 **Backup Solutions** - Quick incremental, full compressed, Kopia setup
- 🚀 **V2 Integration** - Detects and integrates with obsidian-librarian v2
- 🔧 **Advanced Tools** - Performance benchmarks, cache cleaning
- ⚙️ **Settings** - Persistent configuration, customizable options

### 2. Enhanced Vault Manager with ASCII Art
**File:** `vault_manager_enhanced.py` / `obsidian_manager_enhanced`

All features of the standard manager plus:
- 🎨 **ASCII Art Tools** - Convert images to ASCII art
- 📷 **Screenshot to ASCII** - Instant screenshot conversion
- 🖼️ **ASCII Gallery** - Browse pre-loaded ASCII art
- 📚 **Note Integration** - Add ASCII art to your vault notes
- ✨ **Enhanced Interface** - ASCII art decorations throughout

**ASCII Art Styles:**
- `simple` - Basic characters (.:-=+*#%@)
- `standard` - Extended set for more detail
- `blocks` - Unicode blocks (░▒▓█)
- `detailed` - 70+ character set
- `classic` - Traditional ASCII art style
- `matrix` - Matrix-style characters

### 3. Tag Analysis Tool
**File:** `analyze_tags_simple.py`

Standalone tag analyzer that works without dependencies.

**Usage:**
```bash
python3 analyze_tags_simple.py /path/to/vault
python3 analyze_tags_simple.py /path/to/vault --detailed
python3 analyze_tags_simple.py /path/to/vault --output report.json
```

**Reports:**
- Total unique tags and occurrences
- Files with/without tags
- Tag frequency analysis
- Tag hierarchies
- Similar tags detection
- Tag issues (quoted, generic, incomplete)

### 4. Tag Fixing Tool
**File:** `fix_vault_tags.py`

Automated tag standardization and cleanup.

**Usage:**
```bash
# Preview changes
python3 fix_vault_tags.py /path/to/vault --dry-run

# Fix all issues
python3 fix_vault_tags.py /path/to/vault

# Fix specific issues
python3 fix_vault_tags.py /path/to/vault --fix-quoted-only
python3 fix_vault_tags.py /path/to/vault --merge-similar
python3 fix_vault_tags.py /path/to/vault --remove-generic
```

**Fixes:**
- Quoted tags: `"#todo"` → `#todo`
- Similar tags: `#daily-notes` → `#daily-note`
- Generic tags: Removes `#1`, `#2`, `#notes`
- Incomplete hierarchies: `#todo/` → `#todo`

### 5. Backup Tools

#### Python Backup Script
**File:** `backup_vault.py`

Creates timestamped zip archives with metadata.

```bash
python3 backup_vault.py /path/to/vault
```

#### Quick Incremental Backup
**File:** `quick_incremental_backup.sh`

Fast incremental backups using rsync with hard links.

```bash
./quick_incremental_backup.sh /path/to/vault
```

#### Kopia Setup Script
**File:** `kopia_backup_setup.sh`

Advanced backup with deduplication and encryption.

```bash
./kopia_backup_setup.sh /path/to/vault
```

### 6. ASCII Art Converters

#### Better ASCII Converter
**File:** `better_ascii_converter.py`

Creates traditional-style ASCII art with multiple character sets.

```bash
# Convert with specific style
python3 better_ascii_converter.py image.jpg --style classic --width 100

# Try all styles
python3 better_ascii_converter.py image.jpg --style auto

# Invert brightness
python3 better_ascii_converter.py image.jpg --style blocks --invert
```

#### ASCII Magic Converter
**File:** `ascii_magic_converter.py`

High-quality photographic ASCII art (requires ascii-magic).

```bash
python3 ascii_magic_converter.py image.jpg --width 150
```

## 🎮 Using the Interactive Manager

### First Run
1. Launch the manager
2. Enter your vault path when prompted
3. Path is saved for future sessions

### Navigation
- Use number keys (1-9, 0) to select options
- Press 0 to go back or exit
- Follow on-screen prompts
- All actions have confirmation prompts

### Menu Structure
```
MAIN MENU
├── 📊 Analyze Vault
│   ├── View tag statistics
│   ├── Generate detailed report
│   ├── Analyze folder structure
│   ├── Find files without tags
│   └── Export analysis to JSON
├── 🏷️ Manage Tags
│   ├── Preview tag issues
│   ├── Fix all tag issues
│   ├── Fix specific issues
│   └── Auto-tag files (v2)
├── 💾 Backup Vault
│   ├── Quick incremental
│   ├── Full compressed
│   ├── Setup Kopia
│   └── Restore options
├── 🎨 ASCII Art Tools (Enhanced only)
│   ├── Convert images
│   ├── Screenshot to ASCII
│   ├── Gallery browser
│   └── Add to notes
└── ⚙️ Settings
```

## 📦 Installation

### Requirements
- Python 3.6+
- Bash shell
- Optional: PIL/Pillow for ASCII art
- Optional: numpy for image processing
- Optional: Kopia for advanced backups

### Setup
```bash
# Clone or download the tools
git clone <repository>

# Make scripts executable
chmod +x *.py *.sh obsidian_manager*

# Install optional dependencies
pip install pillow numpy  # For ASCII art features
```

## 🏷️ Obsidian Librarian v2 Integration

The managers automatically detect if obsidian-librarian v2 is installed and enable advanced features:

- AI-powered tag suggestions
- Content analysis and quality scoring
- Smart file organization
- Research assistant
- Duplicate detection with ML

To install v2:
```bash
pip install obsidian-librarian
# or
git clone https://github.com/valdez-nick/obsidian-librarian-v2
cd obsidian-librarian-v2
make install
```

## 💡 Tips & Best Practices

### Tag Management
1. Always backup before bulk tag operations
2. Use preview/dry-run mode first
3. Review the tag analysis report
4. Fix quoted tags before merging similar ones

### Backups
1. Use quick incremental for daily backups
2. Create full backups weekly
3. Consider Kopia for large vaults (deduplication)
4. Test restore process periodically

### ASCII Art
1. Use high-contrast images for best results
2. Try different styles to find what works
3. Width 80-120 chars for most uses
4. Save favorites to reuse

### Performance
1. Exclude large directories (.obsidian, Archive)
2. Run analysis during off-hours for large vaults
3. Use the cache for repeated operations
4. Close Obsidian during major operations

## 🔧 Troubleshooting

### Common Issues

**Colors not showing?**
- Ensure terminal supports ANSI colors
- Try different terminal emulator

**Scripts not found?**
- Check file permissions: `chmod +x *.py`
- Ensure all files are in same directory

**Backup failing?**
- Check disk space
- Verify write permissions
- Ensure vault path is correct

**ASCII art not working?**
- Install PIL: `pip install pillow numpy`
- Check image format (JPG, PNG work best)
- Try different styles if output unclear

## 📝 Configuration

Settings stored in: `~/.obsidian_vault_manager.json`

Example:
```json
{
  "last_vault": "/path/to/vault",
  "welcomed": true,
  "ascii_enabled": true,
  "backup_dir": "~/ObsidianBackups",
  "colors_enabled": true
}
```

## 🚀 Advanced Usage

### Automation
```bash
# Daily backup cron job
0 2 * * * /path/to/quick_incremental_backup.sh /vault/path

# Weekly tag cleanup
0 3 * * 0 python3 /path/to/fix_vault_tags.py /vault/path
```

### Custom Scripts
The tools can be imported and used in your own Python scripts:

```python
from analyze_tags_simple import analyze_tags
from fix_vault_tags import TagFixer

# Analyze tags
results = analyze_tags('/path/to/vault')
print(f"Total tags: {results['summary']['total_tags']}")

# Fix tags programmatically
fixer = TagFixer('/path/to/vault')
fixer.fix_all_tags()
```

## 📚 Additional Resources

- [Obsidian Documentation](https://help.obsidian.md)
- [Obsidian Librarian v2](https://github.com/valdez-nick/obsidian-librarian-v2)
- [ASCII Art Guide](ASCII_ART_GUIDE.md)
- [Vault Manager README](VAULT_MANAGER_README.md)

---

*Enjoy managing your Obsidian vault with style! 🎨📚*