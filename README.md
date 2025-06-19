# Obsidian Vault Tools 🛠️

A comprehensive, modern toolkit for managing Obsidian vaults with AI-powered features, smart organization, and delightful user experience.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

## ✨ Features

### 🤖 AI-Powered Intelligence
- **Natural Language Queries**: Ask questions about your vault in plain English
- **Smart Content Analysis**: AI-powered insights into your notes
- **Research Assistant**: Automated research and note creation
- **Intelligent Organization**: Auto-organize files based on content

### 📊 Advanced Analysis
- **Tag Analysis**: Comprehensive tag usage reports and insights
- **Link Analysis**: Visualize connections between notes
- **Content Statistics**: Detailed metrics about your vault
- **Duplicate Detection**: Find and merge duplicate content

### 🎨 Creative Tools
- **ASCII Art Generator**: Convert images to ASCII art in multiple styles
- **Flowchart Generator**: Create ASCII flowcharts from markdown
- **Visual Enhancements**: Add visual elements to your notes

### 🎮 Enhanced Experience
- **Arrow Key Navigation**: Navigate menus with arrow keys
- **Audio Feedback**: Dungeon crawler-themed sound effects
- **Rich CLI Interface**: Beautiful, intuitive command-line interface

### 🔧 Vault Management
- **Smart Tag Management**: Fix, merge, and organize tags
- **Incremental Backups**: Efficient backup system
- **File Organization**: Organize files by content, date, or custom rules
- **Intelligent Cleanup**: AI-powered file cleanup suggestions

## 📦 Installation

### Via pip (Recommended)
```bash
pip install obsidian-vault-tools
```

### Via Homebrew (macOS/Linux)
```bash
brew tap yourusername/obsidian-tools
brew install obsidian-vault-tools
```

### From Source
```bash
git clone https://github.com/yourusername/obsidian-vault-tools.git
cd obsidian-vault-tools
pip install -e .
```

### With AI Features
```bash
pip install obsidian-vault-tools[ai]
```

## 🚀 Quick Start

### Basic Usage
```bash
# Set your vault location
obsidian-tools config set-vault ~/Documents/MyVault

# Or use with any vault
obsidian-tools --vault ~/Documents/MyVault analyze
```

### Common Commands
```bash
# Analyze your vault
ovt analyze

# Query with natural language
ovt query "What are my main project themes?"

# Fix and organize tags
ovt tags fix --preview
ovt tags fix --apply

# Smart file organization
ovt organize --by-content

# Create ASCII art
ovt ascii convert image.jpg --style matrix

# Backup vault
ovt backup --incremental
```

## 🎯 Key Commands

### Analysis & Insights
```bash
ovt analyze                    # Full vault analysis
ovt analyze tags              # Tag analysis
ovt analyze links             # Link analysis
ovt analyze duplicates        # Find duplicates
```

### AI-Powered Features
```bash
ovt query "question"          # Natural language query
ovt research "topic"          # Research and create notes
ovt organize --smart          # AI-powered organization
ovt curate                    # Content curation
```

### Tag Management
```bash
ovt tags preview              # Preview tag fixes
ovt tags fix                  # Apply all fixes
ovt tags merge                # Merge similar tags
ovt tags hierarchy            # Analyze tag hierarchy
```

### Creative Tools
```bash
ovt ascii image.jpg           # Convert image to ASCII
ovt flowchart note.md         # Generate flowchart
ovt ascii gallery             # View ASCII art gallery
```

### Maintenance
```bash
ovt backup                    # Create backup
ovt clean --analyze           # Analyze for cleanup
ovt clean --safe              # Safe cleanup
ovt version                   # File versioning
```

## ⚙️ Configuration

### Initial Setup
```bash
ovt config init               # Interactive setup
ovt config set-vault PATH     # Set default vault
ovt config show               # Show configuration
```

### Configuration File
Configuration is stored in `~/.obsidian-tools/config.yaml`:

```yaml
vault_path: ~/Documents/MyVault
theme: dungeon  # Audio theme
ai_model: gpt-3.5-turbo
backup_location: ~/Backups/obsidian
output_dirs:
  ascii: ./ascii-output
  analysis: ./analysis-output
```

## 🔌 AI Model Support

### Ollama (Local AI)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Configure
ovt config set-model llama2
```

### OpenAI
```bash
export OPENAI_API_KEY="your-key"
ovt config set-model gpt-4
```

## 🎨 Themes & Customization

### Audio Themes
- `dungeon`: Fantasy dungeon crawler (default)
- `scifi`: Sci-fi computer sounds
- `minimal`: Simple beeps
- `silent`: No audio

```bash
ovt config set-theme scifi
```

### ASCII Art Styles
- `simple`: Basic ASCII characters
- `standard`: Extended ASCII set
- `blocks`: Unicode block characters  
- `matrix`: Matrix-style characters
- `detailed`: High detail mode

## 📚 Advanced Usage

### Batch Operations
```bash
# Batch tag operations
ovt tags fix --pattern "#old*" --replace "#new"

# Bulk file operations
ovt organize --by-date --format "YYYY/MM"
```

### Automation
```bash
# Schedule backups
crontab -e
# Add: 0 2 * * * ovt backup --incremental

# Git integration
ovt git commit -m "Daily update"
ovt git sync
```

### Custom Scripts
```python
from obsidian_vault_tools import VaultManager

vault = VaultManager("~/Documents/MyVault")
vault.analyze()
results = vault.query("machine learning notes")
```

## 🐛 Troubleshooting

### Common Issues

**No audio on macOS**:
```bash
brew install pygame
```

**Arrow keys not working**:
```bash
# Check terminal compatibility
ovt config test-navigation
```

**AI features not available**:
```bash
# Install AI dependencies
pip install obsidian-vault-tools[ai]

# Check AI setup
ovt config test-ai
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

```bash
# Development setup
git clone https://github.com/yourusername/obsidian-vault-tools.git
cd obsidian-vault-tools
pip install -e .[dev]
pre-commit install
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with love for the Obsidian community
- Inspired by dungeon crawler games
- Powered by modern AI capabilities

## 📮 Support

- 📧 Email: support@obsidian-vault-tools.com
- 💬 Discord: [Join our server](https://discord.gg/obsidian-tools)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/obsidian-vault-tools/issues)

---

Made with ❤️ by the Obsidian Vault Tools team