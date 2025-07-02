# Obsidian Vault Tools 🛠️

A comprehensive, modern toolkit for managing Obsidian vaults with AI-powered features, smart organization, and delightful user experience.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-v2.3.0-brightgreen)
![Status](https://img.shields.io/badge/status-beta-yellow)

## 🎉 What's New in v2.3.0

**PM Automation Suite** - Transform your PM workflows with AI-powered automation:
- 🤖 **WBR/QBR Automation**: Auto-generate business reviews with insights from Jira, Snowflake, and more
- 📝 **Feature Pipeline**: Convert PRDs to Jira stories in seconds with AI assistance
- 📊 **Analytics Hub**: ML-powered dashboards for tracking PM performance and predicting trends
- 🔐 **OAuth Integration**: Secure authentication for Google Workspace, Atlassian, and more
- 🚨 **Real-time Monitoring**: Get alerts on anomalies and critical PM metrics

## 🚀 Quick Start - Unified Manager

**NEW!** All tools are now available through one unified interface:

```bash
# Launch the unified interactive manager
./obsidian_manager_unified

# Or via the package CLI
ovt              # Launches unified manager
obsidian-tools   # Alternative command
```

The Unified Manager combines ALL features into one cohesive menu system. See [UNIFIED_MANAGER_README.md](UNIFIED_MANAGER_README.md) for details.

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
- **Persistent Configuration**: Automatically remembers vault paths and settings
- **Smart Path Resolution**: Handles common path input issues automatically

### 💼 PM Tools (Product Manager Burnout Prevention)
- **WSJF Task Prioritizer**: Weighted Shortest Job First scoring for all tasks
- **Eisenhower Matrix Classifier**: Automatic urgent/important categorization
- **Burnout Detection System**: Early warning system for overwhelm patterns
- **Content Quality Engine**: Standardize naming and fix incomplete notes
- **Daily Template Generator**: PM-optimized daily planning templates

### 🚀 PM Automation Suite (NEW in v2.3.0)
- **WBR/QBR Automation**: Generate weekly/quarterly business reviews with AI insights
- **Feature Development Pipeline**: Convert PRDs to Jira stories with AI assistance
- **Analytics Hub**: ML-powered PM performance tracking and predictions
- **Real-time Monitoring**: Anomaly detection and alerting for PM metrics
- **Multi-Source Integration**: Connect Jira, Confluence, Google Suite, Snowflake
- **OAuth Authentication**: Secure authentication for all external services

### 🌐 MCP Integration (Model Context Protocol)
- **Dynamic Tool Discovery**: Automatically discover and integrate MCP tools
- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Interactive Tool Execution**: Execute tools with guided parameter input
- **Interactive Configuration**: Configure MCP servers through the unified manager interface
- **Server Management**: Start, stop, and manage MCP servers
- **Tool Categorization**: Organize tools by function and server
- **Execution History**: Track tool usage and performance statistics
- **System Requirements Check**: Verify all dependencies for MCP servers are installed
- **Enhanced Atlassian Setup**: Guided Docker-based setup for Confluence/Jira integration

Supported MCP Servers:
- 🧠 **Obsidian PM Intelligence**: Custom vault intelligence and analysis
- 🔍 **GitHub Integration**: Repository search and management
- 💾 **Memory Server**: Persistent conversation context
- 📚 **Confluence/Jira**: Document and project management
- 🌐 **Web Fetch**: Web content analysis and scraping
- 🤔 **Sequential Thinking**: Structured reasoning tools

## 📦 Installation

### Platform-Specific Guides
- **Ubuntu/Debian**: See [UBUNTU_INSTALL.md](UBUNTU_INSTALL.md) for comprehensive Ubuntu 24.04 LTS instructions with all system dependencies
- **macOS**: Use Homebrew or pip (see below)
- **Windows**: Use pip in PowerShell or WSL

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

**Note**: Linux users may need to install system dependencies first. See [UBUNTU_INSTALL.md](UBUNTU_INSTALL.md) for details.

### With AI Features
```bash
pip install obsidian-vault-tools[ai]
```

### With MCP Integration
```bash
pip install obsidian-vault-tools[mcp]
# or install manually
pip install mcp cryptography
```

### With PM Automation Suite (NEW in v2.3.0)
```bash
pip install obsidian-vault-tools[pm-automation]
# or install all features
pip install obsidian-vault-tools[all]
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

### MCP Integration
```bash
ovt                           # Launch unified manager → Settings → MCP Server Configuration
ovt interactive               # Access MCP tools via interactive menu
ovt mcp start server-name     # Start MCP server
ovt mcp stop server-name      # Stop MCP server
ovt mcp list                  # List configured servers
ovt mcp tools                 # List available tools
```

### Maintenance
```bash
ovt backup                    # Create backup
ovt clean --analyze           # Analyze for cleanup
ovt clean --safe              # Safe cleanup
ovt version                   # File versioning
```

### PM Tools
```bash
ovt pm quality                # Run content quality analysis
ovt pm template               # Generate daily PM template
ovt pm burnout                # Check burnout risk score
ovt pm prioritize             # Run WSJF prioritization
ovt pm eisenhower             # Classify tasks by urgency/importance

# PM Automation Suite (NEW in v2.3.0)
ovt                           # Launch unified manager → PM Tools → PM Automation Suite
ovt-pm wbr --project PROJ     # Generate weekly business review
ovt-pm features prd.pdf -p PROJ  # Convert PRD to Jira stories
ovt-pm analytics --dashboard team  # View PM performance dashboard
ovt-pm monitor --metric velocity   # Real-time metric monitoring
ovt-pm quality                # Analyze content quality
ovt-pm template               # Generate daily PM template
ovt-pm burnout                # Check burnout risk score
```

## ⚙️ Configuration

### Initial Setup
```bash
ovt config set-vault PATH     # Set default vault path
ovt config show               # Show current configuration
ovt config reset              # Reset to defaults

# MCP System Requirements
ovt mcp check-requirements    # Check Docker, Node.js, etc.
```

### Configuration File
Configuration is stored in `~/.obsidian_vault_tools.json`:

```json
{
  "vault_path": "/Users/username/Documents/MyVault",
  "output_directory": "",
  "backup_settings": {
    "auto_backup": false,
    "backup_count": 5
  },
  "ui_settings": {
    "use_colors": true,
    "show_progress": true
  }
}
```

### PM Automation Suite Configuration (NEW in v2.3.0)
Configure through the unified manager:
```
Settings & Configuration → PM Suite Configuration
```

Or set environment variables:
```bash
# Required for PM Automation
export JIRA_URL="https://company.atlassian.net"
export JIRA_EMAIL="your-email@company.com"
export JIRA_API_TOKEN="your-token"
export OPENAI_API_KEY="sk-..."

# Optional integrations
export GOOGLE_CLIENT_ID="your-client-id"
export GOOGLE_CLIENT_SECRET="your-secret"
export SNOWFLAKE_ACCOUNT="your-account"
```

### MCP Configuration

#### Method 1: Interactive Configuration (Recommended)
Launch the unified manager and navigate to:
```
Settings & Configuration → MCP Server Configuration
```

Features:
- Add/edit/remove servers through guided interface
- Test connections before saving
- Secure credential management with masking
- Real-time validation and status indicators

#### Method 2: Manual Configuration
MCP servers can also be configured manually in `~/.obsidian-tools/mcp_config.json`:

```json
{
  "servers": {
    "obsidian-pm-intelligence": {
      "command": "obsidian-pm-intelligence",
      "args": ["--vault", "[YOUR_VAULT_PATH]"],
      "env": {"DEBUG": "false"}
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "[YOUR_GITHUB_TOKEN]"}
    }
  }
}
```

**Security Note**: Replace `[YOUR_*]` placeholders with actual values. Credentials are automatically encrypted when stored.

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

### PM Automation Examples (NEW in v2.3.0)
```python
from pm_automation_suite.wbr import WBROrchestrator
from pm_automation_suite.feature_dev import FeaturePipeline

# Generate weekly business review
orchestrator = WBROrchestrator()
await orchestrator.generate_weekly_review(
    project="PROD",
    output_format="slides"
)

# Convert PRD to Jira stories
pipeline = FeaturePipeline()
stories = await pipeline.prd_to_stories(
    prd_file="requirements.pdf",
    project_key="FEAT"
)
```

### API Server (NEW in v2.3.0)
```bash
# Start the REST API server
python -m obsidian_vault_tools.api_server

# Or with custom host/port
python -m obsidian_vault_tools.api_server --host 0.0.0.0 --port 8080
```

API endpoints available at `http://localhost:8000`:
- `GET /health` - Health check
- `GET /vault/info` - Vault information
- `POST /wbr/generate` - Generate WBR
- `POST /features/pipeline` - Run feature pipeline
- `GET /analytics/quality` - Content quality analysis
- `GET /analytics/dashboard/{type}` - Get dashboard
- `GET /docs` - Interactive API documentation

## 🐛 Troubleshooting

### Common Issues

**Ubuntu/Linux Issues**:
See [UBUNTU_INSTALL.md](UBUNTU_INSTALL.md#troubleshooting) for Ubuntu-specific troubleshooting including:
- Audio device errors
- Docker permissions
- Missing system dependencies
- Virtual environment setup

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