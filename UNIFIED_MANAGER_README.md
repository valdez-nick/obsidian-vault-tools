# 🏰 Unified Obsidian Vault Manager

The Unified Obsidian Vault Manager combines ALL vault management tools into one cohesive interactive menu system, providing a complete toolsuite rather than à la carte features.

## 🚀 Quick Start

### Launch Methods

```bash
# Method 1: Direct execution
./obsidian_manager_unified

# Method 2: Via Python
python3 unified_vault_manager.py

# Method 3: Via package CLI (default when no command given)
ovt

# Method 4: Alternative package command
obsidian-tools
```

## 📋 Main Menu Structure

The unified manager organizes all features into 11 main categories:

### 1. 📊 Vault Analysis & Insights
- Tag statistics and analysis
- Folder structure analysis
- Find untagged files
- Vault growth metrics
- Link analysis
- Content quality scoring
- Export analysis reports

### 2. 🏷️ Tag Management & Organization
- Analyze all tags
- Fix tag issues (quoted, incomplete)
- Merge similar tags
- Remove generic tags
- Bulk tag operations
- Auto-tag with AI (when available)
- Tag hierarchy reports

### 3. 🔍 Search & Query Vault
- Simple text search
- Search with filters
- Advanced query system
- AI-powered semantic search
- Natural language queries
- Recent files browser

### 4. 🤖 AI & Intelligence Features
- Intent-based task execution
- Smart context analysis
- AI research assistant
- Generate content ideas
- Chat with your vault
- AI content summarization
- Auto-organize with AI
- AI writing assistant

### 5. 🛠️ MCP Tools & Integrations
- Dynamic discovery of MCP servers
- Access to 47+ tools across 6 servers:
  - Filesystem operations
  - Git integration
  - Brave search
  - Memory storage
  - Slack integration
  - Google Drive access

### 6. 💾 Backup & Version Control
- Create full backups
- Restore from backup
- List available backups
- File version history
- Compare versions
- Restore specific versions
- Incremental backups
- Backup settings

### 7. 🎨 Creative Tools & ASCII Art
- Generate ASCII art from text
- Create ASCII flowcharts
- ASCII banner generator
- Text effects and styling

### 8. 🔊 Audio System & Effects
- Toggle ambient sounds
- Sound effects testing
- Audio settings
- Wizard greetings
- Success/error sounds
- Dungeon ambiance

### 9. 🛡️ Security & Maintenance
- Run security scans
- Check file permissions
- Vault integrity checks
- Clean temporary files
- Repair broken links

### 10. ⚡ Quick Actions
- Create daily note
- Quick capture
- Open random note
- Today's statistics
- Quick search

### 11. ⚙️ Settings & Configuration
- Change vault path
- Feature status overview
- Export/import configuration
- Reset to defaults

## 🌟 Key Features

### Dynamic Feature Detection
The manager automatically detects which features are available based on installed dependencies and gracefully handles missing components.

### Unified Interface
- Consistent menu navigation across all tools
- Arrow key navigation when available
- Number key fallback for all environments
- Sound effects for enhanced experience
- Color-coded output for clarity

### Integration Benefits
- All tools accessible from one place
- Shared configuration and vault context
- Consistent user experience
- No need to remember separate commands
- Automatic feature discovery

## 📦 Optional Dependencies

While the core functionality works out of the box, additional features become available with these dependencies:

```bash
# Audio system
pip install pygame

# ASCII art generation
pip install art pyfiglet

# Enhanced navigation
pip install blessed

# AI/LLM features
pip install openai langchain

# MCP tools
pip install mcp

# Analysis tools
pip install pandas matplotlib

# Backup features
pip install watchdog
```

## 🔧 Configuration

### Environment Variables
```bash
# Set default vault path
export OBSIDIAN_VAULT_PATH="/path/to/vault"

# Enable specific features
export ENABLE_AUDIO=true
export ENABLE_AI_FEATURES=true
```

### Configuration File
The manager can export/import configuration:
- Export: Settings → Export Configuration
- Import: Settings → Import Configuration

## 🎯 Usage Examples

### Daily Workflow
1. Launch with `ovt`
2. Quick Actions → Daily Note
3. Search → Recent Files
4. Tags → Fix Tag Issues
5. Backup → Create Full Backup

### Research Session
1. AI Features → AI Research Assistant
2. Search → Natural Language Query
3. Analysis → Content Quality Scoring
4. Creative → Generate Flowchart

### Maintenance Tasks
1. Security → Run Security Scan
2. Security → Check File Permissions
3. Tags → Remove Generic Tags
4. Backup → Create Full Backup

## 🚨 Troubleshooting

### Features Not Available
If certain features show as unavailable:
1. Check Settings → Feature Status
2. Install missing dependencies
3. Restart the manager

### Performance Issues
1. Use Quick Actions for common tasks
2. Disable audio if not needed
3. Limit search scope with filters

### Navigation Problems
- Use number keys if arrow navigation fails
- Press 'b' or select last option to go back
- Press Ctrl+C to exit at any time

## 🤝 Contributing

The unified manager is designed to be extensible:

1. Add new features to appropriate categories
2. Follow the existing menu structure pattern
3. Include feature detection with graceful fallback
4. Update feature availability tracking

## 📄 License

Part of the Obsidian Vault Tools project. See main LICENSE file for details.