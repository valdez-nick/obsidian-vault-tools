# 🔗 MCP (Model Context Protocol) Setup Guide

This guide will help you set up MCP servers to extend your Obsidian Vault Tools with powerful integrations like GitHub, Confluence, memory storage, and more.

## 🚀 Quick Start

### 1. Install MCP Dependencies
```bash
# Clone the repository
git clone https://github.com/valdez-nick/obsidian-vault-tools.git
cd obsidian-vault-tools

# Install with MCP support
pip install -e ".[mcp]"
```

### 2. First-Time Setup
```bash
# Set your vault path (if not already done)
ovt config set-vault "/path/to/your/obsidian/vault"

# Run the interactive setup wizard
ovt mcp list
```

The setup wizard will guide you through configuring your first MCP servers.

## 🔧 Manual Server Configuration

### GitHub Server
Access GitHub repositories, issues, and pull requests.

**Required Credentials:**
- GitHub Personal Access Token

**Setup:**
```bash
# Add server
ovt mcp add my-github github

# Set token (option 1: environment variable)
export YOUR_GITHUB_TOKEN="your_github_token_here"

# Set token (option 2: prompted when starting)
ovt mcp start my-github
```

**Create GitHub Token:**
1. Go to [GitHub Settings > Tokens](https://github.com/settings/tokens)
2. Click "Generate new token"
3. Select scopes: `repo`, `read:org`, `read:user`
4. Copy the generated token

### Memory Server
Persistent conversation memory across sessions.

**Required Credentials:**
- Memory storage path

**Setup:**
```bash
# Add server
ovt mcp add my-memory memory

# Set memory path
export YOUR_MEMORY_PATH="/path/to/memory/storage"

# Start server
ovt mcp start my-memory
```

### Web Fetch Server
Fetch and analyze web content.

**Required Credentials:** None

**Setup:**
```bash
# Add and start (no credentials needed)
ovt mcp add web-fetch web-fetch
ovt mcp start web-fetch
```

### Confluence/Jira Server
Access Atlassian Confluence and Jira.

**Required Credentials:**
- Atlassian Cloud ID
- Email address
- Confluence API token
- Jira API token

**Setup:**
```bash
# Add server
ovt mcp add my-confluence confluence

# Set credentials
export YOUR_CLOUD_ID="your_atlassian_cloud_id"
export YOUR_EMAIL="your@email.com"
export YOUR_CONFLUENCE_TOKEN="your_confluence_token"
export YOUR_JIRA_TOKEN="your_jira_token"

# Start server
ovt mcp start my-confluence
```

**Get Atlassian Credentials:**
1. **Cloud ID:** From your Atlassian URL: `https://your-domain.atlassian.net/` → Cloud ID is in the admin settings
2. **API Tokens:** Go to [Atlassian Account Settings](https://id.atlassian.com/manage-profile/security/api-tokens)

### Custom Obsidian PM Intelligence Server
Your custom PM intelligence server (if available).

**Required Credentials:**
- Script path
- Vault path  
- Memory path

**Setup:**
```bash
# Add server with script path
ovt mcp add my-pm obsidian-pm --script-path "/path/to/obsidian-pm-intelligence.js"

# Set paths
export YOUR_VAULT_PATH="/path/to/your/obsidian/vault"
export YOUR_MEMORY_PATH="/path/to/memory/storage"

# Start server
ovt mcp start my-pm
```

## 🔐 Credential Management

### Security Features
- **Encrypted Storage:** All credentials are encrypted locally
- **No Git Exposure:** Credential files are in `.gitignore`
- **Environment Variables:** Support for `.env` files
- **Prompted Input:** Secure credential prompting when needed

### Credential Storage Options

**Option 1: Environment Variables**
```bash
# Create .env file (automatically ignored by git)
echo 'YOUR_GITHUB_TOKEN=your_token_here' >> .env
echo 'YOUR_MEMORY_PATH=/path/to/memory' >> .env
```

**Option 2: Interactive Prompts**
When you start a server, you'll be securely prompted for missing credentials.

**Option 3: Direct Credential Manager**
```bash
# View stored credentials (keys only, not values)
ovt mcp credentials

# Credentials are automatically prompted and stored when needed
```

### Credential Security
- Stored in: `~/.obsidian_vault_tools/credentials.json` (encrypted)
- Encryption key: `~/.obsidian_vault_tools/.cred_key` (secure permissions)
- **Never committed to git** (protected by `.gitignore`)

## 📋 Common Commands

```bash
# List all configured servers
ovt mcp list

# Add a new server from template
ovt mcp add <name> <template>

# Start/stop servers
ovt mcp start <server-name>
ovt mcp stop <server-name>

# Check server status
ovt mcp status <server-name>

# Manage credentials
ovt mcp credentials

# Interactive mode with MCP features
ovt interactive
```

## 🔧 Advanced Configuration

### Custom Configuration File
Copy and modify the example configuration:
```bash
cp examples/mcp_config_example.json ~/.obsidian_vault_tools/mcp_config.json
```

Edit the file to match your setup, using placeholders like `[YOUR_GITHUB_TOKEN]` for credentials.

### Supported Placeholders
- `[YOUR_VAULT_PATH]` - Your Obsidian vault path
- `[YOUR_MEMORY_PATH]` - Memory storage path
- `[YOUR_GITHUB_TOKEN]` - GitHub personal access token
- `[YOUR_CLOUD_ID]` - Atlassian Cloud ID
- `[YOUR_EMAIL]` - Your email address
- `[YOUR_CONFLUENCE_TOKEN]` - Confluence API token
- `[YOUR_JIRA_TOKEN]` - Jira API token

## 🎮 Using MCP in Interactive Mode

Once servers are running, access MCP features through:

```bash
ovt interactive
```

Navigate to: **Advanced Tools → MCP Server Management**

Features available:
- **Server Status:** Real-time monitoring
- **Start/Stop Servers:** Lifecycle management
- **Tools Interface:** Call MCP server tools
- **Resources Access:** Retrieve server resources

## 🐛 Troubleshooting

### "MCP library not available"
```bash
pip install -e ".[mcp]"
```

### "Command not found" (npx/node)
Install Node.js and npm:
```bash
# macOS
brew install node

# Ubuntu/Debian
sudo apt install nodejs npm
```

### "Server not ready" 
Check credentials are configured:
```bash
ovt mcp credentials
ovt mcp list
```

### Docker commands failing
Ensure Docker is installed and running:
```bash
docker --version
docker ps
```

### Permission errors
Check file permissions:
```bash
ls -la ~/.obsidian_vault_tools/
```

## 🔄 Updating

To update to the latest version:
```bash
git pull origin main
pip install -e ".[mcp]"
```

Your credentials and configurations are preserved during updates.

## 🆘 Support

If you encounter issues:

1. **Check logs:** MCP servers output logs when starting
2. **Verify credentials:** Use `ovt mcp credentials` to check stored credentials
3. **Test manually:** Try running MCP server commands directly
4. **Review configuration:** Check `~/.obsidian_vault_tools/mcp_config.json`

## ⚡ Performance Tips

- **Start only needed servers:** Each server uses system resources
- **Use web-fetch sparingly:** Can be network-intensive
- **Regular cleanup:** Clear old memory files periodically
- **Monitor logs:** Check for errors or warnings

---

**Next Steps:** After setup, run `ovt interactive` to explore the enhanced vault management features powered by MCP! 🚀