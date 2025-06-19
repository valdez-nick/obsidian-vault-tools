# MCP Integration Examples

This directory contains example configurations for MCP (Model Context Protocol) integration.

## Quick Start

1. **Install MCP dependencies:**
   ```bash
   pip install -e ".[mcp]"
   ```

2. **Copy the example configuration:**
   ```bash
   cp examples/mcp_config_example.json ~/.obsidian_vault_tools/mcp_config.json
   ```

3. **Set up credentials using environment variables:**
   ```bash
   export GITHUB_PERSONAL_ACCESS_TOKEN="your_token_here"
   export MEMORY_PATH="/path/to/memory/storage"
   export VAULT_PATH="/path/to/your/obsidian/vault"
   ```

   Or use the credential manager (will prompt for values):
   ```bash
   ovt mcp credentials
   ```

4. **Add your first MCP server:**
   ```bash
   ovt mcp add my-github github
   ovt mcp add my-memory memory
   ```

5. **Start a server:**
   ```bash
   ovt mcp start my-memory
   ovt mcp list
   ```

## Available Commands

- `ovt mcp list` - Show configured servers
- `ovt mcp add <name> <template>` - Add server from template
- `ovt mcp start <name>` - Start a server
- `ovt mcp stop <name>` - Stop a server
- `ovt mcp status <name>` - Show server status
- `ovt mcp credentials` - Manage stored credentials

## Configuration Placeholders

The configuration supports these placeholder variables that will be automatically substituted:

- `[GITHUB_PERSONAL_ACCESS_TOKEN]` - Your GitHub personal access token
- `[MEMORY_PATH]` - Path where memory files should be stored
- `[VAULT_PATH]` - Path to your Obsidian vault
- `[CONFLUENCE_CLOUD_ID]` - Your Atlassian Cloud ID
- `[CONFLUENCE_EMAIL]` - Your Confluence email
- `[CONFLUENCE_TOKEN]` - Your Confluence API token
- `[JIRA_TOKEN]` - Your Jira API token

## Security Notes

- **Credentials are encrypted** and stored locally in `~/.obsidian_vault_tools/credentials.json`
- **Never commit** credential files to version control
- Use **environment variables** for CI/CD environments
- The `.gitignore` file excludes all credential files

## Server Templates

### GitHub Server
- **Purpose:** Access GitHub repositories, issues, PRs
- **Required:** `GITHUB_PERSONAL_ACCESS_TOKEN`
- **Example:** `ovt mcp add my-github github`

### Memory Server
- **Purpose:** Persistent conversation memory
- **Required:** `MEMORY_PATH`
- **Example:** `ovt mcp add my-memory memory`

### Web Fetch Server
- **Purpose:** Fetch web content
- **Required:** None
- **Example:** `ovt mcp add web-fetch web-fetch`

### Confluence Server
- **Purpose:** Access Confluence/Jira
- **Required:** `CONFLUENCE_CLOUD_ID`, `CONFLUENCE_EMAIL`, `CONFLUENCE_TOKEN`, `JIRA_TOKEN`
- **Example:** `ovt mcp add my-confluence confluence`

### Obsidian PM Intelligence
- **Purpose:** Custom Obsidian PM intelligence server
- **Required:** `VAULT_PATH`, `MEMORY_PATH`, script path
- **Example:** `ovt mcp add my-pm obsidian-pm --script-path /path/to/script.js`

## Troubleshooting

1. **"MCP library not available"** - Install with `pip install -e ".[mcp]"`
2. **"Command not found"** - Ensure Node.js/npm is installed for npx commands
3. **"Server not ready"** - Check credentials are configured
4. **Docker commands failing** - Ensure Docker is installed and running

## Integration with Interactive Mode

MCP servers can be accessed from the interactive mode (`ovt interactive`) where you can:
- Call MCP tools directly
- Access MCP resources for vault analysis
- Use MCP prompts in workflows

Start the interactive mode and look for MCP-related menu options once servers are running.