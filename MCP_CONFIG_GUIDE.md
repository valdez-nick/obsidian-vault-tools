# 🛠️ MCP Server Configuration Guide

This guide explains how to configure MCP (Model Context Protocol) servers through the Unified Vault Manager's interactive interface.

## 📋 Accessing MCP Configuration

1. Launch the Unified Vault Manager:
   ```bash
   ovt
   # or
   ./obsidian_manager_unified
   ```

2. Navigate to: **Settings & Configuration** → **MCP Server Configuration**

## 🎯 Configuration Options

### 1. View Server Details
- See all configured servers and their status
- Check if servers are ready (command exists, credentials set)
- View current configuration (with sensitive values masked)

### 2. Add New Server
Interactive wizard that guides you through:
- **Server Name**: Unique identifier (e.g., 'github', 'slack')
- **Command**: The command to run (e.g., 'npx', 'node', 'docker')
- **Arguments**: Command arguments in JSON array format
- **Environment Variables**: Optional environment settings

#### Example: Adding a GitHub Server
```
Server name: github
Command: npx
Arguments: ["-y", "@modelcontextprotocol/server-github"]
Environment variables:
  GITHUB_PERSONAL_ACCESS_TOKEN: [YOUR_GITHUB_TOKEN]
```

### 3. Edit Server
Modify existing server configurations:
- Update command or arguments
- Add/modify environment variables
- Keep or clear existing settings

### 4. Remove Server
Delete server configurations you no longer need

### 5. Test Server Connection
Validate that a server configuration is correct:
- Checks if command exists in PATH
- Verifies required credentials are set
- Basic configuration validation

## 📝 Configuration Format

MCP servers require:
- **command**: The executable to run
- **args**: Array of command-line arguments
- **env** (optional): Environment variables

### Example Configuration
```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-memory"],
  "env": {
    "MEMORY_FILE_PATH": "/path/to/memory.json"
  }
}
```

## 🔐 Credentials Management

For sensitive values like tokens and passwords:
- Use placeholder format: `[YOUR_TOKEN_NAME]`
- The system will prompt to set actual values securely
- Credentials are encrypted and stored separately
- Never shown in plain text in the interface

## 🚀 Common MCP Servers

### Memory Server
Provides persistent conversation memory:
```
Command: npx
Args: ["-y", "@modelcontextprotocol/server-memory"]
Env: MEMORY_FILE_PATH=/path/to/memory.json
```

### GitHub Server
Access GitHub repositories and issues:
```
Command: npx
Args: ["-y", "@modelcontextprotocol/server-github"]
Env: GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR_TOKEN]
```

### Web Fetch Server
Fetch and analyze web content:
```
Command: npx
Args: ["-y", "@modelcontextprotocol/server-fetch"]
```

## ❓ Troubleshooting

### "Command not found"
- Ensure the command is installed (e.g., `npm install -g npx`)
- Check that it's in your system PATH

### "Missing credentials"
- Use the test connection option to identify missing credentials
- Set credentials when prompted or through the edit interface

### Server not appearing in MCP Tools menu
- Ensure the server configuration is saved
- Check that all required fields are set
- Verify the server status shows as "ready"

## 📚 Additional Resources

- [MCP Documentation](https://modelcontextprotocol.io)
- Server-specific setup guides from server creators
- Community MCP server list

## 💡 Tips

1. **Copy from Documentation**: MCP server creators provide exact configuration in their docs
2. **Test Before Use**: Always test connection after configuration
3. **Use Placeholders**: For credentials, use `[PLACEHOLDER]` format for security
4. **Check Status**: Regularly check server status to ensure they're ready

The MCP configuration interface makes it easy to add and manage any MCP server without manually editing JSON files!