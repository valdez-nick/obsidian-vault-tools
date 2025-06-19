# 🔒 Security Guidelines

This document outlines security best practices for Obsidian Vault Tools, especially regarding MCP credential management.

## 🛡️ Credential Security

### Encryption
- **All credentials are encrypted** using Fernet symmetric encryption
- **Encryption keys** are stored with restrictive file permissions (600)
- **No credentials in git** - protected by comprehensive .gitignore

### Storage Locations
- **Credentials:** `~/.obsidian_vault_tools/credentials.json` (encrypted)
- **Encryption key:** `~/.obsidian_vault_tools/.cred_key` (secure permissions)
- **Configuration:** `~/.obsidian_vault_tools/mcp_config.json` (templates only)

### Credential Flow
1. **Input:** Environment variables, secure prompts, or .env files
2. **Processing:** Encrypted locally using Fernet
3. **Storage:** Encrypted JSON with restricted file permissions
4. **Usage:** Decrypted only when needed for MCP server connections

## 🚫 What's Never Committed

The following patterns are blocked by .gitignore:

```gitignore
# MCP Credentials (NEVER COMMIT)
**/credentials.json
**/mcp_credentials.json  
**/config.local.json
**/mcp_config.local.json
**/.cred_key
**/memory.json
**/memory/**
**/mcp_memory/**

# Credential patterns (extra security)
**/*token*
**/*secret*
**/*key*
**/*password*
**/config.personal.json
**/personal_*
**/local_*
```

## ✅ Safe Practices

### For Users
1. **Use environment variables** for CI/CD and automation
2. **Use .env files** for local development (auto-ignored)
3. **Use encrypted storage** for interactive workflows
4. **Regular audits** with `ovt mcp audit`

### For Developers
1. **Always use placeholders** like `[YOUR_TOKEN]` in examples
2. **Test with dummy credentials** that are clearly fake
3. **Run audits** before committing: `ovt mcp audit`
4. **Review .gitignore** when adding new credential patterns

## 🔍 Security Auditing

### Automated Audit
```bash
# Run security audit
ovt mcp audit
```

This checks for:
- Known credential patterns (GitHub tokens, API keys)
- Suspicious long strings
- Email addresses in code
- Files not protected by .gitignore

### Manual Verification
```bash
# Check git history for credentials
git log --all --full-history --source -- "*credential*" "*token*" "*key*"

# Check staged files before commit
git diff --cached

# Check for untracked credential files
git status --ignored
```

## 🚨 Incident Response

### If Credentials Are Exposed
1. **Immediately revoke** the exposed credentials
2. **Remove from git history** using BFG or similar tools
3. **Force push** to update remote repository
4. **Generate new credentials** and update securely
5. **Audit access logs** for unauthorized usage

### Prevention
1. **Enable pre-commit hooks** to scan for credentials
2. **Use branch protection** requiring reviews
3. **Regular credential rotation** for high-value tokens
4. **Monitor usage** of API tokens and keys

## 🔧 Configuration Security

### Safe Configuration Examples
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "[YOUR_GITHUB_TOKEN]"
      }
    }
  }
}
```

### Unsafe Patterns to Avoid
```json
{
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_1234567890abcdef..."  ❌ NEVER DO THIS
  }
}
```

## 📋 Security Checklist

Before committing:
- [ ] Run `ovt mcp audit` 
- [ ] Check `git status --ignored` for credential files
- [ ] Verify no real credentials in configuration examples
- [ ] Ensure all placeholders use `[YOUR_*]` format
- [ ] Test setup flow with clean environment

Before releasing:
- [ ] Full repository audit for credential exposure
- [ ] Verify .gitignore effectiveness
- [ ] Test credential encryption/decryption
- [ ] Validate setup wizard security prompts
- [ ] Review all documentation for credential references

## 🆘 Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** create a public issue
2. **Email security concerns** to the maintainers privately
3. **Include details** of the vulnerability and potential impact
4. **Allow time** for responsible disclosure and patching

## 📚 Additional Resources

- [OWASP Credential Management](https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_credentials)
- [GitHub Token Security](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/token-expiration-and-revocation)
- [Git Security Best Practices](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)

---

**Remember: Security is everyone's responsibility. When in doubt, err on the side of caution.** 🔒