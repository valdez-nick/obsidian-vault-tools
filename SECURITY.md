# Security Guidelines for Obsidian Vault Tools

## Overview

This document outlines the security measures implemented in the Obsidian Vault Tools project and provides guidelines for secure deployment and usage.

## Security Features Implemented

### 🔒 **Code Security**

1. **Input Validation**
   - Query length limits (max 1000 characters)
   - Dangerous pattern detection
   - Path traversal prevention
   - Filename sanitization

2. **Safe Import System**
   - Replaced `exec()` with safe `importlib` imports
   - Command argument validation and quoting
   - Subprocess security controls

3. **Secure Serialization**
   - Replaced pickle with JSON serialization
   - Prevents arbitrary code execution
   - Maintains data integrity

### 🛡️ **File System Security**

1. **Path Validation**
   - Base path restrictions
   - Null byte detection
   - Suspicious pattern monitoring
   - Directory traversal prevention

2. **File Operations**
   - Safe filename generation
   - Permission validation
   - Temporary file handling
   - Backup before modification

### 🔐 **Authentication & Authorization**

1. **API Key Management**
   - Secure key generation using `secrets` module
   - HMAC-based signatures
   - Rate limiting on authentication attempts
   - Automatic lockout after failed attempts

2. **Session Management**
   - Secure session tokens
   - Configurable timeout periods
   - Session validation and renewal
   - Proper session termination

### 🚨 **Rate Limiting**

1. **Global Rate Limits**
   - 60 requests per minute (default)
   - 1000 requests per hour
   - Burst protection (10 requests)
   - Per-function rate limiting

2. **Authentication Rate Limits**
   - Login attempts: 10/minute
   - API key generation: 10/minute
   - Session creation: 20/minute

### 📊 **Logging & Monitoring**

1. **Sensitive Data Sanitization**
   - Automatic credential masking
   - Path anonymization
   - PII removal from logs
   - Structured logging with context

2. **Security Event Logging**
   - Failed authentication attempts
   - Suspicious activity detection
   - Rate limit violations
   - File access monitoring

## Configuration

### Security Configuration File

Edit `security_config.yaml` to customize security settings:

```yaml
security:
  enabled: true
  
  input_validation:
    max_query_length: 1000
    max_filename_length: 255
    
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    
  api:
    require_authentication: false  # Set to true in production
```

### Environment Variables

Set these environment variables for production:

```bash
# API Security
export API_KEY="your-secure-api-key"
export JWT_SECRET="your-jwt-secret-key"

# Paths
export OBSIDIAN_PM_SCRIPT_PATH="/path/to/obsidian-pm-intelligence.js"

# Security
export SECURITY_ENABLED=true
export RATE_LIMITING_ENABLED=true
```

## Security Scanning

### Automated Security Checks

Run the security scanner to check for vulnerabilities:

```bash
# Install security tools
pip install -r requirements-security.txt

# Run comprehensive security scan
python security_scan.py

# Scan specific directory
python security_scan.py -d /path/to/scan

# Generate detailed report
python security_scan.py -o security_report.json
```

### Manual Security Checks

1. **Code Security (Bandit)**
   ```bash
   bandit -r . -f json -ll
   ```

2. **Dependency Vulnerabilities (Safety)**
   ```bash
   safety check --json
   ```

3. **Package Vulnerabilities (pip-audit)**
   ```bash
   pip-audit --format=json
   ```

## Deployment Security

### Production Checklist

- [ ] Enable authentication (`require_authentication: true`)
- [ ] Set strong JWT secret key
- [ ] Enable HTTPS for all connections
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Regular security scans
- [ ] Update dependencies regularly
- [ ] Restrict file system access
- [ ] Enable audit logging
- [ ] Configure firewall rules

### HTTPS Configuration

For production deployments with web interfaces:

```python
from obsidian_vault_tools.api_auth import configure_auth

configure_auth(
    require_https=True,
    jwt_secret="your-strong-secret-key",
    session_timeout_minutes=30,
    max_failed_attempts=3
)
```

### Secure Headers

When serving web content, include security headers:

```python
security_headers = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'"
}
```

## API Security

### Authentication

1. **API Key Authentication**
   ```python
   from obsidian_vault_tools.api_auth import require_api_key
   
   @require_api_key(user_id="your_user_id")
   def protected_function():
       return "This function requires authentication"
   ```

2. **Session-based Authentication**
   ```python
   from obsidian_vault_tools.api_auth import require_session
   
   @require_session()
   def session_protected_function(session):
       user_id = session['user_id']
       return f"Hello {user_id}"
   ```

### Rate Limiting

Apply rate limits to sensitive functions:

```python
from obsidian_vault_tools.security import rate_limit

@rate_limit(max_requests=10, window_seconds=60)
def expensive_operation():
    # This function is limited to 10 calls per minute
    pass
```

## Vulnerability Response

### Reporting Vulnerabilities

If you discover a security vulnerability:

1. **Do not** create a public issue
2. Email security concerns to: [security contact]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Process

1. **Acknowledgment**: Within 24 hours
2. **Investigation**: Within 48 hours
3. **Fix Development**: Within 1 week
4. **Patch Release**: As soon as possible
5. **Public Disclosure**: After fix deployment

## Best Practices

### For Developers

1. **Input Validation**
   - Always validate user input
   - Use the provided security utilities
   - Sanitize data before logging

2. **File Operations**
   - Use `validate_path()` for all file paths
   - Sanitize filenames with `sanitize_filename()`
   - Check permissions before file operations

3. **Authentication**
   - Use provided decorators for protection
   - Implement rate limiting on sensitive operations
   - Log security events appropriately

4. **Dependencies**
   - Keep dependencies updated
   - Run security scans regularly
   - Use minimal required permissions

### For Users

1. **API Keys**
   - Keep API keys secure and private
   - Rotate keys regularly
   - Don't commit keys to version control

2. **File Permissions**
   - Set appropriate file permissions
   - Don't run with unnecessary privileges
   - Regularly audit file access

3. **Network Security**
   - Use HTTPS in production
   - Configure firewalls appropriately
   - Monitor network traffic

## Security Updates

### Stay Informed

- Monitor the project's security advisories
- Subscribe to dependency security alerts
- Regular security scans of your deployment

### Update Process

1. **Test updates** in development environment
2. **Backup** your configuration and data
3. **Apply updates** during maintenance windows
4. **Verify** functionality after updates
5. **Monitor** for any issues

## Compliance

### Standards

This project follows security best practices from:

- OWASP Top 10
- NIST Cybersecurity Framework
- Python Security Best Practices
- Secure Coding Guidelines

### Audit Trail

Security events are logged with:

- Timestamp
- User identification
- Action performed
- Result (success/failure)
- Source IP (when applicable)

## Emergency Procedures

### Security Incident Response

1. **Immediate Actions**
   - Isolate affected systems
   - Preserve evidence
   - Assess scope of impact
   - Notify stakeholders

2. **Investigation**
   - Analyze logs and evidence
   - Determine root cause
   - Assess data exposure
   - Document findings

3. **Recovery**
   - Apply fixes and patches
   - Restore from clean backups
   - Update security measures
   - Monitor for recurrence

4. **Post-Incident**
   - Conduct lessons learned review
   - Update security procedures
   - Improve monitoring and detection
   - Communicate with stakeholders

## Contact

For security-related questions or concerns:

- **General Security**: Create an issue with `[SECURITY]` prefix
- **Vulnerability Reports**: Email security team
- **Emergency**: Follow incident response procedures

---

*This document is regularly updated to reflect the current security posture and should be reviewed quarterly.*