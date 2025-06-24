#!/usr/bin/env python3
"""
Security Hardening Bot - Automates Monday Morning Security Tasks
Handles your top WSJF 14.0 and 13.0 security priorities automatically

WSJF Tasks Automated:
- Complete subprocess security fixes (WSJF: 14.0)
- Set strong JWT secret key (WSJF: 13.0)  
- Enable HTTPS for all connections (WSJF: 13.0)
- Configure rate limiting (WSJF: 13.0)

Usage: python security_hardening_bot.py --scan --fix --report
"""

import os
import re
import sys
import json
import secrets
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_hardening.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityIssue:
    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str
    fix_suggestion: str
    auto_fixable: bool

@dataclass
class SecurityReport:
    total_issues: int
    critical_issues: int
    auto_fixed: int
    manual_required: int
    issues: List[SecurityIssue]
    
class SecurityHardeningBot:
    """Automated security hardening for DFP 2.0 codebase"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_found = []
        self.fixes_applied = []
        
        # Dangerous subprocess patterns (WSJF 14.0 priority)
        self.dangerous_subprocess_patterns = [
            r'subprocess\.call\([^)]*shell=True',
            r'subprocess\.run\([^)]*shell=True',
            r'subprocess\.Popen\([^)]*shell=True',
            r'os\.system\(',
            r'os\.popen\(',
            r'commands\.getoutput\(',
            r'commands\.getstatusoutput\('
        ]
        
        # Configuration file patterns
        self.config_patterns = {
            'jwt': [r'jwt.*secret', r'JWT.*SECRET', r'secret.*key'],
            'https': [r'https?://', r'ssl', r'tls', r'certificate'],
            'rate_limit': [r'rate.*limit', r'throttle', r'request.*limit']
        }
    
    def scan_subprocess_security(self) -> List[SecurityIssue]:
        """Scan for dangerous subprocess usage (WSJF 14.0 priority)"""
        logger.info("🔍 Scanning for subprocess security vulnerabilities...")
        issues = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.dangerous_subprocess_patterns:
                        if re.search(pattern, line):
                            issues.append(SecurityIssue(
                                file_path=str(file_path),
                                line_number=line_num,
                                issue_type="subprocess_security",
                                description=f"Dangerous subprocess usage: {line.strip()}",
                                severity="CRITICAL",
                                fix_suggestion="Use subprocess with shell=False and proper argument sanitization",
                                auto_fixable=True
                            ))
                            
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        logger.info(f"Found {len(issues)} subprocess security issues")
        return issues
    
    def fix_subprocess_security(self, issues: List[SecurityIssue]) -> int:
        """Auto-fix subprocess security issues (WSJF 14.0 priority)"""
        fixed_count = 0
        
        for issue in issues:
            if issue.issue_type == "subprocess_security" and issue.auto_fixable:
                try:
                    with open(issue.file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Apply specific fixes
                    original_content = content
                    
                    # Fix shell=True patterns
                    content = re.sub(
                        r'subprocess\.(call|run|Popen)\([^)]*shell=True([^)]*)\)',
                        r'subprocess.\1(\2)',  # Remove shell=True
                        content
                    )
                    
                    # Fix os.system usage
                    content = re.sub(
                        r'os\.system\(([^)]+)\)',
                        r'subprocess.run(shlex.split(\1), check=True)',
                        content
                    )
                    
                    if content != original_content:
                        # Backup original file
                        backup_path = f"{issue.file_path}.security_backup"
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)
                        
                        # Write fixed content
                        with open(issue.file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # Add shlex import if needed
                        if 'shlex.split' in content and 'import shlex' not in content:
                            lines = content.split('\n')
                            # Find import section
                            import_line = -1
                            for i, line in enumerate(lines):
                                if line.startswith('import ') or line.startswith('from '):
                                    import_line = i
                            
                            if import_line >= 0:
                                lines.insert(import_line + 1, 'import shlex')
                                with open(issue.file_path, 'w', encoding='utf-8') as f:
                                    f.write('\n'.join(lines))
                        
                        fixed_count += 1
                        self.fixes_applied.append(f"Fixed subprocess security in {issue.file_path}:{issue.line_number}")
                        logger.info(f"✅ Fixed subprocess security issue in {issue.file_path}")
                        
                except Exception as e:
                    logger.error(f"Failed to fix {issue.file_path}: {e}")
        
        return fixed_count
    
    def generate_jwt_secret(self) -> str:
        """Generate cryptographically secure JWT secret (WSJF 13.0 priority)"""
        return secrets.token_urlsafe(64)
    
    def configure_jwt_security(self) -> Dict[str, Any]:
        """Configure JWT security settings (WSJF 13.0 priority)"""
        logger.info("🔐 Configuring JWT security...")
        
        jwt_secret = self.generate_jwt_secret()
        config_files = []
        
        # Common config file locations
        config_locations = [
            'config.py', 'settings.py', 'config.json', '.env',
            'config/production.py', 'config/config.py'
        ]
        
        for config_file in config_locations:
            config_path = self.project_root / config_file
            if config_path.exists():
                config_files.append(str(config_path))
        
        # Generate secure JWT configuration
        jwt_config = {
            'JWT_SECRET_KEY': jwt_secret,
            'JWT_ACCESS_TOKEN_EXPIRES': 3600,  # 1 hour
            'JWT_REFRESH_TOKEN_EXPIRES': 2592000,  # 30 days
            'JWT_ALGORITHM': 'HS256',
            'JWT_VERIFY_EXPIRATION': True,
            'JWT_VERIFY_SIGNATURE': True,
            'JWT_REQUIRE_CLAIMS': ['exp', 'iat'],
            'JWT_LEEWAY': 10
        }
        
        # Save JWT configuration
        jwt_config_file = self.project_root / 'jwt_security_config.json'
        with open(jwt_config_file, 'w') as f:
            json.dump(jwt_config, f, indent=2)
        
        logger.info(f"✅ Generated JWT security config: {jwt_config_file}")
        
        return {
            'jwt_secret_generated': True,
            'config_file': str(jwt_config_file),
            'config_files_found': config_files,
            'secret_strength': 'cryptographically_secure_512_bits'
        }
    
    def configure_https_security(self) -> Dict[str, Any]:
        """Configure HTTPS security settings (WSJF 13.0 priority)"""
        logger.info("🔒 Configuring HTTPS security...")
        
        https_config = {
            'FORCE_HTTPS': True,
            'HTTPS_REDIRECT': True,
            'SECURE_SSL_REDIRECT': True,
            'SECURE_HSTS_SECONDS': 31536000,  # 1 year
            'SECURE_HSTS_INCLUDE_SUBDOMAINS': True,
            'SECURE_HSTS_PRELOAD': True,
            'SECURE_CONTENT_TYPE_NOSNIFF': True,
            'SECURE_BROWSER_XSS_FILTER': True,
            'SECURE_FRAME_DENY': True,
            'SECURE_PROXY_SSL_HEADER': ('HTTP_X_FORWARDED_PROTO', 'https'),
            'SESSION_COOKIE_SECURE': True,
            'CSRF_COOKIE_SECURE': True
        }
        
        # Generate nginx HTTPS configuration
        nginx_https_config = '''
# HTTPS Security Configuration (Auto-generated)
server {
    listen 80;
    server_name _;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;
    
    # SSL Configuration
    ssl_certificate /path/to/certificate.pem;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Rate limiting
    limit_req zone=api burst=20 nodelay;
}
'''
        
        # Save configurations
        https_config_file = self.project_root / 'https_security_config.json'
        nginx_config_file = self.project_root / 'nginx_https_config.conf'
        
        with open(https_config_file, 'w') as f:
            json.dump(https_config, f, indent=2)
        
        with open(nginx_config_file, 'w') as f:
            f.write(nginx_https_config)
        
        logger.info(f"✅ Generated HTTPS security config: {https_config_file}")
        logger.info(f"✅ Generated nginx HTTPS config: {nginx_config_file}")
        
        return {
            'https_config_generated': True,
            'config_file': str(https_config_file),
            'nginx_config': str(nginx_config_file),
            'security_headers_included': True
        }
    
    def configure_rate_limiting(self) -> Dict[str, Any]:
        """Configure rate limiting (WSJF 13.0 priority)"""
        logger.info("⚡ Configuring rate limiting...")
        
        # Flask rate limiting configuration
        flask_rate_limit_config = {
            'RATELIMIT_STORAGE_URL': 'redis://localhost:6379',
            'RATELIMIT_DEFAULT': '1000/hour',
            'RATELIMIT_HEADERS_ENABLED': True,
            'RATELIMIT_STRATEGY': 'fixed-window',
            'RATE_LIMITS': {
                'api_general': '100/minute',
                'api_auth': '10/minute',
                'api_upload': '5/minute',
                'api_search': '50/minute'
            }
        }
        
        # Generate Flask-Limiter middleware code
        flask_limiter_code = '''
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis

# Rate limiting setup (Auto-generated)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",
    default_limits=["1000/hour", "100/minute"],
    headers_enabled=True
)

# Route-specific rate limits
@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10/minute")
def login():
    pass

@app.route('/api/upload', methods=['POST'])
@limiter.limit("5/minute")
def upload():
    pass

@app.route('/api/search', methods=['GET'])
@limiter.limit("50/minute")
def search():
    pass
'''
        
        # Nginx rate limiting configuration
        nginx_rate_limit_config = '''
# Rate Limiting Configuration (Auto-generated)
http {
    # Define rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=10r/m;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=5r/m;
    
    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=conn:10m;
    
    server {
        # General API rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            limit_conn conn 10;
        }
        
        # Auth endpoints (stricter)
        location /api/auth/ {
            limit_req zone=auth burst=5 nodelay;
        }
        
        # Upload endpoints (very strict)
        location /api/upload/ {
            limit_req zone=upload burst=2 nodelay;
        }
    }
}
'''
        
        # Save configurations
        rate_limit_config_file = self.project_root / 'rate_limiting_config.json'
        flask_limiter_file = self.project_root / 'flask_rate_limiter.py'
        nginx_rate_limit_file = self.project_root / 'nginx_rate_limiting.conf'
        
        with open(rate_limit_config_file, 'w') as f:
            json.dump(flask_rate_limit_config, f, indent=2)
        
        with open(flask_limiter_file, 'w') as f:
            f.write(flask_limiter_code)
        
        with open(nginx_rate_limit_file, 'w') as f:
            f.write(nginx_rate_limit_config)
        
        logger.info(f"✅ Generated rate limiting config: {rate_limit_config_file}")
        logger.info(f"✅ Generated Flask limiter code: {flask_limiter_file}")
        logger.info(f"✅ Generated nginx rate limiting: {nginx_rate_limit_file}")
        
        return {
            'rate_limiting_configured': True,
            'config_file': str(rate_limit_config_file),
            'flask_middleware': str(flask_limiter_file),
            'nginx_config': str(nginx_rate_limit_file),
            'redis_required': True
        }
    
    def generate_security_report(self) -> SecurityReport:
        """Generate comprehensive security report"""
        all_issues = self.scan_subprocess_security()
        
        critical_issues = [issue for issue in all_issues if issue.severity == "CRITICAL"]
        auto_fixable = [issue for issue in all_issues if issue.auto_fixable]
        
        return SecurityReport(
            total_issues=len(all_issues),
            critical_issues=len(critical_issues),
            auto_fixed=len(self.fixes_applied),
            manual_required=len(all_issues) - len(auto_fixable),
            issues=all_issues
        )
    
    def run_full_security_hardening(self) -> Dict[str, Any]:
        """Run complete security hardening (All Monday morning WSJF tasks)"""
        logger.info("🚀 Starting full security hardening...")
        
        start_time = datetime.now()
        
        # 1. Subprocess security fixes (WSJF 14.0)
        subprocess_issues = self.scan_subprocess_security()
        subprocess_fixes = self.fix_subprocess_security(subprocess_issues)
        
        # 2. JWT security (WSJF 13.0)
        jwt_result = self.configure_jwt_security()
        
        # 3. HTTPS security (WSJF 13.0)
        https_result = self.configure_https_security()
        
        # 4. Rate limiting (WSJF 13.0)
        rate_limit_result = self.configure_rate_limiting()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate final report
        report = self.generate_security_report()
        
        result = {
            'execution_time_seconds': duration,
            'subprocess_security': {
                'issues_found': len(subprocess_issues),
                'fixes_applied': subprocess_fixes,
                'status': 'completed'
            },
            'jwt_security': jwt_result,
            'https_security': https_result,
            'rate_limiting': rate_limit_result,
            'security_report': {
                'total_issues': report.total_issues,
                'critical_issues': report.critical_issues,
                'auto_fixed': report.auto_fixed,
                'manual_required': report.manual_required
            },
            'next_steps': [
                'Review generated configuration files',
                'Deploy HTTPS certificates',
                'Set up Redis for rate limiting',
                'Test security configurations',
                'Update deployment scripts'
            ]
        }
        
        # Save results
        results_file = self.project_root / f'security_hardening_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"✅ Security hardening completed in {duration:.2f} seconds")
        logger.info(f"📄 Results saved to: {results_file}")
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Security Hardening Bot - Automate Monday Morning Security Tasks")
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--scan', action='store_true', help='Scan for security issues')
    parser.add_argument('--fix', action='store_true', help='Auto-fix security issues')
    parser.add_argument('--report', action='store_true', help='Generate security report')
    parser.add_argument('--full', action='store_true', help='Run complete security hardening')
    
    args = parser.parse_args()
    
    bot = SecurityHardeningBot(args.project_root)
    
    if args.full or (args.scan and args.fix):
        result = bot.run_full_security_hardening()
        print("\n🎯 SECURITY HARDENING COMPLETE!")
        print(f"⏱️  Execution time: {result['execution_time_seconds']:.2f} seconds")
        print(f"🔧 Subprocess fixes applied: {result['subprocess_security']['fixes_applied']}")
        print(f"🔐 JWT security configured: {result['jwt_security']['jwt_secret_generated']}")
        print(f"🔒 HTTPS security configured: {result['https_security']['https_config_generated']}")
        print(f"⚡ Rate limiting configured: {result['rate_limiting']['rate_limiting_configured']}")
        print("\n📋 Next steps:")
        for step in result['next_steps']:
            print(f"  - {step}")
        
    elif args.scan:
        issues = bot.scan_subprocess_security()
        print(f"Found {len(issues)} security issues")
        for issue in issues:
            print(f"  {issue.severity}: {issue.description}")
    
    elif args.report:
        report = bot.generate_security_report()
        print(f"Security Report:")
        print(f"  Total issues: {report.total_issues}")
        print(f"  Critical issues: {report.critical_issues}")
        print(f"  Auto-fixable: {len([i for i in report.issues if i.auto_fixable])}")

if __name__ == "__main__":
    main()