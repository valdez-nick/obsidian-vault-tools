#!/usr/bin/env python3
"""
Security Scanner for Obsidian Vault Tools

This script runs various security checks on the codebase:
- Bandit for code security issues
- Safety for vulnerable dependencies
- pip-audit for package vulnerabilities
- Custom security checks
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_command(cmd: List[str], capture_output: bool = True) -> tuple:
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"

def check_tool_available(tool: str) -> bool:
    """Check if a security tool is available"""
    returncode, _, _ = run_command([tool, "--version"], capture_output=True)
    return returncode == 0

def run_bandit_scan(directory: str) -> Dict[str, Any]:
    """Run Bandit security scan"""
    print(f"{Colors.BLUE}Running Bandit security scan...{Colors.ENDC}")
    
    if not check_tool_available("bandit"):
        print(f"{Colors.YELLOW}⚠️  Bandit not installed. Install with: pip install bandit{Colors.ENDC}")
        return {"status": "skipped", "reason": "tool not available"}
    
    # Run bandit with JSON output
    cmd = [
        "bandit", 
        "-r", directory,
        "-f", "json",
        "-ll",  # Only report medium and high severity
        "--exclude", "*/test*,*/venv/*,*/.git/*,*/node_modules/*"
    ]
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print(f"{Colors.GREEN}✓ Bandit scan completed - No issues found{Colors.ENDC}")
        return {"status": "passed", "issues": []}
    elif returncode == 1:
        # Bandit found issues
        try:
            result = json.loads(stdout)
            issues = result.get("results", [])
            print(f"{Colors.RED}⚠️  Bandit found {len(issues)} security issues{Colors.ENDC}")
            
            # Print top 5 issues
            for i, issue in enumerate(issues[:5]):
                print(f"  {i+1}. {issue['test_name']}: {issue['issue_text']}")
                print(f"     File: {issue['filename']}:{issue['line_number']}")
                print(f"     Severity: {issue['issue_severity']}")
                print()
            
            if len(issues) > 5:
                print(f"     ... and {len(issues) - 5} more issues")
            
            return {"status": "failed", "issues": issues}
        except json.JSONDecodeError:
            print(f"{Colors.RED}Error parsing Bandit output{Colors.ENDC}")
            return {"status": "error", "error": "Failed to parse output"}
    else:
        print(f"{Colors.RED}Bandit scan failed: {stderr}{Colors.ENDC}")
        return {"status": "error", "error": stderr}

def run_safety_check() -> Dict[str, Any]:
    """Run Safety vulnerability check"""
    print(f"{Colors.BLUE}Running Safety vulnerability check...{Colors.ENDC}")
    
    if not check_tool_available("safety"):
        print(f"{Colors.YELLOW}⚠️  Safety not installed. Install with: pip install safety{Colors.ENDC}")
        return {"status": "skipped", "reason": "tool not available"}
    
    cmd = ["safety", "check", "--json"]
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print(f"{Colors.GREEN}✓ Safety check passed - No vulnerable dependencies{Colors.ENDC}")
        return {"status": "passed", "vulnerabilities": []}
    else:
        try:
            result = json.loads(stdout)
            if isinstance(result, list) and len(result) > 0:
                vulns = result
                print(f"{Colors.RED}⚠️  Found {len(vulns)} vulnerable dependencies{Colors.ENDC}")
                
                for vuln in vulns[:5]:
                    print(f"  • {vuln['package']}: {vuln['vulnerability']}")
                    print(f"    Installed: {vuln['installed_version']}, Fixed in: {vuln['vulnerable_spec']}")
                    print()
                
                return {"status": "failed", "vulnerabilities": vulns}
            else:
                print(f"{Colors.GREEN}✓ Safety check passed{Colors.ENDC}")
                return {"status": "passed", "vulnerabilities": []}
        except json.JSONDecodeError:
            print(f"{Colors.YELLOW}Safety output could not be parsed as JSON{Colors.ENDC}")
            if "No known security vulnerabilities found" in stdout:
                print(f"{Colors.GREEN}✓ Safety check passed{Colors.ENDC}")
                return {"status": "passed", "vulnerabilities": []}
            else:
                return {"status": "error", "error": "Failed to parse output"}

def run_pip_audit() -> Dict[str, Any]:
    """Run pip-audit vulnerability check"""
    print(f"{Colors.BLUE}Running pip-audit vulnerability check...{Colors.ENDC}")
    
    if not check_tool_available("pip-audit"):
        print(f"{Colors.YELLOW}⚠️  pip-audit not installed. Install with: pip install pip-audit{Colors.ENDC}")
        return {"status": "skipped", "reason": "tool not available"}
    
    cmd = ["pip-audit", "--format=json"]
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode == 0:
        print(f"{Colors.GREEN}✓ pip-audit check passed{Colors.ENDC}")
        return {"status": "passed", "vulnerabilities": []}
    else:
        try:
            result = json.loads(stdout)
            vulns = result.get("vulnerabilities", [])
            
            if len(vulns) > 0:
                print(f"{Colors.RED}⚠️  pip-audit found {len(vulns)} vulnerabilities{Colors.ENDC}")
                
                for vuln in vulns[:5]:
                    print(f"  • {vuln['package']}: {vuln['id']}")
                    print(f"    {vuln['description'][:100]}...")
                    print()
                
                return {"status": "failed", "vulnerabilities": vulns}
            else:
                print(f"{Colors.GREEN}✓ pip-audit check passed{Colors.ENDC}")
                return {"status": "passed", "vulnerabilities": []}
                
        except json.JSONDecodeError:
            print(f"{Colors.RED}Error parsing pip-audit output{Colors.ENDC}")
            return {"status": "error", "error": "Failed to parse output"}

def run_custom_checks(directory: str) -> Dict[str, Any]:
    """Run custom security checks"""
    print(f"{Colors.BLUE}Running custom security checks...{Colors.ENDC}")
    
    issues = []
    
    # Check for hardcoded secrets
    secret_patterns = [
        (r'password\s*=\s*["\'][^"\']{3,}["\']', "Potential hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']{10,}["\']', "Potential hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']{10,}["\']', "Potential hardcoded secret"),
        (r'token\s*=\s*["\'][^"\']{10,}["\']', "Potential hardcoded token"),
    ]
    
    import re
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', 'venv', 'node_modules']):
            continue
            
        for file in files:
            if file.endswith(('.py', '.yaml', '.yml', '.json', '.env')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern, description in secret_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Calculate line number
                            line_num = content[:match.start()].count('\n') + 1
                            issues.append({
                                "file": file_path,
                                "line": line_num,
                                "issue": description,
                                "pattern": pattern
                            })
                            
                except Exception as e:
                    # Skip files that can't be read
                    continue
    
    if issues:
        print(f"{Colors.YELLOW}⚠️  Found {len(issues)} potential security issues{Colors.ENDC}")
        for issue in issues[:5]:
            print(f"  • {issue['issue']}")
            print(f"    File: {issue['file']}:{issue['line']}")
            print()
        
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more issues")
    else:
        print(f"{Colors.GREEN}✓ Custom security checks passed{Colors.ENDC}")
    
    return {"status": "passed" if len(issues) == 0 else "warning", "issues": issues}

def check_file_permissions(directory: str) -> Dict[str, Any]:
    """Check for files with overly permissive permissions"""
    print(f"{Colors.BLUE}Checking file permissions...{Colors.ENDC}")
    
    issues = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                stat = os.stat(file_path)
                mode = stat.st_mode & 0o777
                
                # Check for world-writable files
                if mode & 0o002:
                    issues.append({
                        "file": file_path,
                        "mode": oct(mode),
                        "issue": "World-writable file"
                    })
                
                # Check for executable files that shouldn't be
                if file.endswith(('.py', '.yaml', '.json', '.txt', '.md')) and mode & 0o111:
                    issues.append({
                        "file": file_path,
                        "mode": oct(mode),
                        "issue": "Unnecessary execute permission"
                    })
                    
            except (OSError, PermissionError):
                continue
    
    if issues:
        print(f"{Colors.YELLOW}⚠️  Found {len(issues)} file permission issues{Colors.ENDC}")
        for issue in issues[:5]:
            print(f"  • {issue['issue']}: {issue['file']} ({issue['mode']})")
    else:
        print(f"{Colors.GREEN}✓ File permissions look good{Colors.ENDC}")
    
    return {"status": "passed" if len(issues) == 0 else "warning", "issues": issues}

def generate_report(results: Dict[str, Any], output_file: str = None):
    """Generate security scan report"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}Security Scan Report{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    
    total_issues = 0
    total_vulnerabilities = 0
    
    for check, result in results.items():
        print(f"\n{Colors.BOLD}{check.replace('_', ' ').title()}:{Colors.ENDC}")
        
        if result["status"] == "passed":
            print(f"  {Colors.GREEN}✓ Passed{Colors.ENDC}")
        elif result["status"] == "failed":
            if "issues" in result:
                count = len(result["issues"])
                total_issues += count
                print(f"  {Colors.RED}✗ {count} issues found{Colors.ENDC}")
            if "vulnerabilities" in result:
                count = len(result["vulnerabilities"])
                total_vulnerabilities += count
                print(f"  {Colors.RED}✗ {count} vulnerabilities found{Colors.ENDC}")
        elif result["status"] == "warning":
            if "issues" in result:
                count = len(result["issues"])
                print(f"  {Colors.YELLOW}⚠ {count} warnings{Colors.ENDC}")
        elif result["status"] == "skipped":
            print(f"  {Colors.YELLOW}⚠ Skipped - {result.get('reason', 'unknown')}{Colors.ENDC}")
        else:
            print(f"  {Colors.RED}✗ Error - {result.get('error', 'unknown')}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  Security Issues: {total_issues}")
    print(f"  Vulnerabilities: {total_vulnerabilities}")
    
    if total_issues == 0 and total_vulnerabilities == 0:
        print(f"  {Colors.GREEN}Overall Status: ✓ SECURE{Colors.ENDC}")
    elif total_vulnerabilities > 0:
        print(f"  {Colors.RED}Overall Status: ✗ VULNERABLE{Colors.ENDC}")
    else:
        print(f"  {Colors.YELLOW}Overall Status: ⚠ NEEDS ATTENTION{Colors.ENDC}")
    
    # Save detailed report if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n{Colors.GREEN}Detailed report saved to: {output_file}{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.RED}Failed to save report: {e}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="Security scanner for Obsidian Vault Tools")
    parser.add_argument("--directory", "-d", default=".", help="Directory to scan")
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency vulnerability checks")
    parser.add_argument("--skip-code", action="store_true", help="Skip code security checks")
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}Security Scanner for Obsidian Vault Tools{Colors.ENDC}")
    print(f"Scanning directory: {os.path.abspath(args.directory)}")
    print()
    
    results = {}
    
    # Code security checks
    if not args.skip_code:
        results["bandit_scan"] = run_bandit_scan(args.directory)
        results["custom_checks"] = run_custom_checks(args.directory)
        results["file_permissions"] = check_file_permissions(args.directory)
    
    # Dependency vulnerability checks
    if not args.skip_deps:
        results["safety_check"] = run_safety_check()
        results["pip_audit"] = run_pip_audit()
    
    # Generate report
    generate_report(results, args.output)

if __name__ == "__main__":
    main()