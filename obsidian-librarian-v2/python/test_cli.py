#!/usr/bin/env python3
"""
CLI Test Script for Obsidian Librarian

This script tests the CLI functionality to ensure all commands work correctly.
"""

import subprocess
import sys
import tempfile
import shutil
import os
from pathlib import Path

def run_command(cmd, capture_output=True, check=True):
    """Run a CLI command and return the result."""
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, 
        capture_output=capture_output, 
        text=True, 
        check=check,
        env=env
    )
    
    if capture_output:
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    return result

def test_cli():
    """Test the CLI functionality."""
    print("=" * 60)
    print("OBSIDIAN LIBRARIAN CLI TEST")
    print("=" * 60)
    
    # Test 1: Help command
    print("\n1. Testing main help command...")
    result = run_command([sys.executable, "-m", "obsidian_librarian", "--help"])
    assert "Obsidian Librarian" in result.stdout
    assert "Commands" in result.stdout
    print("✓ Main help works")
    
    # Test 2: Version command
    print("\n2. Testing version command...")
    result = run_command([sys.executable, "-m", "obsidian_librarian", "--version"])
    assert "Obsidian Librarian" in result.stdout
    print("✓ Version command works")
    
    # Test 3: Tags help
    print("\n3. Testing tags help...")
    result = run_command([sys.executable, "-m", "obsidian_librarian", "tags", "--help"])
    assert "tag management" in result.stdout.lower()
    print("✓ Tags help works")
    
    # Test 4: Git help
    print("\n4. Testing git help...")
    result = run_command([sys.executable, "-m", "obsidian_librarian", "git", "--help"])
    assert "version control" in result.stdout.lower()
    print("✓ Git help works")
    
    # Test 5: Initialize vault
    print("\n5. Testing vault initialization...")
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir) / "test-vault"
        result = run_command([
            sys.executable, "-m", "obsidian_librarian", 
            "init", str(vault_path), "--name", "Test Vault", "--force"
        ])
        assert vault_path.exists()
        assert (vault_path / ".obsidian").exists()
        assert (vault_path / "README.md").exists()
        print("✓ Vault initialization works")
        
        # Test 6: Stats command
        print("\n6. Testing stats command...")
        result = run_command([
            sys.executable, "-m", "obsidian_librarian", 
            "stats", str(vault_path)
        ])
        assert "Vault Statistics" in result.stdout
        print("✓ Stats command works")
    
    # Test 7: Individual tag commands help
    print("\n7. Testing tag subcommands...")
    tag_commands = ["analyze", "duplicates", "suggest", "auto-tag", "merge", "cleanup", "hierarchy"]
    for cmd in tag_commands:
        result = run_command([
            sys.executable, "-m", "obsidian_librarian", 
            "tags", cmd, "--help"
        ])
        assert "vault" in result.stdout.lower() or "tag" in result.stdout.lower()
        print(f"  ✓ tags {cmd} help works")
    
    # Test 8: Core commands help
    print("\n8. Testing core commands...")
    core_commands = ["analyze", "research", "organize", "curate"]
    for cmd in core_commands:
        result = run_command([
            sys.executable, "-m", "obsidian_librarian", 
            cmd, "--help"
        ])
        assert "vault" in result.stdout.lower()
        print(f"  ✓ {cmd} help works")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("The CLI architecture has been successfully fixed!")
    print("=" * 60)

if __name__ == "__main__":
    test_cli()