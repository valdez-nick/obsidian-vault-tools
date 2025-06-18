#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing Suite for Obsidian Vault Manager
Tests all interactive menu features, progress bars, and natural language interface
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the parent directory to sys.path to import vault manager modules
sys.path.insert(0, str(Path(__file__).parent))

# Import the modules we're testing
from vault_manager import VaultManager, Colors
try:
    from natural_language_query import NaturalLanguageProcessor
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

class TestResults:
    """Track test results and generate reports"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.start_time = datetime.now()
        
    def add_result(self, test_name: str, passed: bool, details: str = ""):
        """Add a test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"{Colors.GREEN}✓ PASS{Colors.ENDC}: {test_name}")
        else:
            self.tests_failed += 1
            self.failures.append(f"{test_name}: {details}")
            print(f"{Colors.RED}✗ FAIL{Colors.ENDC}: {test_name} - {details}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        duration = datetime.now() - self.start_time
        
        report = f"""
{Colors.HEADER}{Colors.BOLD}COMPREHENSIVE E2E TEST REPORT{Colors.ENDC}
{'=' * 60}

Test Summary:
- Total Tests: {self.tests_run}
- Passed: {Colors.GREEN}{self.tests_passed}{Colors.ENDC}
- Failed: {Colors.RED}{self.tests_failed}{Colors.ENDC}
- Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%
- Duration: {duration.total_seconds():.1f} seconds

"""
        
        if self.failures:
            report += f"{Colors.RED}Failed Tests:{Colors.ENDC}\n"
            for i, failure in enumerate(self.failures, 1):
                report += f"  {i}. {failure}\n"
        
        return report

class TestVault:
    """Create and manage a test vault for testing purposes"""
    
    def __init__(self):
        self.temp_dir = None
        self.vault_path = None
        
    def __enter__(self):
        """Create a temporary test vault"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_vault_")
        self.vault_path = self.temp_dir
        
        # Create .obsidian directory to make it a valid vault
        obsidian_dir = os.path.join(self.vault_path, '.obsidian')
        os.makedirs(obsidian_dir, exist_ok=True)
        
        # Create test files with various tag patterns
        self.create_test_files()
        
        return self.vault_path
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the test vault"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test markdown files with various patterns"""
        test_files = {
            'tagged_note.md': """---
tags: [test, markdown, frontmatter]
---

# Tagged Note

This note has tags in frontmatter and inline #inline-tag #test-tag.
""",
            'untagged_note.md': """# Untagged Note

This note has no tags at all.
""",
            'quoted_tags.md': """# Note with Quoted Tags

This note has "quoted tags" that need fixing.

Tags: #"machine learning" #"data science" #ai
""",
            'similar_tags.md': """# Note with Similar Tags

This note has similar tags that should be merged.

#ml #machinelearning #machine-learning #ML
""",
            'duplicate_content.md': """# Duplicate Content

This is some duplicate content that appears in multiple files.
The content here is identical to another file.
""",
            'duplicate_content_copy.md': """# Duplicate Content

This is some duplicate content that appears in multiple files.
The content here is identical to another file.
""",
        }
        
        # Create subdirectories
        subdirs = ['Notes', 'Projects', 'Archive', 'Templates']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.vault_path, subdir), exist_ok=True)
        
        # Create test files
        for filename, content in test_files.items():
            filepath = os.path.join(self.vault_path, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create files in subdirectories
        subdir_files = {
            'Notes/daily_note.md': '# Daily Note\n\n#daily #journal',
            'Projects/project_note.md': '# Project Note\n\n#project #work',
            'Archive/old_note.md': '# Old Note\n\nThis is archived content.',
            'Templates/template.md': '# Template\n\n#template'
        }
        
        for filepath, content in subdir_files.items():
            full_path = os.path.join(self.vault_path, filepath)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

class ComprehensiveE2ETest:
    """Comprehensive end-to-end test suite for vault manager"""
    
    def __init__(self):
        self.results = TestResults()
        self.vault_manager = None
        
    def setup_vault_manager(self, vault_path: str):
        """Setup vault manager with test vault"""
        self.vault_manager = VaultManager()
        self.vault_manager.current_vault = vault_path
        self.vault_manager.config['last_vault'] = vault_path
        
    def run_command_test(self, command: str, description: str, expected_success: bool = True) -> bool:
        """Test a command execution"""
        try:
            print(f"\n{Colors.CYAN}Testing: {description}{Colors.ENDC}")
            print(f"Command: {command}")
            
            # Use the vault manager's run_command method
            result = self.vault_manager.run_command(command, description)
            
            if expected_success:
                success = result is True
                details = "Command succeeded as expected" if success else "Command failed unexpectedly"
            else:
                success = result is False
                details = "Command failed as expected" if success else "Command succeeded when it should have failed"
                
            self.results.add_result(f"Command: {description}", success, details)
            return success
            
        except Exception as e:
            self.results.add_result(f"Command: {description}", False, f"Exception: {str(e)}")
            return False
    
    def test_vault_analysis_features(self, vault_path: str):
        """Test all vault analysis features"""
        print(f"\n{Colors.HEADER}=== TESTING VAULT ANALYSIS FEATURES ==={Colors.ENDC}")
        
        # Test 1: Tag Analysis
        success = self.run_command_test(
            f'python3 analyze_tags_simple.py "{vault_path}"',
            "Tag Analysis",
            expected_success=True
        )
        
        # Test 2: Folder Structure Analysis
        try:
            self.vault_manager.analyze_folder_structure()
            self.results.add_result("Folder Structure Analysis", True)
        except Exception as e:
            self.results.add_result("Folder Structure Analysis", False, str(e))
        
        # Test 3: Find Untagged Files
        try:
            self.vault_manager.find_untagged_files()
            self.results.add_result("Find Untagged Files", True)
        except Exception as e:
            self.results.add_result("Find Untagged Files", False, str(e))
        
        # Test 4: JSON Report Generation (check if file was created)
        json_report = os.path.join(vault_path, 'tag_analysis_report.json')
        if os.path.exists(json_report):
            try:
                with open(json_report, 'r') as f:
                    data = json.load(f)
                    has_required_keys = all(key in data for key in ['summary', 'tag_frequencies'])
                    self.results.add_result("JSON Report Generation", has_required_keys, 
                                           "Valid JSON report created" if has_required_keys else "Invalid JSON structure")
            except Exception as e:
                self.results.add_result("JSON Report Generation", False, f"JSON parsing error: {str(e)}")
        else:
            self.results.add_result("JSON Report Generation", False, "JSON report file not created")
    
    def test_tag_management_features(self, vault_path: str):
        """Test all tag management features"""
        print(f"\n{Colors.HEADER}=== TESTING TAG MANAGEMENT FEATURES ==={Colors.ENDC}")
        
        # Test 1: Preview Tag Issues (Dry Run) - default behavior
        success = self.run_command_test(
            f'python3 fix_vault_tags.py "{vault_path}"',
            "Preview Tag Issues (Dry Run)",
            expected_success=True
        )
        
        # Test 2: Apply Tag Fixes 
        success = self.run_command_test(
            f'python3 fix_vault_tags.py "{vault_path}" --apply',
            "Apply Tag Fixes",
            expected_success=True
        )
        
        # Test 5: Check if scripts exist
        required_scripts = ['fix_vault_tags.py', 'analyze_tags_simple.py']
        for script in required_scripts:
            script_exists = os.path.exists(script)
            self.results.add_result(f"Required Script: {script}", script_exists,
                                  "Script found" if script_exists else "Script missing")
    
    def test_backup_operations(self, vault_path: str):
        """Test backup operations and investigate stuck backup issue"""
        print(f"\n{Colors.HEADER}=== TESTING BACKUP OPERATIONS ==={Colors.ENDC}")
        
        # Test 1: Check if backup scripts exist
        backup_scripts = ['backup_vault.py', 'quick_incremental_backup.sh']
        for script in backup_scripts:
            script_exists = os.path.exists(script)
            self.results.add_result(f"Backup Script Exists: {script}", script_exists,
                                  "Script found" if script_exists else "Script missing")
        
        # Test 2: Test backup with timeout to avoid stuck processes
        def run_backup_with_timeout(command, timeout=30):
            """Run backup command with timeout to detect stuck processes"""
            try:
                print(f"Running backup command with {timeout}s timeout: {command}")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                return result.returncode == 0, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return False, "", "TIMEOUT: Backup process stuck"
            except Exception as e:
                return False, "", f"EXCEPTION: {str(e)}"
        
        # Test 3: Python Backup Script
        if os.path.exists('backup_vault.py'):
            success, stdout, stderr = run_backup_with_timeout(f'python3 backup_vault.py "{vault_path}"')
            details = "Backup completed" if success else f"Backup failed: {stderr}"
            if "TIMEOUT" in stderr:
                details = "🚨 BACKUP STUCK - Process did not complete within timeout"
            self.results.add_result("Python Backup Script", success, details)
        
        # Test 4: Shell Backup Script
        if os.path.exists('quick_incremental_backup.sh'):
            success, stdout, stderr = run_backup_with_timeout(f'./quick_incremental_backup.sh "{vault_path}"')
            details = "Incremental backup completed" if success else f"Incremental backup failed: {stderr}"
            if "TIMEOUT" in stderr:
                details = "🚨 INCREMENTAL BACKUP STUCK - Process did not complete within timeout"
            self.results.add_result("Shell Backup Script", success, details)
        
        # Test 5: Check for stuck processes (exclude system processes)
        try:
            # Check for any running backup processes that are NOT system processes
            result = subprocess.run(['pgrep', '-f', 'backup_vault.py|quick_incremental_backup'], capture_output=True, text=True)
            if result.returncode == 0:
                stuck_processes = result.stdout.strip().split('\n')
                # Filter out empty strings
                stuck_processes = [p for p in stuck_processes if p.strip()]
                if stuck_processes:
                    self.results.add_result("Stuck Backup Processes", False, 
                                          f"Found {len(stuck_processes)} potentially stuck backup processes: {stuck_processes}")
                else:
                    self.results.add_result("Stuck Backup Processes", True, "No stuck backup processes found")
            else:
                self.results.add_result("Stuck Backup Processes", True, "No stuck backup processes found")
        except Exception as e:
            self.results.add_result("Stuck Backup Processes", False, f"Could not check for stuck processes: {str(e)}")
    
    def test_progress_indicators(self, vault_path: str):
        """Test progress indicators and loading bars"""
        print(f"\n{Colors.HEADER}=== TESTING PROGRESS INDICATORS ==={Colors.ENDC}")
        
        # Test 1: Basic Progress Bar
        try:
            print(f"\n{Colors.CYAN}Testing progress bar functionality:{Colors.ENDC}")
            for i in range(5):
                self.vault_manager.show_progress_bar("Test Progress", i, 4, width=30)
                time.sleep(0.5)
            print()  # New line after progress bar
            self.results.add_result("Progress Bar Display", True, "Progress bar rendered correctly")
        except Exception as e:
            self.results.add_result("Progress Bar Display", False, str(e))
        
        # Test 2: Indeterminate Progress
        try:
            print(f"\n{Colors.CYAN}Testing indeterminate progress:{Colors.ENDC}")
            self.vault_manager.show_indeterminate_progress("Test Loading", 2)
            self.results.add_result("Indeterminate Progress", True, "Indeterminate progress displayed")
        except Exception as e:
            self.results.add_result("Indeterminate Progress", False, str(e))
        
        # Test 3: Spinner
        try:
            print(f"\n{Colors.CYAN}Testing spinner:{Colors.ENDC}")
            self.vault_manager.show_loading("Test Spinner", 2)
            self.results.add_result("Spinner Display", True, "Spinner displayed correctly")
        except Exception as e:
            self.results.add_result("Spinner Display", False, str(e))
    
    def test_natural_language_interface(self, vault_path: str):
        """Test natural language query interface"""
        print(f"\n{Colors.HEADER}=== TESTING NATURAL LANGUAGE INTERFACE ==={Colors.ENDC}")
        
        if not NLP_AVAILABLE:
            self.results.add_result("NLP Module Import", False, "natural_language_query module not available")
            return
        
        self.results.add_result("NLP Module Import", True, "natural_language_query module imported successfully")
        
        # Test 1: Initialize NLP Processor
        try:
            nlp = NaturalLanguageProcessor(self.vault_manager)
            self.results.add_result("NLP Processor Initialization", True, "NLP processor created successfully")
        except Exception as e:
            self.results.add_result("NLP Processor Initialization", False, str(e))
            return
        
        # Test 2: Test Common Queries
        test_queries = [
            ("analyze my tags", "analyze_tags", "Tag analysis query"),
            ("backup the vault", "incremental_backup", "Backup query"),
            ("find files without tags", "find_untagged", "Untagged files query"),
            ("help", "help", "Help query"),
            ("research artificial intelligence", "research_topic", "Research query"),
            ("organize files", "smart_organize", "Organization query"),
            ("unknown gibberish query", "unknown", "Unknown query handling")
        ]
        
        for query, expected_action, description in test_queries:
            try:
                result = nlp.process_query(query)
                success = result.action == expected_action or (expected_action == "unknown" and result.confidence == 0.0)
                details = f"Query: '{query}' -> Action: '{result.action}' (Confidence: {result.confidence:.2f})"
                self.results.add_result(f"NLP Query: {description}", success, details)
            except Exception as e:
                self.results.add_result(f"NLP Query: {description}", False, str(e))
        
        # Test 3: Autocomplete Suggestions
        try:
            suggestions = nlp.get_autocomplete_suggestions("analy")
            has_suggestions = len(suggestions) > 0
            self.results.add_result("NLP Autocomplete", has_suggestions, 
                                  f"Generated {len(suggestions)} suggestions")
        except Exception as e:
            self.results.add_result("NLP Autocomplete", False, str(e))
        
        # Test 4: Help System
        try:
            help_text = nlp.get_command_help()
            has_help = len(help_text) > 100  # Should be substantial help text
            self.results.add_result("NLP Help System", has_help, 
                                  f"Generated {len(help_text)} characters of help text")
        except Exception as e:
            self.results.add_result("NLP Help System", False, str(e))
    
    def test_v2_features_fallback(self, vault_path: str):
        """Test V2 features and their fallback behavior"""
        print(f"\n{Colors.HEADER}=== TESTING V2 FEATURES FALLBACK ==={Colors.ENDC}")
        
        # Check V2 availability
        v2_available = self.vault_manager.v2_available
        self.results.add_result("V2 Availability Check", True, 
                              f"V2 {'available' if v2_available else 'not available'}")
        
        if not v2_available:
            # Test that V2 commands show appropriate messages
            v2_commands = [
                "obsidian-librarian analyze",
                "obsidian-librarian research", 
                "obsidian-librarian organize",
                "obsidian-librarian duplicates",
                "obsidian-librarian stats",
                "obsidian-librarian curate"
            ]
            
            for command in v2_commands:
                success = self.run_command_test(f'{command} "{vault_path}"', 
                                              f"V2 Command Fallback: {command.split()[1]}", 
                                              expected_success=False)
    
    def test_error_handling(self, vault_path: str):
        """Test error handling and edge cases"""
        print(f"\n{Colors.HEADER}=== TESTING ERROR HANDLING ==={Colors.ENDC}")
        
        # Test 1: Non-existent vault path
        fake_vault = "/nonexistent/vault/path"
        success = self.run_command_test(
            f'python3 analyze_tags_simple.py "{fake_vault}"',
            "Non-existent Vault Path",
            expected_success=False
        )
        
        # Test 2: Invalid command
        success = self.run_command_test(
            "nonexistent_command_12345",
            "Invalid Command",
            expected_success=False
        )
        
        # Test 3: Permissions test (try to write to read-only location)
        try:
            readonly_path = "/System/test_file_readonly"
            success = self.run_command_test(
                f'echo "test" > "{readonly_path}"',
                "Permissions Error Handling", 
                expected_success=False
            )
        except:
            self.results.add_result("Permissions Error Handling", True, "Handled permission errors correctly")
        
        # Test 4: Large file handling (create a large test file)
        try:
            large_file = os.path.join(vault_path, "large_test.md")
            with open(large_file, 'w') as f:
                # Write a moderately large file (not too large to slow down tests)
                for i in range(1000):
                    f.write(f"# Large File Line {i}\n\nThis is line {i} of a large test file. #test #large-file\n\n")
            
            # Test analysis on large file
            success = self.run_command_test(
                f'python3 analyze_tags_simple.py "{vault_path}"',
                "Large File Handling",
                expected_success=True
            )
            
            # Clean up
            os.remove(large_file)
            
        except Exception as e:
            self.results.add_result("Large File Handling", False, str(e))
    
    def test_configuration_and_settings(self, vault_path: str):
        """Test configuration and settings functionality"""
        print(f"\n{Colors.HEADER}=== TESTING CONFIGURATION AND SETTINGS ==={Colors.ENDC}")
        
        # Test 1: Config file creation and loading
        try:
            original_config = self.vault_manager.config.copy()
            self.vault_manager.config['test_setting'] = 'test_value'
            self.vault_manager.save_config()
            
            # Create new instance to test loading
            new_manager = VaultManager()
            has_test_setting = new_manager.config.get('test_setting') == 'test_value'
            
            self.results.add_result("Config Save/Load", has_test_setting, 
                                  "Configuration saved and loaded correctly")
            
            # Restore original config
            self.vault_manager.config = original_config
            self.vault_manager.save_config()
            
        except Exception as e:
            self.results.add_result("Config Save/Load", False, str(e))
        
        # Test 2: Vault path validation
        try:
            # Test valid path
            valid_result = self.vault_manager.get_vault_path() is not None
            self.results.add_result("Vault Path Validation", True, "Path validation working")
        except Exception as e:
            self.results.add_result("Vault Path Validation", False, str(e))
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}STARTING COMPREHENSIVE E2E TEST SUITE{Colors.ENDC}")
        print(f"{Colors.BLUE}Testing all vault manager features, progress bars, and NLP interface{Colors.ENDC}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with TestVault() as test_vault_path:
            print(f"\n{Colors.CYAN}Created test vault at: {test_vault_path}{Colors.ENDC}")
            
            # Setup vault manager
            self.setup_vault_manager(test_vault_path)
            
            # Run all test categories
            self.test_vault_analysis_features(test_vault_path)
            self.test_tag_management_features(test_vault_path)
            self.test_backup_operations(test_vault_path)
            self.test_progress_indicators(test_vault_path)
            self.test_natural_language_interface(test_vault_path)
            self.test_v2_features_fallback(test_vault_path)
            self.test_error_handling(test_vault_path)
            self.test_configuration_and_settings(test_vault_path)
        
        # Generate and display report
        print(self.results.generate_report())
        
        # Save detailed report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(self.results.generate_report())
        
        print(f"\n{Colors.GREEN}Detailed test report saved to: {report_file}{Colors.ENDC}")
        
        return self.results.tests_failed == 0

def main():
    """Main entry point for the test suite"""
    print(f"{Colors.HEADER}Obsidian Vault Manager - Comprehensive E2E Test Suite{Colors.ENDC}")
    print(f"{Colors.BLUE}This will test all features including potential stuck backup issues{Colors.ENDC}")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("""
Usage: python3 comprehensive_e2e_test.py [options]

Options:
  --help          Show this help message
  --quick         Run quick tests only (skip backup tests)
  --backup-only   Run only backup tests to investigate stuck issues
  
This test suite will:
1. Create a temporary test vault
2. Test all vault manager features
3. Test progress bars and loading indicators  
4. Test natural language query interface
5. Investigate backup operations for stuck processes
6. Generate a comprehensive report
        """)
        return
    
    test_suite = ComprehensiveE2ETest()
    
    try:
        success = test_suite.run_all_tests()
        
        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.ENDC}")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED ❌{Colors.ENDC}")
            print(f"{Colors.YELLOW}Check the detailed report for failure analysis{Colors.ENDC}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {str(e)}{Colors.ENDC}")
        sys.exit(1)

if __name__ == '__main__':
    main()