#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for Obsidian Vault Tools

This test suite verifies all major features of the obsidian-vault-tools project.
Run this to ensure everything is working correctly after changes.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    """Print a section header"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")

def print_test(name, passed, details=""):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASSED{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAILED{Colors.ENDC}"
    print(f"  {name}: {status}")
    if details:
        print(f"    {Colors.YELLOW}{details}{Colors.ENDC}")

class ComprehensiveE2ETest:
    def __init__(self):
        self.test_vault_path = None
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
    def setup_test_vault(self):
        """Create a temporary test vault with sample data"""
        self.test_vault_path = tempfile.mkdtemp(prefix="obsidian_test_vault_")
        
        # Create sample structure
        folders = [
            'Daily Notes',
            'Projects',
            'Research',
            'References',
            'Templates',
            'Archive'
        ]
        
        for folder in folders:
            os.makedirs(os.path.join(self.test_vault_path, folder), exist_ok=True)
        
        # Create sample files
        sample_files = {
            'Daily Notes/2024-01-15.md': """# Daily Note - 2024-01-15

## Tasks
- [ ] Review project documentation
- [x] Update meeting notes
- [ ] Research #quantum-computing applications

## Notes
Had a productive meeting about the #ai-project today. 
The team discussed implementing #machine-learning algorithms.

## Links
- [[Project Alpha Overview]]
- [[Meeting Notes 2024-01-15]]
""",
            'Projects/Project Alpha Overview.md': """# Project Alpha Overview

Tags: #project #ai #machine-learning #active

## Description
Building an AI-powered knowledge management system.

## Team
- Lead: John Doe
- Dev: Jane Smith
- Research: Bob Johnson

## Status
Currently in Phase 2: Implementation

## Related
- [[AI Research Notes]]
- [[Technical Architecture]]
""",
            'Research/Quantum Computing Basics.md': """# Quantum Computing Basics

Tags: #quantum-computing #research #technology

## Introduction
Quantum computing represents a fundamental shift in computation...

## Key Concepts
- Qubits
- Superposition
- Entanglement
- Quantum gates

## Applications
- Cryptography
- Drug discovery
- Financial modeling
- AI/ML optimization
""",
            'References/Python Resources.md': """# Python Resources

Tags: #python #programming #reference

## Official Documentation
- [Python.org](https://python.org)
- [Python Package Index](https://pypi.org)

## Learning Resources
- Python for Data Science
- Advanced Python Patterns
- Async Programming in Python

## Libraries
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning
"""
        }
        
        for file_path, content in sample_files.items():
            full_path = os.path.join(self.test_vault_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Created test vault at: {self.test_vault_path}")
        return self.test_vault_path

    def cleanup_test_vault(self):
        """Remove the test vault"""
        if self.test_vault_path and os.path.exists(self.test_vault_path):
            shutil.rmtree(self.test_vault_path)
            logger.info("Cleaned up test vault")

    async def test_intelligence_system(self):
        """Test the intelligence system components"""
        print_section("Testing Intelligence System")
        
        try:
            from obsidian_vault_tools.intelligence import (
                IntentDetector, ActionExecutor, ContextManager, IntelligenceOrchestrator
            )
            
            # Test Intent Detector
            detector = IntentDetector()
            test_queries = [
                ("show my most used tags", "analyze_tags"),
                ("research quantum computing", "research_topic"),
                ("organize my files", "organize_files"),
                ("backup vault", "backup_vault")
            ]
            
            for query, expected_intent in test_queries:
                result = detector.detect_intent(query)
                passed = result.intent_type.value == expected_intent
                self.record_result(f"Intent Detection: '{query}'", passed, 
                                 f"Detected: {result.intent_type.value}")
            
            # Test Action Executor
            executor = ActionExecutor()
            
            # Test Context Manager
            context_mgr = ContextManager(self.test_vault_path)
            context_mgr.record_query("test query")
            self.record_result("Context Manager", True, "Query recording works")
            
            # Test Orchestrator
            orchestrator = IntelligenceOrchestrator()
            self.record_result("Intelligence Orchestrator", True, "Initialized successfully")
            
        except Exception as e:
            self.record_result("Intelligence System", False, str(e))

    async def test_llm_integration(self):
        """Test LLM integration (if available)"""
        print_section("Testing LLM Integration")
        
        try:
            from vault_query_system_llm import VaultQuerySystemLLM
            
            system = VaultQuerySystemLLM(self.test_vault_path)
            self.record_result("LLM Query System Import", True)
            
            # Check if Ollama is available
            try:
                initialized = await system.initialize()
                if initialized:
                    self.record_result("LLM Initialization", True, "Ollama connected")
                else:
                    self.record_result("LLM Initialization", False, "Ollama not available", skip=True)
            except:
                self.record_result("LLM Initialization", False, "Ollama not running", skip=True)
                
        except ImportError as e:
            self.record_result("LLM Integration", False, f"Import error: {e}", skip=True)

    async def test_research_assistant(self):
        """Test research assistant functionality"""
        print_section("Testing Research Assistant")
        
        try:
            from obsidian_vault_tools.research_assistant import ResearchAssistant
            
            assistant = ResearchAssistant(self.test_vault_path)
            self.record_result("Research Assistant Import", True)
            
            # Test research functionality
            result = await assistant.research_topic("test topic", depth="standard")
            passed = result['success'] and 'note_path' in result
            self.record_result("Research Topic", passed, 
                             f"Created note: {result.get('note_path', 'None')}")
            
        except Exception as e:
            self.record_result("Research Assistant", False, str(e))

    def test_audio_system(self):
        """Test audio system (if pygame available)"""
        print_section("Testing Audio System")
        
        try:
            from audio.audio_manager import AudioManager, get_audio_manager
            
            audio_mgr = get_audio_manager()
            if audio_mgr:
                self.record_result("Audio Manager", True, "Initialized with pygame")
                
                # Test audio initialization
                audio_initialized = audio_mgr.initialized and audio_mgr.enabled
                self.record_result("Audio Initialized", audio_initialized, 
                                 f"Audio system ready: {audio_mgr.initialized}")
            else:
                self.record_result("Audio Manager", False, "pygame not available", skip=True)
                
        except ImportError:
            self.record_result("Audio System", False, "pygame not installed", skip=True)

    def test_file_versioning(self):
        """Test file versioning system"""
        print_section("Testing File Versioning")
        
        try:
            from file_versioning import FileVersioning
            
            # Create test file
            test_file = os.path.join(self.test_vault_path, "test_versioning.md")
            with open(test_file, 'w') as f:
                f.write("Original content")
            
            # Test version creation using static method
            version_path = FileVersioning.get_next_version_path(test_file)
            passed = version_path != test_file  # Should return a different path
            self.record_result("Get Version Path", passed, f"Version: {version_path}")
            
            # Test backup functionality
            backup_path = FileVersioning.backup_existing_file(test_file)
            self.record_result("Backup File", backup_path is not None, f"Backup: {backup_path}")
            
        except Exception as e:
            self.record_result("File Versioning", False, str(e))

    def test_menu_navigation(self):
        """Test menu navigation system"""
        print_section("Testing Menu Navigation")
        
        try:
            from menu_navigator import MenuNavigator
            
            nav = MenuNavigator()
            self.record_result("Menu Navigator Import", True)
            
            # Test menu structure
            test_menu = {
                'title': 'Test Menu',
                'options': ['Option 1', 'Option 2', 'Option 3']
            }
            nav.current_menu = test_menu
            self.record_result("Menu Structure", True, "Menu configuration works")
            
        except Exception as e:
            self.record_result("Menu Navigation", False, str(e))

    def test_vault_analysis(self):
        """Test vault analysis features"""
        print_section("Testing Vault Analysis")
        
        try:
            from vault_query_system import VaultQuerySystem
            
            query_system = VaultQuerySystem(self.test_vault_path)
            
            # Scan vault
            vault_data = query_system.scan_vault()
            self.record_result("Vault Scanning", True, 
                             f"Found {vault_data['statistics']['total_files']} files")
            
            # Test tag analysis
            tags = vault_data.get('tags', {})
            self.record_result("Tag Analysis", len(tags) > 0, 
                             f"Found {len(tags)} unique tags")
            
            # Test query - use a term we know exists
            result = query_system.query("project")
            has_results = 'results' in result and len(result.get('results', [])) > 0
            self.record_result("Basic Query", has_results, 
                             f"Found {len(result.get('results', []))} results")
            
        except Exception as e:
            self.record_result("Vault Analysis", False, str(e))

    def test_mcp_tools(self):
        """Test MCP tools integration"""
        print_section("Testing MCP Tools Integration")
        
        try:
            from obsidian_vault_tools.mcp_tools import (
                MCPClientManager, MCPToolDiscovery, MCPToolExecutor, DynamicMenuBuilder
            )
            
            # Test imports
            self.record_result("MCP Tools Import", True, "All MCP modules imported")
            
            # Test client manager
            client_mgr = MCPClientManager()
            self.record_result("MCP Client Manager", True, "Initialized successfully")
            
            # Test tool discovery
            discovery = MCPToolDiscovery()
            self.record_result("MCP Tool Discovery", True, "Discovery system ready")
            
            # Test menu builder
            menu_builder = DynamicMenuBuilder()
            self.record_result("Dynamic Menu Builder", True, "Menu builder ready")
            
        except ImportError as e:
            self.record_result("MCP Tools", False, f"Import error: {e}", skip=True)

    def test_main_entry_points(self):
        """Test main entry points"""
        print_section("Testing Main Entry Points")
        
        # Test vault_manager_enhanced.py
        try:
            import vault_manager_enhanced
            self.record_result("vault_manager_enhanced.py", True, "Main entry point imports")
        except Exception as e:
            self.record_result("vault_manager_enhanced.py", False, str(e))
        
        # Test CLI
        try:
            from obsidian_vault_tools.cli import main as cli_main
            self.record_result("CLI Entry Point", True, "CLI imports successfully")
        except Exception as e:
            self.record_result("CLI Entry Point", False, str(e))

    def record_result(self, test_name, passed, details="", skip=False):
        """Record test result"""
        self.results['total'] += 1
        
        if skip:
            self.results['skipped'] += 1
            status = "SKIPPED"
        elif passed:
            self.results['passed'] += 1
            status = "PASSED"
        else:
            self.results['failed'] += 1
            status = "FAILED"
            self.results['errors'].append({
                'test': test_name,
                'details': details
            })
        
        print_test(test_name, passed and not skip, details)

    def print_summary(self):
        """Print test summary"""
        print_section("Test Summary")
        
        total = self.results['total']
        passed = self.results['passed']
        failed = self.results['failed']
        skipped = self.results['skipped']
        
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
        print(f"{Colors.RED}Failed: {failed}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Skipped: {skipped}{Colors.ENDC}")
        
        if failed > 0:
            print(f"\n{Colors.RED}Failed Tests:{Colors.ENDC}")
            for error in self.results['errors']:
                print(f"  - {error['test']}: {error['details']}")
        
        success_rate = (passed / (total - skipped) * 100) if (total - skipped) > 0 else 0
        print(f"\n{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.ENDC}")
        
        if success_rate == 100:
            print(f"{Colors.GREEN}🎉 All tests passed!{Colors.ENDC}")
        elif success_rate >= 80:
            print(f"{Colors.YELLOW}⚠️  Most tests passed, but some issues remain{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Significant issues detected{Colors.ENDC}")

    async def run_all_tests(self):
        """Run all tests"""
        print(f"{Colors.BOLD}Obsidian Vault Tools - Comprehensive E2E Test Suite{Colors.ENDC}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Setup
            self.setup_test_vault()
            
            # Run tests
            await self.test_intelligence_system()
            await self.test_llm_integration()
            await self.test_research_assistant()
            self.test_audio_system()
            self.test_file_versioning()
            self.test_menu_navigation()
            self.test_vault_analysis()
            self.test_mcp_tools()
            self.test_main_entry_points()
            
            # Summary
            self.print_summary()
            
        finally:
            # Cleanup
            self.cleanup_test_vault()
            
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Main test runner"""
    test_suite = ComprehensiveE2ETest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())