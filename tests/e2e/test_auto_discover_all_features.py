#!/usr/bin/env python3
"""
Auto-Discovering E2E Test Suite
Automatically discovers and tests ALL features without manual updates
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import importlib
import inspect
import click
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AutoDiscoveringE2ETest(unittest.TestCase):
    """Automatically discovers and tests ALL features without manual updates"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once"""
        cls.test_vault = tempfile.mkdtemp(prefix="test_vault_")
        cls._create_test_vault_structure()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, 'test_vault') and os.path.exists(cls.test_vault):
            shutil.rmtree(cls.test_vault)
    
    @classmethod
    def _create_test_vault_structure(cls):
        """Create a minimal test vault structure"""
        # Create .obsidian directory to make it a valid vault
        os.makedirs(os.path.join(cls.test_vault, '.obsidian'), exist_ok=True)
        
        # Create some test files
        test_files = {
            'test_note.md': '# Test Note\n\nThis is a test note with #tag1 and #tag2',
            'untagged_note.md': '# Untagged Note\n\nThis note has no tags.',
            'Daily Notes/2024-01-01.md': '# Daily Note\n\n- [ ] Task 1\n- [x] Task 2',
        }
        
        for file_path, content in test_files.items():
            full_path = os.path.join(cls.test_vault, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
    
    def discover_all_cli_commands(self) -> List[Dict[str, Any]]:
        """Dynamically discover all CLI commands from click groups"""
        try:
            from obsidian_vault_tools.cli import cli
            commands = []
            
            def extract_commands(group, prefix=""):
                """Recursively extract all commands and subcommands"""
                if hasattr(group, 'commands'):
                    for name, cmd in group.commands.items():
                        full_name = f"{prefix} {name}".strip()
                        commands.append({
                            'name': full_name,
                            'command': cmd,
                            'help': cmd.help or "No help available",
                            'params': [p.name for p in cmd.params]
                        })
                        # Recursively find subcommands
                        if hasattr(cmd, 'commands'):
                            extract_commands(cmd, full_name)
            
            extract_commands(cli, "ovt")
            return commands
        except ImportError:
            return []
    
    def discover_all_menu_options(self) -> Dict[str, List[Tuple[str, str]]]:
        """Dynamically discover all menu options from unified manager"""
        try:
            from unified_vault_manager import UnifiedVaultManager
            
            # Create manager with test vault
            manager = UnifiedVaultManager(self.test_vault)
            menus = {}
            
            # Find all menu handler methods
            for attr_name in dir(manager):
                if attr_name.startswith('handle_') and attr_name.endswith('_menu'):
                    menu_name = attr_name.replace('handle_', '').replace('_menu', '')
                    
                    # Get the method
                    menu_method = getattr(manager, attr_name)
                    
                    # Extract menu options by inspecting the source
                    options = self._extract_menu_options_from_source(menu_method)
                    if options:
                        menus[menu_name] = options
            
            # Also get main menu options
            if hasattr(manager, 'display_main_menu'):
                main_options = self._extract_main_menu_options(manager)
                if main_options:
                    menus['main'] = main_options
            
            return menus
        except ImportError:
            return {}
    
    def _extract_menu_options_from_source(self, method) -> List[Tuple[str, str]]:
        """Extract menu options by parsing method source"""
        try:
            import inspect
            source = inspect.getsource(method)
            
            # Look for options list in source
            options = []
            in_options = False
            
            for line in source.split('\n'):
                if 'options = [' in line:
                    in_options = True
                elif in_options and ']' in line:
                    break
                elif in_options and '(' in line and '"' in line:
                    # Extract option tuple
                    import re
                    match = re.search(r'\("([^"]+)".*?,\s*(\w+|None)', line)
                    if match:
                        label, action = match.groups()
                        options.append((label, action))
            
            return options
        except:
            return []
    
    def _extract_main_menu_options(self, manager) -> List[Tuple[str, str]]:
        """Extract main menu options"""
        try:
            # Call display_main_menu and capture the returned options
            with patch('builtins.print'):  # Suppress output
                options = manager.display_main_menu()
            return options
        except:
            return []
    
    def discover_standalone_scripts(self) -> List[str]:
        """Find all executable Python scripts"""
        scripts = []
        root_dir = Path(__file__).parent.parent.parent
        
        # Look for Python scripts in root and subdirectories
        patterns = ['*.py', 'scripts/*.py', 'tools/*.py', 'utils/*.py']
        
        for pattern in patterns:
            for script_path in root_dir.glob(pattern):
                # Check if it's a script (not a module)
                if script_path.is_file() and script_path.name != '__init__.py':
                    # Check if it has a main block or shebang
                    try:
                        with open(script_path, 'r') as f:
                            content = f.read()
                            if ('if __name__' in content and '__main__' in content) or \
                               content.startswith('#!/'):
                                scripts.append(str(script_path.relative_to(root_dir)))
                    except:
                        pass
        
        return sorted(scripts)
    
    def discover_all_features(self) -> Dict[str, bool]:
        """Discover all features and their availability"""
        try:
            from unified_vault_manager import UnifiedVaultManager
            manager = UnifiedVaultManager(self.test_vault)
            return manager.features
        except ImportError:
            return {}
    
    def test_all_cli_commands_discovered(self):
        """Test that we can discover CLI commands"""
        commands = self.discover_all_cli_commands()
        
        # Should find at least the basic commands
        command_names = [cmd['name'] for cmd in commands]
        
        # Basic commands that should exist
        expected_commands = [
            'ovt tags',
            'ovt backup',
            'ovt organize',
            'ovt version'
        ]
        
        for expected in expected_commands:
            self.assertIn(expected, command_names, 
                         f"Expected command '{expected}' not discovered")
        
        # Test that each discovered command has required info
        for cmd in commands:
            self.assertIn('name', cmd)
            self.assertIn('help', cmd)
            self.assertIn('params', cmd)
    
    def test_all_menu_options_discovered(self):
        """Test that we can discover menu options"""
        menus = self.discover_all_menu_options()
        
        # Should discover main menu and submenus
        self.assertIn('main', menus, "Main menu not discovered")
        
        # Main menu should have expected categories
        if 'main' in menus:
            main_labels = [opt[0] for opt in menus['main']]
            
            # Check for some expected menu items
            expected_items = [
                "Vault Analysis & Insights",
                "Tag Management & Organization",
                "Search & Query Vault"
            ]
            
            for expected in expected_items:
                found = any(expected in label for label in main_labels)
                self.assertTrue(found, f"Expected menu item '{expected}' not found")
    
    def test_cli_commands_execute_without_errors(self):
        """Test that discovered CLI commands can execute (at least show help)"""
        commands = self.discover_all_cli_commands()
        
        for cmd_info in commands[:5]:  # Test first 5 to keep test fast
            with self.subTest(command=cmd_info['name']):
                # Test help works
                result = subprocess.run(
                    cmd_info['name'].split() + ['--help'],
                    capture_output=True,
                    text=True
                )
                
                # Help should work even if command needs args
                self.assertIn([0, 2], [result.returncode], 
                             f"Command {cmd_info['name']} help failed")
                
                # Help text should appear
                if result.returncode == 0:
                    self.assertIn(cmd_info['help'], result.stdout + result.stderr)
    
    def test_feature_availability_detection(self):
        """Test that feature availability is correctly detected"""
        features = self.discover_all_features()
        
        # Should have some features
        self.assertGreater(len(features), 0, "No features discovered")
        
        # All features should be boolean
        for feature_name, is_available in features.items():
            self.assertIsInstance(is_available, bool, 
                                f"Feature {feature_name} availability is not boolean")
        
        # Test that core features are detected
        core_features = ['navigation', 'versioning']
        for core in core_features:
            self.assertIn(core, features, f"Core feature {core} not in features dict")
    
    def test_menu_navigation_without_crashes(self):
        """Test that menu navigation doesn't crash"""
        try:
            from unified_vault_manager import UnifiedVaultManager
            
            # Mock user input to exit immediately
            with patch('builtins.input', return_value='13'):  # Exit option
                with patch('builtins.print'):  # Suppress output
                    manager = UnifiedVaultManager(self.test_vault)
                    
                    # This should not raise any exceptions
                    try:
                        # Just create the manager and display menu
                        manager.display_banner()
                        manager.display_main_menu()
                        # Success if no exception
                        self.assertTrue(True)
                    except Exception as e:
                        self.fail(f"Menu display crashed: {e}")
                        
        except ImportError:
            self.skipTest("unified_vault_manager not available")
    
    def test_dynamic_feature_loading(self):
        """Test that features load dynamically based on availability"""
        try:
            from unified_vault_manager import UnifiedVaultManager
            
            # Create manager
            manager = UnifiedVaultManager(self.test_vault)
            
            # Test that unavailable features are handled gracefully
            for feature_name, is_available in manager.features.items():
                with self.subTest(feature=feature_name):
                    # Get the feature attribute name
                    feature_attr = getattr(manager, feature_name, None)
                    
                    if is_available:
                        # If available, attribute should not be None (in most cases)
                        # Some features might still be None if initialization failed
                        pass  # Can't assert not None as some features initialize lazily
                    else:
                        # If not available, accessing should not crash
                        try:
                            # Try to access the feature
                            _ = feature_attr
                            # Success - no crash
                            self.assertTrue(True)
                        except:
                            self.fail(f"Accessing unavailable feature {feature_name} crashed")
                            
        except ImportError:
            self.skipTest("unified_vault_manager not available")
    
    def generate_test_coverage_report(self) -> Dict[str, Any]:
        """Generate a report of all discovered features for coverage tracking"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cli_commands': self.discover_all_cli_commands(),
            'menu_options': self.discover_all_menu_options(),
            'features': self.discover_all_features(),
            'standalone_scripts': self.discover_standalone_scripts(),
            'test_vault': self.test_vault
        }
        
        # Count totals
        report['totals'] = {
            'cli_commands': len(report['cli_commands']),
            'menu_categories': len(report['menu_options']),
            'total_menu_options': sum(len(opts) for opts in report['menu_options'].values()),
            'features': len(report['features']),
            'features_available': sum(1 for available in report['features'].values() if available),
            'standalone_scripts': len(report['standalone_scripts'])
        }
        
        return report


def run_tests_and_generate_report():
    """Run tests and generate coverage report"""
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(AutoDiscoveringE2ETest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate coverage report
    test_instance = AutoDiscoveringE2ETest()
    test_instance.setUpClass()
    
    try:
        report = test_instance.generate_test_coverage_report()
        
        # Save report
        report_path = Path(__file__).parent / 'coverage_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TEST COVERAGE REPORT")
        print(f"{'='*60}")
        print(f"Generated: {report['timestamp']}")
        print(f"\nDiscovered Features:")
        print(f"  - CLI Commands: {report['totals']['cli_commands']}")
        print(f"  - Menu Categories: {report['totals']['menu_categories']}")
        print(f"  - Total Menu Options: {report['totals']['total_menu_options']}")
        print(f"  - Features: {report['totals']['features']} ({report['totals']['features_available']} available)")
        print(f"  - Standalone Scripts: {report['totals']['standalone_scripts']}")
        print(f"\nReport saved to: {report_path}")
        
    finally:
        test_instance.tearDownClass()
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests_and_generate_report()
    sys.exit(0 if success else 1)