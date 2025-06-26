#!/usr/bin/env python3
"""
CLI Command Auto-Discovery Test
Automatically discovers and tests all CLI commands without hardcoding
"""

import os
import sys
import subprocess
import tempfile
import shutil
import click
from pathlib import Path
from typing import List, Dict, Any, Optional
import unittest
from unittest.mock import patch, MagicMock
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CLIAutoTest(unittest.TestCase):
    """Test all CLI commands without hardcoding them"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_vault = tempfile.mkdtemp(prefix="test_vault_cli_")
        cls._create_test_vault()
        
        # Find the ovt command
        cls.ovt_command = cls._find_ovt_command()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, 'test_vault') and os.path.exists(cls.test_vault):
            shutil.rmtree(cls.test_vault)
    
    @classmethod
    def _create_test_vault(cls):
        """Create a test vault with sample data"""
        # Make it a valid Obsidian vault
        os.makedirs(os.path.join(cls.test_vault, '.obsidian'), exist_ok=True)
        
        # Create test files
        test_files = {
            'note1.md': '# Note 1\n\nContent with #tag1 and #tag2',
            'note2.md': '# Note 2\n\nContent with #tag2 and #tag3',
            'Daily Notes/2024-01-01.md': '# Daily Note\n\n- [ ] Task 1',
        }
        
        for file_path, content in test_files.items():
            full_path = os.path.join(cls.test_vault, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
    
    @classmethod
    def _find_ovt_command(cls) -> List[str]:
        """Find the ovt command - could be installed or in development"""
        # Try direct command first
        result = subprocess.run(['which', 'ovt'], capture_output=True, text=True)
        if result.returncode == 0:
            return ['ovt']
        
        # Try python module
        return [sys.executable, '-m', 'obsidian_vault_tools.cli']
    
    def discover_all_cli_commands(self) -> List[Dict[str, Any]]:
        """Dynamically discover all CLI commands"""
        try:
            # Import the CLI to introspect it
            from obsidian_vault_tools.cli import cli
            
            commands = []
            
            def extract_commands(group, prefix=""):
                """Recursively extract all commands and subcommands"""
                # Get the command name
                cmd_name = prefix if prefix else "ovt"
                
                # Add the group itself if it has a callback
                if hasattr(group, 'callback') and group.callback and prefix:
                    commands.append({
                        'name': cmd_name,
                        'obj': group,
                        'help': group.help or "No help available",
                        'params': getattr(group, 'params', []),
                        'is_group': False
                    })
                
                # Extract subcommands
                if hasattr(group, 'commands'):
                    # If this is a group with commands, add it
                    if prefix:  # Don't add root 'ovt' as a command
                        commands.append({
                            'name': cmd_name,
                            'obj': group,
                            'help': group.help or "No help available",
                            'params': getattr(group, 'params', []),
                            'is_group': True
                        })
                    
                    # Process all subcommands
                    for name, cmd in group.commands.items():
                        subcmd_name = f"{cmd_name} {name}" if prefix else f"ovt {name}"
                        
                        if isinstance(cmd, click.Group):
                            # Recursively process groups
                            extract_commands(cmd, subcmd_name)
                        else:
                            # Add regular command
                            commands.append({
                                'name': subcmd_name,
                                'obj': cmd,
                                'help': cmd.help or "No help available",
                                'params': cmd.params if hasattr(cmd, 'params') else [],
                                'is_group': False
                            })
            
            # Start extraction from root CLI
            extract_commands(cli)
            
            return commands
            
        except ImportError as e:
            print(f"Warning: Could not import CLI: {e}")
            return []
    
    def run_cli_command(self, command: List[str], input_text: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run a CLI command and return result"""
        # Construct full command
        if command[0] == 'ovt':
            full_command = self.ovt_command + command[1:]
        else:
            full_command = command
        
        # Run command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            input=input_text,
            timeout=10  # Prevent hanging
        )
        
        return result
    
    def infer_required_args(self, cmd_info: Dict[str, Any]) -> List[str]:
        """Infer required arguments based on command parameters"""
        args = []
        
        # Check parameters
        for param in cmd_info.get('params', []):
            if param.name == 'vault' and not param.required:
                # Add vault path for commands that can use it
                args.extend(['--vault', self.test_vault])
            elif param.required and param.name not in ['ctx']:  # Skip context params
                # For required params, provide test values
                if param.type == click.Path:
                    args.append(self.test_vault)
                elif param.type == str:
                    args.append('test_value')
                elif param.type == click.Choice:
                    # Use first choice
                    if hasattr(param.type, 'choices'):
                        args.append(param.type.choices[0])
        
        return args
    
    def test_all_cli_commands_help(self):
        """Test that all discovered commands show help"""
        commands = self.discover_all_cli_commands()
        
        self.assertGreater(len(commands), 0, "No CLI commands discovered")
        
        # Test help for each command
        tested = 0
        for cmd_info in commands:
            # Skip the root 'ovt' command and certain problem commands
            if cmd_info['name'] == 'ovt' or 'interactive' in cmd_info['name']:
                continue
                
            with self.subTest(command=cmd_info['name']):
                # Run command with --help
                result = self.run_cli_command(cmd_info['name'].split() + ['--help'])
                
                # Help should work (exit code 0) or show usage (exit code 2)
                self.assertIn(result.returncode, [0, 2],
                            f"Command '{cmd_info['name']}' help failed with code {result.returncode}")
                
                # Should show help text
                output = result.stdout + result.stderr
                self.assertTrue(len(output) > 0, f"No help output for '{cmd_info['name']}'")
                
                # Should contain command name or usage
                self.assertTrue(
                    'Usage:' in output or 'usage:' in output or cmd_info['name'].split()[-1] in output,
                    f"Help output doesn't contain usage info for '{cmd_info['name']}'"
                )
                
                tested += 1
        
        self.assertGreater(tested, 0, "No commands were actually tested")
    
    def test_core_cli_commands_functionality(self):
        """Test core CLI commands with actual functionality"""
        # Test specific commands that should work
        test_cases = [
            {
                'command': ['ovt', 'version'],
                'expected_in_output': 'version',
                'should_succeed': True
            },
            {
                'command': ['ovt', 'tags', '--vault', self.test_vault],
                'expected_in_output': 'Total',
                'should_succeed': True
            },
            {
                'command': ['ovt', 'config', 'show'],
                'expected_in_output': 'Config',
                'should_succeed': True
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(command=' '.join(test_case['command'])):
                result = self.run_cli_command(test_case['command'])
                
                if test_case['should_succeed']:
                    self.assertEqual(result.returncode, 0,
                                   f"Command failed: {result.stderr}")
                    
                    if test_case.get('expected_in_output'):
                        self.assertIn(test_case['expected_in_output'], 
                                    result.stdout + result.stderr,
                                    f"Expected output not found")
    
    def test_mcp_commands_discovery(self):
        """Test that MCP subcommands are discovered"""
        commands = self.discover_all_cli_commands()
        command_names = [cmd['name'] for cmd in commands]
        
        # MCP commands that should be discovered
        expected_mcp_commands = [
            'ovt mcp',
            'ovt mcp list',
            'ovt mcp status',
            'ovt mcp start',
            'ovt mcp stop',
            'ovt mcp audit'
        ]
        
        for expected in expected_mcp_commands:
            self.assertIn(expected, command_names,
                        f"MCP command '{expected}' not discovered")
    
    def test_command_error_handling(self):
        """Test that commands handle errors gracefully"""
        error_cases = [
            {
                'command': ['ovt', 'tags', '--vault', '/nonexistent/path'],
                'description': 'nonexistent vault'
            },
            {
                'command': ['ovt', 'backup', '--vault', '/no/write/permission'],
                'description': 'no write permission'
            },
            {
                'command': ['ovt', 'nonexistent-command'],
                'description': 'nonexistent command'
            }
        ]
        
        for case in error_cases:
            with self.subTest(error=case['description']):
                result = self.run_cli_command(case['command'])
                
                # Should not crash catastrophically
                self.assertIsNotNone(result)
                
                # Should have some error indication
                if result.returncode != 0:
                    # Good - command recognized the error
                    self.assertTrue(True)
                else:
                    # Check if error message is in output
                    output = result.stdout + result.stderr
                    self.assertTrue(
                        any(word in output.lower() 
                            for word in ['error', 'failed', 'not found', 'invalid']),
                        f"No error indication for {case['description']}"
                    )
    
    def test_cli_launches_interactive_mode(self):
        """Test that ovt with no args launches interactive mode"""
        # This is tricky to test as it launches an interactive session
        # We'll just verify it tries to import and run the unified manager
        
        # Run with immediate exit input
        result = self.run_cli_command(['ovt'], input_text='13\n')
        
        # Should either launch interactive or show help
        output = result.stdout + result.stderr
        
        # Check for signs of interactive mode or help
        self.assertTrue(
            any(text in output for text in [
                'Unified Obsidian Vault Manager',
                'Main Menu',
                'Available commands',
                'Usage:'
            ]),
            "ovt without args should launch interactive mode or show help"
        )
    
    def generate_cli_coverage_report(self) -> Dict[str, Any]:
        """Generate a report of all CLI commands discovered"""
        commands = self.discover_all_cli_commands()
        
        report = {
            'total_commands': len(commands),
            'command_list': [cmd['name'] for cmd in commands],
            'commands_by_type': {
                'groups': [cmd['name'] for cmd in commands if cmd.get('is_group')],
                'actions': [cmd['name'] for cmd in commands if not cmd.get('is_group')]
            },
            'help_available': sum(1 for cmd in commands if cmd.get('help') != "No help available")
        }
        
        return report


def run_cli_tests():
    """Run all CLI auto-discovery tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(CLIAutoTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate coverage report
    test_instance = CLIAutoTest()
    test_instance.setUpClass()
    
    try:
        report = test_instance.generate_cli_coverage_report()
        
        print(f"\n{'='*60}")
        print("CLI COVERAGE REPORT")
        print(f"{'='*60}")
        print(f"Total Commands Discovered: {report['total_commands']}")
        print(f"Command Groups: {len(report['commands_by_type']['groups'])}")
        print(f"Action Commands: {len(report['commands_by_type']['actions'])}")
        print(f"Commands with Help: {report['help_available']}")
        print(f"\nAll Commands:")
        for cmd in sorted(report['command_list']):
            print(f"  - {cmd}")
            
    finally:
        test_instance.tearDownClass()
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_cli_tests()
    sys.exit(0 if success else 1)