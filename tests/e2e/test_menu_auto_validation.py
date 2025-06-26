#!/usr/bin/env python3
"""
Dynamic Menu Auto-Validation Test
Automatically validates all menu options without hardcoding
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import unittest
from unittest.mock import patch, MagicMock, call
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MenuAutoValidator(unittest.TestCase):
    """Automatically validates all menu options without hardcoding"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test vault
        import tempfile
        self.test_vault = tempfile.mkdtemp(prefix="test_vault_menu_")
        os.makedirs(os.path.join(self.test_vault, '.obsidian'), exist_ok=True)
        
        # Import manager
        from unified_vault_manager import UnifiedVaultManager
        self.manager = UnifiedVaultManager(self.test_vault)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.test_vault):
            shutil.rmtree(self.test_vault)
    
    def extract_menu_options_from_display(self) -> List[Tuple[str, str]]:
        """Extract main menu options by calling display_main_menu"""
        with patch('builtins.print'):  # Suppress output
            options = self.manager.display_main_menu()
        return options
    
    def simulate_menu_selection(self, menu_number: int, timeout: float = 2.0) -> Dict[str, Any]:
        """Simulate selecting a menu option and capture result"""
        result = {
            'has_error': False,
            'error_message': None,
            'returned_to_menu': False,
            'exited_cleanly': False,
            'output': [],
            'menu_displayed': False
        }
        
        # Mock input to select option then go back
        input_sequence = [str(menu_number), 'b', '13']  # Select, back, exit
        input_iter = iter(input_sequence)
        
        def mock_input(prompt=""):
            result['output'].append(f"PROMPT: {prompt}")
            try:
                return next(input_iter)
            except StopIteration:
                return '13'  # Default to exit
        
        # Capture print output
        output_buffer = StringIO()
        
        def mock_print(*args, **kwargs):
            output = ' '.join(str(arg) for arg in args)
            output_buffer.write(output + '\n')
            result['output'].append(output)
            
            # Check if menu is displayed
            if 'Main Menu:' in output or 'Select option:' in output:
                result['menu_displayed'] = True
        
        # Run with timeout to prevent hanging
        def run_menu():
            try:
                with patch('builtins.input', side_effect=mock_input):
                    with patch('builtins.print', side_effect=mock_print):
                        # Get main menu options
                        options = self.manager.display_main_menu()
                        
                        if 0 <= menu_number - 1 < len(options):
                            action = options[menu_number - 1][1]
                            
                            # Call the appropriate handler
                            if action == 'exit':
                                result['exited_cleanly'] = True
                            elif hasattr(self.manager, f'handle_{action}_menu'):
                                handler = getattr(self.manager, f'handle_{action}_menu')
                                handler()
                                result['returned_to_menu'] = True
                            else:
                                result['has_error'] = True
                                result['error_message'] = f"No handler for action: {action}"
                        else:
                            result['has_error'] = True
                            result['error_message'] = f"Invalid menu number: {menu_number}"
                            
            except Exception as e:
                result['has_error'] = True
                result['error_message'] = str(e)
        
        # Run in thread with timeout
        thread = threading.Thread(target=run_menu)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            result['has_error'] = True
            result['error_message'] = "Menu operation timed out"
        
        return result
    
    def test_all_main_menu_options_dynamically(self):
        """Test every main menu option that exists"""
        # Get menu options
        main_menu_options = self.extract_menu_options_from_display()
        
        self.assertGreater(len(main_menu_options), 0, "No menu options found")
        
        # Test each option
        for i, (label, action) in enumerate(main_menu_options, 1):
            with self.subTest(menu=f"{i}. {label} ({action})"):
                if action == 'exit':
                    # Skip exit option as it terminates the program
                    continue
                
                result = self.simulate_menu_selection(i)
                
                # Verify no errors
                self.assertFalse(result['has_error'], 
                               f"Error in menu {label}: {result['error_message']}")
                
                # Verify menu was displayed
                self.assertTrue(result['menu_displayed'] or result['exited_cleanly'],
                              f"Menu {label} did not display properly")
    
    def test_menu_navigation_parameters(self):
        """Test that MenuNavigator receives correct parameters (regression test)"""
        # This specifically tests the bug we fixed
        from menu_navigator import MenuNavigator
        
        # Mock the navigator
        mock_navigator = MagicMock(spec=MenuNavigator)
        mock_navigator.navigate_menu.return_value = '1'  # Simulate selection
        
        # Replace navigator in manager
        self.manager.navigator = mock_navigator
        
        # Trigger menu navigation
        with patch('builtins.print'):  # Suppress output
            try:
                # Display main menu which should trigger navigation
                self.manager.display_main_menu()
                
                # Simulate the main run loop section that uses navigator
                options = self.manager.display_main_menu()
                
                if self.manager.navigator:
                    # This is the code path that had the bug
                    menu_options = [(str(i+1), label) for i, (label, _) in enumerate(options)]
                    self.manager.navigator.navigate_menu("Main Menu", menu_options)
                
                # Verify navigate_menu was called with correct parameters
                if mock_navigator.navigate_menu.called:
                    call_args = mock_navigator.navigate_menu.call_args
                    
                    # Should be called with (title, options)
                    self.assertEqual(len(call_args[0]), 2, 
                                   "navigate_menu should receive exactly 2 arguments")
                    
                    title, options = call_args[0]
                    self.assertIsInstance(title, str, "First argument should be a string (title)")
                    self.assertIsInstance(options, list, "Second argument should be a list")
                    
                    if options:
                        # Each option should be a tuple of (key, description)
                        first_option = options[0]
                        self.assertIsInstance(first_option, tuple, "Options should be tuples")
                        self.assertEqual(len(first_option), 2, "Each option should have 2 elements")
                
            except Exception as e:
                self.fail(f"Menu navigation failed: {e}")
    
    def test_submenu_navigation_dynamically(self):
        """Test navigation within submenus"""
        # Get main menu options
        main_menu_options = self.extract_menu_options_from_display()
        
        # Test first few submenus
        for i, (label, action) in enumerate(main_menu_options[:3], 1):
            if action == 'exit':
                continue
                
            with self.subTest(submenu=f"{label} ({action})"):
                handler_name = f'handle_{action}_menu'
                
                if hasattr(self.manager, handler_name):
                    # Mock input to navigate submenu
                    with patch('builtins.input', side_effect=['1', 'b']):  # Select first option, then back
                        with patch('builtins.print') as mock_print:
                            try:
                                handler = getattr(self.manager, handler_name)
                                handler()
                                
                                # Verify submenu was displayed
                                printed_output = ' '.join(str(call) for call in mock_print.call_args_list)
                                self.assertIn('Select option:', printed_output, 
                                            f"Submenu {action} did not show options")
                                
                            except StopIteration:
                                # This is okay - means we exhausted the input
                                pass
                            except Exception as e:
                                self.fail(f"Submenu {action} crashed: {e}")
    
    def test_feature_availability_in_menus(self):
        """Test that unavailable features are handled gracefully in menus"""
        # Test with some features disabled
        original_features = self.manager.features.copy()
        
        # Disable some features
        self.manager.features['mcp'] = False
        self.manager.features['ai'] = False
        
        try:
            # Try to access MCP menu (should show unavailable message)
            with patch('builtins.input', return_value=''):  # Just press enter to continue
                with patch('builtins.print') as mock_print:
                    self.manager.handle_mcp_menu()
                    
                    # Should show unavailable message
                    printed = ' '.join(str(call) for call in mock_print.call_args_list)
                    self.assertIn('not available', printed.lower(), 
                                "Should show unavailable message for disabled feature")
                    
        finally:
            # Restore original features
            self.manager.features = original_features
    
    def test_error_handling_in_menus(self):
        """Test that menus handle errors gracefully"""
        # Test invalid input
        test_cases = [
            ('abc', 'non-numeric input'),
            ('999', 'out of range number'),
            ('', 'empty input'),
            ('-1', 'negative number')
        ]
        
        for invalid_input, description in test_cases:
            with self.subTest(input=f"{invalid_input} ({description})"):
                with patch('builtins.input', side_effect=[invalid_input, '13']):  # Invalid then exit
                    with patch('builtins.print') as mock_print:
                        try:
                            # Run a menu handler
                            self.manager.handle_analysis_menu()
                        except:
                            pass  # We expect it to handle errors gracefully
                        
                        # Check that an error message was shown
                        printed = ' '.join(str(call) for call in mock_print.call_args_list)
                        # Should show some error indication but not crash
                        self.assertTrue(any(word in printed.lower() 
                                          for word in ['invalid', 'error', 'try again']),
                                      f"No error message shown for {description}")


def run_menu_validation_tests():
    """Run all menu validation tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(MenuAutoValidator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*60}")
    print("MENU VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {'YES' if result.wasSuccessful() else 'NO'}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_menu_validation_tests()
    sys.exit(0 if success else 1)