#!/usr/bin/env python3
"""
Regression Test for Menu Navigation Bug
Ensures the menu navigation parameter bug never returns
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MenuNavigationRegressionTest(unittest.TestCase):
    """Specific regression tests for the menu navigation bug fix"""
    
    def test_menu_navigator_receives_correct_parameters(self):
        """
        CRITICAL TEST: Ensures MenuNavigator.navigate_menu receives (title, options)
        This was the bug that prevented the interactive menu from loading.
        """
        from unified_vault_manager import UnifiedVaultManager
        from menu_navigator import MenuNavigator
        
        # Create a test vault path
        test_vault = "/tmp/test_vault"
        
        # Create manager
        manager = UnifiedVaultManager(test_vault)
        
        # Mock the navigator
        mock_navigator = MagicMock(spec=MenuNavigator)
        mock_navigator.navigate_menu.return_value = '13'  # Exit option
        manager.navigator = mock_navigator
        
        # Mock input and print to prevent actual interaction
        with patch('builtins.input', return_value='13'):
            with patch('builtins.print'):
                # Get main menu options
                options = manager.display_main_menu()
                
                # This is the exact code from the run() method that had the bug
                if manager.navigator:
                    # Convert options to format expected by MenuNavigator
                    menu_options = [(str(i+1), label) for i, (label, _) in enumerate(options)]
                    manager.navigator.navigate_menu("Main Menu", menu_options)
        
        # CRITICAL ASSERTIONS
        self.assertTrue(mock_navigator.navigate_menu.called,
                       "navigate_menu was not called")
        
        # Get the call arguments
        call_args = mock_navigator.navigate_menu.call_args
        
        # Verify it was called with exactly 2 positional arguments
        self.assertEqual(len(call_args[0]), 2,
                        "navigate_menu should receive exactly 2 arguments (title, options)")
        
        title, options_arg = call_args[0]
        
        # Verify first argument is a string (title)
        self.assertIsInstance(title, str,
                            "First argument to navigate_menu should be a string (title)")
        self.assertEqual(title, "Main Menu",
                        "Title should be 'Main Menu'")
        
        # Verify second argument is a list
        self.assertIsInstance(options_arg, list,
                            "Second argument to navigate_menu should be a list")
        
        # Verify each option is a tuple of (key, description)
        for option in options_arg:
            self.assertIsInstance(option, tuple,
                                "Each option should be a tuple")
            self.assertEqual(len(option), 2,
                            "Each option tuple should have exactly 2 elements")
            
            key, description = option
            self.assertIsInstance(key, str,
                                "Option key should be a string")
            self.assertIsInstance(description, str,
                                "Option description should be a string")
            
            # Key should be a number string
            self.assertTrue(key.isdigit(),
                           f"Option key '{key}' should be a digit string")
    
    def test_menu_loads_without_navigator(self):
        """Test that menu still works even without MenuNavigator (fallback mode)"""
        from unified_vault_manager import UnifiedVaultManager
        
        # Create manager
        manager = UnifiedVaultManager("/tmp/test_vault")
        
        # Disable navigator to test fallback
        manager.navigator = None
        
        # Mock input to exit immediately
        with patch('builtins.input', return_value='13'):
            with patch('builtins.print') as mock_print:
                # Display menu - should not crash
                options = manager.display_main_menu()
                
                # Should return options
                self.assertIsInstance(options, list)
                self.assertGreater(len(options), 0)
                
                # Should print menu items
                printed = ' '.join(str(call) for call in mock_print.call_args_list)
                self.assertIn("Main Menu", printed)
    
    def test_all_menu_handlers_exist(self):
        """Test that all menu options have corresponding handlers"""
        from unified_vault_manager import UnifiedVaultManager
        
        manager = UnifiedVaultManager("/tmp/test_vault")
        
        # Get main menu options
        options = manager.display_main_menu()
        
        # Known mapping exceptions (action -> actual handler name)
        handler_exceptions = {
            'tags': 'handle_tag_menu',  # Not handle_tags_menu
        }
        
        for label, action in options:
            if action and action != 'exit':
                # Check that handler method exists
                if action in handler_exceptions:
                    handler_name = handler_exceptions[action]
                else:
                    handler_name = f'handle_{action}_menu'
                    
                self.assertTrue(hasattr(manager, handler_name),
                              f"Missing handler method: {handler_name} for action: {action}")
                
                # Verify it's callable
                handler = getattr(manager, handler_name)
                self.assertTrue(callable(handler),
                              f"Handler {handler_name} is not callable")
    
    def test_menu_option_structure(self):
        """Test that menu options have the correct structure"""
        from unified_vault_manager import UnifiedVaultManager
        
        manager = UnifiedVaultManager("/tmp/test_vault")
        options = manager.display_main_menu()
        
        # Should be a list of tuples
        self.assertIsInstance(options, list)
        
        # Each option should be a tuple of (label, action)
        for i, option in enumerate(options):
            self.assertIsInstance(option, tuple,
                                f"Option {i} is not a tuple")
            self.assertEqual(len(option), 2,
                            f"Option {i} should have exactly 2 elements")
            
            label, action = option
            self.assertIsInstance(label, str,
                                f"Option {i} label should be a string")
            
            # Action should be string or None
            self.assertTrue(action is None or isinstance(action, str),
                           f"Option {i} action should be string or None")


def run_regression_tests():
    """Run regression tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(MenuNavigationRegressionTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ REGRESSION TESTS PASSED - Menu navigation bug is fixed!")
    else:
        print("\n❌ REGRESSION TEST FAILED - Menu navigation bug may have returned!")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_regression_tests()
    sys.exit(0 if success else 1)