#!/usr/bin/env python3
"""
Arrow Key Menu Navigator
Provides arrow key navigation for the Enhanced Vault Manager
"""

import sys
import termios
import tty
from typing import List, Tuple, Optional, Callable
from vault_manager import Colors

class MenuNavigator:
    """
    Arrow key navigation system for menus
    """
    
    def __init__(self, audio_callback: Optional[Callable] = None):
        self.audio_callback = audio_callback
        self.selected_index = 0
        
    def get_key(self) -> str:
        """Get a single keypress from the user"""
        if sys.platform == 'win32':
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':  # Arrow key prefix on Windows
                key = msvcrt.getch()
                if key == b'H':
                    return 'UP'
                elif key == b'P':
                    return 'DOWN'
                elif key == b'K':
                    return 'LEFT'
                elif key == b'M':
                    return 'RIGHT'
            elif key == b'\r':
                return 'ENTER'
            elif key == b'\x1b':
                return 'ESC'
            else:
                return key.decode('utf-8', errors='ignore')
        else:
            # Unix/Linux/macOS
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                
                if key == '\x1b':  # ESC sequence
                    key += sys.stdin.read(2)
                    if key == '\x1b[A':
                        return 'UP'
                    elif key == '\x1b[B':
                        return 'DOWN'
                    elif key == '\x1b[C':
                        return 'RIGHT'
                    elif key == '\x1b[D':
                        return 'LEFT'
                    else:
                        return 'ESC'
                elif key == '\r' or key == '\n':
                    return 'ENTER'
                elif key == '\x03':  # Ctrl+C
                    return 'CTRL_C'
                elif key == 'q' or key == 'Q':
                    return 'QUIT'
                elif key == '0':
                    return 'ZERO'
                else:
                    return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def display_menu(self, title: str, options: List[Tuple[str, str]], selected_index: int = 0):
        """Display menu with highlighted selection"""
        # Clear screen and move cursor to top
        print('\033[2J\033[H', end='')
        
        # Display title
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{title.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")
        
        # Display options
        for i, (key, description) in enumerate(options):
            if i == selected_index:
                # Highlighted option
                print(f"{Colors.CYAN}  ► {Colors.YELLOW}{key}. {description}{Colors.ENDC}")
            else:
                # Normal option
                print(f"    {Colors.BLUE}{key}.{Colors.ENDC} {description}")
        
        # Instructions
        print(f"\n{Colors.GREEN}Navigation:{Colors.ENDC}")
        print(f"  {Colors.CYAN}↑/↓{Colors.ENDC} Move selection")
        print(f"  {Colors.CYAN}1-9{Colors.ENDC} Quick select (no Enter needed)")
        print(f"  {Colors.CYAN}Enter{Colors.ENDC} Select highlighted option")
        print(f"  {Colors.CYAN}0 or ESC{Colors.ENDC} Back/Exit")
        print(f"  {Colors.CYAN}Q{Colors.ENDC} Quit")
    
    def navigate_menu(self, title: str, options: List[Tuple[str, str]]) -> Optional[str]:
        """
        Navigate menu with arrow keys
        
        Args:
            title: Menu title
            options: List of (key, description) tuples
            
        Returns:
            Selected option key or None if cancelled
        """
        if not options:
            return None
        
        selected_index = 0
        
        while True:
            # Display menu
            self.display_menu(title, options, selected_index)
            
            # Get user input
            key = self.get_key()
            
            if key == 'UP':
                # Move selection up
                selected_index = (selected_index - 1) % len(options)
                if self.audio_callback:
                    self.audio_callback('menu_blip')
                    
            elif key == 'DOWN':
                # Move selection down
                selected_index = (selected_index + 1) % len(options)
                if self.audio_callback:
                    self.audio_callback('menu_blip')
                    
            elif key == 'ENTER':
                # Select current option
                if self.audio_callback:
                    self.audio_callback('menu_select')
                return options[selected_index][0]
                
            elif key == 'ESC' or key == 'ZERO':
                # Back/Cancel
                if self.audio_callback:
                    self.audio_callback('menu_back')
                return '0'
                
            elif key == 'QUIT' or key == 'CTRL_C':
                # Quit application
                if self.audio_callback:
                    self.audio_callback('menu_back')
                return 'quit'
                
            elif key.isdigit() and key != '0':
                # Direct number selection
                option_num = int(key) - 1
                if 0 <= option_num < len(options):
                    if self.audio_callback:
                        self.audio_callback('menu_select')
                    return options[option_num][0]
                else:
                    # Invalid number
                    if self.audio_callback:
                        self.audio_callback('error_chord')
    
    def show_message(self, title: str, message: str, auto_clear: bool = False):
        """Show a message to the user"""
        print('\033[2J\033[H', end='')  # Clear screen
        
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{title.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")
        
        print(message)
        
        if not auto_clear:
            print(f"\n{Colors.YELLOW}Press any key to continue...{Colors.ENDC}")
            self.get_key()
    
    def get_text_input(self, prompt: str) -> str:
        """Get text input from user with proper terminal reset"""
        # Restore normal terminal mode for text input
        if sys.platform != 'win32':
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return input(f"{Colors.YELLOW}{prompt}{Colors.ENDC}")
            finally:
                pass  # Keep normal mode for subsequent input
        else:
            return input(f"{Colors.YELLOW}{prompt}{Colors.ENDC}")

# Test the menu navigator
if __name__ == "__main__":
    def test_audio(sound_name):
        print(f"🔊 Playing: {sound_name}")
    
    navigator = MenuNavigator(audio_callback=test_audio)
    
    test_options = [
        ('1', '🎨 Convert image to ASCII art'),
        ('2', '📷 Convert screenshot to ASCII'),
        ('3', '🖼️  View ASCII art gallery'),
        ('4', '📊 Generate flowchart from document'),
        ('5', '📚 Add art to vault notes'),
        ('6', '⚙️  ASCII art settings'),
    ]
    
    print("Testing Arrow Key Navigation...")
    result = navigator.navigate_menu("TEST MENU", test_options)
    print(f"\nSelected: {result}")