#!/usr/bin/env python3
"""
Enhanced Obsidian Vault Manager with ASCII Art Integration
Includes ASCII art generation capabilities throughout the interface
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import time
from typing import Optional, Dict, Any
import random

# Try to import required libraries
try:
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import the better ASCII converter
try:
    from better_ascii_converter import BetterASCIIConverter
    BETTER_ASCII_AVAILABLE = True
except ImportError:
    BETTER_ASCII_AVAILABLE = False

# Import the original vault manager components
from vault_manager import Colors, VaultManager

class ASCIIArtManager:
    """Manages ASCII art generation and display"""
    
    def __init__(self):
        self.pil_available = PIL_AVAILABLE
        self.better_ascii_available = BETTER_ASCII_AVAILABLE
        self.art_cache = {}
        self.default_style = 'standard'
        self.default_width = 80
        
        # Pre-defined ASCII art pieces
        self.ascii_collection = {
            'wizard': """
                    .-.
                   (o o)
                   | O \\
                  |    |
        |\\_/|    |     |
        |a a|".-.;     |
        | | | '-.__.-'(
       =\\t /=    (_(__))
        ) (     /     )
       /   \\   /      \\
       |   |  ((______))
       |   |   )      (
       |___|  /        \\
        |||  (_________)
       /_|_\\   Digital Librarian""",
            
            'books': """
       ___________________________
      /                           /|
     /                           / |
    /___________________________/  |
    |  ______________________  |  /
    | |                      | | /
    | |    OBSIDIAN VAULT    | |/
    | |   ================   | |
    | |                      | |
    | |   📚 Knowledge Base  | |
    | |______________________| |
    |___________________________|""",
            
            'success': """
         ✨ ✨ ✨
       ╔═══════╗
       ║   ✓   ║
       ║SUCCESS║
       ╚═══════╝
         ✨ ✨ ✨""",
            
            'thinking': """
       🤔
      .oOo.
     /     \\
    | () () |
     \\  ~  /
      |>o<|
     /|---|\\
    / |   | \\
   ---|   |---""",
            
            'warning': """
      ⚠️  ⚠️  ⚠️
     ╔═════════╗
     ║ WARNING ║
     ╚═════════╝
      ⚠️  ⚠️  ⚠️""",
            
            'portal': """
       ╔══════════════╗
      ╱                ╲
     ╱   ░▒▓█▓▒░       ╲
    │    ▒▓███▓▒        │
    │    ▓█████▓        │
    │    ▒▓███▓▒        │
     ╲   ░▒▓█▓▒░       ╱
      ╲                ╱
       ╚══════════════╝
         Magic Portal"""
        }
    
    def generate_from_image(self, image_path: str, width: int = 80, style: str = None) -> Optional[str]:
        """Generate ASCII art from an image file using the better converter"""
        if not self.better_ascii_available or not self.pil_available:
            return None
            
        try:
            # Check cache first
            style = style or self.default_style
            cache_key = f"{image_path}_{width}_{style}"
            if cache_key in self.art_cache:
                return self.art_cache[cache_key]
            
            # Create converter
            converter = BetterASCIIConverter(char_set=style, width=width)
            
            # Convert to ASCII
            ascii_str = converter.convert(image_path)
            
            # Cache the result
            self.art_cache[cache_key] = ascii_str
            
            return ascii_str
            
        except Exception as e:
            print(f"{Colors.RED}Error generating ASCII art: {e}{Colors.ENDC}")
            return None
    
    def get_random_art(self) -> str:
        """Get a random ASCII art from the collection"""
        key = random.choice(list(self.ascii_collection.keys()))
        return self.ascii_collection[key]
    
    def get_art(self, name: str) -> str:
        """Get specific ASCII art by name"""
        return self.ascii_collection.get(name, self.ascii_collection['wizard'])

class EnhancedVaultManager(VaultManager):
    """Enhanced Vault Manager with ASCII Art Integration"""
    
    def __init__(self):
        super().__init__()
        self.ascii_manager = ASCIIArtManager()
        self.ascii_enabled = True
        
    def show_welcome_art(self):
        """Display enhanced welcome ASCII art"""
        # Show the original logo
        super().show_welcome_art()
        
        # Add a random decorative element
        if self.ascii_enabled and random.random() > 0.5:
            print(f"\n{Colors.BLUE}{self.ascii_manager.get_random_art()}{Colors.ENDC}")
    
    def show_ascii_menu(self):
        """ASCII Art tools submenu"""
        while True:
            options = [
                ('1', '🎨 Convert image to ASCII art'),
                ('2', '📷 Convert screenshot to ASCII'),
                ('3', '🖼️  View ASCII art gallery'),
                ('4', '📊 Generate flowchart from document'),
                ('5', '📚 Add art to vault notes'),
                ('6', '⚙️  ASCII art settings'),
                ('0', '← Back to main menu')
            ]
            
            self.show_menu('ASCII ART TOOLS', options)
            choice = input().strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.convert_image_to_ascii()
            elif choice == '2':
                self.convert_screenshot_to_ascii()
            elif choice == '3':
                self.view_ascii_gallery()
            elif choice == '4':
                self.generate_flowchart_from_document()
            elif choice == '5':
                self.add_ascii_to_notes()
            elif choice == '6':
                self.ascii_art_settings()
    
    def convert_image_to_ascii(self):
        """Convert user-selected image to ASCII art"""
        print(f"\n{Colors.CYAN}Convert Image to ASCII Art{Colors.ENDC}")
        
        if not self.ascii_manager.pil_available:
            print(f"{Colors.YELLOW}⚠️  PIL/Pillow not installed{Colors.ENDC}")
            print(f"{Colors.BLUE}Install with: pip install pillow numpy{Colors.ENDC}")
            return
        
        image_path = input(f"\n{Colors.YELLOW}Enter image path (or drag & drop): {Colors.ENDC}").strip()
        image_path = image_path.strip('"\'')  # Remove quotes if dragged
        
        if not os.path.exists(image_path):
            print(f"{Colors.RED}❌ File not found: {image_path}{Colors.ENDC}")
            return
        
        # Show available styles
        print(f"\n{Colors.CYAN}Available styles:{Colors.ENDC}")
        styles = ['simple', 'standard', 'blocks', 'detailed', 'classic', 'matrix']
        for i, style in enumerate(styles):
            print(f"  {i+1}. {style}")
        
        style_choice = input(f"\n{Colors.YELLOW}Choose style (1-{len(styles)}) or press Enter for standard: {Colors.ENDC}").strip()
        style = styles[int(style_choice)-1] if style_choice.isdigit() and 1 <= int(style_choice) <= len(styles) else 'standard'
        
        width = input(f"{Colors.YELLOW}Width in characters (default 80): {Colors.ENDC}").strip()
        width = int(width) if width else 80
        
        print(f"\n{Colors.CYAN}Converting with {style} style...{Colors.ENDC}")
        self.show_loading("Generating ASCII art", 2)
        
        ascii_art = self.ascii_manager.generate_from_image(image_path, width, style)
        
        if ascii_art:
            print(f"\n{Colors.GREEN}✓ ASCII Art Generated:{Colors.ENDC}\n")
            # Show preview (first 30 lines)
            lines = ascii_art.split('\n')
            for line in lines[:30]:
                print(line)
            if len(lines) > 30:
                print(f"\n{Colors.YELLOW}... ({len(lines)-30} more lines){Colors.ENDC}")
            
            # Offer to save
            save = input(f"\n{Colors.YELLOW}Save to file? (y/n): {Colors.ENDC}").lower()
            if save == 'y':
                output_path = Path(image_path).stem + f"_ascii_{style}.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(ascii_art)
                print(f"{Colors.GREEN}✓ Saved to: {output_path}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Failed to generate ASCII art{Colors.ENDC}")
    
    def convert_screenshot_to_ascii(self):
        """Take a screenshot and convert it to ASCII art"""
        print(f"\n{Colors.CYAN}Screenshot to ASCII Art{Colors.ENDC}")
        
        if not self.ascii_manager.pil_available:
            print(f"{Colors.YELLOW}⚠️  PIL/Pillow not installed{Colors.ENDC}")
            return
        
        print(f"{Colors.BLUE}Taking screenshot in 3 seconds...{Colors.ENDC}")
        print(f"{Colors.BLUE}Position your window now!{Colors.ENDC}")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # Take screenshot based on OS
        screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        if sys.platform == 'darwin':  # macOS
            cmd = f'screencapture -i "{screenshot_path}"'
        elif sys.platform == 'win32':  # Windows
            cmd = f'snippingtool /clip'  # Will need manual save
            print(f"{Colors.YELLOW}Please save the screenshot as: {screenshot_path}{Colors.ENDC}")
        else:  # Linux
            cmd = f'gnome-screenshot -a -f "{screenshot_path}"'
        
        subprocess.run(cmd, shell=True)
        
        if os.path.exists(screenshot_path):
            width = 100
            style = 'standard'
            ascii_art = self.ascii_manager.generate_from_image(screenshot_path, width, style)
            
            if ascii_art:
                print(f"\n{Colors.GREEN}✓ Screenshot converted:{Colors.ENDC}\n")
                lines = ascii_art.split('\n')[:20]
                for line in lines:
                    print(line)
                print(f"\n{Colors.YELLOW}... (preview of first 20 lines){Colors.ENDC}")
                
                # Offer to save
                save = input(f"\n{Colors.YELLOW}Save full ASCII art? (y/n): {Colors.ENDC}").lower()
                if save == 'y':
                    output_path = f"screenshot_ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(ascii_art)
                    print(f"{Colors.GREEN}✓ Saved to: {output_path}{Colors.ENDC}")
                
                # Clean up
                os.remove(screenshot_path)
    
    def generate_flowchart_from_document(self):
        """Generate ASCII flowchart from a document"""
        print(f"\n{Colors.CYAN}Generate ASCII Flowchart from Document{Colors.ENDC}")
        
        if not self.current_vault:
            print(f"{Colors.RED}❌ No vault selected{Colors.ENDC}")
            return
        
        # List recent markdown files
        print(f"\n{Colors.BLUE}Recent markdown files in your vault:{Colors.ENDC}")
        
        recent_files = []
        for root, dirs, files in os.walk(self.current_vault):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    recent_files.append((os.path.getmtime(file_path), file_path))
        
        recent_files.sort(reverse=True)
        
        # Show first 15 files
        for i, (_, file_path) in enumerate(recent_files[:15]):
            rel_path = os.path.relpath(file_path, self.current_vault)
            print(f"  {i+1:2d}. {rel_path}")
        
        # Get user selection
        print(f"\n{Colors.YELLOW}Options:{Colors.ENDC}")
        print(f"  • Enter number (1-{min(15, len(recent_files))}) to select from list")
        print(f"  • Enter relative path to specific file")
        print(f"  • Press Enter to cancel")
        
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.ENDC}").strip()
        
        if not choice:
            return
        
        # Determine input file
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= min(15, len(recent_files)):
                input_file = recent_files[choice_num - 1][1]
            else:
                print(f"{Colors.RED}❌ Invalid selection{Colors.ENDC}")
                return
        else:
            input_file = os.path.join(self.current_vault, choice)
            if not os.path.exists(input_file):
                print(f"{Colors.RED}❌ File not found: {choice}{Colors.ENDC}")
                return
        
        # Get output filename
        input_name = Path(input_file).stem
        output_file = os.path.join(self.current_vault, f"{input_name}_flowchart.md")
        
        custom_output = input(f"\n{Colors.YELLOW}Output filename (default: {Path(output_file).name}): {Colors.ENDC}").strip()
        if custom_output:
            if not custom_output.endswith('.md'):
                custom_output += '.md'
            output_file = os.path.join(self.current_vault, custom_output)
        
        # Generate flowchart
        print(f"\n{Colors.CYAN}Analyzing document and generating flowchart...{Colors.ENDC}")
        self.show_loading("Processing document", 3)
        
        try:
            # Import the flowchart generator
            from ascii_flowchart_generator import generate_flowchart_from_file
            
            # Generate the flowchart
            result = generate_flowchart_from_file(input_file, output_file)
            
            print(f"\n{Colors.GREEN}✓ Flowchart generated successfully!{Colors.ENDC}")
            print(f"{Colors.BLUE}Saved to: {os.path.relpath(output_file, self.current_vault)}{Colors.ENDC}")
            
            # Show preview
            preview = input(f"\n{Colors.YELLOW}Show preview? (y/n): {Colors.ENDC}").lower()
            if preview == 'y':
                print(f"\n{Colors.CYAN}Preview:{Colors.ENDC}")
                lines = result.split('\n')
                for line in lines[:30]:  # Show first 30 lines
                    print(line)
                if len(lines) > 30:
                    print(f"\n{Colors.YELLOW}... ({len(lines)-30} more lines in full file){Colors.ENDC}")
            
        except ImportError:
            print(f"{Colors.RED}❌ Flowchart generator not available{Colors.ENDC}")
            print(f"{Colors.BLUE}Please ensure ascii_flowchart_generator.py is in the same directory{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error generating flowchart: {str(e)}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def view_ascii_gallery(self):
        """View collection of ASCII art"""
        arts = list(self.ascii_manager.ascii_collection.keys())
        current = 0
        
        while True:
            self.clear_screen()
            print(f"{Colors.HEADER}{Colors.BOLD}{'ASCII ART GALLERY'.center(60)}{Colors.ENDC}")
            print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")
            
            art_name = arts[current]
            art = self.ascii_manager.get_art(art_name)
            
            print(f"{Colors.CYAN}[{current + 1}/{len(arts)}] {art_name.upper()}{Colors.ENDC}\n")
            print(f"{Colors.GREEN}{art}{Colors.ENDC}")
            
            print(f"\n{Colors.YELLOW}[N]ext  [P]revious  [Q]uit{Colors.ENDC}")
            
            choice = input(f"{Colors.YELLOW}Choice: {Colors.ENDC}").lower()
            
            if choice == 'n':
                current = (current + 1) % len(arts)
            elif choice == 'p':
                current = (current - 1) % len(arts)
            elif choice == 'q':
                break
    
    def ascii_art_settings(self):
        """Configure ASCII art settings"""
        print(f"\n{Colors.CYAN}ASCII Art Settings{Colors.ENDC}")
        
        print(f"\n1. PIL/Pillow Available: {Colors.GREEN if self.ascii_manager.pil_available else Colors.RED}"
              f"{'Yes' if self.ascii_manager.pil_available else 'No'}{Colors.ENDC}")
        
        print(f"2. Better ASCII Converter: {Colors.GREEN if self.ascii_manager.better_ascii_available else Colors.RED}"
              f"{'Available' if self.ascii_manager.better_ascii_available else 'Not Found'}{Colors.ENDC}")
        
        print(f"3. ASCII Art in Menus: {Colors.GREEN if self.ascii_enabled else Colors.RED}"
              f"{'Enabled' if self.ascii_enabled else 'Disabled'}{Colors.ENDC}")
        
        print(f"4. Default Style: {Colors.YELLOW}{self.ascii_manager.default_style}{Colors.ENDC}")
        print(f"5. Default Width: {Colors.YELLOW}{self.ascii_manager.default_width}{Colors.ENDC}")
        
        toggle = input(f"\n{Colors.YELLOW}Toggle ASCII art in menus? (y/n): {Colors.ENDC}").lower()
        if toggle == 'y':
            self.ascii_enabled = not self.ascii_enabled
            self.config['ascii_enabled'] = self.ascii_enabled
            self.save_config()
            print(f"{Colors.GREEN}✓ ASCII art {'enabled' if self.ascii_enabled else 'disabled'}{Colors.ENDC}")
    
    def add_ascii_to_notes(self):
        """Add ASCII art to vault notes"""
        print(f"\n{Colors.CYAN}Add ASCII Art to Notes{Colors.ENDC}")
        
        if not self.current_vault:
            print(f"{Colors.RED}❌ No vault selected{Colors.ENDC}")
            return
        
        # List recent notes
        print(f"\n{Colors.BLUE}Recent notes in your vault:{Colors.ENDC}")
        
        recent_files = []
        for root, dirs, files in os.walk(self.current_vault):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    recent_files.append((os.path.getmtime(file_path), file_path))
        
        recent_files.sort(reverse=True)
        
        for i, (_, file_path) in enumerate(recent_files[:10]):
            rel_path = os.path.relpath(file_path, self.current_vault)
            print(f"  {i+1}. {rel_path}")
        
        choice = input(f"\n{Colors.YELLOW}Select note (1-10) or enter path: {Colors.ENDC}").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= 10:
            note_path = recent_files[int(choice)-1][1]
        else:
            note_path = os.path.join(self.current_vault, choice)
        
        if not os.path.exists(note_path):
            print(f"{Colors.RED}❌ Note not found{Colors.ENDC}")
            return
        
        # Select ASCII art
        print(f"\n{Colors.BLUE}Select ASCII art to add:{Colors.ENDC}")
        arts = list(self.ascii_manager.ascii_collection.keys())
        for i, art in enumerate(arts):
            print(f"  {i+1}. {art}")
        
        art_choice = input(f"\n{Colors.YELLOW}Choice: {Colors.ENDC}").strip()
        
        if art_choice.isdigit() and 1 <= int(art_choice) <= len(arts):
            art_name = arts[int(art_choice)-1]
            ascii_art = self.ascii_manager.get_art(art_name)
            
            # Add to note
            with open(note_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n```\n{ascii_art}\n```\n")
            
            print(f"{Colors.GREEN}✓ ASCII art added to note!{Colors.ENDC}")
    
    def show_menu_with_art(self, title, options, footer=None):
        """Show menu with optional ASCII art decoration"""
        if self.ascii_enabled and random.random() > 0.7:
            # Occasionally show ASCII art in menus
            art = self.ascii_manager.get_art('portal' if 'V2' in title else 'books')
            print(f"{Colors.BLUE}{art}{Colors.ENDC}\n")
        
        # Call parent method
        super().show_menu(title, options, footer)
    
    def run(self):
        """Enhanced main application loop"""
        # Show enhanced welcome
        if not self.config.get('welcomed', False):
            self.clear_screen()
            self.show_welcome_art()
            print(f"\n{Colors.CYAN}Welcome to the Enhanced Obsidian Vault Manager!{Colors.ENDC}")
            print(f"{Colors.BLUE}Now with ASCII art integration! 🎨{Colors.ENDC}")
            self.config['welcomed'] = True
            self.save_config()
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
        
        # Get vault path if not set
        if not self.current_vault or not os.path.exists(self.current_vault):
            self.clear_screen()
            print(f"\n{Colors.YELLOW}No vault configured.{Colors.ENDC}")
            if not self.get_vault_path():
                print(f"\n{Colors.RED}Cannot proceed without a vault path.{Colors.ENDC}")
                return
        
        # Enhanced main menu loop
        while True:
            options = [
                ('1', '📊 Analyze Vault'),
                ('2', '🏷️  Manage Tags'),
                ('3', '💾 Backup Vault'),
                ('4', '🚀 V2 Features' if self.v2_available else '🚀 V2 Features (Not Installed)'),
                ('5', '🎨 ASCII Art Tools'),  # New menu option
                ('6', '🔧 Advanced Tools'),
                ('7', '📚 Help & Documentation'),
                ('8', '⚙️  Settings'),
                ('0', '👋 Exit')
            ]
            
            # Use enhanced menu display
            self.show_menu_with_art('MAIN MENU', options, 'Choose an option to get started!')
            choice = input().strip()
            
            if choice == '0':
                # Show farewell art
                if self.ascii_enabled:
                    print(f"\n{Colors.GREEN}{self.ascii_manager.get_art('success')}{Colors.ENDC}")
                print(f"\n{Colors.GREEN}Thanks for using Enhanced Obsidian Vault Manager!{Colors.ENDC}")
                print(f"{Colors.BLUE}Your vault is in good hands. 📚🎨{Colors.ENDC}\n")
                break
            elif choice == '1':
                self.analyze_vault_menu()
            elif choice == '2':
                self.manage_tags_menu()
            elif choice == '3':
                self.backup_vault_menu()
            elif choice == '4':
                self.v2_features_menu()
            elif choice == '5':
                self.show_ascii_menu()  # New ASCII art menu
            elif choice == '6':
                self.advanced_tools_menu()
            elif choice == '7':
                self.show_help()
            elif choice == '8':
                self.settings_menu()
            else:
                print(f"\n{Colors.RED}Invalid choice. Please try again.{Colors.ENDC}")
                time.sleep(1)

def main():
    """Entry point for enhanced vault manager"""
    try:
        manager = EnhancedVaultManager()
        manager.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user.{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.ENDC}")
        sys.exit(1)

if __name__ == '__main__':
    main()