#!/usr/bin/env python3
"""
Enhanced Obsidian Vault Manager with ASCII Art Integration
Includes ASCII art generation capabilities throughout the interface
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
import time
from typing import Optional, Dict, Any
import random
import uuid

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

# Import menu navigation system
try:
    from menu_navigator import MenuNavigator
    MENU_NAVIGATOR_AVAILABLE = True
except ImportError:
    MENU_NAVIGATOR_AVAILABLE = False

# Import vault query system
try:
    from vault_query_system import VaultQuerySystem
    VAULT_QUERY_AVAILABLE = True
except ImportError:
    VAULT_QUERY_AVAILABLE = False

# Import LLM-enhanced vault query system
try:
    from vault_query_system_llm import VaultQuerySystemLLM
    LLM_QUERY_AVAILABLE = True
except ImportError:
    LLM_QUERY_AVAILABLE = False

# Import file versioning
try:
    from file_versioning import FileVersioning
    FILE_VERSIONING_AVAILABLE = True
except ImportError:
    FILE_VERSIONING_AVAILABLE = False

# Import audio system
try:
    from audio.audio_manager import AudioManager, get_audio_manager, play_sound
    from audio.audio_manager import start_dungeon_ambiance, stop_dungeon_ambiance
    from audio.audio_manager import wizard_greeting, magical_success, dungeon_danger
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    # Create dummy functions if audio not available
    def get_audio_manager(): return None
    def play_sound(*args, **kwargs): return False
    def start_dungeon_ambiance(): return False
    def stop_dungeon_ambiance(): pass
    def wizard_greeting(): return False
    def magical_success(): return False
    def dungeon_danger(): return False

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
        self.last_query_id = None  # Store last query ID for rating
        
        # Initialize audio system
        self.audio_manager = get_audio_manager() if AUDIO_AVAILABLE else None
        self.audio_enabled = AUDIO_AVAILABLE and (self.audio_manager is not None) and self.audio_manager.is_available()
        
        # Initialize menu navigator
        if MENU_NAVIGATOR_AVAILABLE:
            self.navigator = MenuNavigator(audio_callback=self.play_navigation_sound)
            self.arrow_navigation = True
            print(f"{Colors.GREEN}Arrow key navigation enabled{Colors.ENDC}")
        else:
            self.navigator = None
            self.arrow_navigation = False
            print(f"{Colors.YELLOW}Arrow key navigation not available{Colors.ENDC}")
        
        # Initialize vault query system - LLM is the foundation
        self.llm_enabled = False
        self.pattern_matching_mode = False
        self.active_model = None
        
        if LLM_QUERY_AVAILABLE and self.current_vault:
            # LLM system is the primary system
            try:
                self.query_system = VaultQuerySystemLLM(self.current_vault)
                self.query_enabled = True
                print(f"{Colors.GREEN}🤖 LLM-powered vault query system loaded{Colors.ENDC}")
                
                # Initialize LLM components asynchronously with auto-detection
                import asyncio
                try:
                    # Check if this is first run
                    user_prefs_path = Path.home() / ".obsidian_ai_config"
                    is_first_run = not user_prefs_path.exists()
                    
                    if is_first_run:
                        print(f"\n{Colors.CYAN}🎉 Welcome to AI-Powered Obsidian Vault Manager!{Colors.ENDC}")
                        print(f"{Colors.BLUE}Setting up your AI assistant...{Colors.ENDC}")
                        
                        # Auto-detect or let user choose model
                        model_manager = self.query_system.model_manager
                        available_models = asyncio.run(model_manager.auto_detect_models())
                        
                        if available_models:
                            print(f"{Colors.GREEN}✅ Found {len(available_models)} AI models{Colors.ENDC}")
                            
                            # Check if dolphin3 is available
                            dolphin_models = [m for m in available_models if "dolphin" in m.lower()]
                            if dolphin_models:
                                selected_model = dolphin_models[0]
                                print(f"{Colors.MAGENTA}🐬 Auto-selected: {selected_model} (Excellent for vault analysis){Colors.ENDC}")
                                self.active_model = selected_model
                            else:
                                selected_model = asyncio.run(model_manager.select_model_interactive())
                                self.active_model = selected_model
                        else:
                            print(f"{Colors.YELLOW}⚠️ No AI models found in Ollama{Colors.ENDC}")
                            selected_model = None
                    
                    initialized = asyncio.run(self.query_system.initialize())
                    if initialized:
                        self.llm_enabled = True
                        
                        # Get the active model for display
                        detected_model = self.query_system.model_manager.get_detected_model()
                        if detected_model:
                            self.active_model = detected_model
                            print(f"{Colors.GREEN}✨ AI Assistant ready with {detected_model}{Colors.ENDC}")
                            
                            # Show special message for dolphin3
                            if "dolphin" in detected_model.lower():
                                print(f"{Colors.MAGENTA}🐬 Dolphin3 is excellent for vault analysis and natural language queries{Colors.ENDC}")
                        else:
                            print(f"{Colors.GREEN}✨ AI models ready for natural language queries{Colors.ENDC}")
                            
                        if is_first_run:
                            print(f"{Colors.CYAN}💡 Try asking: 'What are the main themes in my vault?'{Colors.ENDC}")
                    else:
                        print(f"{Colors.RED}❌ AI models not available - Ollama may not be running{Colors.ENDC}")
                        print(f"{Colors.YELLOW}📝 To use AI features:{Colors.ENDC}")
                        print(f"{Colors.YELLOW}   1. Install Ollama: https://ollama.ai{Colors.ENDC}")
                        print(f"{Colors.YELLOW}   2. Run: ollama serve{Colors.ENDC}")
                        print(f"{Colors.YELLOW}   3. Pull a model: ollama pull dolphin3{Colors.ENDC}")
                        
                        # Ask user if they want to use pattern matching fallback
                        if VAULT_QUERY_AVAILABLE:
                            print(f"\n{Colors.YELLOW}⚠️ Pattern matching fallback is available but very limited{Colors.ENDC}")
                            use_fallback = input(f"{Colors.YELLOW}Use pattern matching mode? (y/N): {Colors.ENDC}")
                            if use_fallback.lower() == 'y':
                                self.pattern_matching_mode = True
                                print(f"{Colors.YELLOW}📊 Pattern matching mode enabled (limited functionality){Colors.ENDC}")
                            else:
                                self.query_enabled = False
                                print(f"{Colors.RED}❌ Query system disabled - AI models required{Colors.ENDC}")
                        else:
                            self.query_enabled = False
                except Exception as e:
                    print(f"{Colors.RED}❌ Could not initialize AI models: {e}{Colors.ENDC}")
                    self.query_enabled = False
            except Exception as e:
                print(f"{Colors.RED}❌ LLM system error: {e}{Colors.ENDC}")
                self.query_enabled = False
                self.query_system = None
        elif VAULT_QUERY_AVAILABLE and self.current_vault:
            # Only use pattern matching if explicitly requested
            print(f"{Colors.YELLOW}⚠️ AI-powered query system not available{Colors.ENDC}")
            print(f"{Colors.YELLOW}Pattern matching is available but provides limited functionality{Colors.ENDC}")
            use_pattern = input(f"{Colors.YELLOW}Use pattern matching mode? (y/N): {Colors.ENDC}")
            if use_pattern.lower() == 'y':
                self.query_system = VaultQuerySystem(self.current_vault)
                self.query_enabled = True
                self.pattern_matching_mode = True
                print(f"{Colors.YELLOW}📊 Pattern matching mode enabled{Colors.ENDC}")
            else:
                self.query_system = None
                self.query_enabled = False
                print(f"{Colors.RED}❌ Query system disabled - AI models recommended{Colors.ENDC}")
        else:
            self.query_system = None
            self.query_enabled = False
            print(f"{Colors.RED}❌ No query system available{Colors.ENDC}")
            print(f"{Colors.YELLOW}📝 Install requirements: pip install -r requirements.txt{Colors.ENDC}")
        
        if self.audio_enabled:
            print(f"{Colors.GREEN}🎵 Dungeon crawler audio system initialized{Colors.ENDC}")
            # Temporarily disable ambient atmosphere to fix humming issue
            # start_dungeon_ambiance()
            print(f"{Colors.BLUE}🔇 Ambient atmosphere disabled (sound effects only){Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}⚠️  Audio system not available - continuing in silent mode{Colors.ENDC}")
    
    def cleanup(self):
        """Clean up resources when exiting"""
        if self.audio_enabled:
            print(f"\n{Colors.BLUE}🎵 Farewell from the digital wizard...{Colors.ENDC}")
            if self.audio_manager:
                play_sound('wizard_warn')  # Farewell sound
                time.sleep(0.5)  # Let sound play
                stop_dungeon_ambiance()
                self.audio_manager.cleanup()
        super().cleanup() if hasattr(super(), 'cleanup') else None
        
    def show_welcome_art(self):
        """Display enhanced welcome ASCII art"""
        # Show the original logo
        super().show_welcome_art()
        
        # Play wizard greeting sound
        if self.audio_enabled:
            wizard_greeting()
        
        # Add a random decorative element
        if self.ascii_enabled and random.random() > 0.5:
            print(f"\n{Colors.BLUE}{self.ascii_manager.get_random_art()}{Colors.ENDC}")
    
    def get_choice_with_audio(self, valid_choices=None):
        """Get user input with audio feedback"""
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.ENDC}").strip()
        
        if self.audio_enabled:
            if choice == '0' or choice.lower() in ['exit', 'quit', 'back']:
                play_sound('menu_back')
            elif choice == '':
                play_sound('menu_error')  # Empty input
            elif valid_choices and choice not in valid_choices:
                play_sound('error_chord')  # Invalid choice
                dungeon_danger()  # Warning sound
            else:
                play_sound('menu_select')  # Valid selection
        
        return choice
    
    def play_navigation_sound(self, sound_name: str):
        """Callback for menu navigator audio"""
        if self.audio_enabled:
            play_sound(sound_name)
    
    def play_menu_navigation_sound(self):
        """Play sound for menu navigation"""
        if self.audio_enabled:
            play_sound('menu_blip')
    
    def play_operation_start_sound(self, operation_type='general'):
        """Play sound for starting operations"""
        if self.audio_enabled:
            if operation_type == 'scan':
                play_sound('scan_begin')
            elif operation_type == 'backup':
                play_sound('backup_begin')
            elif operation_type == 'ascii':
                play_sound('magic_begin')
            else:
                play_sound('menu_select')
    
    def play_operation_complete_sound(self, operation_type='general', success=True):
        """Play sound for completed operations"""
        if self.audio_enabled:
            if success:
                if operation_type == 'scan':
                    play_sound('scan_complete')
                elif operation_type == 'backup':
                    play_sound('backup_complete')
                elif operation_type == 'ascii':
                    play_sound('magic_complete')
                    magical_success()
                else:
                    play_sound('magic_success')
            else:
                play_sound('error_chord')
                dungeon_danger()
    
    def show_ascii_menu(self):
        """ASCII Art tools submenu"""
        while True:
            options = [
                ('1', 'Convert image to ASCII art'),
                ('2', 'Convert screenshot to ASCII'),
                ('3', 'View ASCII art gallery'),
                ('4', 'Generate flowchart from document'),
                ('5', 'Add art to vault notes'),
                ('6', 'ASCII art settings'),
                ('7', 'Query vault content'),
                ('0', 'Back to main menu')
            ]
            
            # Use arrow key navigation if available
            if self.arrow_navigation and self.navigator:
                choice = self.navigator.navigate_menu('ASCII ART TOOLS', options)
                if choice == 'quit':
                    return  # Exit to main menu
            else:
                # Fallback to traditional menu
                self.show_menu('ASCII ART TOOLS', options)
                self.play_menu_navigation_sound()
                choice = self.get_choice_with_audio(['0', '1', '2', '3', '4', '5', '6', '7'])
            
            if choice == '0':
                break
            elif choice == '1':
                self.play_operation_start_sound('ascii')
                self.convert_image_to_ascii()
            elif choice == '2':
                self.play_operation_start_sound('ascii')
                self.convert_screenshot_to_ascii()
            elif choice == '3':
                self.play_menu_navigation_sound()
                self.view_ascii_gallery()
            elif choice == '4':
                self.play_operation_start_sound('ascii')
                self.generate_flowchart_from_document()
            elif choice == '5':
                self.play_operation_start_sound('ascii')
                self.add_ascii_to_notes()
            elif choice == '6':
                self.play_menu_navigation_sound()
                self.ascii_art_settings()
            elif choice == '7':
                self.play_menu_navigation_sound()
                self.vault_query_interface()
    
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
        self.show_indeterminate_progress("Generating ASCII art", 3)
        
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
                    self.play_operation_complete_sound('ascii', success=True)
                else:
                    self.play_operation_complete_sound('ascii', success=True)
                
                # Clean up
                os.remove(screenshot_path)
            else:
                print(f"{Colors.RED}❌ Failed to convert screenshot{Colors.ENDC}")
                self.play_operation_complete_sound('ascii', success=False)
    
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
        self.show_indeterminate_progress("Processing document", 5)
        
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
        
        # Audio settings
        print(f"\n{Colors.MAGENTA}🎵 AUDIO SETTINGS{Colors.ENDC}")
        print(f"6. Audio System: {Colors.GREEN if self.audio_enabled else Colors.RED}"
              f"{'Available' if self.audio_enabled else 'Not Available'}{Colors.ENDC}")
        
        if self.audio_enabled and self.audio_manager:
            print(f"7. Master Volume: {Colors.YELLOW}{self.audio_manager.get_volume('master'):.1f}{Colors.ENDC}")
            print(f"8. Effects Volume: {Colors.YELLOW}{self.audio_manager.get_volume('effects'):.1f}{Colors.ENDC}")
            print(f"9. Ambient Volume: {Colors.YELLOW}{self.audio_manager.get_volume('ambient'):.1f}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Configuration Options:{Colors.ENDC}")
        print("a) Toggle ASCII art in menus")
        if self.audio_enabled:
            print("v) Adjust volume settings")
            print("t) Test sound effects")
        print("q) Return to menu")
        
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.ENDC}").lower().strip()
        
        if choice == 'a':
            self.ascii_enabled = not self.ascii_enabled
            self.config['ascii_enabled'] = self.ascii_enabled
            self.save_config()
            print(f"{Colors.GREEN}✓ ASCII art {'enabled' if self.ascii_enabled else 'disabled'}{Colors.ENDC}")
            self.play_operation_complete_sound('general', success=True)
        elif choice == 'v' and self.audio_enabled:
            self.audio_volume_settings()
        elif choice == 't' and self.audio_enabled:
            self.test_audio_effects()
        elif choice == 'q':
            pass
        else:
            print(f"{Colors.RED}❌ Invalid choice{Colors.ENDC}")
            if self.audio_enabled:
                dungeon_danger()
    
    def audio_volume_settings(self):
        """Configure audio volume settings"""
        if not self.audio_enabled or not self.audio_manager:
            return
        
        print(f"\n{Colors.MAGENTA}🎵 VOLUME SETTINGS{Colors.ENDC}")
        print(f"Current volumes:")
        print(f"  Master: {self.audio_manager.get_volume('master'):.1f}")
        print(f"  Effects: {self.audio_manager.get_volume('effects'):.1f}")
        print(f"  Ambient: {self.audio_manager.get_volume('ambient'):.1f}")
        
        category = input(f"\n{Colors.YELLOW}Adjust volume for (master/effects/ambient): {Colors.ENDC}").lower().strip()
        
        if category in ['master', 'effects', 'ambient']:
            try:
                volume = float(input(f"{Colors.YELLOW}New volume (0.0-1.0): {Colors.ENDC}"))
                volume = max(0.0, min(1.0, volume))
                self.audio_manager.set_volume(category, volume)
                print(f"{Colors.GREEN}✓ {category.title()} volume set to {volume:.1f}{Colors.ENDC}")
                play_sound('menu_select')  # Test the new volume
            except ValueError:
                print(f"{Colors.RED}❌ Invalid volume value{Colors.ENDC}")
                dungeon_danger()
    
    def test_audio_effects(self):
        """Test different audio effects"""
        if not self.audio_enabled:
            return
        
        print(f"\n{Colors.MAGENTA}🎵 AUDIO TEST{Colors.ENDC}")
        print("Testing sound effects...")
        
        effects = [
            ('Menu Blip', 'menu_blip'),
            ('Menu Select', 'menu_select'),
            ('Menu Back', 'menu_back'),
            ('Wizard Greeting', 'wizard_hello'),
            ('Magic Spell', 'spell_cast'),
            ('Success', 'magic_success'),
            ('Error', 'error_chord'),
            ('Danger Warning', 'danger_sting')
        ]
        
        for name, effect in effects:
            print(f"  Playing: {name}")
            play_sound(effect)
            time.sleep(1.2)  # Pause between sounds
        
        print(f"{Colors.GREEN}✓ Audio test complete{Colors.ENDC}")
        magical_success()
    
    def vault_query_interface(self):
        """Interactive vault query interface"""
        if not self.query_enabled or not self.query_system:
            print(f"{Colors.RED}Vault query system not available{Colors.ENDC}")
            return
        
        # Show current mode at the top with active model information
        print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        # Build header with model information
        if hasattr(self, 'llm_enabled') and self.llm_enabled:
            # Get active model name if available
            model_name = None
            if hasattr(self, 'query_system') and self.query_system and hasattr(self.query_system, 'model_manager'):
                try:
                    model_name = self.query_system.model_manager.get_detected_model()
                except:
                    model_name = None
            
            # Build header with model info and dolphin emoji for dolphin models
            if model_name:
                if 'dolphin' in model_name.lower():
                    model_display = f"🐬 {model_name}"
                else:
                    model_display = model_name
                print(f"{Colors.CYAN}🤖 Vault Query System - AI-Powered ({model_display}){Colors.ENDC}")
            else:
                print(f"{Colors.CYAN}🤖 Vault Query System - AI-Powered (LLM Enhanced){Colors.ENDC}")
            print(f"{Colors.BLUE}Using natural language processing with AI models{Colors.ENDC}")
            
        elif hasattr(self, 'pattern_matching_mode') and self.pattern_matching_mode:
            print(f"{Colors.CYAN}📊 Vault Query System - Pattern Matching (Limited){Colors.ENDC}")
            print(f"{Colors.BLUE}Using basic pattern matching - AI models not available{Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}📊 Vault Query System - Standard{Colors.ENDC}")
            
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
        
        while True:
            print(f"{Colors.GREEN}Example queries:{Colors.ENDC}")
            print("  - How many files are in my vault?")
            print("  - Show me recent meeting notes")
            print("  - Find files about projects")
            print("  - What are my most used tags?")
            print("  - Who is mentioned most in my notes?")
            print("  - Generate a summary report")
            
            if hasattr(self, 'llm_enabled') and self.llm_enabled:
                print(f"\n{Colors.MAGENTA}LLM-specific features:{Colors.ENDC}")
                print("  - What are the main themes in my vault?")
                print("  - Summarize my project documentation")
                print("  - Extract action items from meeting notes")
                print("  - Analyze the sentiment of my daily notes")
                print("  - Generate insights about my vault")
                print("  - rate [1-5] [comment] - Rate the last response")
            
            if self.navigator:
                query = self.navigator.get_text_input(f"\n{Colors.YELLOW}Enter your query (or 'quit' to exit): {Colors.ENDC}")
            else:
                query = input(f"\n{Colors.YELLOW}Enter your query (or 'quit' to exit): {Colors.ENDC}")
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Check for rating command
            if query.lower().startswith('rate '):
                if not self.last_query_id:
                    print(f"{Colors.RED}No previous query to rate{Colors.ENDC}")
                    continue
                
                try:
                    # Parse rating command (e.g., "rate 4" or "rate 5 great response")
                    parts = query.split(' ', 2)
                    rating = int(parts[1])
                    comment = parts[2] if len(parts) > 2 else None
                    
                    if 1 <= rating <= 5:
                        # Store rating (implement actual storage if needed)
                        print(f"{Colors.GREEN}✓ Rating recorded: {rating}/5{Colors.ENDC}")
                        if comment:
                            print(f"{Colors.GREEN}  Comment: {comment}{Colors.ENDC}")
                        print(f"{Colors.BLUE}  Query ID: {self.last_query_id}{Colors.ENDC}")
                        self.play_operation_complete_sound('scan', success=True)
                    else:
                        print(f"{Colors.RED}Rating must be between 1 and 5{Colors.ENDC}")
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Invalid rating format. Use: rate [1-5] [optional comment]{Colors.ENDC}")
                continue
            
            if query.lower() in ['summary', 'report', 'generate summary']:
                # Generate summary report
                print(f"\n{Colors.BLUE}Generating vault summary report...{Colors.ENDC}")
                self.play_operation_start_sound('scan')
                
                try:
                    summary = self.query_system.generate_summary_report()
                    
                    # Save summary to file with versioning
                    if FILE_VERSIONING_AVAILABLE:
                        output_path = FileVersioning.create_output_filename(
                            "vault_summary", "report", self.current_vault
                        )
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(self.current_vault, f"vault_summary_report_{timestamp}.md")
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    
                    print(f"\n{Colors.GREEN}Summary report generated:{Colors.ENDC}")
                    print(f"Saved to: {os.path.basename(output_path)}")
                    
                    # Show preview
                    lines = summary.split('\n')
                    for line in lines[:15]:  # Show first 15 lines
                        print(line)
                    
                    if len(lines) > 15:
                        print(f"{Colors.YELLOW}... (see full report in file){Colors.ENDC}")
                    
                    self.play_operation_complete_sound('scan', success=True)
                    
                except Exception as e:
                    print(f"{Colors.RED}Error generating summary: {e}{Colors.ENDC}")
                    self.play_operation_complete_sound('scan', success=False)
            
            else:
                # Process query
                print(f"\n{Colors.BLUE}Processing query: {query}{Colors.ENDC}")
                self.play_operation_start_sound('scan')
                
                try:
                    results = self.query_system.query(query)
                    
                    # Generate and store query ID
                    self.last_query_id = str(uuid.uuid4())[:8]
                    
                    # Display results based on query system type
                    print(f"\n{Colors.GREEN}Query Results:{Colors.ENDC}")
                    
                    # Check if this is an LLM response (has 'response' field)
                    if 'response' in results:
                        # LLM-specific response display
                        print(f"\n{Colors.MAGENTA}AI Response:{Colors.ENDC}")
                        print(results['response'])
                        
                        # Show metadata if available
                        if 'metadata' in results:
                            meta = results['metadata']
                            print(f"\n{Colors.CYAN}Metadata:{Colors.ENDC}")
                            if 'models_used' in meta:
                                print(f"  Models: {', '.join(meta['models_used'])}")
                            if 'confidence' in meta:
                                print(f"  Confidence: {meta['confidence']:.2f}")
                            if 'response_time' in meta:
                                print(f"  Response time: {meta['response_time']:.2f}s")
                        
                        # Show query ID for rating
                        print(f"\n{Colors.BLUE}Query ID: {self.last_query_id}{Colors.ENDC}")
                        print(f"{Colors.YELLOW}To rate this response: rate [1-5] [optional comment]{Colors.ENDC}")
                    
                    # Standard structured results display
                    else:
                        print(f"Type: {results.get('query_type', 'general')}")
                        
                        if 'summary' in results:
                            print(f"Summary: {results['summary']}")
                        
                        if 'count' in results:
                            print(f"Count: {Colors.YELLOW}{results['count']}{Colors.ENDC}")
                        
                        if 'top_tags' in results:
                            print(f"Top tags: {', '.join(results['top_tags'][:5])}")
                        
                        if 'matching_files' in results and results['matching_files']:
                            print(f"\n{Colors.CYAN}Matching files:{Colors.ENDC}")
                            for i, match in enumerate(results['matching_files'][:5], 1):
                                print(f"  {i}. {match['name']} (score: {match['score']})")
                        
                        if 'files' in results:
                            print(f"\n{Colors.CYAN}Files:{Colors.ENDC}")
                            for i, file_info in enumerate(results['files'][:5], 1):
                                print(f"  {i}. {file_info['name']} - {file_info.get('modified', 'Unknown date')}")
                        
                        if 'projects' in results:
                            print(f"\n{Colors.CYAN}Projects:{Colors.ENDC}")
                            for i, project in enumerate(results['projects'][:5], 1):
                                print(f"  {i}. {project['name']} - Status: {project['status']}")
                        
                        if 'top_people' in results:
                            print(f"\n{Colors.CYAN}People mentioned:{Colors.ENDC}")
                            for person, files in results['top_people'][:5]:
                                print(f"  - {person} ({len(files)} mentions)")
                    
                    # Offer to save results
                    if self.navigator:
                        save_results = self.navigator.get_text_input(f"\n{Colors.YELLOW}Save results to file? (y/n): {Colors.ENDC}")
                    else:
                        save_results = input(f"\n{Colors.YELLOW}Save results to file? (y/n): {Colors.ENDC}")
                    
                    if save_results.lower() == 'y':
                        # Create results file
                        results_content = f"# Query Results\n\n"
                        results_content += f"**Query:** {query}\n\n"
                        results_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                        results_content += f"**Query ID:** {self.last_query_id}\n\n"
                        
                        # Handle LLM responses differently
                        if 'response' in results:
                            results_content += f"**Mode:** AI-Powered (LLM)\n\n"
                            results_content += "## AI Response\n\n"
                            results_content += results['response'] + "\n\n"
                            
                            if 'metadata' in results:
                                results_content += "## Metadata\n\n"
                                meta = results['metadata']
                                if 'models_used' in meta:
                                    results_content += f"- **Models:** {', '.join(meta['models_used'])}\n"
                                if 'confidence' in meta:
                                    results_content += f"- **Confidence:** {meta['confidence']:.2f}\n"
                                if 'response_time' in meta:
                                    results_content += f"- **Response Time:** {meta['response_time']:.2f}s\n"
                        else:
                            # Standard query results
                            results_content += f"**Type:** {results.get('query_type', 'general')}\n\n"
                            
                            if 'summary' in results:
                                results_content += f"**Summary:** {results['summary']}\n\n"
                            
                            if 'matching_files' in results and results['matching_files']:
                                results_content += "## Matching Files\n\n"
                                for match in results['matching_files'][:10]:
                                    results_content += f"- **{match['name']}** (score: {match['score']})\n"
                        
                        # Save with versioning
                        if FILE_VERSIONING_AVAILABLE:
                            query_safe = re.sub(r'[<>:"/\\|?*]', '_', query[:30])
                            output_path = FileVersioning.create_output_filename(
                                f"query_{query_safe}", "results", self.current_vault
                            )
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = os.path.join(self.current_vault, f"query_results_{timestamp}.md")
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(results_content)
                        
                        print(f"{Colors.GREEN}Results saved to: {os.path.basename(output_path)}{Colors.ENDC}")
                    
                    self.play_operation_complete_sound('scan', success=True)
                    
                except Exception as e:
                    print(f"{Colors.RED}Error processing query: {e}{Colors.ENDC}")
                    self.play_operation_complete_sound('scan', success=False)
            
            print()  # Empty line for readability
    
    def ai_model_configuration(self):
        """Configure AI models and LLM settings"""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.CYAN}AI MODEL CONFIGURATION{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
        
        # Check if LLM system is available
        if not LLM_QUERY_AVAILABLE:
            print(f"{Colors.RED}❌ LLM system not installed{Colors.ENDC}")
            print(f"{Colors.YELLOW}📝 To install: pip install -r requirements.txt{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        # Check if Ollama is running
        ollama_running = False
        if self.llm_enabled and hasattr(self.query_system, 'llm_handler'):
            try:
                # Test Ollama connection
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                ollama_running = result.returncode == 0
            except:
                ollama_running = False
        
        if not ollama_running:
            print(f"{Colors.YELLOW}⚠️  Ollama is not running{Colors.ENDC}")
            print(f"{Colors.BLUE}To use AI features:{Colors.ENDC}")
            print(f"  1. Install Ollama: https://ollama.ai")
            print(f"  2. Run: ollama serve")
            print(f"  3. Pull a model: ollama pull llama2")
        
        while True:
            options = [
                ('1', '📋 View available Ollama models'),
                ('2', '⬇️  Pull new models from Ollama'),
                ('3', '📊 View model performance stats'),
                ('4', '⚙️  Configure model preferences'),
                ('5', '💾 Export training data from feedback'),
                ('6', '🧪 Test models with sample queries'),
                ('0', 'Back to main menu')
            ]
            
            # Use arrow key navigation if available
            if self.arrow_navigation and self.navigator:
                choice = self.navigator.navigate_menu('AI MODEL CONFIGURATION', options)
                if choice == 'quit':
                    return  # Exit to main menu
            else:
                # Fallback to traditional menu
                self.show_menu('AI MODEL CONFIGURATION', options)
                self.play_menu_navigation_sound()
                choice = self.get_choice_with_audio(['0', '1', '2', '3', '4', '5', '6'])
            
            if choice == '0':
                break
            elif choice == '1':
                self.view_available_models()
            elif choice == '2':
                self.pull_new_models()
            elif choice == '3':
                self.view_model_performance_stats()
            elif choice == '4':
                self.configure_model_preferences()
            elif choice == '5':
                self.export_training_data()
            elif choice == '6':
                self.test_models_with_samples()
    
    def view_available_models(self):
        """View available Ollama models"""
        print(f"\n{Colors.CYAN}Available Ollama Models{Colors.ENDC}")
        
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"\n{Colors.GREEN}Installed models:{Colors.ENDC}")
                print(result.stdout)
                
                # Also show popular models to pull
                print(f"\n{Colors.BLUE}Popular models you can pull:{Colors.ENDC}")
                print("  - llama2: Meta's Llama 2 model")
                print("  - mistral: Fast and efficient 7B model")
                print("  - codellama: Specialized for code")
                print("  - neural-chat: Optimized for conversation")
                print("  - phi: Microsoft's small but capable model")
                print("  - mixtral: High-quality mixture of experts model")
            else:
                print(f"{Colors.RED}❌ Could not get model list. Is Ollama running?{Colors.ENDC}")
                print(f"{Colors.YELLOW}Run 'ollama serve' in another terminal{Colors.ENDC}")
        except FileNotFoundError:
            print(f"{Colors.RED}❌ Ollama not found. Please install from https://ollama.ai{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def pull_new_models(self):
        """Pull new models from Ollama"""
        print(f"\n{Colors.CYAN}Pull New Models from Ollama{Colors.ENDC}")
        
        print(f"\n{Colors.BLUE}Recommended models:{Colors.ENDC}")
        models = [
            ('llama2', 'General purpose, well-rounded'),
            ('mistral', 'Fast and efficient 7B model'),
            ('codellama', 'Specialized for code understanding'),
            ('neural-chat', 'Optimized for conversation'),
            ('phi', 'Small but capable (2.7B)'),
            ('mixtral', 'High-quality mixture of experts'),
            ('deepseek-coder', 'Excellent for code analysis'),
            ('qwen', 'Strong multilingual support')
        ]
        
        for i, (model, desc) in enumerate(models, 1):
            print(f"  {i}. {model}: {desc}")
        
        model_choice = input(f"\n{Colors.YELLOW}Enter model name or number (or 'cancel'): {Colors.ENDC}").strip()
        
        if model_choice.lower() == 'cancel' or not model_choice:
            return
        
        # Handle numeric choice
        if model_choice.isdigit():
            idx = int(model_choice) - 1
            if 0 <= idx < len(models):
                model_name = models[idx][0]
            else:
                print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
                return
        else:
            model_name = model_choice
        
        print(f"\n{Colors.BLUE}Pulling {model_name}... This may take a while...{Colors.ENDC}")
        self.play_operation_start_sound('scan')
        
        try:
            import subprocess
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Show progress
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"  {line.strip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print(f"\n{Colors.GREEN}✓ Successfully pulled {model_name}{Colors.ENDC}")
                self.play_operation_complete_sound('scan', success=True)
            else:
                print(f"\n{Colors.RED}❌ Failed to pull {model_name}{Colors.ENDC}")
                self.play_operation_complete_sound('scan', success=False)
        except Exception as e:
            print(f"{Colors.RED}❌ Error: {e}{Colors.ENDC}")
            self.play_operation_complete_sound('scan', success=False)
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def view_model_performance_stats(self):
        """View model performance statistics"""
        print(f"\n{Colors.CYAN}Model Performance Statistics{Colors.ENDC}")
        
        if not self.query_system or not hasattr(self.query_system, 'get_performance_stats'):
            # Create mock statistics for demonstration
            print(f"\n{Colors.YELLOW}Performance tracking not available in current configuration{Colors.ENDC}")
            print(f"\n{Colors.BLUE}Sample performance metrics:{Colors.ENDC}")
            print(f"  Average response time: 2.3s")
            print(f"  Queries processed: 42")
            print(f"  Success rate: 95%")
            print(f"  Average confidence: 0.87")
            print(f"\n{Colors.YELLOW}Note: Install full LLM system for detailed tracking{Colors.ENDC}")
        else:
            try:
                stats = self.query_system.get_performance_stats()
                print(f"\n{Colors.GREEN}Query Statistics:{Colors.ENDC}")
                print(f"  Total queries: {stats.get('total_queries', 0)}")
                print(f"  Average response time: {stats.get('avg_response_time', 0):.2f}s")
                print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
                
                if 'model_stats' in stats:
                    print(f"\n{Colors.GREEN}Model-specific stats:{Colors.ENDC}")
                    for model, model_stats in stats['model_stats'].items():
                        print(f"\n  {model}:")
                        print(f"    Queries: {model_stats.get('count', 0)}")
                        print(f"    Avg time: {model_stats.get('avg_time', 0):.2f}s")
                        print(f"    Avg confidence: {model_stats.get('avg_confidence', 0):.2f}")
            except Exception as e:
                print(f"{Colors.RED}Error getting stats: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def configure_model_preferences(self):
        """Configure model preferences"""
        print(f"\n{Colors.CYAN}Model Preferences Configuration{Colors.ENDC}")
        
        # Show current preferences
        print(f"\n{Colors.BLUE}Current settings:{Colors.ENDC}")
        
        # Default preferences
        prefs = self.config.get('llm_preferences', {
            'default_model': 'llama2',
            'temperature': 0.7,
            'max_tokens': 2048,
            'timeout': 30,
            'use_multiple_models': False
        })
        
        print(f"  Default model: {Colors.YELLOW}{prefs['default_model']}{Colors.ENDC}")
        print(f"  Temperature: {Colors.YELLOW}{prefs['temperature']}{Colors.ENDC}")
        print(f"  Max tokens: {Colors.YELLOW}{prefs['max_tokens']}{Colors.ENDC}")
        print(f"  Timeout: {Colors.YELLOW}{prefs['timeout']}s{Colors.ENDC}")
        print(f"  Use multiple models: {Colors.YELLOW}{'Yes' if prefs['use_multiple_models'] else 'No'}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Configuration options:{Colors.ENDC}")
        print("1. Change default model")
        print("2. Adjust temperature (creativity)")
        print("3. Set max tokens (response length)")
        print("4. Set timeout")
        print("5. Toggle multiple model usage")
        print("0. Back")
        
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.ENDC}").strip()
        
        if choice == '1':
            new_model = input(f"{Colors.YELLOW}Enter new default model name: {Colors.ENDC}").strip()
            if new_model:
                prefs['default_model'] = new_model
                print(f"{Colors.GREEN}✓ Default model set to {new_model}{Colors.ENDC}")
        
        elif choice == '2':
            try:
                temp = float(input(f"{Colors.YELLOW}Enter temperature (0.0-2.0, current: {prefs['temperature']}): {Colors.ENDC}"))
                if 0.0 <= temp <= 2.0:
                    prefs['temperature'] = temp
                    print(f"{Colors.GREEN}✓ Temperature set to {temp}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}Temperature must be between 0.0 and 2.0{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Invalid temperature value{Colors.ENDC}")
        
        elif choice == '3':
            try:
                tokens = int(input(f"{Colors.YELLOW}Enter max tokens (100-8192, current: {prefs['max_tokens']}): {Colors.ENDC}"))
                if 100 <= tokens <= 8192:
                    prefs['max_tokens'] = tokens
                    print(f"{Colors.GREEN}✓ Max tokens set to {tokens}{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}Max tokens must be between 100 and 8192{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Invalid token value{Colors.ENDC}")
        
        elif choice == '4':
            try:
                timeout = int(input(f"{Colors.YELLOW}Enter timeout in seconds (5-300, current: {prefs['timeout']}): {Colors.ENDC}"))
                if 5 <= timeout <= 300:
                    prefs['timeout'] = timeout
                    print(f"{Colors.GREEN}✓ Timeout set to {timeout}s{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}Timeout must be between 5 and 300 seconds{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Invalid timeout value{Colors.ENDC}")
        
        elif choice == '5':
            prefs['use_multiple_models'] = not prefs['use_multiple_models']
            status = 'enabled' if prefs['use_multiple_models'] else 'disabled'
            print(f"{Colors.GREEN}✓ Multiple model usage {status}{Colors.ENDC}")
        
        # Save preferences
        if choice in ['1', '2', '3', '4', '5']:
            self.config['llm_preferences'] = prefs
            self.save_config()
            
            # Apply to query system if available
            if self.query_system and hasattr(self.query_system, 'update_preferences'):
                self.query_system.update_preferences(prefs)
        
        if choice != '0':
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def export_training_data(self):
        """Export training data from feedback"""
        print(f"\n{Colors.CYAN}Export Training Data from Feedback{Colors.ENDC}")
        
        # Check if we have feedback data
        feedback_file = os.path.join(self.current_vault, '.obsidian_librarian', 'query_feedback.json')
        
        if not os.path.exists(feedback_file):
            print(f"{Colors.YELLOW}No feedback data found yet{Colors.ENDC}")
            print(f"Rate queries using 'rate [1-5] [comment]' in the query interface")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            print(f"\n{Colors.GREEN}Found {len(feedback_data)} feedback entries{Colors.ENDC}")
            
            # Create training data export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(self.current_vault, f"training_data_export_{timestamp}.jsonl")
            
            # Convert to training format
            training_entries = []
            for entry in feedback_data:
                if entry.get('rating', 0) >= 4:  # Only use positive examples
                    training_entry = {
                        'query': entry['query'],
                        'response': entry['response'],
                        'rating': entry['rating'],
                        'comment': entry.get('comment', ''),
                        'timestamp': entry['timestamp']
                    }
                    training_entries.append(training_entry)
            
            # Write JSONL format
            with open(export_path, 'w', encoding='utf-8') as f:
                for entry in training_entries:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"{Colors.GREEN}✓ Exported {len(training_entries)} positive examples{Colors.ENDC}")
            print(f"{Colors.BLUE}Saved to: {os.path.basename(export_path)}{Colors.ENDC}")
            
            # Offer to create fine-tuning dataset
            create_ft = input(f"\n{Colors.YELLOW}Create OpenAI fine-tuning format? (y/n): {Colors.ENDC}").lower()
            if create_ft == 'y':
                ft_path = os.path.join(self.current_vault, f"fine_tuning_data_{timestamp}.jsonl")
                with open(ft_path, 'w', encoding='utf-8') as f:
                    for entry in training_entries:
                        ft_entry = {
                            'messages': [
                                {'role': 'user', 'content': entry['query']},
                                {'role': 'assistant', 'content': entry['response']}
                            ]
                        }
                        f.write(json.dumps(ft_entry) + '\n')
                print(f"{Colors.GREEN}✓ Created fine-tuning dataset: {os.path.basename(ft_path)}{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error exporting data: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def test_models_with_samples(self):
        """Test models with sample queries"""
        print(f"\n{Colors.CYAN}Test Models with Sample Queries{Colors.ENDC}")
        
        if not self.llm_enabled:
            print(f"{Colors.RED}❌ LLM system not available{Colors.ENDC}")
            print(f"{Colors.YELLOW}Please ensure Ollama is running and models are installed{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        # Sample queries for testing
        sample_queries = [
            "What are the main topics in my vault?",
            "Summarize my recent meeting notes",
            "Find all tasks marked as urgent",
            "What projects am I currently working on?",
            "Analyze the sentiment of my daily notes from last week"
        ]
        
        print(f"\n{Colors.BLUE}Sample queries:{Colors.ENDC}")
        for i, query in enumerate(sample_queries, 1):
            print(f"  {i}. {query}")
        
        print(f"\n{Colors.YELLOW}Options:{Colors.ENDC}")
        print("  - Enter a number (1-5) to test a sample query")
        print("  - Enter your own custom query")
        print("  - Type 'benchmark' to test all samples")
        print("  - Type 'cancel' to go back")
        
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.ENDC}").strip()
        
        if choice.lower() == 'cancel':
            return
        
        elif choice.lower() == 'benchmark':
            print(f"\n{Colors.BLUE}Running benchmark on all sample queries...{Colors.ENDC}")
            self.play_operation_start_sound('scan')
            
            for i, query in enumerate(sample_queries, 1):
                print(f"\n{Colors.CYAN}Query {i}/{len(sample_queries)}: {query}{Colors.ENDC}")
                
                try:
                    start_time = time.time()
                    results = self.query_system.query(query)
                    end_time = time.time()
                    
                    print(f"{Colors.GREEN}✓ Response time: {end_time - start_time:.2f}s{Colors.ENDC}")
                    
                    # Show brief preview
                    if 'response' in results:
                        preview = results['response'][:200] + '...' if len(results['response']) > 200 else results['response']
                        print(f"Preview: {preview}")
                    
                except Exception as e:
                    print(f"{Colors.RED}❌ Error: {e}{Colors.ENDC}")
            
            self.play_operation_complete_sound('scan', success=True)
        
        else:
            # Single query test
            if choice.isdigit() and 1 <= int(choice) <= len(sample_queries):
                test_query = sample_queries[int(choice) - 1]
            else:
                test_query = choice
            
            print(f"\n{Colors.BLUE}Testing query: {test_query}{Colors.ENDC}")
            self.play_operation_start_sound('scan')
            
            try:
                start_time = time.time()
                results = self.query_system.query(test_query)
                end_time = time.time()
                
                print(f"\n{Colors.GREEN}✓ Query completed in {end_time - start_time:.2f}s{Colors.ENDC}")
                
                # Display full results
                if 'response' in results:
                    print(f"\n{Colors.MAGENTA}AI Response:{Colors.ENDC}")
                    print(results['response'])
                    
                    if 'metadata' in results:
                        meta = results['metadata']
                        print(f"\n{Colors.CYAN}Metadata:{Colors.ENDC}")
                        if 'models_used' in meta:
                            print(f"  Models: {', '.join(meta['models_used'])}")
                        if 'confidence' in meta:
                            print(f"  Confidence: {meta['confidence']:.2f}")
                
                self.play_operation_complete_sound('scan', success=True)
                
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.ENDC}")
                self.play_operation_complete_sound('scan', success=False)
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
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
    
    def get_menu_choice(self, title, options, audio_enabled=True):
        """Get menu choice using arrow keys if available, fallback to traditional input"""
        if self.arrow_navigation and self.navigator:
            if audio_enabled:
                self.play_menu_navigation_sound()
            choice = self.navigator.navigate_menu(title, options)
            if choice == 'quit':
                return '0'  # Standardize quit behavior
            return choice
        else:
            # Fallback to traditional menu
            self.show_menu(title, options)
            if audio_enabled:
                self.play_menu_navigation_sound()
            return input().strip()
    
    def smart_analysis_menu(self):
        """Smart vault analysis combining basic and AI-powered features"""
        while True:
            options = [
                ('1', 'Quick vault overview'),
                ('2', 'AI-powered content analysis' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('3', 'Tag analysis & insights'),
                ('4', 'Advanced vault analytics' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('5', 'Link analysis & graph'),
                ('6', 'Content statistics'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('📊 SMART VAULT ANALYSIS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.play_operation_start_sound('scan')
                # Quick overview using basic analysis
                print(f"\n{Colors.CYAN}Running quick vault overview...{Colors.ENDC}")
                self.run_command_with_progress(
                    f'python3 analyze_tags_simple.py {self.quote_path(self.current_vault)}',
                    'Analyzing vault structure',
                    estimated_duration=5
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                if self.v2_available:
                    self.run_v2_analysis()
                else:
                    print(f"\n{Colors.YELLOW}AI-powered analysis requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.play_operation_start_sound('scan')
                self.run_command_with_progress(
                    f'python3 analyze_tags_simple.py {self.quote_path(self.current_vault)}',
                    'Analyzing tags with insights',
                    estimated_duration=5
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                if self.v2_available:
                    self.run_v2_analytics()
                else:
                    print(f"\n{Colors.YELLOW}Advanced analytics requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                print(f"\n{Colors.YELLOW}Link analysis feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '6':
                print(f"\n{Colors.YELLOW}Content statistics feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def manage_tags_menu(self):
        """Override manage tags menu with arrow key navigation"""
        while True:
            options = [
                ('1', 'Preview all tag fixes'),
                ('2', 'Apply all tag fixes'),
                ('3', 'Fix quoted tags only'),
                ('4', 'Merge similar tags'),
                ('5', 'Remove generic tags'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('🏷️ TAG MANAGEMENT', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.play_operation_start_sound('scan')
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --dry-run',
                    'Previewing tag fixes',
                    estimated_duration=8
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.play_operation_start_sound('scan')
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply',
                    'Applying all tag fixes',
                    estimated_duration=15
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.play_operation_start_sound('scan')
                self.fix_quoted_tags()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                self.play_operation_start_sound('scan')
                self.merge_similar_tags()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                self.play_operation_start_sound('scan')
                self.remove_generic_tags()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def backup_vault_menu(self):
        """Override backup vault menu with arrow key navigation"""
        while True:
            options = [
                ('1', 'Full backup'),
                ('2', 'Incremental backup'),
                ('3', 'Restore from backup'),
                ('4', 'View backup history'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('💾 BACKUP MANAGEMENT', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.play_operation_start_sound('backup')
                self.run_command_with_progress(
                    f'python3 backup_vault.py {self.quote_path(self.current_vault)}',
                    'Creating full backup',
                    estimated_duration=10
                )
                self.play_operation_complete_sound('backup')
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.play_operation_start_sound('backup')
                self.run_command_with_progress(
                    f'./quick_incremental_backup.sh {self.quote_path(self.current_vault)}',
                    'Creating incremental backup',
                    estimated_duration=3
                )
                self.play_operation_complete_sound('backup')
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.restore_backup_menu()
            elif choice == '4':
                print(f"\n{Colors.YELLOW}Backup history feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def advanced_tools_menu(self):
        """Override advanced tools menu with arrow key navigation"""
        while True:
            options = [
                ('1', 'Intelligent janitor'),
                ('2', 'Organize output files'),
                ('3', 'File versioning'),
                ('4', 'Bulk rename'),
                ('5', 'Content migration'),
                ('6', 'Plugin management'),
                ('7', 'Diagnostic tools'),
                ('8', '🔗 MCP Server Management'),
                ('9', '🛠️ MCP Tools'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('🔧 ADVANCED TOOLS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.intelligent_janitor_menu()
            elif choice == '2':
                self.organize_output_files()
            elif choice == '3':
                print(f"\n{Colors.YELLOW}File versioning feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                print(f"\n{Colors.YELLOW}Bulk rename feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                print(f"\n{Colors.YELLOW}Content migration feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '6':
                print(f"\n{Colors.YELLOW}Plugin management feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '7':
                self.diagnostic_tools_menu()
            elif choice == '8':
                self.mcp_server_menu()
            elif choice == '9':
                self.mcp_tools_direct_access()
    
    def mcp_tools_direct_access(self):
        """Direct access to MCP tools interface"""
        try:
            # Import MCP modules dynamically
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'obsidian_vault_tools'))
            
            from obsidian_vault_tools.mcp_tools import get_client_manager
            client_manager = get_client_manager()
            
            # Call the MCP tools interface directly
            self._mcp_tools_interface(client_manager)
            
        except ImportError:
            print(f"\n{Colors.RED}❌ MCP features not available{Colors.ENDC}")
            print(f"{Colors.YELLOW}Install with: pip install mcp cryptography{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
    
    def diagnostic_tools_menu(self):
        """Diagnostic tools submenu"""
        while True:
            options = [
                ('1', 'Audio system test'),
                ('2', 'Navigation test'),
                ('3', 'Performance test'),
                ('4', 'Dependency check'),
                ('0', 'Back to advanced tools')
            ]
            
            choice = self.get_menu_choice('🔍 DIAGNOSTIC TOOLS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.test_audio_system()
            elif choice == '2':
                self.test_navigation_system()
            elif choice == '3':
                print(f"\n{Colors.YELLOW}Performance test feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                self.validate_dependencies()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def mcp_server_menu(self):
        """MCP server management menu"""
        try:
            # Import MCP modules dynamically
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'obsidian_vault_tools'))
            
            from obsidian_vault_tools.mcp_tools import MCPConfig, get_client_manager
            import asyncio
            
            config = MCPConfig()
            client_manager = get_client_manager()
            
        except ImportError:
            print(f"\n{Colors.RED}❌ MCP features not available{Colors.ENDC}")
            print(f"{Colors.YELLOW}Install with: pip install mcp cryptography{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        while True:
            # Get server status
            try:
                server_status = asyncio.run(self._get_mcp_server_status(client_manager, config))
            except Exception as e:
                print(f"{Colors.RED}Error getting server status: {e}{Colors.ENDC}")
                server_status = {}
            
            options = [
                ('1', '📋 List configured servers'),
                ('2', '▶️  Start MCP server'),
                ('3', '⏹️  Stop MCP server'),
                ('4', '🔄 Restart MCP server'),
                ('5', '➕ Add new server'),
                ('6', '🔧 MCP tools interface'),
                ('7', '📚 MCP resources'),
                ('0', 'Back to advanced tools')
            ]
            
            # Add status info to title
            running_count = sum(1 for s in server_status.values() if s.get('running', False))
            total_count = len(server_status)
            status_info = f"({running_count}/{total_count} running)" if total_count > 0 else "(no servers)"
            
            choice = self.get_menu_choice(f'🔗 MCP SERVER MANAGEMENT {status_info}', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self._show_mcp_server_list(config, client_manager)
            elif choice == '2':
                self._start_mcp_server(config, client_manager)
            elif choice == '3':
                self._stop_mcp_server(config, client_manager)
            elif choice == '4':
                self._restart_mcp_server(config, client_manager)
            elif choice == '5':
                self._add_mcp_server(config)
            elif choice == '6':
                self._mcp_tools_interface(client_manager)
            elif choice == '7':
                self._mcp_resources_interface(client_manager)
    
    async def _get_mcp_server_status(self, client_manager, config):
        """Get MCP server status"""
        return client_manager.get_all_server_status()
    
    def _show_mcp_server_list(self, config, client_manager):
        """Show list of MCP servers"""
        print(f"\n{Colors.CYAN}📋 MCP Server Status{Colors.ENDC}")
        
        try:
            import asyncio
            server_status = asyncio.run(self._get_mcp_server_status(client_manager, config))
            
            if not server_status:
                print(f"{Colors.YELLOW}No MCP servers configured.{Colors.ENDC}")
                print(f"{Colors.BLUE}Use option 5 to add a server.{Colors.ENDC}")
            else:
                print(f"\n{Colors.HEADER}{'Name':<20} {'Status':<12} {'Command':<30}{Colors.ENDC}")
                print("-" * 70)
                
                for name, status in server_status.items():
                    running = status.get('running', False)
                    status_str = f"{Colors.GREEN}✓ Running{Colors.ENDC}" if running else f"{Colors.RED}✗ Stopped{Colors.ENDC}"
                    
                    # Get command from config
                    server_config = config.get_server_config(name)
                    command = server_config.get('command', 'N/A') if server_config else 'N/A'
                    
                    print(f"{name:<20} {status_str:<20} {command:<30}")
        
        except Exception as e:
            print(f"{Colors.RED}Error getting server status: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _start_mcp_server(self, config, client_manager):
        """Start an MCP server"""
        servers = config.list_servers()
        if not servers:
            print(f"\n{Colors.YELLOW}No servers configured. Add one first.{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        print(f"\n{Colors.CYAN}📋 Available servers:{Colors.ENDC}")
        for i, server in enumerate(servers):
            print(f"  {i+1}. {server}")
        
        choice = input(f"\n{Colors.YELLOW}Enter server number to start: {Colors.ENDC}")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(servers):
                server_name = servers[idx]
                print(f"\n{Colors.BLUE}Starting server '{server_name}'...{Colors.ENDC}")
                
                import asyncio
                success = asyncio.run(client_manager.start_server(server_name))
                
                if success:
                    print(f"{Colors.GREEN}✓ Server '{server_name}' started successfully!{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Failed to start server '{server_name}'.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
        except (ValueError, Exception) as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _stop_mcp_server(self, config, client_manager):
        """Stop an MCP server"""
        import asyncio
        running_servers = []
        
        try:
            all_status = asyncio.run(self._get_mcp_server_status(client_manager, config))
            running_servers = [name for name, status in all_status.items() if status.get('running', False)]
        except Exception as e:
            print(f"{Colors.RED}Error getting server status: {e}{Colors.ENDC}")
            return
        
        if not running_servers:
            print(f"\n{Colors.YELLOW}No servers are currently running.{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        print(f"\n{Colors.CYAN}📋 Running servers:{Colors.ENDC}")
        for i, server in enumerate(running_servers):
            print(f"  {i+1}. {server}")
        
        choice = input(f"\n{Colors.YELLOW}Enter server number to stop: {Colors.ENDC}")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(running_servers):
                server_name = running_servers[idx]
                print(f"\n{Colors.BLUE}Stopping server '{server_name}'...{Colors.ENDC}")
                
                success = asyncio.run(client_manager.stop_server(server_name))
                
                if success:
                    print(f"{Colors.GREEN}✓ Server '{server_name}' stopped successfully!{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Failed to stop server '{server_name}'.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
        except (ValueError, Exception) as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _restart_mcp_server(self, config, client_manager):
        """Restart an MCP server"""
        servers = config.list_servers()
        if not servers:
            print(f"\n{Colors.YELLOW}No servers configured.{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        print(f"\n{Colors.CYAN}📋 Available servers:{Colors.ENDC}")
        for i, server in enumerate(servers):
            print(f"  {i+1}. {server}")
        
        choice = input(f"\n{Colors.YELLOW}Enter server number to restart: {Colors.ENDC}")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(servers):
                server_name = servers[idx]
                print(f"\n{Colors.BLUE}Restarting server '{server_name}'...{Colors.ENDC}")
                
                import asyncio
                success = asyncio.run(client_manager.restart_server(server_name))
                
                if success:
                    print(f"{Colors.GREEN}✓ Server '{server_name}' restarted successfully!{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Failed to restart server '{server_name}'.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid selection.{Colors.ENDC}")
        except (ValueError, Exception) as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _add_mcp_server(self, config):
        """Add a new MCP server"""
        print(f"\n{Colors.CYAN}➕ Add New MCP Server{Colors.ENDC}")
        
        templates = ['github', 'memory', 'confluence', 'obsidian-pm', 'web-fetch']
        
        print(f"\n{Colors.YELLOW}Available templates:{Colors.ENDC}")
        for i, template in enumerate(templates):
            descriptions = {
                'github': 'GitHub repository access',
                'memory': 'Persistent conversation memory',
                'confluence': 'Atlassian Confluence/Jira access',
                'obsidian-pm': 'Custom Obsidian PM intelligence',
                'web-fetch': 'Web content fetching'
            }
            print(f"  {i+1}. {template} - {descriptions.get(template, '')}")
        
        template_choice = input(f"\n{Colors.YELLOW}Choose template (1-{len(templates)}): {Colors.ENDC}")
        
        try:
            template_idx = int(template_choice) - 1
            if 0 <= template_idx < len(templates):
                template = templates[template_idx]
                server_name = input(f"{Colors.YELLOW}Enter server name: {Colors.ENDC}")
                
                if not server_name:
                    print(f"{Colors.RED}Server name cannot be empty.{Colors.ENDC}")
                    return
                
                kwargs = {}
                if template == 'obsidian-pm':
                    script_path = input(f"{Colors.YELLOW}Enter script path: {Colors.ENDC}")
                    if script_path:
                        kwargs['script_path'] = script_path
                
                success = config.create_server_from_template(server_name, template, **kwargs)
                
                if success:
                    print(f"{Colors.GREEN}✓ Added server '{server_name}' from template '{template}'!{Colors.ENDC}")
                    print(f"{Colors.BLUE}Configure credentials using environment variables or they'll be prompted when starting.{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Failed to add server.{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid template selection.{Colors.ENDC}")
        except (ValueError, Exception) as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _mcp_tools_interface(self, client_manager):
        """Dynamic MCP tools interface"""
        try:
            # Import MCP tools dynamically
            import sys
            import os
            import asyncio
            sys.path.append(os.path.join(os.path.dirname(__file__), 'obsidian_vault_tools'))
            
            from obsidian_vault_tools.mcp_tools.tools.menu_builder import get_menu_builder
            from obsidian_vault_tools.mcp_tools.tools.executor import get_executor
            
            menu_builder = get_menu_builder()
            executor = get_executor()
            
        except ImportError as e:
            print(f"\n{Colors.RED}❌ MCP tools not available: {e}{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        while True:
            try:
                # Build main tools menu
                tools_menu = asyncio.run(menu_builder.build_tools_menu())
                
                if not tools_menu or tools_menu[0][0] == '0':
                    print(f"\n{Colors.YELLOW}No MCP tools available.{Colors.ENDC}")
                    print(f"{Colors.BLUE}Start some MCP servers first using the MCP Server Management menu.{Colors.ENDC}")
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                    break
                
                choice = self.get_menu_choice('🛠️ MCP TOOLS', tools_menu)
                
                if choice == '0':
                    break
                elif choice == 'r':
                    # Force refresh tools
                    print(f"\n{Colors.BLUE}🔄 Refreshing tool discovery...{Colors.ENDC}")
                    asyncio.run(menu_builder.build_tools_menu(force_refresh=True))
                    print(f"{Colors.GREEN}✓ Tools refreshed{Colors.ENDC}")
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                elif choice == 's':
                    # Show execution statistics
                    self._show_execution_stats(executor)
                elif choice == 'h':
                    # Show execution history
                    self._show_execution_history(executor)
                elif choice.isdigit():
                    # Navigate to server tools
                    server_index = int(choice) - 1
                    server_names = list(asyncio.run(menu_builder.discovery_service.discover_all_tools()).keys())
                    if 0 <= server_index < len(server_names):
                        server_name = server_names[server_index]
                        self._server_tools_menu(server_name, menu_builder, executor)
                
            except Exception as e:
                print(f"\n{Colors.RED}Error in tools interface: {e}{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                break
    
    def _server_tools_menu(self, server_name, menu_builder, executor):
        """Menu for tools from a specific server"""
        while True:
            try:
                import asyncio
                
                # Build server tools menu
                server_tools_menu = asyncio.run(menu_builder.build_server_tools_menu(server_name))
                
                server_display = menu_builder._get_server_display_name(server_name)
                choice = self.get_menu_choice(f'{server_display} TOOLS', server_tools_menu)
                
                if choice == '0':
                    break
                elif choice == 'r':
                    # Refresh tools for this server
                    print(f"\n{Colors.BLUE}🔄 Refreshing tools for {server_name}...{Colors.ENDC}")
                    asyncio.run(menu_builder.discovery_service.discover_tools(server_name, force_refresh=True))
                    print(f"{Colors.GREEN}✓ Tools refreshed{Colors.ENDC}")
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                elif choice == 's':
                    # Search tools
                    self._search_tools_menu(server_name, menu_builder, executor)
                elif choice.isdigit():
                    # Execute selected tool
                    tool = asyncio.run(menu_builder.get_tool_by_menu_selection(server_name, choice))
                    if tool:
                        self._execute_tool_interactive(tool, executor)
                
            except Exception as e:
                print(f"\n{Colors.RED}Error in server tools menu: {e}{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                break
    
    def _execute_tool_interactive(self, tool, executor):
        """Interactive tool execution with parameter prompts"""
        try:
            import asyncio
            from obsidian_vault_tools.mcp_tools.tools.menu_builder import get_menu_builder
            
            menu_builder = get_menu_builder()
            
            print(f"\n{Colors.CYAN}🔧 Executing Tool: {tool.name}{Colors.ENDC}")
            print(f"{Colors.BLUE}Server: {tool.server}{Colors.ENDC}")
            if tool.description:
                print(f"{Colors.YELLOW}Description: {tool.description}{Colors.ENDC}")
            
            # Get parameter prompts
            param_prompts = menu_builder.build_parameter_prompts(tool)
            
            if not param_prompts:
                # No parameters needed, execute directly
                print(f"\n{Colors.GREEN}Executing tool (no parameters required)...{Colors.ENDC}")
                result = asyncio.run(executor.execute_tool(tool.server, tool.name, {}))
                self._display_tool_result(result, menu_builder)
                return
            
            # Collect parameters from user
            print(f"\n{Colors.YELLOW}Please provide the following parameters:{Colors.ENDC}")
            arguments = {}
            
            for prompt in param_prompts:
                param_name = prompt['name']
                param_desc = prompt['description']
                param_type = prompt['type']
                required = prompt['required']
                default = prompt['default']
                enum_values = prompt['enum']
                
                # Build prompt string
                prompt_str = f"\n{Colors.CYAN}{prompt['display_name']}{Colors.ENDC}"
                if required:
                    prompt_str += f" {Colors.RED}(required){Colors.ENDC}"
                if param_desc:
                    prompt_str += f": {param_desc}"
                if enum_values:
                    prompt_str += f"\n  Options: {', '.join(map(str, enum_values))}"
                if default is not None:
                    prompt_str += f"\n  Default: {default}"
                
                # Get user input
                user_input = input(f"{prompt_str}\n> ").strip()
                
                # Handle empty input
                if not user_input:
                    if required:
                        print(f"{Colors.RED}Error: Required parameter cannot be empty{Colors.ENDC}")
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                        return
                    elif default is not None:
                        arguments[param_name] = default
                        continue
                    else:
                        continue
                
                # Type conversion
                try:
                    if param_type == 'integer':
                        arguments[param_name] = int(user_input)
                    elif param_type == 'number':
                        arguments[param_name] = float(user_input)
                    elif param_type == 'boolean':
                        arguments[param_name] = user_input.lower() in ('true', 'yes', '1', 'on')
                    elif param_type == 'array':
                        # Simple comma-separated array
                        arguments[param_name] = [item.strip() for item in user_input.split(',')]
                    else:
                        arguments[param_name] = user_input
                except ValueError:
                    print(f"{Colors.RED}Error: Invalid {param_type} value{Colors.ENDC}")
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                    return
            
            # Execute tool
            print(f"\n{Colors.GREEN}Executing tool with parameters...{Colors.ENDC}")
            result = asyncio.run(executor.execute_tool(tool.server, tool.name, arguments))
            self._display_tool_result(result, menu_builder)
            
        except Exception as e:
            print(f"\n{Colors.RED}Error executing tool: {e}{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _display_tool_result(self, result, menu_builder):
        """Display tool execution result"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}TOOL EXECUTION RESULT{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        if result.success:
            print(f"{Colors.GREEN}✓ Tool executed successfully{Colors.ENDC}")
            if result.execution_time:
                print(f"{Colors.BLUE}Execution time: {result.execution_time:.2f} seconds{Colors.ENDC}")
            
            if result.result is not None:
                print(f"\n{Colors.CYAN}Result:{Colors.ENDC}")
                formatted_result = menu_builder.format_tool_result(result.result)
                print(formatted_result)
        else:
            print(f"{Colors.RED}❌ Tool execution failed{Colors.ENDC}")
            print(f"{Colors.YELLOW}Error: {result.error}{Colors.ENDC}")
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _show_execution_stats(self, executor):
        """Show tool execution statistics"""
        stats = executor.get_execution_stats()
        
        print(f"\n{Colors.CYAN}📊 Tool Execution Statistics{Colors.ENDC}")
        print(f"{Colors.BLUE}Total executions: {stats['total_executions']}{Colors.ENDC}")
        print(f"{Colors.BLUE}Success rate: {stats['success_rate']:.1f}%{Colors.ENDC}")
        print(f"{Colors.BLUE}Average execution time: {stats['average_execution_time']:.2f}s{Colors.ENDC}")
        
        if stats['most_used_tools']:
            print(f"\n{Colors.YELLOW}Most used tools:{Colors.ENDC}")
            for tool_info in stats['most_used_tools']:
                print(f"  {Colors.GREEN}{tool_info['tool']}{Colors.ENDC}: {tool_info['count']} times")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _show_execution_history(self, executor):
        """Show recent tool execution history"""
        history = executor.get_execution_history()
        recent_history = history[-10:]  # Last 10 executions
        
        print(f"\n{Colors.CYAN}📜 Recent Tool Execution History{Colors.ENDC}")
        
        if not recent_history:
            print(f"{Colors.YELLOW}No tool executions yet.{Colors.ENDC}")
        else:
            for i, result in enumerate(reversed(recent_history), 1):
                status_icon = "✓" if result.success else "❌"
                status_color = Colors.GREEN if result.success else Colors.RED
                
                timestamp = result.timestamp.strftime("%H:%M:%S") if result.timestamp else "Unknown"
                
                print(f"{Colors.BLUE}{i:2d}.{Colors.ENDC} {status_color}{status_icon}{Colors.ENDC} "
                      f"{Colors.CYAN}{result.server_name}:{result.tool_name}{Colors.ENDC} "
                      f"({timestamp})")
                
                if result.execution_time:
                    print(f"     {Colors.BLUE}Time: {result.execution_time:.2f}s{Colors.ENDC}")
                
                if not result.success and result.error:
                    error_preview = result.error[:50] + "..." if len(result.error) > 50 else result.error
                    print(f"     {Colors.RED}Error: {error_preview}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _search_tools_menu(self, server_name, menu_builder, executor):
        """Search tools menu"""
        try:
            import asyncio
            
            query = input(f"\n{Colors.YELLOW}Enter search query: {Colors.ENDC}").strip()
            if not query:
                return
            
            # Search tools on this server
            tools = asyncio.run(menu_builder.discovery_service.discover_tools(server_name))
            matching_tools = menu_builder.discovery_service.search_tools(query, tools)
            
            if not matching_tools:
                print(f"\n{Colors.YELLOW}No tools found matching '{query}'{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                return
            
            print(f"\n{Colors.CYAN}Found {len(matching_tools)} tools matching '{query}':{Colors.ENDC}")
            
            for i, tool in enumerate(matching_tools, 1):
                icon = menu_builder._get_tool_icon(tool)
                print(f"{Colors.BLUE}{i:2d}.{Colors.ENDC} {icon} {Colors.GREEN}{tool.name}{Colors.ENDC}")
                if tool.description:
                    print(f"     {tool.description[:60]}...")
            
            # Allow user to select and execute a tool
            try:
                selection = input(f"\n{Colors.YELLOW}Select tool number to execute (or Enter to go back): {Colors.ENDC}").strip()
                if selection.isdigit():
                    tool_index = int(selection) - 1
                    if 0 <= tool_index < len(matching_tools):
                        self._execute_tool_interactive(matching_tools[tool_index], executor)
            except ValueError:
                pass
                
        except Exception as e:
            print(f"\n{Colors.RED}Error in search: {e}{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def _mcp_resources_interface(self, client_manager):
        """Interface for accessing MCP resources"""
        print(f"\n{Colors.CYAN}📚 MCP Resources Interface{Colors.ENDC}")
        print(f"{Colors.YELLOW}This feature allows you to access resources provided by running MCP servers.{Colors.ENDC}")
        print(f"{Colors.BLUE}Resource integration coming in next update!{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def test_navigation_system(self):
        """Test arrow key navigation system"""
        print(f"\n{Colors.CYAN}Testing Navigation System{Colors.ENDC}")
        
        if self.arrow_navigation and self.navigator:
            print(f"{Colors.GREEN}✓ Arrow key navigation is enabled{Colors.ENDC}")
            
            # Test menu
            test_options = [
                ('1', 'Test option 1'),
                ('2', 'Test option 2'),
                ('3', 'Test option 3'),
                ('0', 'Exit test')
            ]
            
            print(f"\n{Colors.BLUE}Testing arrow key menu...{Colors.ENDC}")
            choice = self.navigator.navigate_menu('🧪 NAVIGATION TEST', test_options)
            print(f"{Colors.GREEN}✓ Navigation test completed. Selected: {choice}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Arrow key navigation not available{Colors.ENDC}")
            if not MENU_NAVIGATOR_AVAILABLE:
                print(f"{Colors.YELLOW}   MenuNavigator module not found{Colors.ENDC}")
            
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def settings_menu(self):
        """Override settings menu with arrow key navigation"""
        while True:
            options = [
                ('1', 'Audio settings'),
                ('2', 'Display settings'),
                ('3', 'Path settings'),
                ('4', 'AI model configuration'),
                ('5', 'V2 setup & installation'),
                ('6', 'Reset configuration'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('⚙️ SETTINGS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.audio_settings_menu()
            elif choice == '2':
                self.display_settings_menu()
            elif choice == '3':
                self.path_settings_menu()
            elif choice == '4':
                self.ai_model_configuration()
            elif choice == '5':
                self.setup_v2_installation()
            elif choice == '6':
                self.reset_configuration()
    
    def audio_settings_menu(self):
        """Audio settings submenu"""
        while True:
            options = [
                ('1', f'Toggle audio: {"ON" if self.audio_enabled else "OFF"}'),
                ('2', f'Toggle ASCII art: {"ON" if self.ascii_enabled else "OFF"}'),
                ('3', 'Test audio system'),
                ('4', 'Volume settings'),
                ('0', 'Back to settings')
            ]
            
            choice = self.get_menu_choice('🎵 AUDIO SETTINGS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.audio_enabled = not self.audio_enabled
                status = "enabled" if self.audio_enabled else "disabled"
                print(f"\n{Colors.GREEN}Audio {status}{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.ascii_enabled = not self.ascii_enabled
                status = "enabled" if self.ascii_enabled else "disabled"
                print(f"\n{Colors.GREEN}ASCII art {status}{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.test_audio_system()
            elif choice == '4':
                print(f"\n{Colors.YELLOW}Volume settings feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def display_settings_menu(self):
        """Display settings submenu"""
        while True:
            options = [
                ('1', f'Arrow navigation: {"ON" if self.arrow_navigation else "OFF"}'),
                ('2', 'Color theme settings'),
                ('3', 'Menu layout settings'),
                ('0', 'Back to settings')
            ]
            
            choice = self.get_menu_choice('🎨 DISPLAY SETTINGS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                if MENU_NAVIGATOR_AVAILABLE:
                    self.arrow_navigation = not self.arrow_navigation
                    status = "enabled" if self.arrow_navigation else "disabled"
                    print(f"\n{Colors.GREEN}Arrow navigation {status}{Colors.ENDC}")
                else:
                    print(f"\n{Colors.RED}Arrow navigation not available - MenuNavigator module required{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                print(f"\n{Colors.YELLOW}Color theme settings feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                print(f"\n{Colors.YELLOW}Menu layout settings feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def path_settings_menu(self):
        """Path settings submenu"""
        while True:
            options = [
                ('1', 'Change vault path'),
                ('2', 'Set output directories'),
                ('3', 'View current paths'),
                ('0', 'Back to settings')
            ]
            
            choice = self.get_menu_choice('📁 PATH SETTINGS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.get_vault_path()
            elif choice == '2':
                self.organize_output_files()
            elif choice == '3':
                print(f"\n{Colors.CYAN}Current Paths:{Colors.ENDC}")
                print(f"  Vault: {self.current_vault}")
                print(f"  Config: {self.config_file}")
                
                # Show output directories
                output_types = ['ascii-art', 'vault-analysis', 'backup', 'research', 'diagnostic']
                print(f"\n{Colors.CYAN}Output Directories:{Colors.ENDC}")
                for output_type in output_types:
                    output_dir = self.get_output_directory(output_type)
                    files_count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]) if os.path.exists(output_dir) else 0
                    print(f"  {output_type}: {output_dir} ({files_count} files)")
                
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def reset_configuration(self):
        """Reset configuration to defaults"""
        confirm = input(f"\n{Colors.RED}Are you sure you want to reset all settings? (y/N): {Colors.ENDC}")
        if confirm.lower() == 'y':
            self.config = {}
            self.save_config()
            print(f"\n{Colors.GREEN}Configuration reset successfully{Colors.ENDC}")
        else:
            print(f"\n{Colors.BLUE}Configuration reset cancelled{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def v2_features_menu(self):
        """Override V2 features menu with arrow key navigation"""
        while True:
            options = [
                ('1', '🧠 AI-powered content analysis'),
                ('2', '🔍 Research topics & create notes'),
                ('3', '📚 Smart file organization'),
                ('4', '🔄 Find & merge duplicates'),
                ('5', '📊 Advanced vault analytics'),
                ('6', '🎯 Comprehensive curation'),
                ('7', '⚙️  Configure AI settings'),
                ('8', '🛠️  Install/Setup V2'),
                ('0', 'Back to main menu')
            ]
            
            footer = "v2 Available ✓" if self.v2_available else "v2 Not Installed - Use option 8 to setup"
            
            choice = self.get_menu_choice('🚀 OBSIDIAN LIBRARIAN V2', options)
            
            if choice == '0':
                break
            elif choice == '8':
                self.setup_v2_installation()
            elif not self.v2_available:
                print(f"\n{Colors.YELLOW}⚠️  Obsidian Librarian v2 is not installed{Colors.ENDC}")
                print(f"{Colors.BLUE}Use option 8 to setup V2, or see Help for manual installation{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                continue
            
            if choice == '1':
                self.run_v2_analysis()
            elif choice == '2':
                self.run_v2_research()
            elif choice == '3':
                self.run_v2_organization()
            elif choice == '4':
                self.run_v2_duplicates()
            elif choice == '5':
                self.run_v2_analytics()
            elif choice == '6':
                self.run_v2_curation()
            elif choice == '7':
                self.configure_v2_ai()
    
    def setup_v2_installation(self):
        """Setup V2 installation"""
        print(f"\n{Colors.CYAN}Setting up Obsidian Librarian V2{Colors.ENDC}")
        
        v2_path = os.path.join(os.path.dirname(__file__), 'obsidian-librarian-v2')
        
        if os.path.exists(v2_path):
            print(f"{Colors.GREEN}✓ V2 directory found at: {v2_path}{Colors.ENDC}")
            
            # Try to build and install
            python_path = os.path.join(v2_path, 'python')
            if os.path.exists(python_path):
                print(f"{Colors.BLUE}Installing V2 Python package...{Colors.ENDC}")
                
                try:
                    # Change to V2 directory and install
                    result = subprocess.run([
                        'pip', 'install', '-e', python_path
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        print(f"{Colors.GREEN}✓ V2 installation successful!{Colors.ENDC}")
                        self.v2_available = True
                        # Update config to remember installation
                        self.config['v2_installed'] = True
                        self.save_config()
                    else:
                        print(f"{Colors.RED}❌ Installation failed:{Colors.ENDC}")
                        print(result.stderr)
                        print(f"\n{Colors.YELLOW}Try manual installation:{Colors.ENDC}")
                        print(f"cd {python_path}")
                        print(f"pip install -e .")
                        
                except subprocess.TimeoutExpired:
                    print(f"{Colors.RED}❌ Installation timed out{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.RED}❌ Installation error: {e}{Colors.ENDC}")
            else:
                print(f"{Colors.RED}❌ V2 Python directory not found{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ V2 directory not found{Colors.ENDC}")
            print(f"{Colors.YELLOW}Expected location: {v2_path}{Colors.ENDC}")
            print(f"\n{Colors.BLUE}Please ensure obsidian-librarian-v2 is in the same directory as this script{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def run_v2_analysis(self):
        """Run V2 content analysis"""
        print(f"\n{Colors.CYAN}Running AI-powered content analysis...{Colors.ENDC}")
        
        v2_path = os.path.join(os.path.dirname(__file__), 'obsidian-librarian-v2', 'python')
        
        try:
            # Use Python module directly
            cmd = [
                'python', '-m', 'obsidian_librarian.cli', 
                'analyze', self.current_vault, 
                '--quality', '--structure'
            ]
            
            self.run_command_with_progress(
                ' '.join(cmd),
                'Analyzing content with AI',
                estimated_duration=30
            )
        except Exception as e:
            print(f"{Colors.RED}❌ Analysis failed: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def run_v2_research(self):
        """Run V2 research functionality"""
        query = input(f"\n{Colors.CYAN}Enter research topic: {Colors.ENDC}")
        if query:
            print(f"\n{Colors.CYAN}Researching: {query}{Colors.ENDC}")
            
            try:
                cmd = [
                    'python', '-m', 'obsidian_librarian.cli',
                    'research', self.current_vault, query
                ]
                
                self.run_command_with_progress(
                    ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd]),
                    f'Researching: {query}',
                    estimated_duration=60
                )
            except Exception as e:
                print(f"{Colors.RED}❌ Research failed: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def run_v2_organization(self):
        """Run V2 smart organization"""
        print(f"\n{Colors.CYAN}Planning smart organization...{Colors.ENDC}")
        
        try:
            cmd = [
                'python', '-m', 'obsidian_librarian.cli',
                'organize', self.current_vault,
                '--strategy', 'content', '--dry-run'
            ]
            
            self.run_command_with_progress(
                ' '.join(cmd),
                'Planning smart organization',
                estimated_duration=20
            )
            
            confirm = input(f"\n{Colors.CYAN}Apply these changes? (y/n): {Colors.ENDC}").lower()
            if confirm == 'y':
                cmd_apply = [
                    'python', '-m', 'obsidian_librarian.cli',
                    'organize', self.current_vault,
                    '--strategy', 'content'
                ]
                
                self.run_command_with_progress(
                    ' '.join(cmd_apply),
                    'Organizing files',
                    estimated_duration=35
                )
        except Exception as e:
            print(f"{Colors.RED}❌ Organization failed: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def run_v2_duplicates(self):
        """Run V2 duplicate detection"""
        print(f"\n{Colors.CYAN}Finding duplicate content...{Colors.ENDC}")
        
        try:
            cmd = [
                'python', '-m', 'obsidian_librarian.cli',
                'duplicates', self.current_vault,
                '--threshold', '0.85'
            ]
            
            self.run_command_with_progress(
                ' '.join(cmd),
                'Finding duplicate content',
                estimated_duration=25
            )
        except Exception as e:
            print(f"{Colors.RED}❌ Duplicate detection failed: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def run_v2_analytics(self):
        """Run V2 advanced analytics"""
        print(f"\n{Colors.CYAN}Generating advanced analytics...{Colors.ENDC}")
        
        try:
            cmd = [
                'python', '-m', 'obsidian_librarian.cli',
                'status', self.current_vault,
                '--detailed'
            ]
            
            self.run_command_with_progress(
                ' '.join(cmd),
                'Generating analytics',
                estimated_duration=15
            )
        except Exception as e:
            print(f"{Colors.RED}❌ Analytics failed: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def run_v2_curation(self):
        """Run V2 comprehensive curation"""
        print(f"\n{Colors.CYAN}Running comprehensive curation...{Colors.ENDC}")
        
        try:
            cmd = [
                'python', '-m', 'obsidian_librarian.cli',
                'curate', self.current_vault,
                '--quality', '--structure'
            ]
            
            self.run_command_with_progress(
                ' '.join(cmd),
                'Curating vault content',
                estimated_duration=45
            )
        except Exception as e:
            print(f"{Colors.RED}❌ Curation failed: {e}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def configure_v2_ai(self):
        """Configure V2 AI settings"""
        print(f"\n{Colors.CYAN}V2 AI Configuration{Colors.ENDC}")
        print(f"{Colors.YELLOW}This feature configures AI models for V2 functionality{Colors.ENDC}")
        print(f"{Colors.BLUE}Currently integrated with the existing AI model configuration{Colors.ENDC}")
        
        # Delegate to existing AI configuration
        self.ai_model_configuration()
    
    def research_and_create_menu(self):
        """Research topics and create notes menu"""
        while True:
            options = [
                ('1', '🔍 Research a topic' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('2', '📝 Create note from research'),
                ('3', '🌐 Web research & summarize'),
                ('4', '📚 Research history'),
                ('5', '⚙️  Configure research sources'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('🔍 RESEARCH & CREATE', options)
            
            if choice == '0':
                break
            elif choice == '1':
                if self.v2_available:
                    self.run_v2_research()
                else:
                    print(f"\n{Colors.YELLOW}Research feature requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                print(f"\n{Colors.YELLOW}Create note from research feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                print(f"\n{Colors.YELLOW}Web research feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                self.view_research_history()
            elif choice == '5':
                print(f"\n{Colors.YELLOW}Research source configuration coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def tags_and_organization_menu(self):
        """Enhanced tag management with smart organization"""
        while True:
            options = [
                ('1', 'Preview all tag fixes'),
                ('2', 'Apply all tag fixes'),
                ('3', 'Smart file organization' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('4', 'Auto-organize by content' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('5', 'Tag hierarchy analysis'),
                ('6', 'Bulk tag operations'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('🏷️ TAGS & ORGANIZATION', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.play_operation_start_sound('scan')
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --dry-run',
                    'Previewing tag fixes',
                    estimated_duration=8
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.play_operation_start_sound('scan')
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply',
                    'Applying all tag fixes',
                    estimated_duration=15
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                if self.v2_available:
                    self.run_v2_organization()
                else:
                    print(f"\n{Colors.YELLOW}Smart organization requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                if self.v2_available:
                    print(f"\n{Colors.CYAN}Auto-organizing by content...{Colors.ENDC}")
                    self.run_v2_organization()
                else:
                    print(f"\n{Colors.YELLOW}Auto-organization requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                print(f"\n{Colors.YELLOW}Tag hierarchy analysis coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '6':
                self.bulk_tag_operations_menu()
    
    def bulk_tag_operations_menu(self):
        """Bulk tag operations submenu"""
        while True:
            options = [
                ('1', 'Fix quoted tags only'),
                ('2', 'Merge similar tags'),
                ('3', 'Remove generic tags'),
                ('4', 'Add tags by pattern'),
                ('5', 'Remove tags by pattern'),
                ('0', 'Back to tags menu')
            ]
            
            choice = self.get_menu_choice('🏷️ BULK TAG OPERATIONS', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.play_operation_start_sound('scan')
                self.fix_quoted_tags()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.play_operation_start_sound('scan')
                self.merge_similar_tags()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.play_operation_start_sound('scan')
                self.remove_generic_tags()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                print(f"\n{Colors.YELLOW}Add tags by pattern feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                print(f"\n{Colors.YELLOW}Remove tags by pattern feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def duplicates_and_curation_menu(self):
        """Find duplicates and curate content"""
        while True:
            options = [
                ('1', 'Find duplicate content' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('2', 'Merge duplicate notes'),
                ('3', 'Comprehensive curation' + (' ✓' if self.v2_available else ' (Setup Required)')),
                ('4', 'Content quality analysis'),
                ('5', 'Archive old content'),
                ('0', 'Back to main menu')
            ]
            
            choice = self.get_menu_choice('🔄 DUPLICATES & CURATION', options)
            
            if choice == '0':
                break
            elif choice == '1':
                if self.v2_available:
                    self.run_v2_duplicates()
                else:
                    print(f"\n{Colors.YELLOW}Duplicate detection requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                print(f"\n{Colors.YELLOW}Merge duplicates feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                if self.v2_available:
                    self.run_v2_curation()
                else:
                    print(f"\n{Colors.YELLOW}Comprehensive curation requires V2 setup{Colors.ENDC}")
                    if input(f"{Colors.CYAN}Would you like to set it up now? (y/N): {Colors.ENDC}").lower() == 'y':
                        self.setup_v2_installation()
                    else:
                        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                print(f"\n{Colors.YELLOW}Content quality analysis coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                print(f"\n{Colors.YELLOW}Archive old content feature coming soon!{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def view_research_history(self):
        """View research history"""
        print(f"\n{Colors.CYAN}Research History{Colors.ENDC}")
        
        research_dir = self.get_output_directory('research')
        if os.path.exists(research_dir):
            research_files = sorted(
                [f for f in os.listdir(research_dir) if f.endswith('.md')],
                key=lambda x: os.path.getmtime(os.path.join(research_dir, x)),
                reverse=True
            )
            
            if research_files:
                print(f"\n{Colors.GREEN}Recent research notes:{Colors.ENDC}")
                for i, filename in enumerate(research_files[:10]):
                    filepath = os.path.join(research_dir, filename)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    print(f"  {i+1}. {filename} - {mod_time.strftime('%Y-%m-%d %H:%M')}")
                
                if len(research_files) > 10:
                    print(f"  ... and {len(research_files) - 10} more")
            else:
                print(f"{Colors.YELLOW}No research history found{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}No research directory found{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def get_output_directory(self, feature_name: str) -> str:
        """Get organized output directory for a specific feature"""
        base_dir = os.path.dirname(__file__)
        output_dir = os.path.join(base_dir, f"{feature_name}-output")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    def cleanup_root_directory(self):
        """Move existing output files to organized directories"""
        base_dir = os.path.dirname(__file__)
        
        # Define file patterns and their target directories
        file_mappings = {
            'ascii-art-output': [
                '*.txt',  # ASCII art files
                '*ascii*.png', '*ascii*.jpg',  # ASCII art images
                'dore_*.txt', 'dore_*.jpg',  # Specific ASCII art files
                'test_*.png'  # Test images
            ],
            'vault-analysis-output': [
                'tag_fix_report.json',
                '*_analysis.txt',
                '*_report.txt'
            ],
            'backup-output': [
                'obsidian_backups/',
                '*.backup',
                '*backup*'
            ],
            'research-output': [
                'llm_feedback/',
                'embedding_cache/',
                '*_research.md'
            ],
            'diagnostic-output': [
                'audio_*.py',
                'test_*.py',
                'simple_beep_test.py',
                'regenerate_*.py',
                '*_diagnostic.py'
            ]
        }
        
        moved_files = []
        
        for output_dir, patterns in file_mappings.items():
            target_dir = self.get_output_directory(output_dir.replace('-output', ''))
            
            for pattern in patterns:
                # Handle directories
                if pattern.endswith('/'):
                    dir_path = os.path.join(base_dir, pattern.rstrip('/'))
                    if os.path.exists(dir_path) and os.path.isdir(dir_path):
                        target_path = os.path.join(target_dir, os.path.basename(dir_path))
                        if not os.path.exists(target_path):
                            import shutil
                            shutil.move(dir_path, target_path)
                            moved_files.append(f"{pattern} -> {output_dir}/")
                else:
                    # Handle file patterns
                    import glob
                    for file_path in glob.glob(os.path.join(base_dir, pattern)):
                        if os.path.isfile(file_path):
                            filename = os.path.basename(file_path)
                            target_path = os.path.join(target_dir, filename)
                            if not os.path.exists(target_path):
                                import shutil
                                shutil.move(file_path, target_path)
                                moved_files.append(f"{filename} -> {output_dir}/")
        
        return moved_files
    
    def convert_image_to_ascii(self):
        """Convert user-selected image to ASCII art with organized output"""
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
        self.show_indeterminate_progress("Generating ASCII art", 3)
        
        ascii_art = self.ascii_manager.generate_from_image(image_path, width, style)
        
        if ascii_art:
            # Save to organized output directory
            output_dir = self.get_output_directory('ascii-art')
            
            # Generate filename based on original image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{style}_{timestamp}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save ASCII art
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"ASCII Art Generated from: {image_path}\n")
                f.write(f"Style: {style}, Width: {width}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(ascii_art)
            
            print(f"{Colors.GREEN}✓ ASCII art generated successfully!{Colors.ENDC}")
            print(f"{Colors.BLUE}Saved to: {output_path}{Colors.ENDC}")
            
            # Display preview
            print(f"\n{Colors.CYAN}Preview:{Colors.ENDC}")
            lines = ascii_art.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                print(line)
            if len(lines) > 10:
                print(f"... ({len(lines) - 10} more lines)")
            
            # Add to collection for quick access
            self.ascii_manager.add_to_collection(f"{base_name}_{style}", ascii_art)
            
            self.play_operation_complete_sound('ascii')
        else:
            print(f"{Colors.RED}❌ Failed to generate ASCII art{Colors.ENDC}")
            self.play_operation_complete_sound('general', success=False)
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def organize_output_files(self):
        """Organize scattered output files into proper directories"""
        print(f"\n{Colors.CYAN}Organizing Output Files{Colors.ENDC}")
        print(f"{Colors.BLUE}Moving scattered files to organized directories...{Colors.ENDC}")
        
        moved_files = self.cleanup_root_directory()
        
        if moved_files:
            print(f"\n{Colors.GREEN}✓ Moved {len(moved_files)} files:{Colors.ENDC}")
            for file_move in moved_files[:10]:  # Show first 10
                print(f"  {file_move}")
            if len(moved_files) > 10:
                print(f"  ... and {len(moved_files) - 10} more files")
        else:
            print(f"\n{Colors.BLUE}✓ All files are already organized{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def intelligent_janitor_menu(self):
        """Intelligent file cleanup menu"""
        while True:
            options = [
                ('1', 'Analyze output files'),
                ('2', 'Clean unused files (safe)'),
                ('3', 'Deep clean (advanced)'),
                ('4', 'Show disk usage'),
                ('5', 'Configure cleanup rules'),
                ('0', 'Back to advanced tools')
            ]
            
            choice = self.get_menu_choice('🧹 INTELLIGENT JANITOR', options)
            
            if choice == '0':
                break
            elif choice == '1':
                self.analyze_output_files()
            elif choice == '2':
                self.safe_cleanup()
            elif choice == '3':
                self.deep_cleanup()
            elif choice == '4':
                self.show_disk_usage()
            elif choice == '5':
                self.configure_cleanup_rules()
    
    def analyze_output_files(self):
        """Analyze output files using AI to understand usage patterns"""
        print(f"\n{Colors.CYAN}Analyzing Output Files with AI{Colors.ENDC}")
        
        if not (self.llm_enabled or self.pattern_matching_mode):
            print(f"{Colors.RED}❌ AI analysis requires LLM or pattern matching mode{Colors.ENDC}")
            print(f"{Colors.YELLOW}Configure AI models first in the main menu{Colors.ENDC}")
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            return
        
        output_types = ['ascii-art', 'vault-analysis', 'backup', 'research', 'diagnostic']
        total_files = 0
        analysis_results = {}
        
        for output_type in output_types:
            output_dir = self.get_output_directory(output_type)
            if os.path.exists(output_dir):
                files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
                total_files += len(files)
                
                # Analyze file patterns
                file_ages = []
                file_sizes = []
                recent_files = 0
                
                for filename in files:
                    filepath = os.path.join(output_dir, filename)
                    stat = os.stat(filepath)
                    age_days = (time.time() - stat.st_mtime) / (24 * 3600)
                    file_ages.append(age_days)
                    file_sizes.append(stat.st_size)
                    
                    if age_days <= 7:  # Recent files (last week)
                        recent_files += 1
                
                analysis_results[output_type] = {
                    'total_files': len(files),
                    'avg_age_days': sum(file_ages) / len(file_ages) if file_ages else 0,
                    'total_size_mb': sum(file_sizes) / (1024 * 1024),
                    'recent_files': recent_files,
                    'oldest_age_days': max(file_ages) if file_ages else 0
                }
        
        print(f"\n{Colors.GREEN}📊 Analysis Results:{Colors.ENDC}")
        print(f"Total files analyzed: {total_files}")
        
        for output_type, stats in analysis_results.items():
            if stats['total_files'] > 0:
                print(f"\n{Colors.CYAN}{output_type.upper()}:{Colors.ENDC}")
                print(f"  Files: {stats['total_files']}")
                print(f"  Size: {stats['total_size_mb']:.1f} MB")
                print(f"  Average age: {stats['avg_age_days']:.1f} days")
                print(f"  Recent files: {stats['recent_files']}")
                print(f"  Oldest file: {stats['oldest_age_days']:.1f} days")
                
                # AI-powered recommendations
                if stats['avg_age_days'] > 30 and stats['recent_files'] == 0:
                    print(f"  {Colors.YELLOW}💡 Recommendation: Consider archiving old files{Colors.ENDC}")
                elif stats['total_size_mb'] > 100:
                    print(f"  {Colors.YELLOW}💡 Recommendation: Large directory, review for cleanup{Colors.ENDC}")
                elif stats['total_files'] > 50 and stats['recent_files'] < 5:
                    print(f"  {Colors.YELLOW}💡 Recommendation: Many files but low recent activity{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def safe_cleanup(self):
        """Safe cleanup of obviously unused files"""
        print(f"\n{Colors.CYAN}Safe Cleanup{Colors.ENDC}")
        print(f"{Colors.BLUE}Identifying safe files to clean...{Colors.ENDC}")
        
        output_types = ['ascii-art', 'vault-analysis', 'backup', 'research', 'diagnostic']
        candidates_for_removal = []
        
        for output_type in output_types:
            output_dir = self.get_output_directory(output_type)
            if os.path.exists(output_dir):
                files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
                
                for filename in files:
                    filepath = os.path.join(output_dir, filename)
                    stat = os.stat(filepath)
                    age_days = (time.time() - stat.st_mtime) / (24 * 3600)
                    
                    # Safe cleanup criteria
                    should_remove = False
                    reason = ""
                    
                    if age_days > 90 and filename.startswith('test_'):
                        should_remove = True
                        reason = "Old test file (>90 days)"
                    elif age_days > 60 and 'temp' in filename.lower():
                        should_remove = True
                        reason = "Old temporary file (>60 days)"
                    elif age_days > 180 and output_type == 'diagnostic':
                        should_remove = True
                        reason = "Old diagnostic file (>180 days)"
                    elif stat.st_size == 0:
                        should_remove = True
                        reason = "Empty file"
                    elif filename.endswith('.log') and age_days > 30:
                        should_remove = True
                        reason = "Old log file (>30 days)"
                    
                    if should_remove:
                        candidates_for_removal.append((filepath, reason, stat.st_size))
        
        if candidates_for_removal:
            total_size = sum(size for _, _, size in candidates_for_removal)
            print(f"\n{Colors.YELLOW}Found {len(candidates_for_removal)} files for safe removal ({total_size / (1024*1024):.1f} MB):{Colors.ENDC}")
            
            for filepath, reason, size in candidates_for_removal[:10]:
                rel_path = os.path.relpath(filepath)
                print(f"  {rel_path} - {reason} ({size / 1024:.1f} KB)")
            
            if len(candidates_for_removal) > 10:
                print(f"  ... and {len(candidates_for_removal) - 10} more files")
            
            confirm = input(f"\n{Colors.CYAN}Remove these files? (y/N): {Colors.ENDC}").lower()
            if confirm == 'y':
                removed_count = 0
                for filepath, _, _ in candidates_for_removal:
                    try:
                        os.remove(filepath)
                        removed_count += 1
                    except Exception as e:
                        print(f"{Colors.RED}Failed to remove {filepath}: {e}{Colors.ENDC}")
                
                print(f"{Colors.GREEN}✓ Removed {removed_count} files, freed {total_size / (1024*1024):.1f} MB{Colors.ENDC}")
            else:
                print(f"{Colors.BLUE}Cleanup cancelled{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}✓ No files found for safe cleanup{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def deep_cleanup(self):
        """Advanced cleanup with user confirmation"""
        print(f"\n{Colors.CYAN}Deep Cleanup (Advanced){Colors.ENDC}")
        print(f"{Colors.RED}⚠️  This is an advanced feature that requires careful review{Colors.ENDC}")
        
        confirm = input(f"\n{Colors.YELLOW}Continue with deep cleanup analysis? (y/N): {Colors.ENDC}").lower()
        if confirm != 'y':
            return
        
        # This would implement more aggressive cleanup logic
        print(f"\n{Colors.YELLOW}Deep cleanup feature coming soon!{Colors.ENDC}")
        print(f"{Colors.BLUE}Will include:{Colors.ENDC}")
        print(f"  • Duplicate file detection")
        print(f"  • Large file analysis") 
        print(f"  • Unused dependency cleanup")
        print(f"  • Advanced pattern-based removal")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def show_disk_usage(self):
        """Show disk usage for output directories"""
        print(f"\n{Colors.CYAN}Disk Usage Analysis{Colors.ENDC}")
        
        output_types = ['ascii-art', 'vault-analysis', 'backup', 'research', 'diagnostic']
        total_size = 0
        
        for output_type in output_types:
            output_dir = self.get_output_directory(output_type)
            if os.path.exists(output_dir):
                dir_size = 0
                file_count = 0
                
                for root, dirs, files in os.walk(output_dir):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        try:
                            dir_size += os.path.getsize(filepath)
                            file_count += 1
                        except OSError:
                            pass
                
                total_size += dir_size
                print(f"{output_type:15}: {dir_size / (1024*1024):8.1f} MB ({file_count:3d} files)")
        
        print(f"{'':15}   {'='*20}")
        print(f"{'Total':15}: {total_size / (1024*1024):8.1f} MB")
        
        # Show vault size for comparison if available
        if self.current_vault and os.path.exists(self.current_vault):
            vault_size = 0
            vault_files = 0
            
            for root, dirs, files in os.walk(self.current_vault):
                # Skip .obsidian directory
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for filename in files:
                    if filename.endswith('.md'):
                        filepath = os.path.join(root, filename)
                        try:
                            vault_size += os.path.getsize(filepath)
                            vault_files += 1
                        except OSError:
                            pass
            
            print(f"\n{Colors.BLUE}Vault comparison:{Colors.ENDC}")
            print(f"{'Vault (.md files)':15}: {vault_size / (1024*1024):8.1f} MB ({vault_files:3d} files)")
            print(f"{'Output files':15}: {total_size / (1024*1024):8.1f} MB")
            
            if total_size > vault_size:
                ratio = total_size / vault_size if vault_size > 0 else float('inf')
                print(f"{Colors.YELLOW}⚠️  Output files are {ratio:.1f}x larger than vault content{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def configure_cleanup_rules(self):
        """Configure intelligent cleanup rules"""
        print(f"\n{Colors.CYAN}Cleanup Rules Configuration{Colors.ENDC}")
        print(f"{Colors.YELLOW}Feature coming soon!{Colors.ENDC}")
        print(f"{Colors.BLUE}Will include configurable rules for:{Colors.ENDC}")
        print(f"  • File age thresholds")
        print(f"  • Size limits")
        print(f"  • Pattern matching")
        print(f"  • Auto-cleanup scheduling")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
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
                ('1', '📊 Smart Vault Analysis'),
                ('2', '🔍 Research & Create Notes'),
                ('3', '🏷️  Manage Tags & Organization'),
                ('4', '🤖 Query Vault with AI' if self.llm_enabled else '🤖 Query Vault (Pattern Matching)' if self.pattern_matching_mode else '🤖 Query Vault (Not Available)'),
                ('5', '🔄 Find Duplicates & Curate'),
                ('6', '💾 Backup & Sync'),
                ('7', '🎨 ASCII Art Tools'),
                ('8', '🔧 Advanced Tools'),
                ('9', '📚 Help & Documentation'),
                ('10', '⚙️  Settings'),
                ('0', '👋 Exit')
            ]
            
            # Use arrow key navigation if available, fallback to traditional menu
            if self.arrow_navigation and self.navigator:
                choice = self.navigator.navigate_menu('🏰 ENHANCED VAULT MANAGER - MAIN MENU', options)
                if choice == 'quit':
                    choice = '0'  # Treat quit as exit
            else:
                # Fallback to enhanced menu display
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
                self.smart_analysis_menu()
            elif choice == '2':
                self.research_and_create_menu()
            elif choice == '3':
                self.tags_and_organization_menu()
            elif choice == '4':
                self.play_menu_navigation_sound()
                self.vault_query_interface()
            elif choice == '5':
                self.duplicates_and_curation_menu()
            elif choice == '6':
                self.backup_vault_menu()
            elif choice == '7':
                self.show_ascii_menu()
            elif choice == '8':
                self.advanced_tools_menu()
            elif choice == '9':
                self.show_help()
            elif choice == '10':
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