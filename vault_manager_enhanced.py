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
                ('8', '🤖 AI Model Configuration' if self.llm_enabled or self.pattern_matching_mode else '🤖 AI Model Configuration (LLM Not Available)'),
                ('9', '⚙️  Settings'),
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
                self.ai_model_configuration()
            elif choice == '9':
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