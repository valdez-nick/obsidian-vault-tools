#!/usr/bin/env python3
"""
Demo of Enhanced Vault Manager with ASCII Art
Shows the new ASCII art integration features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vault_manager_enhanced import EnhancedVaultManager, Colors, ASCIIArtManager

def demo():
    """Show demo of enhanced features"""
    
    # Create managers
    manager = EnhancedVaultManager()
    ascii_mgr = ASCIIArtManager()
    
    # Show enhanced welcome
    print("=== ENHANCED WELCOME SCREEN ===\n")
    manager.show_welcome_art()
    
    print("\n\n=== ASCII ART GALLERY ===\n")
    
    # Show some ASCII art from the collection
    for art_name in ['books', 'portal', 'thinking', 'success']:
        print(f"{Colors.CYAN}--- {art_name.upper()} ---{Colors.ENDC}")
        print(f"{Colors.GREEN}{ascii_mgr.get_art(art_name)}{Colors.ENDC}\n")
    
    print("\n\n=== NEW ASCII ART MENU ===\n")
    
    options = [
        ('1', '🎨 Convert image to ASCII art'),
        ('2', '📷 Convert screenshot to ASCII'),
        ('3', '🖼️  View ASCII art gallery'),
        ('4', '⚙️  ASCII art settings'),
        ('5', '📚 Add art to vault notes'),
        ('0', '← Back to main menu')
    ]
    
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'ASCII ART TOOLS'.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")
    
    for key, desc in options:
        print(f"  {Colors.GREEN}{key}{Colors.ENDC}. {desc}")
    
    print(f"\n{Colors.YELLOW}Enter your choice: {Colors.ENDC}")
    
    print("\n\n=== EXAMPLE: ASCII ART FROM DORÉ'S LUCIFER ===\n")
    
    # Show a sample of the converted Doré image
    if os.path.exists('dore_lucifer_ascii_magic.txt'):
        with open('dore_lucifer_ascii_magic.txt', 'r') as f:
            ascii_art = f.read()
            # Show first 20 lines
            lines = ascii_art.split('\n')[:20]
            for line in lines:
                print(line[:120])  # Limit width for demo
            print("... [truncated for demo]")
    else:
        print(f"{Colors.YELLOW}Run the converter first to see Doré's Lucifer in ASCII!{Colors.ENDC}")
        print(f"{Colors.BLUE}python3 ascii_magic_converter.py dore_lucifer.jpg --width 120{Colors.ENDC}")
    
    print(f"\n\n{Colors.GREEN}✨ The Enhanced Vault Manager now includes:{Colors.ENDC}")
    print("• ASCII art throughout the interface")
    print("• Image to ASCII conversion")
    print("• Screenshot to ASCII conversion")
    print("• ASCII art gallery")
    print("• Add ASCII art to your vault notes")
    print("• Customizable ASCII art settings")
    
    print(f"\n{Colors.CYAN}ASCII Magic integration allows you to:{Colors.ENDC}")
    print("• Convert any image to detailed ASCII art")
    print("• Customize character width (80-200+ chars)")
    print("• Save ASCII art to files")
    print("• Add artistic elements to your vault")

if __name__ == '__main__':
    demo()