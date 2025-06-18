#!/usr/bin/env python3
"""
Demo of the Vault Manager Interface
Shows what the interactive menu system looks like
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vault_manager import VaultManager, Colors

def demo():
    """Show demo of the interface"""
    manager = VaultManager()
    
    # Show welcome screen with wizard
    manager.show_welcome_art()
    print(f"\n{Colors.CYAN}Welcome to the Obsidian Vault Manager!{Colors.ENDC}")
    print(f"{Colors.BLUE}This tool helps you organize, analyze, and backup your vault.{Colors.ENDC}")
    
    print("\n\n--- Main Menu Example ---")
    
    # Show main menu
    options = [
        ('1', '📊 Analyze Vault'),
        ('2', '🏷️  Manage Tags'),
        ('3', '💾 Backup Vault'),
        ('4', '🚀 V2 Features'),
        ('5', '🔧 Advanced Tools'),
        ('6', '📚 Help & Documentation'),
        ('7', '⚙️  Settings'),
        ('0', '👋 Exit')
    ]
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'MAIN MENU'.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}Current Vault: {Colors.YELLOW}/Users/example/MyVault{Colors.ENDC}\n")
    
    for i, (key, desc) in enumerate(options):
        print(f"  {Colors.GREEN}{key}{Colors.ENDC}. {desc}")
    
    print(f"\n{Colors.BLUE}Choose an option to get started!{Colors.ENDC}")
    print(f"\n{Colors.YELLOW}Enter your choice: {Colors.ENDC}")
    
    print("\n\n--- Tag Management Menu Example ---")
    
    tag_options = [
        ('1', '👀 Preview tag issues'),
        ('2', '🚀 Fix all tag issues (auto)'),
        ('3', '📝 Fix quoted tags only'),
        ('4', '🔄 Merge similar tags'),
        ('5', '🗑️  Remove generic tags'),
        ('6', '🏷️  Auto-tag untagged files (v2)'),
        ('0', '← Back to main menu')
    ]
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'TAG MANAGEMENT'.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")
    
    for i, (key, desc) in enumerate(tag_options):
        print(f"  {Colors.GREEN}{key}{Colors.ENDC}. {desc}")
    
    print(f"\n{Colors.YELLOW}Enter your choice: {Colors.ENDC}")
    
    print("\n\n--- Sample Analysis Output ---")
    
    print(f"\n{Colors.CYAN}Analyzing tags...{Colors.ENDC}")
    print(f"{Colors.GREEN}✓ Success{Colors.ENDC}")
    print(f"""
{Colors.GREEN}Tag Analysis Summary:{Colors.ENDC}
Total unique tags: 204
Total tag occurrences: 423  
Files with tags: 123 (29.9%)
Files without tags: 289 (70.1%)

{Colors.CYAN}Top 10 most used tags:{Colors.ENDC}
  #identity (12 occurrences)
  #project/active (11 occurrences)
  #daily-note (10 occurrences)
  #meeting/team (9 occurrences)
  #todo (7 occurrences)
  #api-platform (6 occurrences)
  #ideas (5 occurrences)
  #payments (5 occurrences)
  #Payments (5 occurrences)
  #documentation (4 occurrences)

{Colors.YELLOW}Tag Issues Found:{Colors.ENDC}
  • Quoted tags: 15 instances (e.g., "#todo" → #todo)
  • Similar tags: 8 pairs (e.g., #payments vs #Payments)
  • Generic tags: 12 instances (#1, #2, #notes)
  • Incomplete hierarchies: 7 instances (e.g., #todo/)
""")
    
    print(f"\n{Colors.GREEN}✨ The Vault Manager provides an intuitive menu-driven interface{Colors.ENDC}")
    print(f"{Colors.GREEN}   for managing your Obsidian vault without memorizing commands!{Colors.ENDC}")
    
if __name__ == '__main__':
    demo()