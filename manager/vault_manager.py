#!/usr/bin/env python3
"""
Obsidian Vault Manager - Interactive Menu System
A user-friendly interface for managing your Obsidian vault
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

# Import the natural language query system
try:
    from natural_language_query import NaturalLanguageProcessor, QueryResult
    NLQ_AVAILABLE = True
except ImportError:
    NLQ_AVAILABLE = False

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class VaultManager:
    def __init__(self):
        self.config_file = Path.home() / '.obsidian_vault_manager.json'
        self.config = self.load_config()
        self.current_vault = self.config.get('last_vault', '')
        self.v2_available = self.check_v2_installation()
        
        # Initialize natural language processor
        self.nlp = None
        if NLQ_AVAILABLE:
            self.nlp = NaturalLanguageProcessor(self)
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def check_v2_installation(self) -> bool:
        """Check if obsidian-librarian v2 is installed"""
        try:
            # First check if the Python package is available
            import importlib.util
            
            # Check for the v2 directory structure
            v2_path = os.path.join(os.path.dirname(__file__), 'obsidian-librarian-v2')
            python_path = os.path.join(v2_path, 'python')
            
            if os.path.exists(v2_path) and os.path.exists(python_path):
                # Check if the Python package is importable
                spec = importlib.util.spec_from_file_location(
                    "obsidian_librarian", 
                    os.path.join(python_path, "obsidian_librarian", "__init__.py")
                )
                if spec is not None:
                    return True
            
            # Fallback: try to import as installed package
            try:
                import obsidian_librarian
                return True
            except ImportError:
                pass
            
            # Last resort: check CLI command
            result = subprocess.run(
                ['obsidian-librarian', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            # If there's any error, assume v2 is not available
            return False
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def show_welcome_art(self):
        """Display welcome ASCII art"""
        art = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════╗
║     {Colors.YELLOW} ___  _         _     _ _                          {Colors.CYAN}║
║     {Colors.YELLOW}/ _ \\| |__  ___(_) __| (_) __ _ _ __              {Colors.CYAN}║
║    {Colors.YELLOW}| | | | '_ \\/ __| |/ _` | |/ _` | '_ \\             {Colors.CYAN}║
║    {Colors.YELLOW}| |_| | |_) \\__ \\ | (_| | | (_| | | | |            {Colors.CYAN}║
║     {Colors.YELLOW}\\___/|_.__/|___/_|\\__,_|_|\\__,_|_| |_|            {Colors.CYAN}║
║                                                       {Colors.CYAN}║
║     {Colors.GREEN}__     __         _ _                              {Colors.CYAN}║
║     {Colors.GREEN}\\ \\   / /_ _ _   _| | |_                           {Colors.CYAN}║
║      {Colors.GREEN}\\ \\ / / _` | | | | | __|                          {Colors.CYAN}║
║       {Colors.GREEN}\\ V / (_| | |_| | | |_                           {Colors.CYAN}║
║        {Colors.GREEN}\\_/ \\__,_|\\__,_|_|\\__|                          {Colors.CYAN}║
║                                                       {Colors.CYAN}║
║      {Colors.BLUE}__  __                                            {Colors.CYAN}║
║     {Colors.BLUE}|  \\/  | __ _ _ __   __ _  __ _  ___ _ __         {Colors.CYAN}║
║     {Colors.BLUE}| |\\/| |/ _` | '_ \\ / _` |/ _` |/ _ \\ '__|        {Colors.CYAN}║
║     {Colors.BLUE}| |  | | (_| | | | | (_| | (_| |  __/ |           {Colors.CYAN}║
║     {Colors.BLUE}|_|  |_|\\__,_|_| |_|\\__,_|\\__, |\\___|_|           {Colors.CYAN}║
║                               {Colors.BLUE}|___/                    {Colors.CYAN}║
╚═══════════════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.BLUE}                    .-.
                   (o o)
                   | O \\
                  |    |
        |\\_/|    |     |
        |a a|".-.;     |
        | | | '-.__.-'(
       =\\t /=    {Colors.CYAN}(_{Colors.BLUE}(__)__)
        ) (     /     )
       /   \\   /      \\
       |   |  ((______))
       |   |   )      (
       |___|  /        \\
        |||  (_________)
       /_|_\\  {Colors.YELLOW} Your Digital Librarian{Colors.ENDC}
        """
        print(art)
    
    def show_loading(self, message="Loading", duration=2):
        """Show animated loading spinner"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        end_time = time.time() + duration
        i = 0
        while time.time() < end_time:
            print(f"\r{Colors.CYAN}{message} {spinner[i % len(spinner)]}{Colors.ENDC}", end='', flush=True)
            time.sleep(0.1)
            i += 1
        print(f"\r{message} ✓", flush=True)
    
    def show_progress_bar(self, message, current, total, width=50, show_percentage=True):
        """Display a progress bar with optional percentage"""
        if total == 0:
            percent = 1.0
        else:
            percent = current / total
        filled = int(width * percent)
        bar = f"{Colors.GREEN}{'█' * filled}{Colors.YELLOW}{'░' * (width - filled)}{Colors.ENDC}"
        
        if show_percentage:
            percentage_text = f" {percent*100:.1f}% ({current}/{total})"
        else:
            percentage_text = f" ({current}/{total})"
        
        print(f"\r{Colors.CYAN}{message}{Colors.ENDC} [{bar}]{percentage_text}", end='', flush=True)
        
        if current >= total:
            print()  # New line when complete
    
    def show_indeterminate_progress(self, message, duration=None):
        """Show an indeterminate progress bar that moves back and forth"""
        if duration is None:
            duration = 5
        
        width = 40
        end_time = time.time() + duration
        position = 0
        direction = 1
        
        while time.time() < end_time:
            # Create a moving block
            bar = ['░'] * width
            block_size = 8
            start = max(0, position - block_size // 2)
            end = min(width, position + block_size // 2)
            
            for i in range(start, end):
                intensity = 1 - abs(i - position) / (block_size // 2)
                if intensity > 0.7:
                    bar[i] = '█'
                elif intensity > 0.3:
                    bar[i] = '▓'
                else:
                    bar[i] = '▒'
            
            bar_str = f"{Colors.GREEN}{''.join(bar)}{Colors.ENDC}"
            print(f"\r{Colors.CYAN}{message}{Colors.ENDC} [{bar_str}]", end='', flush=True)
            
            # Update position
            position += direction
            if position >= width - 1 or position <= 0:
                direction *= -1
            
            time.sleep(0.1)
        
        # Show completion
        completed_bar = f"{Colors.GREEN}{'█' * width}{Colors.ENDC}"
        print(f"\r{Colors.CYAN}{message}{Colors.ENDC} [{completed_bar}] ✓")
    
    def run_command_with_progress(self, command, description=None, estimated_duration=None):
        """Run a command with a progress indicator"""
        import threading
        import subprocess
        
        if description:
            print(f"\n{Colors.CYAN}{description}...{Colors.ENDC}")
        
        # Debug output for troubleshooting
        if self.config.get('debug', False):
            print(f"{Colors.YELLOW}Debug - Running command: {command}{Colors.ENDC}")
        
        result = {'completed': False, 'returncode': None, 'stdout': '', 'stderr': ''}
        
        def run_process():
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                result['returncode'] = proc.returncode
                result['stdout'] = proc.stdout
                result['stderr'] = proc.stderr
            except Exception as e:
                result['error'] = str(e)
            finally:
                result['completed'] = True
        
        # Start the process in a separate thread
        thread = threading.Thread(target=run_process)
        thread.start()
        
        # Show progress while the command runs
        if estimated_duration:
            # Show determinate progress for known duration
            start_time = time.time()
            while not result['completed']:
                elapsed = time.time() - start_time
                if elapsed >= estimated_duration:
                    progress = 0.9  # Cap at 90% until actually complete
                else:
                    progress = elapsed / estimated_duration
                
                self.show_progress_bar(description or "Processing", 
                                     int(progress * 100), 100, 
                                     show_percentage=True)
                time.sleep(0.1)
        else:
            # Show indeterminate progress
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            i = 0
            while not result['completed']:
                print(f"\r{Colors.CYAN}{description or 'Processing'} {spinner[i % len(spinner)]}{Colors.ENDC}", 
                      end='', flush=True)
                time.sleep(0.1)
                i += 1
        
        # Wait for completion
        thread.join()
        
        # Show final result
        if 'error' in result:
            print(f"\r{Colors.RED}❌ Unexpected error: {result['error']}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Command: {command}{Colors.ENDC}")
            return False
        elif result['returncode'] == 0:
            print(f"\r{Colors.GREEN}✓ {description or 'Process'} completed successfully{Colors.ENDC}")
            if result['stdout']:
                print(result['stdout'])
            return True
        else:
            print(f"\r{Colors.RED}❌ Command failed (exit code: {result['returncode']}){Colors.ENDC}")
            if result['stderr']:
                print(f"{Colors.RED}Error details: {result['stderr'].strip()}{Colors.ENDC}")
            if result['stdout']:
                print(f"{Colors.YELLOW}Output: {result['stdout'].strip()}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Command that failed: {command}{Colors.ENDC}")
            return False
    
    def show_progress(self, current, total, width=50):
        """Display a progress bar"""
        percent = current / total
        filled = int(width * percent)
        bar = f"{Colors.GREEN}{'█' * filled}{Colors.ENDC}{'░' * (width - filled)}"
        print(f"\r[{bar}] {percent*100:.1f}%", end='', flush=True)
    
    def show_menu(self, title, options, footer=None):
        """Display a formatted menu"""
        self.clear_screen()
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")
        
        if self.current_vault:
            print(f"{Colors.CYAN}Current Vault: {Colors.YELLOW}{self.current_vault}{Colors.ENDC}\n")
        
        for i, (key, desc) in enumerate(options):
            print(f"  {Colors.GREEN}{key}{Colors.ENDC}. {desc}")
        
        if footer:
            print(f"\n{Colors.BLUE}{footer}{Colors.ENDC}")
        
        print(f"\n{Colors.YELLOW}Enter your choice: {Colors.ENDC}", end='')
    
    def get_vault_path(self):
        """Get vault path from user"""
        print(f"\n{Colors.CYAN}Please enter the path to your Obsidian vault:{Colors.ENDC}")
        print(f"{Colors.BLUE}(Example: /Users/username/Documents/MyVault){Colors.ENDC}")
        
        vault_path = input(f"\n{Colors.YELLOW}Vault path: {Colors.ENDC}").strip()
        
        if not vault_path:
            return None
            
        vault_path = os.path.expanduser(vault_path)
        
        if not os.path.exists(vault_path):
            print(f"\n{Colors.RED}❌ Path does not exist: {vault_path}{Colors.ENDC}")
            return None
        
        if not os.path.isdir(vault_path):
            print(f"\n{Colors.RED}❌ Path is not a directory: {vault_path}{Colors.ENDC}")
            return None
        
        # Check if it's likely an Obsidian vault
        obsidian_dir = os.path.join(vault_path, '.obsidian')
        if not os.path.exists(obsidian_dir):
            print(f"\n{Colors.YELLOW}⚠️  Warning: No .obsidian directory found.{Colors.ENDC}")
            print(f"{Colors.YELLOW}This might not be an Obsidian vault.{Colors.ENDC}")
            confirm = input(f"\n{Colors.CYAN}Continue anyway? (y/n): {Colors.ENDC}").lower()
            if confirm != 'y':
                return None
        
        # Save to config
        self.current_vault = vault_path
        self.config['last_vault'] = vault_path
        self.save_config()
        
        print(f"\n{Colors.GREEN}✓ Vault set to: {vault_path}{Colors.ENDC}")
        return vault_path
    
    def quote_path(self, path):
        """Properly quote a path for shell command usage"""
        import shlex
        return shlex.quote(path)
    
    def validate_dependencies(self):
        """Validate that required scripts and dependencies exist"""
        dependencies = {
            'analyze_tags_simple.py': 'Tag analysis functionality',
            'fix_vault_tags.py': 'Tag fixing functionality', 
            'backup_vault.py': 'Backup functionality',
            'quick_incremental_backup.sh': 'Incremental backup functionality'
        }
        
        missing = []
        for dep, description in dependencies.items():
            if not os.path.exists(dep):
                missing.append(f"{dep} ({description})")
        
        if missing:
            print(f"\n{Colors.YELLOW}⚠️  Missing dependencies:{Colors.ENDC}")
            for item in missing:
                print(f"  • {item}")
            print(f"\n{Colors.BLUE}Some features may not work correctly.{Colors.ENDC}")
            return False
        return True
    
    def run_command(self, command, description=None):
        """Run a shell command and handle output"""
        import shlex
        
        if description:
            print(f"\n{Colors.CYAN}{description}...{Colors.ENDC}")
        
        # Convert string command to list for security
        if isinstance(command, str):
            try:
                command_list = shlex.split(command)
            except ValueError as e:
                print(f"{Colors.RED}Error parsing command: {e}{Colors.ENDC}")
                return False
        else:
            command_list = command
        
        try:
            # Debug output for troubleshooting
            if self.config.get('debug', False):
                print(f"{Colors.YELLOW}Debug - Running command: {' '.join(command_list)}{Colors.ENDC}")
            
            result = subprocess.run(
                command_list,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                print(f"{Colors.GREEN}✓ Success{Colors.ENDC}")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"{Colors.RED}❌ Command failed (exit code: {result.returncode}){Colors.ENDC}")
                if result.stderr:
                    print(f"{Colors.RED}Error details: {result.stderr.strip()}{Colors.ENDC}")
                if result.stdout:
                    print(f"{Colors.YELLOW}Output: {result.stdout.strip()}{Colors.ENDC}")
                
                # Show command for debugging
                print(f"{Colors.YELLOW}Command that failed: {command}{Colors.ENDC}")
                return False
        except FileNotFoundError as e:
            print(f"{Colors.RED}❌ Command not found: {str(e)}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Make sure the required script/tool is in the same directory{Colors.ENDC}")
            return False
        except PermissionError as e:
            print(f"{Colors.RED}❌ Permission denied: {str(e)}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Check file permissions for the command{Colors.ENDC}")
            return False
        except Exception as e:
            print(f"{Colors.RED}❌ Unexpected error: {str(e)}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Command: {command}{Colors.ENDC}")
            return False
    
    def natural_language_interface(self):
        """Interactive natural language query interface"""
        if not NLQ_AVAILABLE:
            print(f"{Colors.RED}❌ Natural language processing not available{Colors.ENDC}")
            print(f"{Colors.YELLOW}Make sure natural_language_query.py is in the same directory{Colors.ENDC}")
            return
        
        self.clear_screen()
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'🤖 NATURAL LANGUAGE INTERFACE'.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Welcome to the AI-powered vault manager!{Colors.ENDC}")
        print(f"{Colors.BLUE}Ask me anything about your vault in natural language.{Colors.ENDC}")
        print(f"\n{Colors.YELLOW}Examples:{Colors.ENDC}")
        print(f"  • \"analyze my tags\"")
        print(f"  • \"backup the vault\"") 
        print(f"  • \"find files without tags\"")
        print(f"  • \"research artificial intelligence\"")
        print(f"  • \"help\" or \"what can you do?\"")
        print(f"\n{Colors.GREEN}Type 'menu' to return to the standard menu interface{Colors.ENDC}")
        print(f"{Colors.GREEN}Type 'exit' to quit the application{Colors.ENDC}")
        
        while True:
            print(f"\n{Colors.CYAN}Vault: {Colors.YELLOW}{self.current_vault or 'Not set'}{Colors.ENDC}")
            query = input(f"{Colors.BOLD}🤖 Ask me: {Colors.ENDC}").strip()
            
            if not query:
                continue
            
            if query.lower() in ['menu', 'back', 'return']:
                break
                
            if query.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{Colors.GREEN}Thanks for using the Natural Language Vault Manager!{Colors.ENDC}")
                print(f"{Colors.BLUE}Your vault is in good hands. 🤖📚{Colors.ENDC}\n")
                sys.exit(0)
            
            # Process the query
            result = self.nlp.process_query(query)
            self.handle_query_result(result, query)
    
    def handle_query_result(self, result: QueryResult, original_query: str):
        """Handle the result of a natural language query"""
        if result.confidence == 0.0:
            print(f"\n{Colors.RED}❌ I didn't understand: \"{original_query}\"{Colors.ENDC}")
            if result.suggestions:
                print(f"\n{Colors.YELLOW}💡 Did you mean:{Colors.ENDC}")
                for i, suggestion in enumerate(result.suggestions[:3], 1):
                    print(f"  {i}. {suggestion}")
                    
                choice = input(f"\n{Colors.CYAN}Select option (1-{len(result.suggestions[:3])}) or press Enter to try again: {Colors.ENDC}").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(result.suggestions[:3]):
                    # Re-process the selected suggestion
                    selected = result.suggestions[int(choice)-1]
                    new_result = self.nlp.process_query(selected)
                    self.handle_query_result(new_result, selected)
            else:
                print(f"{Colors.YELLOW}Try 'help' to see available commands{Colors.ENDC}")
            return
        
        if result.confidence < 0.6:
            print(f"\n{Colors.YELLOW}🤔 I think you want: {result.description}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Confidence: {result.confidence*100:.1f}%{Colors.ENDC}")
            confirm = input(f"{Colors.CYAN}Is this correct? (y/n): {Colors.ENDC}").lower()
            if confirm != 'y':
                if result.suggestions:
                    print(f"\n{Colors.YELLOW}💡 Other possibilities:{Colors.ENDC}")
                    for suggestion in result.suggestions[:3]:
                        print(f"  • {suggestion}")
                return
        
        # Execute the function
        print(f"\n{Colors.GREEN}✓ Executing: {result.description}{Colors.ENDC}")
        if result.function:
            try:
                if result.action == 'research_topics' and 'research_topic' in result.parameters:
                    # Special handling for research
                    self.execute_research_query(result.parameters['research_topic'])
                elif result.action == 'show_help':
                    self.show_nlp_help()
                elif result.action == 'exit_application':
                    print(f"\n{Colors.GREEN}Thanks for using the Natural Language Vault Manager!{Colors.ENDC}")
                    print(f"{Colors.BLUE}Your vault is in good hands. 🤖📚{Colors.ENDC}\n")
                    sys.exit(0)
                else:
                    # Execute the mapped function
                    result.function()
            except Exception as e:
                print(f"{Colors.RED}❌ Error executing command: {str(e)}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}❌ Function not implemented: {result.action}{Colors.ENDC}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def execute_research_query(self, topic: str):
        """Execute a research query with the extracted topic"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian research {self.quote_path(self.current_vault)} {self.quote_path(topic)}',
                f'Researching: {topic}',
                estimated_duration=60
            )
        else:
            print(f"{Colors.YELLOW}⚠️  Research feature requires obsidian-librarian v2{Colors.ENDC}")
            print(f"{Colors.BLUE}See the Advanced Tools menu for installation instructions{Colors.ENDC}")
    
    def show_nlp_help(self):
        """Show natural language processing help"""
        if self.nlp:
            help_text = self.nlp.get_command_help()
            print(f"\n{help_text}")
        else:
            self.show_help()
    
    # Mapping functions for NLP system
    def analyze_vault_tags(self):
        """Wrapper for tag analysis"""
        self.run_command_with_progress(
            f'python3 analyze_tags_simple.py {self.quote_path(self.current_vault)}',
            'Analyzing vault tags',
            estimated_duration=10
        )
    
    def generate_analysis_report(self):
        """Wrapper for analysis report generation"""
        self.run_command_with_progress(
            f'python3 analyze_tags_simple.py {self.quote_path(self.current_vault)}',
            'Generating comprehensive analysis report',
            estimated_duration=12
        )
        print(f"{Colors.CYAN}Note: JSON report saved as 'tag_analysis_report.json' in your vault directory{Colors.ENDC}")
    
    def preview_tag_issues(self):
        """Wrapper for tag issue preview"""
        self.run_command_with_progress(
            f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --dry-run',
            'Analyzing tag issues',
            estimated_duration=8
        )
    
    def fix_all_tag_issues(self):
        """Wrapper for fixing all tag issues"""
        print(f"\n{Colors.YELLOW}⚠️  This will modify your vault files!{Colors.ENDC}")
        confirm = input(f"{Colors.CYAN}Are you sure? (y/n): {Colors.ENDC}").lower()
        if confirm == 'y':
            self.run_command_with_progress(
                f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)}',
                'Fixing all tag issues',
                estimated_duration=15
            )
    
    def fix_quoted_tags(self):
        """Wrapper for fixing quoted tags"""
        self.run_command_with_progress(
            f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply --fix-quoted-only',
            'Fixing quoted tags',
            estimated_duration=6
        )
    
    def merge_similar_tags(self):
        """Wrapper for merging similar tags"""
        self.run_command_with_progress(
            f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply --merge-similar',
            'Merging similar tags',
            estimated_duration=12
        )
    
    def remove_generic_tags(self):
        """Wrapper for removing generic tags"""
        self.run_command_with_progress(
            f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply --remove-generic',
            'Removing generic tags',
            estimated_duration=8
        )
    
    def auto_tag_files(self):
        """Wrapper for auto-tagging files"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian tags auto-tag {self.quote_path(self.current_vault)}',
                'Auto-tagging untagged files with AI',
                estimated_duration=25
            )
        else:
            print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
    
    def create_incremental_backup(self):
        """Wrapper for incremental backup"""
        self.run_command_with_progress(
            f'./quick_incremental_backup.sh {self.quote_path(self.current_vault)}',
            'Creating incremental backup',
            estimated_duration=20
        )
    
    def create_full_backup(self):
        """Wrapper for full backup"""
        self.run_command_with_progress(
            f'python3 backup_vault.py {self.quote_path(self.current_vault)}',
            'Creating full backup',
            estimated_duration=45
        )
    
    def ai_content_analysis(self):
        """Wrapper for AI content analysis"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian analyze {self.quote_path(self.current_vault)} --quality --structure',
                'Analyzing content with AI',
                estimated_duration=30
            )
        else:
            print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
    
    def research_topics(self):
        """Wrapper for research functionality"""
        query = input(f"\n{Colors.CYAN}Enter research topic: {Colors.ENDC}")
        if query:
            self.execute_research_query(query)
    
    def smart_file_organization(self):
        """Wrapper for smart organization"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian organize {self.quote_path(self.current_vault)} --strategy content --dry-run',
                'Planning smart organization',
                estimated_duration=20
            )
            confirm = input(f"\n{Colors.CYAN}Apply these changes? (y/n): {Colors.ENDC}").lower()
            if confirm == 'y':
                self.run_command_with_progress(
                    f'obsidian-librarian organize {self.quote_path(self.current_vault)} --strategy content',
                    'Organizing files',
                    estimated_duration=35
                )
        else:
            print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
    
    def find_duplicate_content(self):
        """Wrapper for duplicate detection"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian duplicates {self.quote_path(self.current_vault)} --threshold 0.85',
                'Finding duplicate content',
                estimated_duration=25
            )
        else:
            print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
    
    def generate_advanced_analytics(self):
        """Wrapper for advanced analytics"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian stats {self.quote_path(self.current_vault)} --detailed',
                'Generating advanced analytics',
                estimated_duration=15
            )
        else:
            print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
    
    def comprehensive_vault_curation(self):
        """Wrapper for comprehensive curation"""
        if self.v2_available:
            self.run_command_with_progress(
                f'obsidian-librarian curate {self.quote_path(self.current_vault)} --duplicates --quality --structure --dry-run',
                'Planning comprehensive curation',
                estimated_duration=40
            )
            confirm = input(f"\n{Colors.CYAN}Apply curation? (y/n): {Colors.ENDC}").lower()
            if confirm == 'y':
                self.run_command_with_progress(
                    f'obsidian-librarian curate {self.quote_path(self.current_vault)} --duplicates --quality --structure',
                    'Curating vault',
                    estimated_duration=90
                )
        else:
            print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
    
    def install_v2_features(self):
        """Wrapper for v2 installation"""
        self.show_v2_setup()
    
    def check_for_updates(self):
        """Wrapper for update checking"""
        print(f"\n{Colors.CYAN}Checking for updates...{Colors.ENDC}")
        self.run_command('pip list --outdated | grep obsidian', 'Checking pip packages')
    
    def enable_debug_mode(self):
        """Wrapper for debug mode"""
        print(f"\n{Colors.CYAN}Debug mode enabled for next command{Colors.ENDC}")
        self.config['debug'] = True
        self.save_config()
    
    def clean_cache_files(self):
        """Wrapper for cache cleaning"""
        self.clean_cache()
    
    def change_vault_location(self):
        """Wrapper for changing vault location"""
        self.get_vault_path()
    
    def toggle_color_output(self):
        """Wrapper for toggling colors"""
        self.toggle_colors()
    
    def configure_backup_settings(self):
        """Wrapper for backup settings"""
        self.configure_backup_settings()
    
    def reset_to_defaults(self):
        """Wrapper for resetting settings"""
        self.reset_settings()
    
    def exit_application(self):
        """Wrapper for exiting"""
        print(f"\n{Colors.GREEN}Thanks for using Obsidian Vault Manager!{Colors.ENDC}")
        print(f"{Colors.BLUE}Your vault is in good hands. 📚{Colors.ENDC}\n")
        sys.exit(0)
    
    def analyze_vault_menu(self):
        """Vault analysis submenu"""
        while True:
            options = [
                ('1', '📊 Analyze vault tags'),
                ('2', '📁 Analyze folder structure'),
                ('3', '🔍 Find files without tags'),
                ('4', '💾 Generate tag analysis report'),
                ('0', '← Back to main menu')
            ]
            
            self.show_menu('VAULT ANALYSIS', options)
            choice = input().strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.run_command_with_progress(
                    f'python3 analyze_tags_simple.py {self.quote_path(self.current_vault)}',
                    'Analyzing vault tags',
                    estimated_duration=10
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.analyze_folder_structure()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.find_untagged_files()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                self.run_command_with_progress(
                    f'python3 analyze_tags_simple.py {self.quote_path(self.current_vault)}',
                    'Generating comprehensive analysis report',
                    estimated_duration=12
                )
                print(f"{Colors.CYAN}Note: JSON report saved as 'tag_analysis_report.json' in your vault directory{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def manage_tags_menu(self):
        """Tag management submenu"""
        while True:
            options = [
                ('1', '👀 Preview tag issues'),
                ('2', '🚀 Fix all tag issues (auto)'),
                ('3', '📝 Fix quoted tags only'),
                ('4', '🔄 Merge similar tags'),
                ('5', '🗑️  Remove generic tags'),
                ('6', '🏷️  Auto-tag untagged files (v2)' if self.v2_available else '🏷️  Auto-tag untagged files ⚠️'),
                ('0', '← Back to main menu')
            ]
            
            self.show_menu('TAG MANAGEMENT', options)
            choice = input().strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --dry-run',
                    'Analyzing tag issues',
                    estimated_duration=8
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                print(f"\n{Colors.YELLOW}⚠️  This will modify your vault files!{Colors.ENDC}")
                confirm = input(f"{Colors.CYAN}Are you sure? (y/n): {Colors.ENDC}").lower()
                if confirm == 'y':
                    self.run_command_with_progress(
                        f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)}',
                        'Fixing all tag issues',
                        estimated_duration=15
                    )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply --fix-quoted-only',
                    'Fixing quoted tags',
                    estimated_duration=6
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply --merge-similar',
                    'Merging similar tags',
                    estimated_duration=12
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                self.run_command_with_progress(
                    f'python3 fix_vault_tags.py {self.quote_path(self.current_vault)} --apply --remove-generic',
                    'Removing generic tags',
                    estimated_duration=8
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '6':
                if self.v2_available:
                    self.run_command_with_progress(
                        f'obsidian-librarian tags auto-tag {self.quote_path(self.current_vault)}',
                        'Auto-tagging untagged files with AI',
                        estimated_duration=25
                    )
                else:
                    print(f"\n{Colors.YELLOW}⚠️  This feature requires obsidian-librarian v2{Colors.ENDC}")
                    print(f"{Colors.BLUE}See the Advanced Tools menu for installation instructions{Colors.ENDC}")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def backup_vault_menu(self):
        """Backup management submenu"""
        while True:
            options = [
                ('1', '⚡ Quick backup (incremental)'),
                ('2', '📦 Full backup (compressed)'),
                ('3', '🚀 Setup Kopia (advanced)'),
                ('4', '📊 View backup history'),
                ('5', '♻️  Restore from backup'),
                ('0', '← Back to main menu')
            ]
            
            self.show_menu('BACKUP & RESTORE', options)
            choice = input().strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.run_command_with_progress(
                    f'./quick_incremental_backup.sh {self.quote_path(self.current_vault)}',
                    'Creating incremental backup',
                    estimated_duration=20
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                self.run_command_with_progress(
                    f'python3 backup_vault.py {self.quote_path(self.current_vault)}',
                    'Creating full backup',
                    estimated_duration=45
                )
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.setup_kopia()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                self.view_backup_history()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                self.restore_backup_menu()
    
    def v2_features_menu(self):
        """Advanced v2 features submenu"""
        while True:
            options = [
                ('1', '🧠 AI-powered content analysis'),
                ('2', '🔍 Research topics & create notes'),
                ('3', '📚 Smart file organization'),
                ('4', '🔄 Find & merge duplicates'),
                ('5', '📊 Advanced vault analytics'),
                ('6', '🎯 Comprehensive curation'),
                ('7', '⚙️  Configure AI settings'),
                ('0', '← Back to main menu')
            ]
            
            footer = "v2 Available ✓" if self.v2_available else "v2 Not Installed - See Help for setup"
            self.show_menu('OBSIDIAN LIBRARIAN V2', options, footer)
            choice = input().strip()
            
            if choice == '0':
                break
            elif not self.v2_available:
                print(f"\n{Colors.YELLOW}⚠️  Obsidian Librarian v2 is not installed{Colors.ENDC}")
                print(f"{Colors.BLUE}Would you like to see installation instructions? (y/n): {Colors.ENDC}", end='')
                if input().lower() == 'y':
                    self.show_v2_setup()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
                continue
            
            if choice == '1':
                self.run_command_with_progress(
                    f'obsidian-librarian analyze {self.quote_path(self.current_vault)} --quality --structure',
                    'Analyzing content with AI',
                    estimated_duration=30
                )
            elif choice == '2':
                query = input(f"\n{Colors.CYAN}Enter research topic: {Colors.ENDC}")
                if query:
                    self.run_command_with_progress(
                        f'obsidian-librarian research {self.quote_path(self.current_vault)} {self.quote_path(query)}',
                        f'Researching: {query}',
                        estimated_duration=60
                    )
            elif choice == '3':
                self.run_command_with_progress(
                    f'obsidian-librarian organize {self.quote_path(self.current_vault)} --strategy content --dry-run',
                    'Planning smart organization',
                    estimated_duration=20
                )
                confirm = input(f"\n{Colors.CYAN}Apply these changes? (y/n): {Colors.ENDC}").lower()
                if confirm == 'y':
                    self.run_command_with_progress(
                        f'obsidian-librarian organize {self.quote_path(self.current_vault)} --strategy content',
                        'Organizing files',
                        estimated_duration=35
                    )
            elif choice == '4':
                self.run_command_with_progress(
                    f'obsidian-librarian duplicates {self.quote_path(self.current_vault)} --threshold 0.85',
                    'Finding duplicate content',
                    estimated_duration=25
                )
            elif choice == '5':
                self.run_command_with_progress(
                    f'obsidian-librarian stats {self.quote_path(self.current_vault)} --detailed',
                    'Generating advanced analytics',
                    estimated_duration=15
                )
            elif choice == '6':
                self.run_command_with_progress(
                    f'obsidian-librarian curate {self.quote_path(self.current_vault)} --duplicates --quality --structure --dry-run',
                    'Planning comprehensive curation',
                    estimated_duration=40
                )
                confirm = input(f"\n{Colors.CYAN}Apply curation? (y/n): {Colors.ENDC}").lower()
                if confirm == 'y':
                    self.run_command_with_progress(
                        f'obsidian-librarian curate {self.quote_path(self.current_vault)} --duplicates --quality --structure',
                        'Curating vault',
                        estimated_duration=90
                    )
            elif choice == '7':
                self.configure_ai_settings()
            
            if choice in '123456':
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def advanced_tools_menu(self):
        """Advanced tools submenu"""
        while True:
            options = [
                ('1', '🔧 Install Obsidian Librarian v2'),
                ('2', '🔄 Check for updates'),
                ('3', '📊 Performance benchmarks'),
                ('4', '🛠️  Debug mode'),
                ('5', '🧹 Clean cache files'),
                ('0', '← Back to main menu')
            ]
            
            self.show_menu('ADVANCED TOOLS', options)
            choice = input().strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.show_v2_setup()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '2':
                print(f"\n{Colors.CYAN}Checking for updates...{Colors.ENDC}")
                self.run_command('pip list --outdated | grep obsidian', 'Checking pip packages')
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '3':
                self.run_performance_benchmark()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '4':
                print(f"\n{Colors.CYAN}Debug mode enabled for next command{Colors.ENDC}")
                self.config['debug'] = True
                self.save_config()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
            elif choice == '5':
                self.clean_cache()
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def settings_menu(self):
        """Settings submenu"""
        while True:
            options = [
                ('1', f'📁 Change vault location (current: {self.current_vault or "Not set"})'),
                ('2', '🎨 Toggle color output'),
                ('3', '💾 Backup settings'),
                ('4', '🔄 Reset to defaults'),
                ('0', '← Back to main menu')
            ]
            
            self.show_menu('SETTINGS', options)
            choice = input().strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.get_vault_path()
            elif choice == '2':
                self.toggle_colors()
            elif choice == '3':
                self.configure_backup_settings()
            elif choice == '4':
                self.reset_settings()
    
    def show_help(self):
        """Display help information"""
        self.clear_screen()
        help_text = f"""
{Colors.HEADER}{Colors.BOLD}OBSIDIAN VAULT MANAGER - HELP{Colors.ENDC}
{'=' * 60}

{Colors.CYAN}Overview:{Colors.ENDC}
This interactive tool helps you manage your Obsidian vault with
powerful features for tag management, backup, and organization.

{Colors.CYAN}Key Features:{Colors.ENDC}
• {Colors.GREEN}Tag Analysis & Cleanup{Colors.ENDC} - Find and fix tag issues
• {Colors.GREEN}Backup Solutions{Colors.ENDC} - Multiple backup strategies
• {Colors.GREEN}File Organization{Colors.ENDC} - AI-powered organization (v2)
• {Colors.GREEN}Research Assistant{Colors.ENDC} - Create notes from web sources (v2)

{Colors.CYAN}Navigation:{Colors.ENDC}
• Use number keys to select menu options
• Press 0 to go back or exit
• Follow on-screen prompts

{Colors.CYAN}First Time Setup:{Colors.ENDC}
1. The tool will ask for your vault location
2. This is saved for future sessions
3. You can change it anytime in Settings

{Colors.CYAN}Safety:{Colors.ENDC}
• Always backup before major operations
• Use preview/dry-run options first
• Check the confirmation prompts

{Colors.CYAN}For more information:{Colors.ENDC}
• GitHub: https://github.com/valdez-nick/obsidian-librarian
• Docs: See VAULT_MANAGER_README.md
"""
        print(help_text)
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.ENDC}")
    
    def show_v2_setup(self):
        """Show v2 installation instructions"""
        self.clear_screen()
        setup_text = f"""
{Colors.HEADER}{Colors.BOLD}INSTALLING OBSIDIAN LIBRARIAN V2{Colors.ENDC}
{'=' * 60}

{Colors.CYAN}Option 1: Quick Install (Python components only){Colors.ENDC}
{Colors.GREEN}$ pip install obsidian-librarian{Colors.ENDC}

{Colors.CYAN}Option 2: Full Install with Rust Performance (Recommended){Colors.ENDC}
{Colors.GREEN}$ git clone https://github.com/valdez-nick/obsidian-librarian-v2
$ cd obsidian-librarian-v2
$ make install{Colors.ENDC}

{Colors.CYAN}Requirements:{Colors.ENDC}
• Python 3.9+ (you have: {sys.version.split()[0]})
• Rust 1.70+ (optional, for performance)
• 500MB disk space

{Colors.CYAN}What v2 Provides:{Colors.ENDC}
• {Colors.YELLOW}10-100x faster{Colors.ENDC} performance
• {Colors.YELLOW}AI-powered{Colors.ENDC} content understanding
• {Colors.YELLOW}Research assistant{Colors.ENDC} that creates notes
• {Colors.YELLOW}Smart organization{Colors.ENDC} based on content
• Support for {Colors.YELLOW}100,000+ notes{Colors.ENDC}

{Colors.CYAN}After Installation:{Colors.ENDC}
1. Restart this vault manager
2. The v2 features will be automatically detected
3. New options will appear in the menus
"""
        print(setup_text)
    
    # Helper methods
    def analyze_folder_structure(self):
        """Analyze and display folder structure"""
        print(f"\n{Colors.CYAN}Analyzing folder structure...{Colors.ENDC}")
        
        try:
            # First pass: count total directories to show progress
            total_dirs_to_scan = sum(1 for _, dirs, _ in os.walk(self.current_vault) 
                                   if not any(d.startswith('.') for d in dirs))
            
            total_dirs = 0
            total_files = 0
            folder_stats = {}
            processed_dirs = 0
            
            for root, dirs, files in os.walk(self.current_vault):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                processed_dirs += 1
                self.show_progress_bar("Scanning directories", processed_dirs, 
                                     total_dirs_to_scan, width=40, show_percentage=True)
                
                total_dirs += len(dirs)
                md_files = [f for f in files if f.endswith('.md')]
                total_files += len(md_files)
                
                if md_files:
                    rel_path = os.path.relpath(root, self.current_vault)
                    folder_stats[rel_path] = len(md_files)
                
                time.sleep(0.02)  # Small delay to show progress
            
            print(f"\n{Colors.GREEN}Folder Structure Analysis:{Colors.ENDC}")
            print(f"Total folders: {total_dirs}")
            print(f"Total markdown files: {total_files}")
            print(f"\n{Colors.CYAN}Top folders by file count:{Colors.ENDC}")
            
            sorted_folders = sorted(folder_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            for folder, count in sorted_folders:
                print(f"  {folder}: {count} files")
                
        except Exception as e:
            print(f"{Colors.RED}Error analyzing structure: {str(e)}{Colors.ENDC}")
    
    def find_untagged_files(self):
        """Find files without tags"""
        print(f"\n{Colors.CYAN}Finding files without tags...{Colors.ENDC}")
        
        try:
            # First pass: count total markdown files
            print(f"{Colors.YELLOW}Counting files...{Colors.ENDC}")
            all_md_files = []
            for root, dirs, files in os.walk(self.current_vault):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    if file.endswith('.md'):
                        all_md_files.append(os.path.join(root, file))
            
            total_files = len(all_md_files)
            untagged = []
            processed = 0
            
            print(f"{Colors.CYAN}Scanning {total_files} files for tags...{Colors.ENDC}")
            
            for file_path in all_md_files:
                processed += 1
                self.show_progress_bar("Checking for tags", processed, 
                                     total_files, width=40, show_percentage=True)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Check for inline tags (#tag) and frontmatter tags
                        if not re.search(r'#[\w\-/]+', content) and not re.search(r'^tags:\s*\[.*\]', content, re.MULTILINE):
                            rel_path = os.path.relpath(file_path, self.current_vault)
                            untagged.append(rel_path)
                except Exception:
                    # Skip files that can't be read
                    continue
                
                time.sleep(0.01)  # Small delay to show progress
            
            print(f"\n{Colors.GREEN}Untagged Files Analysis:{Colors.ENDC}")
            print(f"Total files scanned: {total_files}")
            print(f"Files without tags: {len(untagged)}")
            
            if untagged:
                print(f"\n{Colors.CYAN}First 20 untagged files:{Colors.ENDC}")
                for file in untagged[:20]:
                    print(f"  • {file}")
                if len(untagged) > 20:
                    print(f"  ... and {len(untagged) - 20} more")
                    
        except Exception as e:
            print(f"{Colors.RED}Error finding untagged files: {str(e)}{Colors.ENDC}")
    
    def view_backup_history(self):
        """View backup history"""
        print(f"\n{Colors.CYAN}Checking backup history...{Colors.ENDC}")
        
        backup_base = os.path.expanduser("~/ObsidianBackups")
        if not os.path.exists(backup_base):
            print(f"{Colors.YELLOW}No backups found yet.{Colors.ENDC}")
            return
        
        vault_name = os.path.basename(self.current_vault)
        vault_backup_dir = os.path.join(backup_base, vault_name)
        
        if not os.path.exists(vault_backup_dir):
            print(f"{Colors.YELLOW}No backups found for this vault.{Colors.ENDC}")
            return
        
        # List all backups
        backups = []
        for item in os.listdir(vault_backup_dir):
            item_path = os.path.join(vault_backup_dir, item)
            if os.path.isdir(item_path) or item.endswith('.zip'):
                stat = os.stat(item_path)
                backups.append((item, stat.st_mtime, stat.st_size))
        
        backups.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{Colors.GREEN}Backup History:{Colors.ENDC}")
        for name, mtime, size in backups[:10]:
            date = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            size_mb = size / 1024 / 1024
            print(f"  • {name} - {date} ({size_mb:.1f} MB)")
    
    def restore_backup_menu(self):
        """Restore from backup submenu"""
        print(f"\n{Colors.YELLOW}⚠️  Restore will overwrite current vault contents!{Colors.ENDC}")
        print(f"{Colors.CYAN}Choose backup type to restore from:{Colors.ENDC}")
        print("  1. Incremental backup")
        print("  2. Zip archive")
        print("  0. Cancel")
        
        choice = input(f"\n{Colors.YELLOW}Choice: {Colors.ENDC}").strip()
        
        if choice == '1':
            # List incremental backups
            backup_base = os.path.expanduser("~/ObsidianBackups")
            vault_name = os.path.basename(self.current_vault)
            vault_backup_dir = os.path.join(backup_base, vault_name)
            
            if os.path.exists(vault_backup_dir):
                backups = [d for d in os.listdir(vault_backup_dir) 
                          if os.path.isdir(os.path.join(vault_backup_dir, d)) and d != 'latest']
                backups.sort(reverse=True)
                
                if backups:
                    print(f"\n{Colors.CYAN}Available backups:{Colors.ENDC}")
                    for i, backup in enumerate(backups[:10]):
                        print(f"  {i+1}. {backup}")
                    
                    backup_choice = input(f"\n{Colors.YELLOW}Select backup (number): {Colors.ENDC}").strip()
                    try:
                        idx = int(backup_choice) - 1
                        if 0 <= idx < len(backups):
                            selected = backups[idx]
                            confirm = input(f"\n{Colors.RED}Restore from {selected}? This will overwrite current vault! (y/n): {Colors.ENDC}").lower()
                            if confirm == 'y':
                                source = os.path.join(vault_backup_dir, selected)
                                self.run_command(
                                    f'rsync -av --delete "{source}/" "{self.current_vault}/"',
                                    f'Restoring from {selected}'
                                )
                    except ValueError:
                        print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
        
        elif choice == '2':
            print(f"\n{Colors.CYAN}Enter path to backup zip file:{Colors.ENDC}")
            zip_path = input(f"{Colors.YELLOW}Path: {Colors.ENDC}").strip()
            
            if os.path.exists(zip_path) and zip_path.endswith('.zip'):
                confirm = input(f"\n{Colors.RED}Restore from {os.path.basename(zip_path)}? This will overwrite current vault! (y/n): {Colors.ENDC}").lower()
                if confirm == 'y':
                    # Create temp dir for extraction
                    temp_dir = f"/tmp/obsidian_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.makedirs(temp_dir)
                    
                    self.run_command(
                        f'unzip -q "{zip_path}" -d "{temp_dir}"',
                        'Extracting backup'
                    )
                    
                    # Find the vault directory in the extract
                    extracted = os.listdir(temp_dir)
                    if extracted:
                        source = os.path.join(temp_dir, extracted[0])
                        self.run_command(
                            f'rsync -av --delete "{source}/" "{self.current_vault}/"',
                            'Restoring files'
                        )
                        
                        # Cleanup
                        self.run_command(f'rm -rf "{temp_dir}"', 'Cleaning up')
    
    def setup_kopia(self):
        """Setup Kopia for advanced backups"""
        print(f"\n{Colors.CYAN}Kopia Setup{Colors.ENDC}")
        print("\nKopia provides advanced backup features:")
        print("• Deduplication across all backups")
        print("• Compression and encryption")
        print("• Snapshot-based versioning")
        print("• Cloud storage support")
        
        print(f"\n{Colors.YELLOW}Install Kopia first:{Colors.ENDC}")
        print("• macOS: brew install kopia")
        print("• Linux: See https://kopia.io/docs/installation/")
        print("• Windows: Download from https://github.com/kopia/kopia/releases")
        
        # Check if Kopia is installed
        try:
            result = subprocess.run(['kopia', '--version'], capture_output=True)
            if result.returncode == 0:
                print(f"\n{Colors.GREEN}✓ Kopia is installed{Colors.ENDC}")
                
                # Initialize repository
                repo_path = os.path.expanduser("~/ObsidianBackups/kopia-repo")
                if not os.path.exists(os.path.join(repo_path, '.kopia')):
                    print(f"\n{Colors.CYAN}Initializing Kopia repository...{Colors.ENDC}")
                    os.makedirs(repo_path, exist_ok=True)
                    
                    password = input(f"\n{Colors.YELLOW}Enter a password for the repository: {Colors.ENDC}")
                    if password:
                        self.run_command(
                            f'kopia repository create filesystem --path="{repo_path}" --password="{password}"',
                            'Creating repository'
                        )
                        
                        # Create snapshot policy
                        self.run_command(
                            f'kopia policy set --global --keep-latest=10 --keep-daily=7 --keep-weekly=4 --keep-monthly=12',
                            'Setting retention policy'
                        )
                        
                        print(f"\n{Colors.GREEN}✓ Kopia setup complete!{Colors.ENDC}")
                        print(f"\nTo create a backup: kopia snapshot create \"{self.current_vault}\"")
                else:
                    print(f"\n{Colors.GREEN}✓ Kopia repository already exists{Colors.ENDC}")
            else:
                print(f"\n{Colors.YELLOW}Kopia is not installed. Please install it first.{Colors.ENDC}")
        except FileNotFoundError:
            print(f"\n{Colors.YELLOW}Kopia is not installed. Please install it first.{Colors.ENDC}")
    
    def run_performance_benchmark(self):
        """Run performance benchmarks"""
        print(f"\n{Colors.CYAN}Running performance benchmarks...{Colors.ENDC}")
        
        start_time = time.time()
        
        # Count files
        file_count = 0
        for root, dirs, files in os.walk(self.current_vault):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            file_count += len([f for f in files if f.endswith('.md')])
        
        file_time = time.time() - start_time
        
        # Tag parsing benchmark
        start_time = time.time()
        tag_count = 0
        sample_files = 0
        
        for root, dirs, files in os.walk(self.current_vault):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.md') and sample_files < 100:  # Sample first 100 files
                    sample_files += 1
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            tags = re.findall(r'#[\w\-/]+', content)
                            tag_count += len(tags)
                    except:
                        pass
        
        tag_time = time.time() - start_time
        
        print(f"\n{Colors.GREEN}Performance Results:{Colors.ENDC}")
        print(f"Vault size: {file_count} markdown files")
        print(f"File scanning: {file_time:.2f} seconds ({file_count/file_time:.0f} files/sec)")
        print(f"Tag parsing (100 files): {tag_time:.2f} seconds")
        print(f"Tags found in sample: {tag_count}")
        
        if self.v2_available:
            print(f"\n{Colors.CYAN}With v2 installed, these operations would be 10-100x faster!{Colors.ENDC}")
    
    def clean_cache(self):
        """Clean cache files"""
        print(f"\n{Colors.CYAN}Cleaning cache files...{Colors.ENDC}")
        
        cache_locations = [
            os.path.join(self.current_vault, '.obsidian', 'cache'),
            os.path.join(self.current_vault, '.trash'),
            os.path.expanduser('~/.obsidian_vault_manager_cache')
        ]
        
        total_size = 0
        for location in cache_locations:
            if os.path.exists(location):
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(location)
                          for filename in filenames)
                total_size += size
                print(f"Found cache: {location} ({size/1024/1024:.1f} MB)")
        
        if total_size > 0:
            confirm = input(f"\n{Colors.YELLOW}Remove {total_size/1024/1024:.1f} MB of cache? (y/n): {Colors.ENDC}").lower()
            if confirm == 'y':
                for location in cache_locations:
                    if os.path.exists(location) and location != os.path.join(self.current_vault, '.obsidian', 'cache'):
                        self.run_command(f'rm -rf "{location}"', f'Removing {os.path.basename(location)}')
                print(f"\n{Colors.GREEN}✓ Cache cleaned{Colors.ENDC}")
        else:
            print(f"{Colors.GREEN}No cache files found{Colors.ENDC}")
    
    def toggle_colors(self):
        """Toggle color output"""
        colors_enabled = self.config.get('colors_enabled', True)
        self.config['colors_enabled'] = not colors_enabled
        self.save_config()
        
        if not self.config['colors_enabled']:
            # Disable colors
            for attr in dir(Colors):
                if not attr.startswith('_'):
                    setattr(Colors, attr, '')
        
        print(f"\n{Colors.GREEN if self.config['colors_enabled'] else ''}✓ Colors {'enabled' if self.config['colors_enabled'] else 'disabled'}{Colors.ENDC}")
    
    def configure_backup_settings(self):
        """Configure backup settings"""
        print(f"\n{Colors.CYAN}Backup Settings{Colors.ENDC}")
        
        current_dir = self.config.get('backup_dir', '~/ObsidianBackups')
        print(f"\nCurrent backup directory: {current_dir}")
        
        new_dir = input(f"\n{Colors.YELLOW}Enter new backup directory (or press Enter to keep current): {Colors.ENDC}").strip()
        if new_dir:
            self.config['backup_dir'] = new_dir
            self.save_config()
            print(f"{Colors.GREEN}✓ Backup directory updated{Colors.ENDC}")
        
        # Auto-backup settings
        auto_backup = self.config.get('auto_backup', False)
        print(f"\nAuto-backup is currently: {'enabled' if auto_backup else 'disabled'}")
        
        toggle = input(f"{Colors.YELLOW}Toggle auto-backup? (y/n): {Colors.ENDC}").lower()
        if toggle == 'y':
            self.config['auto_backup'] = not auto_backup
            self.save_config()
            print(f"{Colors.GREEN}✓ Auto-backup {'enabled' if self.config['auto_backup'] else 'disabled'}{Colors.ENDC}")
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        confirm = input(f"\n{Colors.YELLOW}Reset all settings to defaults? (y/n): {Colors.ENDC}").lower()
        if confirm == 'y':
            self.config = {}
            self.current_vault = ''
            self.save_config()
            print(f"\n{Colors.GREEN}✓ Settings reset to defaults{Colors.ENDC}")
    
    def configure_ai_settings(self):
        """Configure AI settings for v2"""
        print(f"\n{Colors.CYAN}AI Configuration{Colors.ENDC}")
        print("\nNote: These settings are saved to your vault's .obsidian-librarian/config.yaml")
        
        config_path = os.path.join(self.current_vault, '.obsidian-librarian', 'config.yaml')
        
        print(f"\n{Colors.YELLOW}Choose AI provider:{Colors.ENDC}")
        print("  1. Local (Ollama) - Private, no API key needed")
        print("  2. OpenAI - Requires API key")
        print("  3. Anthropic - Requires API key")
        
        choice = input(f"\n{Colors.YELLOW}Choice (1-3): {Colors.ENDC}").strip()
        
        if choice == '1':
            print(f"\n{Colors.CYAN}To use local AI with Ollama:{Colors.ENDC}")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Run: ollama pull llama2")
            print("3. The librarian will use it automatically")
        elif choice in ['2', '3']:
            provider = 'openai' if choice == '2' else 'anthropic'
            print(f"\n{Colors.YELLOW}Enter your {provider.upper()} API key:{Colors.ENDC}")
            api_key = input().strip()
            
            if api_key:
                # Create config directory
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                # Save to environment or config
                env_var = f"{provider.upper()}_API_KEY"
                print(f"\n{Colors.CYAN}Add this to your shell profile:{Colors.ENDC}")
                print(f"export {env_var}='{api_key}'")
                print(f"\n{Colors.GREEN}✓ Configuration saved{Colors.ENDC}")
    
    def cleanup(self):
        """Clean up resources when exiting (base implementation)"""
        # Base cleanup - can be overridden by subclasses
        pass
    
    def run(self):
        """Main application loop"""
        # Validate dependencies on first run
        if not self.config.get('deps_checked', False):
            self.validate_dependencies()
            self.config['deps_checked'] = True
            self.save_config()
        
        # Show welcome on first run
        if not self.config.get('welcomed', False):
            self.clear_screen()
            self.show_welcome_art()
            print(f"\n{Colors.CYAN}Welcome to the Obsidian Vault Manager!{Colors.ENDC}")
            print(f"{Colors.BLUE}This tool helps you organize, analyze, and backup your vault.{Colors.ENDC}")
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
        
        # Main menu loop
        while True:
            nlp_status = "🤖 AI Query Interface" if NLQ_AVAILABLE else "🤖 AI Query Interface (Not Available)"
            options = [
                ('1', '📊 Analyze Vault'),
                ('2', '🏷️  Manage Tags'),
                ('3', '💾 Backup Vault'),
                ('4', '🚀 V2 Features' if self.v2_available else '🚀 V2 Features (Not Installed)'),
                ('5', '🔧 Advanced Tools'),
                ('6', '📚 Help & Documentation'),
                ('7', '⚙️  Settings'),
                ('8', nlp_status),
                ('0', '👋 Exit')
            ]
            
            self.show_menu('MAIN MENU', options, 'Choose an option to get started!')
            choice = input().strip()
            
            if choice == '0':
                print(f"\n{Colors.GREEN}Thanks for using Obsidian Vault Manager!{Colors.ENDC}")
                print(f"{Colors.BLUE}Your vault is in good hands. 📚{Colors.ENDC}\n")
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
                self.advanced_tools_menu()
            elif choice == '6':
                self.show_help()
            elif choice == '7':
                self.settings_menu()
            elif choice == '8':
                if NLQ_AVAILABLE:
                    self.natural_language_interface()
                else:
                    print(f"\n{Colors.RED}❌ Natural language processing not available{Colors.ENDC}")
                    print(f"{Colors.YELLOW}Make sure natural_language_query.py is in the same directory{Colors.ENDC}")
                    time.sleep(2)
            else:
                print(f"\n{Colors.RED}Invalid choice. Please try again.{Colors.ENDC}")
                time.sleep(1)


def main():
    """Entry point"""
    try:
        manager = VaultManager()
        manager.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user.{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.ENDC}")
        sys.exit(1)


if __name__ == '__main__':
    main()
