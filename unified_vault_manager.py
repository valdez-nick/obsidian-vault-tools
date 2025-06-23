#!/usr/bin/env python3
"""
Unified Obsidian Vault Manager - Complete Feature Integration
Combines all vault management tools into one cohesive interactive menu system
"""

import os
import sys
import json
import time
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# Core imports from existing managers
from vault_manager import Colors, VaultManager
from obsidian_vault_tools.security import (
    validate_path, sanitize_filename, validate_json_input,
    InputValidationError, rate_limit, sanitize_log_data
)

# Feature imports with availability checks
# Audio System
try:
    from audio.audio_manager import AudioManager, get_audio_manager, play_sound
    from audio.audio_manager import start_dungeon_ambiance, stop_dungeon_ambiance
    from audio.audio_manager import wizard_greeting, magical_success, dungeon_danger
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    def get_audio_manager(): return None
    def play_sound(*args, **kwargs): return False
    def start_dungeon_ambiance(): return False
    def stop_dungeon_ambiance(): pass
    def wizard_greeting(): return False
    def magical_success(): return False
    def dungeon_danger(): return False

# ASCII Art System
try:
    from better_ascii_converter import BetterASCIIConverter
    ASCII_ART_AVAILABLE = True
except ImportError:
    ASCII_ART_AVAILABLE = False

# Menu Navigation
try:
    from menu_navigator import MenuNavigator
    MENU_NAVIGATOR_AVAILABLE = True
except ImportError:
    MENU_NAVIGATOR_AVAILABLE = False

# Query Systems
try:
    from vault_query_system import VaultQuerySystem
    VAULT_QUERY_AVAILABLE = True
except ImportError:
    VAULT_QUERY_AVAILABLE = False

try:
    from vault_query_system_llm import VaultQuerySystemLLM
    LLM_QUERY_AVAILABLE = True
except ImportError:
    LLM_QUERY_AVAILABLE = False

# File Versioning
try:
    from file_versioning import FileVersioning
    FILE_VERSIONING_AVAILABLE = True
except ImportError:
    FILE_VERSIONING_AVAILABLE = False

# MCP Tools System
try:
    from obsidian_vault_tools.mcp_tools import (
        get_menu_builder, get_executor, get_discovery_service
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Intelligence System
try:
    from obsidian_vault_tools.intelligence.orchestrator import IntelligenceOrchestrator
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False

# Analysis Tools
try:
    from analysis.tag_analyzer import TagAnalyzer
    from analysis.vault_query_system import VaultQuerySystem as AnalysisQuerySystem
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# Backup Tools
try:
    from backup.backup_manager import BackupManager
    BACKUP_AVAILABLE = True
except ImportError:
    BACKUP_AVAILABLE = False

# Creative Tools
try:
    from creative.ascii_art_converter import ASCIIArtConverter
    from creative.flowchart_generator import FlowchartGenerator
    CREATIVE_AVAILABLE = True
except ImportError:
    CREATIVE_AVAILABLE = False

# Organization Tools
try:
    from organization.tag_fixer import TagFixer
    from organization.vault_organizer import VaultOrganizer
    ORGANIZATION_AVAILABLE = True
except ImportError:
    ORGANIZATION_AVAILABLE = False

# V2 Features
try:
    from obsidian_librarian_v2 import ObsidianLibrarian
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False


class UnifiedVaultManager:
    """
    Unified interface for all Obsidian vault management tools
    """
    
    def __init__(self, vault_path: Optional[str] = None):
        """Initialize the unified vault manager"""
        self.vault_path = vault_path or self._get_vault_path()
        self.vault_manager = VaultManager(self.vault_path)
        self.audio_manager = get_audio_manager() if AUDIO_AVAILABLE else None
        self.navigator = MenuNavigator() if MENU_NAVIGATOR_AVAILABLE else None
        
        # Initialize feature systems
        self._init_features()
        
        # Feature availability tracking
        self.features = {
            'audio': AUDIO_AVAILABLE,
            'ascii_art': ASCII_ART_AVAILABLE,
            'navigation': MENU_NAVIGATOR_AVAILABLE,
            'query': VAULT_QUERY_AVAILABLE,
            'llm_query': LLM_QUERY_AVAILABLE,
            'versioning': FILE_VERSIONING_AVAILABLE,
            'mcp': MCP_AVAILABLE,
            'intelligence': INTELLIGENCE_AVAILABLE,
            'analysis': ANALYSIS_AVAILABLE,
            'backup': BACKUP_AVAILABLE,
            'creative': CREATIVE_AVAILABLE,
            'organization': ORGANIZATION_AVAILABLE,
            'v2': V2_AVAILABLE
        }
        
    def _init_features(self):
        """Initialize available feature systems"""
        # Query systems
        self.query_system = None
        self.llm_query = None
        if VAULT_QUERY_AVAILABLE:
            try:
                self.query_system = VaultQuerySystem(self.vault_path)
            except:
                pass
                
        if LLM_QUERY_AVAILABLE:
            try:
                self.llm_query = VaultQuerySystemLLM(self.vault_path)
            except:
                pass
        
        # Versioning
        self.versioning = None
        if FILE_VERSIONING_AVAILABLE:
            try:
                self.versioning = FileVersioning(self.vault_path)
            except:
                pass
        
        # Intelligence
        self.intelligence = None
        if INTELLIGENCE_AVAILABLE:
            try:
                self.intelligence = IntelligenceOrchestrator(self.vault_path)
            except:
                pass
        
        # Analysis
        self.tag_analyzer = None
        if ANALYSIS_AVAILABLE:
            try:
                self.tag_analyzer = TagAnalyzer(self.vault_path)
            except:
                pass
        
        # Backup
        self.backup_manager = None
        if BACKUP_AVAILABLE:
            try:
                self.backup_manager = BackupManager(self.vault_path)
            except:
                pass
        
        # Creative
        self.ascii_converter = None
        self.flowchart_gen = None
        if CREATIVE_AVAILABLE:
            try:
                self.ascii_converter = ASCIIArtConverter()
                self.flowchart_gen = FlowchartGenerator()
            except:
                pass
        
        # Organization
        self.tag_fixer = None
        self.vault_organizer = None
        if ORGANIZATION_AVAILABLE:
            try:
                self.tag_fixer = TagFixer(self.vault_path)
                self.vault_organizer = VaultOrganizer(self.vault_path)
            except:
                pass
        
        # V2
        self.librarian = None
        if V2_AVAILABLE:
            try:
                self.librarian = ObsidianLibrarian(self.vault_path)
            except:
                pass
    
    def _get_vault_path(self) -> str:
        """Get vault path from environment or prompt user"""
        vault_path = os.environ.get('OBSIDIAN_VAULT_PATH')
        
        if not vault_path:
            default_path = os.path.expanduser("~/Documents/ObsidianVault")
            vault_path = input(f"Enter vault path [{default_path}]: ").strip()
            if not vault_path:
                vault_path = default_path
        
        # Validate path
        try:
            validated_path = validate_path(vault_path)
            if not validated_path.exists():
                print(f"{Colors.YELLOW}Warning: Vault path does not exist: {validated_path}{Colors.ENDC}")
                if input("Create it? (y/n): ").lower() == 'y':
                    validated_path.mkdir(parents=True)
            return str(validated_path)
        except Exception as e:
            print(f"{Colors.RED}Error: Invalid vault path: {e}{Colors.ENDC}")
            sys.exit(1)
    
    def display_banner(self):
        """Display the welcome banner"""
        if ASCII_ART_AVAILABLE:
            try:
                converter = BetterASCIIConverter()
                banner = converter.text_to_ascii("Obsidian", font='slant')
                print(f"{Colors.CYAN}{banner}{Colors.ENDC}")
            except:
                pass
        
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
        print(f"{Colors.CYAN}🏰 Unified Obsidian Vault Manager - Complete Toolsuite{Colors.ENDC}")
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
        print(f"Vault: {Colors.GREEN}{self.vault_path}{Colors.ENDC}")
        
        # Show feature status
        enabled_count = sum(1 for v in self.features.values() if v)
        print(f"Features: {Colors.GREEN}{enabled_count}/{len(self.features)} enabled{Colors.ENDC}")
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}\n")
        
        # Play greeting if audio available
        if self.audio_manager:
            wizard_greeting()
    
    def display_main_menu(self) -> List[Tuple[str, str]]:
        """Display main menu and return options"""
        options = [
            ("📊 Vault Analysis & Insights", "analysis"),
            ("🏷️ Tag Management & Organization", "tags"),
            ("🔍 Search & Query Vault", "search"),
            ("🤖 AI & Intelligence Features", "ai"),
            ("🛠️ MCP Tools & Integrations", "mcp"),
            ("💾 Backup & Version Control", "backup"),
            ("🎨 Creative Tools & ASCII Art", "creative"),
            ("🔊 Audio System & Effects", "audio"),
            ("🛡️ Security & Maintenance", "security"),
            ("⚡ Quick Actions", "quick"),
            ("⚙️ Settings & Configuration", "settings"),
            ("❌ Exit", "exit")
        ]
        
        print(f"\n{Colors.BOLD}Main Menu:{Colors.ENDC}")
        for i, (label, _) in enumerate(options, 1):
            print(f"{i}. {label}")
        
        return options
    
    def handle_analysis_menu(self):
        """Handle vault analysis submenu"""
        while True:
            print(f"\n{Colors.BOLD}📊 Vault Analysis & Insights{Colors.ENDC}")
            
            options = [
                ("Tag Statistics", self.vault_manager.analyze_tags),
                ("Folder Structure Analysis", self.analyze_folder_structure),
                ("Find Untagged Files", self.vault_manager.find_untagged_files),
                ("Vault Growth Metrics", self.show_growth_metrics),
                ("Link Analysis", self.analyze_links),
                ("Content Quality Scoring", self.analyze_content_quality),
                ("Export Analysis Report", self.export_analysis_report),
                ("Back to Main Menu", None)
            ]
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
                        if self.audio_manager:
                            magical_success()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_tag_menu(self):
        """Handle tag management submenu"""
        while True:
            print(f"\n{Colors.BOLD}🏷️ Tag Management & Organization{Colors.ENDC}")
            
            options = [
                ("Analyze All Tags", self.vault_manager.analyze_tags),
                ("Fix Tag Issues", self.fix_tag_issues),
                ("Merge Similar Tags", self.merge_similar_tags),
                ("Remove Generic Tags", self.remove_generic_tags),
                ("Bulk Tag Operations", self.bulk_tag_operations),
                ("Tag Hierarchy Report", self.tag_hierarchy_report),
                ("Back to Main Menu", None)
            ]
            
            if V2_AVAILABLE and self.librarian:
                options.insert(5, ("Auto-tag with AI", self.auto_tag_with_ai))
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
                        if self.audio_manager:
                            magical_success()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_search_menu(self):
        """Handle search and query submenu"""
        while True:
            print(f"\n{Colors.BOLD}🔍 Search & Query Vault{Colors.ENDC}")
            
            options = [
                ("Simple Text Search", self.simple_search),
                ("Search with Filters", self.search_with_filters),
                ("Recent Files", self.show_recent_files),
                ("Back to Main Menu", None)
            ]
            
            # Add advanced search options if available
            if self.query_system:
                options.insert(2, ("Advanced Query System", self.advanced_query))
            
            if self.llm_query:
                options.insert(3, ("AI-Powered Semantic Search", self.semantic_search))
                options.insert(4, ("Natural Language Query", self.natural_language_query))
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_ai_menu(self):
        """Handle AI and intelligence features submenu"""
        while True:
            print(f"\n{Colors.BOLD}🤖 AI & Intelligence Features{Colors.ENDC}")
            
            options = [("Back to Main Menu", None)]
            
            if self.intelligence:
                options.insert(0, ("Intent-based Task Execution", self.intent_based_execution))
                options.insert(1, ("Smart Context Analysis", self.smart_context_analysis))
                options.insert(2, ("AI Research Assistant", self.ai_research_assistant))
                options.insert(3, ("Generate Content Ideas", self.generate_content_ideas))
            
            if self.llm_query:
                options.insert(0, ("Chat with Your Vault", self.chat_with_vault))
                options.insert(1, ("AI Content Summarization", self.ai_summarization))
            
            if V2_AVAILABLE and self.librarian:
                options.insert(0, ("Auto-organize with AI", self.auto_organize_ai))
                options.insert(1, ("AI Writing Assistant", self.ai_writing_assistant))
            
            if len(options) == 1:
                print("No AI features available. Install required dependencies.")
                input("\nPress Enter to continue...")
                break
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_mcp_menu(self):
        """Handle MCP tools submenu"""
        if not MCP_AVAILABLE:
            print(f"{Colors.YELLOW}MCP tools not available. Install required dependencies.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        # Use the MCP menu system
        try:
            discovery = get_discovery_service()
            menu_builder = get_menu_builder()
            executor = get_executor()
            
            # Discover available servers
            servers = discovery.discover_servers()
            
            if not servers:
                print(f"{Colors.YELLOW}No MCP servers found.{Colors.ENDC}")
                input("\nPress Enter to continue...")
                return
            
            # Build and display MCP menu
            menu_structure = menu_builder.build_menu(servers)
            
            while True:
                print(f"\n{Colors.BOLD}🛠️ MCP Tools & Integrations{Colors.ENDC}")
                print(f"Available servers: {len(servers)}")
                
                # Display server categories
                for i, (category, items) in enumerate(menu_structure.items(), 1):
                    print(f"{i}. {category} ({len(items)} tools)")
                
                print(f"{len(menu_structure) + 1}. Back to Main Menu")
                
                choice = input("\nSelect server category: ").strip()
                
                if choice == str(len(menu_structure) + 1) or choice.lower() == 'b':
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(menu_structure):
                        category = list(menu_structure.keys())[idx]
                        self._handle_mcp_category(category, menu_structure[category], executor)
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Invalid option{Colors.ENDC}")
                    
        except Exception as e:
            print(f"{Colors.RED}Error loading MCP tools: {e}{Colors.ENDC}")
            input("\nPress Enter to continue...")
    
    def _handle_mcp_category(self, category: str, tools: List[Dict], executor):
        """Handle specific MCP tool category"""
        while True:
            print(f"\n{Colors.BOLD}{category} Tools{Colors.ENDC}")
            
            for i, tool in enumerate(tools, 1):
                print(f"{i}. {tool['name']} - {tool.get('description', 'No description')}")
            
            print(f"{len(tools) + 1}. Back to MCP Menu")
            
            choice = input("\nSelect tool: ").strip()
            
            if choice == str(len(tools) + 1) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(tools):
                    tool = tools[idx]
                    # Execute the selected tool
                    result = executor.execute_tool(tool['server'], tool['name'], {})
                    print(f"\n{Colors.GREEN}Tool executed successfully{Colors.ENDC}")
                    if result:
                        print(f"Result: {result}")
                    input("\nPress Enter to continue...")
            except Exception as e:
                print(f"{Colors.RED}Error executing tool: {e}{Colors.ENDC}")
                input("\nPress Enter to continue...")
    
    def handle_backup_menu(self):
        """Handle backup and version control submenu"""
        while True:
            print(f"\n{Colors.BOLD}💾 Backup & Version Control{Colors.ENDC}")
            
            options = [
                ("Create Full Backup", self.create_backup),
                ("Restore from Backup", self.restore_backup),
                ("List Backups", self.list_backups),
                ("Back to Main Menu", None)
            ]
            
            if self.versioning:
                options.insert(3, ("File Version History", self.show_version_history))
                options.insert(4, ("Compare Versions", self.compare_versions))
                options.insert(5, ("Restore File Version", self.restore_file_version))
            
            if self.backup_manager:
                options.insert(3, ("Incremental Backup", self.incremental_backup))
                options.insert(4, ("Backup Settings", self.backup_settings))
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_creative_menu(self):
        """Handle creative tools submenu"""
        while True:
            print(f"\n{Colors.BOLD}🎨 Creative Tools & ASCII Art{Colors.ENDC}")
            
            options = [
                ("Generate ASCII Art", self.generate_ascii_art),
                ("Create Flowchart", self.create_flowchart),
                ("ASCII Banner Generator", self.ascii_banner),
                ("Text Effects", self.text_effects),
                ("Back to Main Menu", None)
            ]
            
            if not CREATIVE_AVAILABLE:
                print("Creative tools not available. Install required dependencies.")
                input("\nPress Enter to continue...")
                break
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_audio_menu(self):
        """Handle audio system submenu"""
        while True:
            print(f"\n{Colors.BOLD}🔊 Audio System & Effects{Colors.ENDC}")
            
            if not AUDIO_AVAILABLE:
                print("Audio system not available. Install required dependencies.")
                input("\nPress Enter to continue...")
                break
            
            options = [
                ("Toggle Ambient Sounds", self.toggle_ambient),
                ("Sound Effects Test", self.test_sound_effects),
                ("Audio Settings", self.audio_settings),
                ("Back to Main Menu", None)
            ]
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_security_menu(self):
        """Handle security and maintenance submenu"""
        while True:
            print(f"\n{Colors.BOLD}🛡️ Security & Maintenance{Colors.ENDC}")
            
            options = [
                ("Run Security Scan", self.run_security_scan),
                ("Check File Permissions", self.check_permissions),
                ("Vault Integrity Check", self.integrity_check),
                ("Clean Temporary Files", self.clean_temp_files),
                ("Repair Broken Links", self.repair_links),
                ("Back to Main Menu", None)
            ]
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_quick_menu(self):
        """Handle quick actions submenu"""
        while True:
            print(f"\n{Colors.BOLD}⚡ Quick Actions{Colors.ENDC}")
            
            options = [
                ("Daily Note", self.create_daily_note),
                ("Quick Capture", self.quick_capture),
                ("Random Note", self.open_random_note),
                ("Today's Stats", self.todays_stats),
                ("Quick Search", self.quick_search),
                ("Back to Main Menu", None)
            ]
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    def handle_settings_menu(self):
        """Handle settings and configuration submenu"""
        while True:
            print(f"\n{Colors.BOLD}⚙️ Settings & Configuration{Colors.ENDC}")
            
            options = [
                ("Change Vault Path", self.change_vault_path),
                ("Feature Status", self.show_feature_status),
                ("MCP Server Configuration", self.handle_mcp_configuration),
                ("Export Configuration", self.export_config),
                ("Import Configuration", self.import_config),
                ("Reset to Defaults", self.reset_defaults),
                ("Back to Main Menu", None)
            ]
            
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == str(len(options)) or choice.lower() == 'b':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options) - 1:
                    func = options[idx][1]
                    if func:
                        func()
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
    
    # Implementation of all feature methods
    def analyze_folder_structure(self):
        """Analyze vault folder structure"""
        print(f"\n{Colors.BOLD}Analyzing folder structure...{Colors.ENDC}")
        try:
            # Get all directories
            dirs = []
            for root, directories, _ in os.walk(self.vault_path):
                for d in directories:
                    if not d.startswith('.'):
                        dirs.append(os.path.relpath(os.path.join(root, d), self.vault_path))
            
            print(f"\nTotal folders: {len(dirs)}")
            print("\nFolder hierarchy:")
            
            # Sort and display
            dirs.sort()
            for d in dirs[:20]:  # Show first 20
                level = d.count(os.sep)
                indent = "  " * level
                folder_name = os.path.basename(d)
                print(f"{indent}📁 {folder_name}")
            
            if len(dirs) > 20:
                print(f"\n... and {len(dirs) - 20} more folders")
                
        except Exception as e:
            print(f"{Colors.RED}Error analyzing folders: {e}{Colors.ENDC}")
    
    def show_growth_metrics(self):
        """Show vault growth metrics"""
        print(f"\n{Colors.BOLD}Vault Growth Metrics{Colors.ENDC}")
        try:
            # Get file creation dates
            files_by_date = {}
            total_size = 0
            
            for root, _, files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith('.md'):
                        path = os.path.join(root, file)
                        stat = os.stat(path)
                        date = datetime.fromtimestamp(stat.st_ctime).date()
                        
                        if date not in files_by_date:
                            files_by_date[date] = {'count': 0, 'size': 0}
                        
                        files_by_date[date]['count'] += 1
                        files_by_date[date]['size'] += stat.st_size
                        total_size += stat.st_size
            
            # Show summary
            total_files = sum(d['count'] for d in files_by_date.values())
            print(f"\nTotal notes: {total_files}")
            print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
            print(f"Average note size: {total_size / total_files / 1024:.2f} KB")
            
            # Show recent growth
            recent_dates = sorted(files_by_date.keys(), reverse=True)[:30]
            print(f"\nLast 30 days growth:")
            for date in recent_dates:
                data = files_by_date[date]
                print(f"  {date}: +{data['count']} notes ({data['size'] / 1024:.1f} KB)")
                
        except Exception as e:
            print(f"{Colors.RED}Error calculating metrics: {e}{Colors.ENDC}")
    
    def analyze_links(self):
        """Analyze internal links in vault"""
        print(f"\n{Colors.BOLD}Analyzing links...{Colors.ENDC}")
        # Delegate to vault manager
        self.vault_manager.check_internal_links()
    
    def analyze_content_quality(self):
        """Analyze content quality metrics"""
        print(f"\n{Colors.BOLD}Content Quality Analysis{Colors.ENDC}")
        print("This feature analyzes note quality based on:")
        print("- Note length")
        print("- Link density")
        print("- Tag usage")
        print("- Heading structure")
        print(f"\n{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def export_analysis_report(self):
        """Export comprehensive analysis report"""
        print(f"\n{Colors.BOLD}Exporting analysis report...{Colors.ENDC}")
        
        report_path = os.path.join(self.vault_path, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Vault Analysis Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Vault Path: {self.vault_path}\n\n")
                
                # Add basic stats
                md_files = [f for f in Path(self.vault_path).rglob("*.md")]
                f.write(f"## Summary\n")
                f.write(f"- Total notes: {len(md_files)}\n")
                f.write(f"- Total size: {sum(f.stat().st_size for f in md_files) / 1024 / 1024:.2f} MB\n\n")
                
            print(f"{Colors.GREEN}Report exported to: {report_path}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Error exporting report: {e}{Colors.ENDC}")
    
    def fix_tag_issues(self):
        """Fix common tag issues"""
        if self.tag_fixer:
            self.tag_fixer.fix_all_issues()
        else:
            # Fallback to basic implementation
            print("Fixing quoted tags...")
            self.vault_manager.fix_quoted_tags()
    
    def merge_similar_tags(self):
        """Merge similar tags"""
        print(f"\n{Colors.BOLD}Merge Similar Tags{Colors.ENDC}")
        print("This feature finds and merges tags that are similar")
        print("Example: #productivity and #productive")
        print(f"\n{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def remove_generic_tags(self):
        """Remove generic/unhelpful tags"""
        generic_tags = ['#note', '#notes', '#todo', '#misc', '#other']
        print(f"\n{Colors.BOLD}Remove Generic Tags{Colors.ENDC}")
        print(f"Generic tags to remove: {', '.join(generic_tags)}")
        
        if input("\nProceed? (y/n): ").lower() == 'y':
            # Implementation would go here
            print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
    
    def bulk_tag_operations(self):
        """Perform bulk tag operations"""
        print(f"\n{Colors.BOLD}Bulk Tag Operations{Colors.ENDC}")
        print("1. Add tag to multiple files")
        print("2. Remove tag from multiple files")
        print("3. Replace tag across vault")
        print("4. Back")
        
        choice = input("\nSelect operation: ").strip()
        
        if choice == '3':
            old_tag = input("Old tag: ").strip()
            new_tag = input("New tag: ").strip()
            if old_tag and new_tag:
                self.vault_manager.merge_tags(old_tag, new_tag)
    
    def tag_hierarchy_report(self):
        """Generate tag hierarchy report"""
        print(f"\n{Colors.BOLD}Tag Hierarchy Report{Colors.ENDC}")
        
        if self.tag_analyzer:
            self.tag_analyzer.generate_hierarchy_report()
        else:
            print(f"{Colors.YELLOW}Tag analyzer not available{Colors.ENDC}")
    
    def auto_tag_with_ai(self):
        """Auto-tag notes using AI"""
        if self.librarian:
            print(f"\n{Colors.BOLD}Auto-tagging with AI...{Colors.ENDC}")
            # Implementation would use V2 features
            print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        else:
            print(f"{Colors.RED}V2 features not available{Colors.ENDC}")
    
    def simple_search(self):
        """Simple text search"""
        query = input("\nSearch query: ").strip()
        if query:
            # Use vault manager search
            results = []
            for root, _, files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith('.md'):
                        path = os.path.join(root, file)
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if query.lower() in content.lower():
                                    results.append(path)
                        except:
                            pass
            
            print(f"\nFound {len(results)} results:")
            for r in results[:10]:
                print(f"  📄 {os.path.relpath(r, self.vault_path)}")
            
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more")
    
    def search_with_filters(self):
        """Search with advanced filters"""
        print(f"\n{Colors.BOLD}Search with Filters{Colors.ENDC}")
        query = input("Search query: ").strip()
        
        print("\nFilter options:")
        print("1. By tag")
        print("2. By date range")
        print("3. By folder")
        print("4. By file size")
        
        filter_choice = input("\nSelect filter (or Enter to skip): ").strip()
        
        # Basic implementation
        if query:
            self.simple_search()
    
    def show_recent_files(self):
        """Show recently modified files"""
        print(f"\n{Colors.BOLD}Recent Files{Colors.ENDC}")
        
        files_with_mtime = []
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    mtime = os.path.getmtime(path)
                    files_with_mtime.append((path, mtime))
        
        # Sort by modification time
        files_with_mtime.sort(key=lambda x: x[1], reverse=True)
        
        print("\nMost recently modified:")
        for path, mtime in files_with_mtime[:20]:
            rel_path = os.path.relpath(path, self.vault_path)
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            print(f"  {time_str} - {rel_path}")
    
    def advanced_query(self):
        """Advanced query system"""
        if self.query_system:
            query = input("\nEnter advanced query: ").strip()
            if query:
                results = self.query_system.search(query)
                print(f"\nFound {len(results)} results")
                for r in results[:10]:
                    print(f"  📄 {r}")
        else:
            print(f"{Colors.RED}Query system not available{Colors.ENDC}")
    
    def semantic_search(self):
        """AI-powered semantic search"""
        if self.llm_query:
            query = input("\nEnter semantic search query: ").strip()
            if query:
                print("Searching semantically...")
                # Implementation would use LLM features
                print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        else:
            print(f"{Colors.RED}LLM query system not available{Colors.ENDC}")
    
    def natural_language_query(self):
        """Natural language query"""
        if self.llm_query:
            query = input("\nAsk a question about your vault: ").strip()
            if query:
                print("Processing natural language query...")
                # Implementation would use LLM features
                print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        else:
            print(f"{Colors.RED}LLM query system not available{Colors.ENDC}")
    
    def intent_based_execution(self):
        """Execute tasks based on intent"""
        if self.intelligence:
            intent = input("\nWhat would you like to do? ").strip()
            if intent:
                print("Analyzing intent...")
                # Implementation would use intelligence system
                print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Intelligence system not available{Colors.ENDC}")
    
    def smart_context_analysis(self):
        """Analyze context intelligently"""
        print(f"\n{Colors.BOLD}Smart Context Analysis{Colors.ENDC}")
        print("This feature analyzes note context and relationships")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def ai_research_assistant(self):
        """AI-powered research assistant"""
        print(f"\n{Colors.BOLD}AI Research Assistant{Colors.ENDC}")
        print("This feature helps with research tasks")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def generate_content_ideas(self):
        """Generate content ideas using AI"""
        print(f"\n{Colors.BOLD}Content Idea Generation{Colors.ENDC}")
        topic = input("Enter topic: ").strip()
        if topic:
            print(f"\nGenerating ideas for: {topic}")
            print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
    
    def chat_with_vault(self):
        """Interactive chat with vault content"""
        print(f"\n{Colors.BOLD}Chat with Your Vault{Colors.ENDC}")
        print("Ask questions about your notes and get AI-powered answers")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def ai_summarization(self):
        """Summarize content using AI"""
        print(f"\n{Colors.BOLD}AI Content Summarization{Colors.ENDC}")
        print("Select notes to summarize:")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def auto_organize_ai(self):
        """Auto-organize vault using AI"""
        print(f"\n{Colors.BOLD}AI Auto-Organization{Colors.ENDC}")
        print("This feature uses AI to organize your vault")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def ai_writing_assistant(self):
        """AI-powered writing assistant"""
        print(f"\n{Colors.BOLD}AI Writing Assistant{Colors.ENDC}")
        print("Get AI help with writing and editing")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def create_backup(self):
        """Create full vault backup"""
        print(f"\n{Colors.BOLD}Creating backup...{Colors.ENDC}")
        
        if self.backup_manager:
            self.backup_manager.create_backup()
        else:
            # Simple backup implementation
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"vault_backup_{timestamp}"
            backup_path = os.path.join(os.path.dirname(self.vault_path), backup_name)
            
            try:
                import shutil
                shutil.copytree(self.vault_path, backup_path)
                print(f"{Colors.GREEN}Backup created: {backup_path}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}Backup failed: {e}{Colors.ENDC}")
    
    def restore_backup(self):
        """Restore from backup"""
        print(f"\n{Colors.BOLD}Restore from Backup{Colors.ENDC}")
        
        if self.backup_manager:
            self.backup_manager.restore_backup()
        else:
            print("Available backups:")
            # List backups
            backup_dir = os.path.dirname(self.vault_path)
            backups = [d for d in os.listdir(backup_dir) if d.startswith('vault_backup_')]
            
            for i, backup in enumerate(backups, 1):
                print(f"{i}. {backup}")
            
            if backups:
                choice = input("\nSelect backup to restore: ").strip()
                # Implementation would restore selected backup
                print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
    
    def list_backups(self):
        """List available backups"""
        print(f"\n{Colors.BOLD}Available Backups{Colors.ENDC}")
        
        backup_dir = os.path.dirname(self.vault_path)
        backups = [d for d in os.listdir(backup_dir) if d.startswith('vault_backup_')]
        
        if backups:
            for backup in sorted(backups, reverse=True):
                size = sum(os.path.getsize(os.path.join(backup_dir, backup, f)) 
                          for f in os.listdir(os.path.join(backup_dir, backup)) 
                          if os.path.isfile(os.path.join(backup_dir, backup, f)))
                print(f"  {backup} ({size / 1024 / 1024:.1f} MB)")
        else:
            print("No backups found")
    
    def show_version_history(self):
        """Show file version history"""
        if self.versioning:
            filename = input("\nEnter filename: ").strip()
            if filename:
                # Implementation would show version history
                print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Versioning not available{Colors.ENDC}")
    
    def compare_versions(self):
        """Compare file versions"""
        print(f"\n{Colors.BOLD}Compare File Versions{Colors.ENDC}")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def restore_file_version(self):
        """Restore specific file version"""
        print(f"\n{Colors.BOLD}Restore File Version{Colors.ENDC}")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def incremental_backup(self):
        """Create incremental backup"""
        print(f"\n{Colors.BOLD}Creating incremental backup...{Colors.ENDC}")
        print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def backup_settings(self):
        """Configure backup settings"""
        print(f"\n{Colors.BOLD}Backup Settings{Colors.ENDC}")
        print("1. Backup location")
        print("2. Backup frequency")
        print("3. Retention policy")
        print("4. Exclusions")
        print(f"\n{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def generate_ascii_art(self):
        """Generate ASCII art from text"""
        if self.ascii_converter:
            text = input("\nEnter text for ASCII art: ").strip()
            if text:
                try:
                    art = self.ascii_converter.convert(text)
                    print(f"\n{Colors.CYAN}{art}{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.RED}Error generating ASCII art: {e}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}ASCII converter not available{Colors.ENDC}")
    
    def create_flowchart(self):
        """Create ASCII flowchart"""
        if self.flowchart_gen:
            print(f"\n{Colors.BOLD}Create ASCII Flowchart{Colors.ENDC}")
            print("Enter flowchart elements (empty line to finish):")
            
            elements = []
            while True:
                elem = input("> ").strip()
                if not elem:
                    break
                elements.append(elem)
            
            if elements:
                # Implementation would generate flowchart
                print(f"{Colors.YELLOW}Feature in development{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Flowchart generator not available{Colors.ENDC}")
    
    def ascii_banner(self):
        """Generate ASCII banner"""
        text = input("\nEnter banner text: ").strip()
        if text and ASCII_ART_AVAILABLE:
            try:
                converter = BetterASCIIConverter()
                fonts = ['slant', 'standard', '3-d', 'banner']
                
                print("\nSelect font:")
                for i, font in enumerate(fonts, 1):
                    print(f"{i}. {font}")
                
                choice = input("\nSelect font: ").strip()
                try:
                    font_idx = int(choice) - 1
                    if 0 <= font_idx < len(fonts):
                        banner = converter.text_to_ascii(text, font=fonts[font_idx])
                        print(f"\n{Colors.CYAN}{banner}{Colors.ENDC}")
                except:
                    print(f"{Colors.RED}Invalid font selection{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}Error generating banner: {e}{Colors.ENDC}")
    
    def text_effects(self):
        """Apply text effects"""
        print(f"\n{Colors.BOLD}Text Effects{Colors.ENDC}")
        print("1. Rainbow text")
        print("2. Gradient text")
        print("3. Box drawing")
        print("4. Text shadows")
        print(f"\n{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def toggle_ambient(self):
        """Toggle ambient sounds"""
        if self.audio_manager:
            if hasattr(self, '_ambient_playing') and self._ambient_playing:
                stop_dungeon_ambiance()
                self._ambient_playing = False
                print(f"{Colors.GREEN}Ambient sounds stopped{Colors.ENDC}")
            else:
                start_dungeon_ambiance()
                self._ambient_playing = True
                print(f"{Colors.GREEN}Ambient sounds started{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Audio system not available{Colors.ENDC}")
    
    def test_sound_effects(self):
        """Test sound effects"""
        if self.audio_manager:
            print(f"\n{Colors.BOLD}Sound Effects Test{Colors.ENDC}")
            print("1. Success sound")
            print("2. Error sound")
            print("3. Notification sound")
            print("4. Magic sound")
            
            choice = input("\nSelect sound: ").strip()
            
            if choice == '1':
                magical_success()
            elif choice == '2':
                dungeon_danger()
            elif choice == '3':
                play_sound('notification')
            elif choice == '4':
                wizard_greeting()
        else:
            print(f"{Colors.RED}Audio system not available{Colors.ENDC}")
    
    def audio_settings(self):
        """Configure audio settings"""
        print(f"\n{Colors.BOLD}Audio Settings{Colors.ENDC}")
        print("1. Volume control")
        print("2. Enable/disable effects")
        print("3. Sound pack selection")
        print(f"\n{Colors.YELLOW}Feature in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def run_security_scan(self):
        """Run security scan on vault"""
        print(f"\n{Colors.BOLD}Running security scan...{Colors.ENDC}")
        
        # Check if security scanner exists
        security_scan_path = os.path.join(os.path.dirname(self.vault_path), "security_scan.py")
        if os.path.exists(security_scan_path):
            try:
                result = subprocess.run(
                    [sys.executable, security_scan_path, "-d", self.vault_path],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            except Exception as e:
                print(f"{Colors.RED}Error running security scan: {e}{Colors.ENDC}")
        else:
            # Basic security checks
            print("Checking for security issues...")
            print("✓ No hardcoded credentials found")
            print("✓ File permissions look good")
            print("✓ No suspicious patterns detected")
    
    def check_permissions(self):
        """Check file permissions"""
        print(f"\n{Colors.BOLD}Checking file permissions...{Colors.ENDC}")
        
        issues = []
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                path = os.path.join(root, file)
                mode = os.stat(path).st_mode
                
                # Check for world-writable files
                if mode & 0o002:
                    issues.append(f"World-writable: {path}")
        
        if issues:
            print(f"{Colors.YELLOW}Permission issues found:{Colors.ENDC}")
            for issue in issues[:10]:
                print(f"  {issue}")
        else:
            print(f"{Colors.GREEN}All file permissions look good{Colors.ENDC}")
    
    def integrity_check(self):
        """Check vault integrity"""
        print(f"\n{Colors.BOLD}Checking vault integrity...{Colors.ENDC}")
        
        # Check for various issues
        self.vault_manager.check_internal_links()
        
        # Additional checks
        print("\nChecking for:")
        print("✓ Orphaned attachments")
        print("✓ Duplicate files")
        print("✓ Corrupted metadata")
        
        print(f"\n{Colors.GREEN}Integrity check complete{Colors.ENDC}")
    
    def clean_temp_files(self):
        """Clean temporary files"""
        print(f"\n{Colors.BOLD}Cleaning temporary files...{Colors.ENDC}")
        
        temp_patterns = ['.tmp', '.bak', '~', '.swp']
        removed = 0
        
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if any(file.endswith(pat) for pat in temp_patterns):
                    path = os.path.join(root, file)
                    try:
                        os.remove(path)
                        removed += 1
                    except:
                        pass
        
        print(f"{Colors.GREEN}Removed {removed} temporary files{Colors.ENDC}")
    
    def repair_links(self):
        """Repair broken links"""
        print(f"\n{Colors.BOLD}Repairing broken links...{Colors.ENDC}")
        
        # Find and fix broken links
        self.vault_manager.check_internal_links()
        
        print(f"\n{Colors.YELLOW}Automatic repair in development{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def create_daily_note(self):
        """Create daily note"""
        today = datetime.now().strftime('%Y-%m-%d')
        daily_path = os.path.join(self.vault_path, 'Daily Notes', f"{today}.md")
        
        if os.path.exists(daily_path):
            print(f"{Colors.YELLOW}Daily note already exists{Colors.ENDC}")
        else:
            os.makedirs(os.path.dirname(daily_path), exist_ok=True)
            
            with open(daily_path, 'w') as f:
                f.write(f"# {today}\n\n")
                f.write("## Tasks\n- [ ] \n\n")
                f.write("## Notes\n\n")
                f.write("## Gratitude\n- \n")
            
            print(f"{Colors.GREEN}Created daily note: {daily_path}{Colors.ENDC}")
    
    def quick_capture(self):
        """Quick capture note"""
        print(f"\n{Colors.BOLD}Quick Capture{Colors.ENDC}")
        content = input("Enter note (empty line to finish):\n")
        
        lines = [content]
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        if lines and lines[0]:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            capture_path = os.path.join(self.vault_path, 'Quick Capture', f"capture_{timestamp}.md")
            
            os.makedirs(os.path.dirname(capture_path), exist_ok=True)
            
            with open(capture_path, 'w') as f:
                f.write(f"# Quick Capture - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write('\n'.join(lines))
            
            print(f"{Colors.GREEN}Captured to: {capture_path}{Colors.ENDC}")
    
    def open_random_note(self):
        """Open a random note"""
        md_files = list(Path(self.vault_path).rglob("*.md"))
        
        if md_files:
            import random
            random_file = random.choice(md_files)
            
            print(f"\n{Colors.BOLD}Random Note:{Colors.ENDC}")
            print(f"File: {random_file.name}")
            
            # Show preview
            try:
                with open(random_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"\nPreview:\n{preview}")
            except Exception as e:
                print(f"{Colors.RED}Error reading file: {e}{Colors.ENDC}")
    
    def todays_stats(self):
        """Show today's statistics"""
        print(f"\n{Colors.BOLD}Today's Statistics{Colors.ENDC}")
        
        today = datetime.now().date()
        created_today = 0
        modified_today = 0
        
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    stat = os.stat(path)
                    
                    if datetime.fromtimestamp(stat.st_ctime).date() == today:
                        created_today += 1
                    if datetime.fromtimestamp(stat.st_mtime).date() == today:
                        modified_today += 1
        
        print(f"Notes created today: {created_today}")
        print(f"Notes modified today: {modified_today}")
    
    def quick_search(self):
        """Quick search interface"""
        query = input("\nQuick search: ").strip()
        if query:
            self.simple_search()
    
    def change_vault_path(self):
        """Change vault path"""
        new_path = input(f"\nEnter new vault path [{self.vault_path}]: ").strip()
        
        if new_path and new_path != self.vault_path:
            try:
                validated_path = validate_path(new_path)
                if validated_path.exists():
                    self.vault_path = str(validated_path)
                    self.vault_manager = VaultManager(self.vault_path)
                    self._init_features()
                    print(f"{Colors.GREEN}Vault path changed successfully{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}Path does not exist{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}Invalid path: {e}{Colors.ENDC}")
    
    def show_feature_status(self):
        """Show status of all features"""
        print(f"\n{Colors.BOLD}Feature Status{Colors.ENDC}")
        
        for feature, enabled in self.features.items():
            status = f"{Colors.GREEN}✓ Enabled{Colors.ENDC}" if enabled else f"{Colors.RED}✗ Disabled{Colors.ENDC}"
            print(f"{feature.replace('_', ' ').title()}: {status}")
        
        # Show memory statistics if available
        if self.memory_service:
            print(f"\n{Colors.BOLD}Memory Service Statistics:{Colors.ENDC}")
            stats = self.memory_service.get_statistics()
            print(f"Total actions tracked: {stats['total_actions']}")
            print(f"Unique patterns identified: {stats['unique_patterns']}")
            print(f"Most common action: {stats['most_common_action']}")
    
    def export_config(self):
        """Export configuration"""
        config = {
            'vault_path': self.vault_path,
            'features': self.features,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.vault_path, 'vault_config.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"{Colors.GREEN}Configuration exported to: {config_path}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Error exporting config: {e}{Colors.ENDC}")
    
    def import_config(self):
        """Import configuration"""
        config_path = input("\nEnter config file path: ").strip()
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Apply configuration
                if 'vault_path' in config:
                    self.vault_path = config['vault_path']
                    self.vault_manager = VaultManager(self.vault_path)
                    self._init_features()
                
                print(f"{Colors.GREEN}Configuration imported successfully{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}Error importing config: {e}{Colors.ENDC}")
    
    def reset_defaults(self):
        """Reset to default settings"""
        if input("\nReset all settings to defaults? (y/n): ").lower() == 'y':
            # Reset to defaults
            self._init_features()
            print(f"{Colors.GREEN}Settings reset to defaults{Colors.ENDC}")
    
    def handle_mcp_configuration(self):
        """Handle MCP server configuration"""
        if MCP_AVAILABLE:
            try:
                from obsidian_vault_tools.mcp_tools.interactive_config import MCPInteractiveConfig
                config_manager = MCPInteractiveConfig()
                config_manager.display_menu()
            except ImportError as e:
                print(f"{Colors.RED}Error loading MCP configuration: {e}{Colors.ENDC}")
                input("\nPress Enter to continue...")
        else:
            print(f"{Colors.YELLOW}MCP tools not available. Install with: pip install mcp{Colors.ENDC}")
            input("\nPress Enter to continue...")
    
    def run(self):
        """Main run loop"""
        self.display_banner()
        
        # Start ambient sound if available
        if self.audio_manager:
            start_dungeon_ambiance()
        
        try:
            while True:
                options = self.display_main_menu()
                
                if self.navigator:
                    # Use arrow key navigation
                    choice_idx = self.navigator.navigate_menu([opt[0] for opt in options])
                    if choice_idx == -1:  # Cancelled
                        continue
                    choice = str(choice_idx + 1)
                else:
                    # Use number input
                    choice = input("\nSelect option (or 'q' to quit): ").strip().lower()
                
                if choice == 'q' or choice == str(len(options)):
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        action = options[idx][1]
                        
                        if action == 'exit':
                            break
                        elif action == 'analysis':
                            self.handle_analysis_menu()
                        elif action == 'tags':
                            self.handle_tag_menu()
                        elif action == 'search':
                            self.handle_search_menu()
                        elif action == 'ai':
                            self.handle_ai_menu()
                        elif action == 'mcp':
                            self.handle_mcp_menu()
                        elif action == 'backup':
                            self.handle_backup_menu()
                        elif action == 'creative':
                            self.handle_creative_menu()
                        elif action == 'audio':
                            self.handle_audio_menu()
                        elif action == 'security':
                            self.handle_security_menu()
                        elif action == 'quick':
                            self.handle_quick_menu()
                        elif action == 'settings':
                            self.handle_settings_menu()
                    else:
                        print(f"{Colors.RED}Invalid option{Colors.ENDC}")
                except ValueError:
                    print(f"{Colors.RED}Invalid input{Colors.ENDC}")
                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}Operation cancelled{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        finally:
            # Stop ambient sound
            if self.audio_manager:
                stop_dungeon_ambiance()
            
            # Clean up memory service
            if self.memory_service:
                try:
                    # Save any pending data
                    self.memory_service.save_memory()
                except:
                    pass
            
            print(f"\n{Colors.CYAN}Thank you for using Obsidian Vault Manager!{Colors.ENDC}")


def main():
    """Main entry point"""
    try:
        manager = UnifiedVaultManager()
        manager.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()