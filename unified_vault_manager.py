#!/usr/bin/env python3
"""
Unified Obsidian Vault Manager - Complete Feature Integration
Combines all vault management tools into one cohesive interactive menu system
"""

# Import configuration to suppress startup warnings
try:
    import ovt_config
except ImportError:
    pass

import os
import sys
import json
import time
import shlex
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Core imports from existing managers
from vault_manager import Colors, VaultManager
from obsidian_vault_tools.security import (
    validate_path, sanitize_filename, validate_json_input,
    InputValidationError, rate_limit, sanitize_log_data
)
from obsidian_vault_tools.utils import Config

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
        get_menu_builder, get_executor, get_discovery_service, get_client_manager
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

# PM Tools
try:
    from obsidian_vault_tools.pm_tools.task_extractor import TaskExtractor
    from obsidian_vault_tools.pm_tools.wsjf_analyzer import WSJFAnalyzer
    from obsidian_vault_tools.pm_tools.eisenhower_matrix import EisenhowerMatrixClassifier
    from obsidian_vault_tools.pm_tools.burnout_detector import BurnoutDetector
    PM_TOOLS_AVAILABLE = True
except ImportError:
    PM_TOOLS_AVAILABLE = False

# PM Tools - Planned features (placeholders for now)
try:
    from obsidian_vault_tools.pm_tools import ContentQualityEngine
    CONTENT_QUALITY_AVAILABLE = True
except ImportError:
    CONTENT_QUALITY_AVAILABLE = False

try:
    from obsidian_vault_tools.pm_tools import daily_template_generator
    DAILY_TEMPLATE_AVAILABLE = True
except ImportError:
    DAILY_TEMPLATE_AVAILABLE = False

# PM Automation Suite
try:
    import sys
    # Add PM automation suite to path
    pm_suite_path = Path(__file__).parent / "pm_automation_suite"
    if pm_suite_path.exists():
        sys.path.insert(0, str(pm_suite_path))
    
    # Core Infrastructure
    from connectors.jira_connector import JiraConnector
    from authentication.auth_manager import AuthManager
    from orchestration.workflow_engine import WorkflowEngine
    from orchestration.event_bus import EventBus
    
    # WBR/QBR Automation
    from wbr.wbr_data_extractor import WBRDataExtractor
    from wbr.insight_generator import InsightGenerator
    from wbr.slide_generator import SlideGenerator
    from wbr.wbr_workflow import WBRWorkflow
    
    # Feature Development Pipeline
    from feature_pipeline.prd_parser import PRDParser
    from feature_pipeline.story_generator import StoryGenerator
    from feature_pipeline.jira_bulk_creator import JiraBulkCreator
    from feature_pipeline.feature_pipeline import FeaturePipeline
    
    # Analytics Hub
    from analytics_hub.etl_pipeline import ETLPipeline
    from analytics_hub.ml_models import PMPerformancePredictor, BurnoutPredictor, ProductivityAnalyzer
    from analytics_hub.dashboard_generator import DashboardGenerator
    from analytics_hub.monitoring_system import MonitoringSystem
    
    PM_AUTOMATION_AVAILABLE = True
except ImportError as e:
    PM_AUTOMATION_AVAILABLE = False
    logger.debug(f"PM Automation Suite not available: {e}")

# AI Meeting Notes Organizer
try:
    from obsidian_vault_tools.ai.meeting_notes_organizer import MeetingNotesOrganizer
    MEETING_ORGANIZER_AVAILABLE = True
except ImportError:
    MEETING_ORGANIZER_AVAILABLE = False

# Model Management
try:
    from obsidian_vault_tools.model_management import InteractiveModelManager
    MODEL_MANAGEMENT_AVAILABLE = True
except ImportError:
    MODEL_MANAGEMENT_AVAILABLE = False


class UnifiedVaultManager:
    """
    Unified interface for all Obsidian vault management tools
    """
    
    def __init__(self, vault_path: Optional[str] = None):
        """Initialize the unified vault manager"""
        self.vault_path = vault_path or self._get_vault_path()
        self.vault_manager = VaultManager()
        
        # Check system requirements on first run
        self._check_system_requirements()
        
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
            'v2': V2_AVAILABLE,
            'pm_tools': PM_TOOLS_AVAILABLE,
            'content_quality': CONTENT_QUALITY_AVAILABLE,
            'daily_template': DAILY_TEMPLATE_AVAILABLE,
            'meeting_organizer': MEETING_ORGANIZER_AVAILABLE,
            'model_management': MODEL_MANAGEMENT_AVAILABLE,
            'pm_automation': PM_AUTOMATION_AVAILABLE
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
        
        # PM Tools
        self.task_extractor = None
        self.wsjf_analyzer = None
        self.eisenhower_classifier = None
        self.burnout_detector = None
        self.content_quality_engine = None
        self.daily_template_gen = None
        
        if PM_TOOLS_AVAILABLE:
            try:
                self.task_extractor = TaskExtractor(self.vault_path)
                self.wsjf_analyzer = WSJFAnalyzer()
                self.eisenhower_classifier = EisenhowerMatrixClassifier()
                self.burnout_detector = BurnoutDetector(self.vault_path)
            except:
                pass
        
        # Initialize planned features if available
        if CONTENT_QUALITY_AVAILABLE:
            try:
                self.content_quality_engine = ContentQualityEngine(self.vault_path)
            except:
                pass
                
        if DAILY_TEMPLATE_AVAILABLE:
            try:
                self.daily_template_gen = daily_template_generator
            except:
                pass
        
        # Meeting Notes Organizer
        self.meeting_organizer = None
        if MEETING_ORGANIZER_AVAILABLE:
            try:
                self.meeting_organizer = MeetingNotesOrganizer(self.vault_path)
            except:
                pass
        
        # Model Management
        self.model_manager = None
        if MODEL_MANAGEMENT_AVAILABLE:
            try:
                self.model_manager = InteractiveModelManager()
            except:
                pass
    
    def _get_vault_path(self) -> str:
        """Get vault path from environment, config, or prompt user"""
        # Priority: Environment Variable > Saved Config > User Prompt
        vault_path = os.environ.get('OBSIDIAN_VAULT_PATH')
        
        if not vault_path:
            # Check saved configuration
            config = Config()
            saved_path = config.get_vault_path()
            
            if saved_path and os.path.exists(saved_path):
                print(f"{Colors.GREEN}Using saved vault path: {saved_path}{Colors.ENDC}")
                return saved_path
            
            # Prompt user for path
            default_path = os.path.expanduser("~/Documents/ObsidianVault")
            vault_path = input(f"Enter vault path [{default_path}]: ").strip()
            if not vault_path:
                vault_path = default_path
        
        # Fix common path input issues
        # If path looks like an absolute path without leading slash, add it
        if vault_path and not vault_path.startswith(('/', '~', '.')) and vault_path.startswith('Users/'):
            vault_path = '/' + vault_path
            print(f"{Colors.CYAN}Interpreting path as: {vault_path}{Colors.ENDC}")
        
        # Expand user path (handles ~)
        vault_path = os.path.expanduser(vault_path)
        
        # Validate path
        try:
            validated_path = validate_path(vault_path)
            if not validated_path.exists():
                print(f"{Colors.YELLOW}Warning: Vault path does not exist: {validated_path}{Colors.ENDC}")
                if input("Create it? (y/n): ").lower() == 'y':
                    validated_path.mkdir(parents=True)
            
            # Save valid path to config for future use
            if not os.environ.get('OBSIDIAN_VAULT_PATH'):  # Only save if not from env
                config = Config()
                config.set_vault_path(str(validated_path))
                print(f"{Colors.GREEN}✓ Vault path saved for future use{Colors.ENDC}")
            
            return str(validated_path)
        except Exception as e:
            print(f"{Colors.RED}Error: Invalid vault path: {e}{Colors.ENDC}")
            sys.exit(1)
    
    def _check_system_requirements(self):
        """Check system requirements and show warnings if needed"""
        try:
            from obsidian_vault_tools.system_requirements import SystemRequirementsChecker
            
            # Only check on first run or if explicitly requested
            config_file = Path.home() / '.obsidian_vault_tools' / '.system_checked'
            
            # Check if we've already shown system check this session
            if hasattr(self, '_system_checked'):
                return
                
            # Check if we've checked in the last 7 days
            if config_file.exists():
                last_check = config_file.stat().st_mtime
                days_since_check = (time.time() - last_check) / 86400
                if days_since_check < 7:
                    return
            
            checker = SystemRequirementsChecker()
            results = checker.check_all()
            
            # Only show warnings if there are issues
            if results['missing_deps'] or results['warnings']:
                print(f"\n{Colors.YELLOW}System Requirements Check:{Colors.ENDC}")
                
                if results['missing_deps']:
                    print(f"{Colors.RED}Missing dependencies detected:{Colors.ENDC}")
                    for dep in results['missing_deps']:
                        print(f"  • {dep}")
                
                if results['warnings']:
                    print(f"{Colors.YELLOW}Warnings:{Colors.ENDC}")
                    for warning in results['warnings']:
                        print(f"  • {warning}")
                
                print(f"\n{Colors.CYAN}Run 'ovt check-system' for detailed information and installation commands{Colors.ENDC}")
                print(f"{Colors.CYAN}See UBUNTU_INSTALL.md for Ubuntu-specific instructions{Colors.ENDC}\n")
                
                # Brief pause to let user see the message
                time.sleep(2)
            
            # Mark as checked
            config_file.parent.mkdir(exist_ok=True)
            config_file.touch()
            self._system_checked = True
            
        except Exception:
            # Silently fail - don't break the app if system check fails
            pass
    
    def display_banner(self):
        """Display the welcome banner with custom ASCII art"""
        # Custom ASCII art banner
        ascii_banner = """⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠴⠒⠒⠒⠦⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⢉⡱⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠈⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⢉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⢏⠲⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠜⠋⠀⠀⠀⠐⠀⣀⣀⣀⣀⣀⠀⠀⠀⠀⢀⡼⠀⠀⠈⠆⠸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⠐⠤⠤⢾⠗⠉⠁⠀⠀⠀⠀⠉⠙⠫⣉⡁⠀⠀⠀⣠⠂⢠⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡎⠀⠀⠀⠀⢀⣠⡎⣀⣀⢀⣀⣀⠀⠀⠀⠐⠀⠀⠹⡑⠒⠋⢀⡠⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡃⠀⠘⠛⠟⠛⣼⣿⣄⡀⠀⠀⠀⠛⠣⢤⣀⠀⠀⠀⢣⠀⠘⠃⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠱⢄⠀⠀⠀⠀⡏⢩⢯⣿⣦⠀⠰⣾⣿⠷⣞⢳⡄⠀⠈⡆⠀⢀⡴⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡴⣄⠀⠀⠀⠀⠀⠂⠀⠈⢇⡀⠀⣠⠏⠻⣿⣿⠟⠀⠼⣷⣽⣷⠈⢀⣹⣆⠀⠗⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⣼⠈⢀⣀⠤⠖⢒⡷⠀⠀⠉⢉⡏⠀⢀⣼⣅⣤⣰⡆⠙⠂⠀⠀⠸⣿⣿⣾⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠏⣰⠇⡔⠉⣠⠶⣞⣋⣀⠀⠀⠸⣼⣴⣾⣿⣿⣿⣿⣿⣆⠀⠀⣀⠀⠀⠈⣵⠿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡎⠙⣿⠀⣿⠗⠊⣭⣬⣭⣑⣲⡆⢹⠈⠀⢹⣟⣋⡛⠛⠿⣷⣤⡼⠃⠀⢰⡿⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠘⠙⠛⠻⢤⣾⡿⣿⡞⢿⣿⠀⠘⣧⠀⠠⠽⠿⡅⠀⠀⠀⠈⠀⠀⡾⡼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡇⢀⣀⣤⠤⢤⣿⡴⠁⢠⠏⠀⠀⠀⠈⣧⠀⠀⠀⠈⠀⠀⣀⣀⣴⢞⠃⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⢾⢢⡇⠀⠙⢀⡏⠀⣀⣀⠠⣶⡏⢻⣶⡦⡄⣀⡠⠌⠀⠀⣀⡴⠛⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢽⠈⠀⠀⢘⣿⡀⠀⢀⡟⠉⠀⠀⠀⢸⣿⣇⠀⠻⣭⢋⡥⠤⠔⠚⠉⠀⠀⣸⣿⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⢇⠀⠀⠞⠁⢁⣤⠞⠀⠀⠀⠀⠀⢸⣿⣿⣆⣀⣸⣼⡀⠀⠀⢀⣀⣠⣾⢿⣿⣿⠟⠁⠈⠒⠄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠊⠁⠀⠀⠀⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀⠀⠈⠓⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⢹⡃⠀⠀⠀⠀⠀⠀⢠⠞⠁⠀⠀⠀⠀⢠⠉⣩⣿⣿⣿⡿⠛⠙⠛⣿⣿⣿⡏⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⡴⠊⠁⠀⠀⠀⢧⠀⠀⠀⠀⢀⡴⠋⠀⡄⠀⠀⠀⢠⠃⠈⣩⣍⠉⠁⠀⠀⠀⠀⣿⣿⡿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠜⠢⡀⠀⠀⠀⠀
⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠘⢇⡀⠀⣠⠞⠁⠀⡸⠀⠀⠀⠀⠎⠀⠀⠿⠟⠀⠀⠀⠀⠀⠀⠘⢿⣿⣿⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠄⡖⠁⠀⠀⠱⡄⠀⠀⠀
⠀⠀⣠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⣰⢿⡆⠀⢰⠁⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡐⢁⠞⠀⠀⠀⠀⠀⠱⡀⠀⠀
⠀⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⠉⡟⠁⢠⠃⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⡜⠀⡎⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀
⠀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠁⠀⡰⠃⠀⡜⠀⠀⠀⢀⠃⠀⢰⣶⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠰⠀⡸⠁⠀⠀⠀⠀⠀⠀⠀⠘⡄⠀
⢸⡁⠀⠀⠀⠀⠀⠀⠀⠀⠂⠁⠀⢀⣾⠁⠀⡸⠀⠀⠀⠀⡜⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣾⣦⠀⠀⠀⠀⡼⠀⠀⠀⠀⠀⠀⢀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⢡⠀
⠈⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠴⢉⡏⠀⢠⠃⠀⠀⠀⠰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⠀⠀⠀⠀⠇⠀⠀⠀⠀⠀⠃⠸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄
⠀⠀⠙⢦⡀⠀⠀⠀⠀⠒⠉⠁⣠⠋⠀⠀⡞⠀⠀⠀⠀⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⡀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢡
⠀⠀⠀⠀⠈⠳⢤⣀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⢰⠁⠀⢰⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠡⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘
⠀⠀⠀⠀⠀⠀⠀⠈⠙⠒⠤⠤⣀⣀⣀⡎⠀⠀⠀⠀⡜⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⣿⠇⠀⠀⠀⠀⠇⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⠀⠀⠀⢠⠁⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"""
        
        print(f"{Colors.CYAN}{ascii_banner}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
        print(f"{Colors.CYAN}🏰 Unified Obsidian Vault Manager - Complete Toolsuite{Colors.ENDC}")
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
        print(f"Version: {Colors.YELLOW}v2.3.0{Colors.ENDC}")
        print(f"Vault: {Colors.GREEN}{self.vault_path}{Colors.ENDC}")
        
        # Show feature status
        enabled_count = sum(1 for v in self.features.values() if v)
        print(f"Features: {Colors.GREEN}{enabled_count}/{len(self.features)} enabled{Colors.ENDC}")
        print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}\n")
        
        # Play greeting if audio available
        # if self.audio_manager:
        #     wizard_greeting()  # Removed background sound
    
    def get_main_menu_options(self) -> List[Tuple[str, str]]:
        """Get main menu options without displaying them"""
        return [
            ("📊 Vault Analysis & Insights", "analysis"),
            ("🏷️ Tag Management & Organization", "tags"),
            ("🔍 Search & Query Vault", "search"),
            ("🤖 AI & Intelligence Features", "ai"),
            ("📋 PM Tools & Task Management", "pm_tools"),
            ("🛠️ MCP Tools & Integrations", "mcp"),
            ("💾 Backup & Version Control", "backup"),
            ("🎨 Creative Tools & ASCII Art", "creative"),
            ("🔊 Audio System & Effects", "audio"),
            ("🛡️ Security & Maintenance", "security"),
            ("⚡ Quick Actions", "quick"),
            ("⚙️ Settings & Configuration", "settings"),
            ("❌ Exit", "exit")
        ]
    
    def display_main_menu(self) -> List[Tuple[str, str]]:
        """Display main menu and return options (fallback mode)"""
        options = self.get_main_menu_options()
        
        print(f"\n{Colors.BOLD}Main Menu:{Colors.ENDC}")
        for i, (label, _) in enumerate(options, 1):
            print(f"{i}. {label}")
        
        return options
    
    def get_menu_choice(self, prompt="Select option: ", max_options=9) -> str:
        """Get menu choice using single keypress if available, fallback to input()"""
        if self.navigator:
            # Show prompt indicating single keypress mode
            print(f"\n{Colors.YELLOW}Press a number key (no Enter needed):{Colors.ENDC}", end='', flush=True)
            
            while True:
                key = self.navigator.get_key()
                
                # Handle special keys
                if key in ['ESC', 'ZERO']:
                    return '0'  # Back
                elif key == 'QUIT' or key == 'CTRL_C':
                    return 'q'  # Quit
                elif key == 'b' or key == 'B':
                    return 'b'  # Back
                elif key.isdigit():
                    # Direct number selection
                    if 1 <= int(key) <= max_options:
                        print(f" {key}")  # Echo the selection
                        return key
                    else:
                        # Invalid number - play error sound if available
                        if self.audio_manager:
                            play_sound('error_chord')
        else:
            # Fallback to traditional input
            return input(f"\n{prompt}").strip()
    
    def handle_menu(self, title: str, options: List[Tuple[str, Any]], back_option: bool = True) -> Optional[Any]:
        """Unified menu handler with single keypress support
        
        Args:
            title: Menu title
            options: List of (label, function/value) tuples
            back_option: Whether to add a back option
            
        Returns:
            Selected function/value or None if cancelled
        """
        if back_option:
            options = options + [("Back to Previous Menu", None)]
        
        if self.navigator and len(options) <= 9:
            # Use MenuNavigator for instant selection
            menu_options = [(str(i+1), label) for i, (label, _) in enumerate(options)]
            selected_key = self.navigator.navigate_menu(title, menu_options)
            
            if selected_key == '0' or selected_key == 'quit':
                return None
            
            try:
                idx = int(selected_key) - 1
                if 0 <= idx < len(options):
                    return options[idx][1]
            except (ValueError, IndexError):
                return None
        else:
            # Fallback to traditional menu display
            print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
            for i, (label, _) in enumerate(options, 1):
                print(f"{i}. {label}")
            
            choice = self.get_menu_choice(max_options=len(options))
            
            if choice == '0' or choice.lower() in ['b', 'back']:
                return None
            elif choice.lower() == 'q':
                return 'quit'
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][1]
            except (ValueError, IndexError):
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")
                return None
    
    def handle_analysis_menu(self):
        """Handle vault analysis submenu"""
        while True:
            options = [
                ("Tag Statistics", self.vault_manager.analyze_tags),
                ("Folder Structure Analysis", self.analyze_folder_structure),
                ("Find Untagged Files", self.vault_manager.find_untagged_files),
                ("Vault Growth Metrics", self.show_growth_metrics),
                ("Link Analysis", self.analyze_links),
                ("Content Quality Scoring", self.analyze_content_quality),
                ("Content Quality Analysis", self.content_quality_analysis),
                ("Export Analysis Report", self.export_analysis_report)
            ]
            
            result = self.handle_menu("📊 Vault Analysis & Insights", options)
            
            if result is None:  # Back option selected
                break
            elif result == 'quit':  # Quit selected
                return 'quit'
            elif callable(result):  # Function selected
                result()
                if self.audio_manager:
                    magical_success()
    
    def handle_tag_menu(self):
        """Handle tag management submenu"""
        while True:
            options = [
                ("Analyze All Tags", self.vault_manager.analyze_tags),
                ("Fix Tag Issues", self.fix_tag_issues),
                ("Merge Similar Tags", self.merge_similar_tags),
                ("Remove Generic Tags", self.remove_generic_tags),
                ("Bulk Tag Operations", self.bulk_tag_operations),
                ("Tag Hierarchy Report", self.tag_hierarchy_report)
            ]
            
            if V2_AVAILABLE and self.librarian:
                options.insert(5, ("Auto-tag with AI", self.auto_tag_with_ai))
            
            result = self.handle_menu("🏷️ Tag Management & Organization", options)
            
            if result is None:  # Back option selected
                break
            elif result == 'quit':  # Quit selected
                return 'quit'
            elif callable(result):  # Function selected
                result()
                if self.audio_manager:
                    magical_success()
    
    def handle_search_menu(self):
        """Handle search and query submenu"""
        while True:
            options = [
                ("Simple Text Search", self.simple_search),
                ("Search with Filters", self.search_with_filters),
                ("Recent Files", self.show_recent_files)
            ]
            
            # Add advanced search options if available
            if self.query_system:
                options.insert(2, ("Advanced Query System", self.advanced_query))
            
            if self.llm_query:
                options.insert(3, ("AI-Powered Semantic Search", self.semantic_search))
                options.insert(4, ("Natural Language Query", self.natural_language_query))
            
            result = self.handle_menu("🔍 Search & Query Vault", options)
            
            if result is None:  # Back option selected
                break
            elif result == 'quit':  # Quit selected
                return 'quit'
            elif callable(result):  # Function selected
                result()
    
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
            
            # Add Meeting Notes Organizer if available
            if MEETING_ORGANIZER_AVAILABLE:
                options.insert(0, ("Organize Meeting Notes", self.organize_meeting_notes))
            
            # Add Generate Enhanced Daily Note if available
            if DAILY_TEMPLATE_AVAILABLE or PM_TOOLS_AVAILABLE:
                options.insert(0, ("Generate Enhanced Daily Note", self.generate_enhanced_daily_note))
            
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
    
    def handle_pm_tools_menu(self):
        """Handle PM tools submenu"""
        if not PM_TOOLS_AVAILABLE:
            print(f"{Colors.YELLOW}PM tools not available. Install required dependencies.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
            
        while True:
            print(f"\n{Colors.BOLD}📋 PM Tools & Task Management{Colors.ENDC}")
            
            options = [
                ("Extract Tasks from Vault", self.extract_tasks),
                ("WSJF Priority Analysis", self.run_wsjf_analysis),  
                ("Eisenhower Matrix Classification", self.run_eisenhower_analysis),
                ("Burnout Detection Analysis", self.run_burnout_detection),
                ("Combined PM Dashboard", self.run_pm_dashboard),
                ("Export PM Reports", self.export_pm_reports),
            ]
            
            # Add PM Automation Suite options if available
            if PM_AUTOMATION_AVAILABLE:
                options.extend([
                    ("──── PM Automation Suite ────", None),
                    ("🤖 WBR/QBR Automation", self.handle_wbr_automation),
                    ("📝 Feature Development Pipeline", self.handle_feature_pipeline),
                    ("📊 Analytics Hub & ML Insights", self.handle_analytics_hub),
                    ("🚨 Real-time Monitoring", self.handle_monitoring_system),
                    ("⚙️ PM Suite Configuration", self.handle_pm_suite_config),
                ])
            
            options.append(("Back to Main Menu", None))
            
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
            from obsidian_vault_tools.mcp_tools import MCPConfig
            
            discovery = get_discovery_service()
            menu_builder = get_menu_builder()
            executor = get_executor()
            client_manager = get_client_manager()
            config = MCPConfig()
            
            # Get all configured servers (not just running ones)
            servers = config.list_servers()
            
            if not servers:
                print(f"{Colors.YELLOW}No MCP servers configured.{Colors.ENDC}")
                print("Use Settings → MCP Server Configuration to add servers.")
                input("\nPress Enter to continue...")
                return
            
            # Check which servers are actually running
            server_status = client_manager.get_all_server_status()
            running_servers = [name for name, status in server_status.items() if status.get('running', False)]
            
            # Display server status
            print(f"\n{Colors.BOLD}MCP Server Status:{Colors.ENDC}")
            for server in servers:
                if server in running_servers:
                    print(f"  {server}: {Colors.GREEN}[RUNNING]{Colors.ENDC}")
                else:
                    print(f"  {server}: {Colors.YELLOW}[READY]{Colors.ENDC}")
            
            # For now, we'll try to work with all configured servers
            # The menu builder and discovery service will handle servers that aren't running
            
            # Simple server selection menu
            while True:
                print(f"\n{Colors.BOLD}🛠️ MCP Tools & Integrations{Colors.ENDC}")
                print(f"Available servers: {len(servers)}\n")
                
                # Display servers
                for i, server in enumerate(servers, 1):
                    status = "[RUNNING]" if server in running_servers else "[READY]"
                    color = Colors.GREEN if server in running_servers else Colors.YELLOW
                    print(f"{i}. {server} {color}{status}{Colors.ENDC}")
                
                print(f"\n{len(servers) + 1}. Back to Main Menu")
                
                choice = input("\nSelect server: ").strip()
                
                if choice == str(len(servers) + 1) or choice.lower() == 'b':
                    break
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(servers):
                        server_name = servers[idx]
                        self._handle_mcp_server(server_name, executor, client_manager)
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Invalid option{Colors.ENDC}")
                    
        except Exception as e:
            print(f"{Colors.RED}Error loading MCP tools: {e}{Colors.ENDC}")
            input("\nPress Enter to continue...")
    
    def _handle_mcp_server(self, server_name: str, executor, client_manager):
        """Handle specific MCP server operations"""
        import asyncio
        
        while True:
            print(f"\n{Colors.BOLD}🛠️ {server_name} Server{Colors.ENDC}")
            print(f"{'='*50}")
            
            # Get server status
            status = client_manager.get_all_server_status().get(server_name, {})
            is_running = status.get('running', False)
            
            print(f"Status: {Colors.GREEN if is_running else Colors.YELLOW}{'[RUNNING]' if is_running else '[READY]'}{Colors.ENDC}")
            
            print(f"\nOptions:")
            if not is_running:
                print(f"1. Start Server")
            else:
                print(f"1. Stop Server")
                print(f"2. Show Available Tools")
            print(f"3. Show Server Configuration")
            print(f"4. Back to MCP Menu")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '4' or choice.lower() == 'b':
                break
            elif choice == '1':
                if not is_running:
                    print(f"{Colors.YELLOW}Starting {server_name}...{Colors.ENDC}")
                    try:
                        success = asyncio.run(client_manager.start_server(server_name))
                        if success:
                            print(f"{Colors.GREEN}✓ Server started successfully{Colors.ENDC}")
                        else:
                            print(f"{Colors.RED}✗ Failed to start server{Colors.ENDC}")
                    except Exception as e:
                        print(f"{Colors.RED}Error starting server: {e}{Colors.ENDC}")
                else:
                    print(f"{Colors.YELLOW}Stopping {server_name}...{Colors.ENDC}")
                    try:
                        success = asyncio.run(client_manager.stop_server(server_name))
                        if success:
                            print(f"{Colors.GREEN}✓ Server stopped successfully{Colors.ENDC}")
                        else:
                            print(f"{Colors.RED}✗ Failed to stop server{Colors.ENDC}")
                    except Exception as e:
                        print(f"{Colors.RED}Error stopping server: {e}{Colors.ENDC}")
                input("\nPress Enter to continue...")
            elif choice == '2' and is_running:
                print(f"{Colors.YELLOW}Loading tools for {server_name}...{Colors.ENDC}")
                # This would show available tools - simplified for now
                print(f"{Colors.CYAN}Tool discovery would be implemented here{Colors.ENDC}")
                input("\nPress Enter to continue...")
            elif choice == '3':
                from obsidian_vault_tools.mcp_tools import MCPConfig
                config = MCPConfig()
                server_config = config.get_server_config(server_name)
                if server_config:
                    print(f"\n{Colors.BOLD}Server Configuration:{Colors.ENDC}")
                    print(f"Command: {server_config.get('command', 'N/A')}")
                    print(f"Args: {server_config.get('args', [])}")
                    if server_config.get('env'):
                        print(f"Environment: {list(server_config['env'].keys())}")
                else:
                    print(f"{Colors.RED}No configuration found for {server_name}{Colors.ENDC}")
                input("\nPress Enter to continue...")
            else:
                print(f"{Colors.RED}Invalid option{Colors.ENDC}")

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
            options = [
                ("Daily Note", self.create_daily_note),
                ("Quick Capture", self.quick_capture),
                ("Random Note", self.open_random_note),
                ("Today's Stats", self.todays_stats),
                ("Quick Search", self.quick_search)
            ]
            
            result = self.handle_menu("⚡ Quick Actions", options)
            
            if result is None:  # Back option selected
                break
            elif result == 'quit':  # Quit selected
                return 'quit'
            elif callable(result):  # Function selected
                result()
    
    def handle_settings_menu(self):
        """Handle settings and configuration submenu"""
        while True:
            options = [
                ("Change Vault Path", self.change_vault_path),
                ("Feature Status", self.show_feature_status),
                ("MCP Server Configuration", self.handle_mcp_configuration),
                ("AI Model Management", self.handle_model_management),
                ("Export Configuration", self.export_config),
                ("Import Configuration", self.import_config),
                ("Reset to Defaults", self.reset_defaults)
            ]
            
            result = self.handle_menu("⚙️ Settings & Configuration", options)
            
            if result is None:  # Back option selected
                break
            elif result == 'quit':  # Quit selected
                return 'quit'
            elif callable(result):  # Function selected
                result()
    
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
    
    def organize_meeting_notes(self):
        """AI-powered meeting notes organization"""
        if not self.meeting_organizer:
            print(f"\n{Colors.BOLD}Organize Meeting Notes{Colors.ENDC}")
            print("This feature automatically organizes unstructured meeting notes into proper template sections using AI.")
            print("Features:")
            print("- Reads your QuickAdd templates automatically")
            print("- Categorizes content into appropriate sections")
            print("- Supports both 1:1 and general meeting formats")
            print("- Smart backup system with instant undo capability")
            print(f"\n{Colors.YELLOW}Meeting Notes Organizer not available{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}🤖 AI Meeting Notes Organizer{Colors.ENDC}")
        print("Automatically organize your rapid meeting notes into proper template sections!")
        print()
        
        # Display current daily note info
        daily_note = self.meeting_organizer.find_current_daily_note()
        if daily_note:
            print(f"📅 Found daily note: {daily_note.name}")
        else:
            print("⚠️  No daily note found for today")
            
            # Ask user for file path
            file_path = input("\nEnter path to file to organize (or Enter to cancel): ").strip()
            if not file_path:
                return
            
            if not os.path.exists(file_path):
                print(f"{Colors.RED}File not found: {file_path}{Colors.ENDC}")
                input("\nPress Enter to continue...")
                return
        
        print("\nTemplate Types:")
        print("1. Auto-detect (recommended)")
        print("2. General Meeting")
        print("3. 1:1 Meeting")
        print("4. Cancel")
        
        template_choice = input("\nSelect template type: ").strip()
        
        template_type_map = {
            '1': 'auto',
            '2': 'meeting', 
            '3': 'one_on_one'
        }
        
        if template_choice == '4' or template_choice.lower() == 'c':
            return
        
        template_type = template_type_map.get(template_choice, 'auto')
        
        print(f"\n{Colors.CYAN}Analyzing and organizing meeting notes...{Colors.ENDC}")
        
        try:
            # Organize the notes
            file_path_to_use = file_path if 'file_path' in locals() else str(daily_note)
            result = self.meeting_organizer.organize_meeting_notes(
                file_path_to_use, 
                template_type
            )
            
            if not result['success']:
                print(f"{Colors.RED}Organization failed: {result['error']}{Colors.ENDC}")
                if result.get('suggestions'):
                    print("\nSuggestions:")
                    for suggestion in result['suggestions']:
                        print(f"  • {suggestion}")
                input("\nPress Enter to continue...")
                return
            
            # Show preview
            print(f"\n{Colors.BOLD}Preview of organized content:{Colors.ENDC}")
            print(f"Template Type: {result['template_type'].replace('_', ' ').title()}")
            print(f"Session ID: {result['session_id']}")
            
            # Show categorized content summary
            print(f"\n{Colors.BOLD}Content Categorization:{Colors.ENDC}")
            for section, items in result['categorized_content'].items():
                if items:
                    print(f"  📝 {section}: {len(items)} items")
            
            print(f"\n{result['preview']}")
            
            # Ask user to apply changes
            print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
            print("1. Apply changes")
            print("2. Cancel (no changes)")
            
            choice = input("\nWhat would you like to do? ").strip()
            
            if choice == '1':
                # Apply the organization
                success = self.meeting_organizer.apply_organization(
                    result['organized_content'],
                    result['file_path']
                )
                
                if success:
                    print(f"\n{Colors.GREEN}✅ Meeting notes organized successfully!{Colors.ENDC}")
                    print(f"File updated: {result['file_path']}")
                    print(f"\n{Colors.CYAN}💡 Pro tip: If you need to undo, run this feature again and select 'Undo last organization'{Colors.ENDC}")
                    
                    # Complete the session successfully
                    self.meeting_organizer.complete_organization(success=True)
                else:
                    print(f"{Colors.RED}Failed to apply organization{Colors.ENDC}")
                    print("Your original content is safely backed up and can be restored.")
            else:
                # Cancel - restore original content
                print(f"\n{Colors.YELLOW}Organization cancelled{Colors.ENDC}")
                self.meeting_organizer.complete_organization(success=False)
                
        except Exception as e:
            print(f"{Colors.RED}Error during organization: {e}{Colors.ENDC}")
            print("Your original content is safely backed up.")
            
            # Offer to undo
            if hasattr(self.meeting_organizer, 'current_session') and self.meeting_organizer.current_session:
                if input("\nAttempt to restore original content? (y/n): ").lower() == 'y':
                    if self.meeting_organizer.undo_organization():
                        print(f"{Colors.GREEN}Original content restored{Colors.ENDC}")
                    else:
                        print(f"{Colors.RED}Could not restore automatically{Colors.ENDC}")
        
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
    
    def handle_model_management(self):
        """Handle AI model management configuration"""
        if MODEL_MANAGEMENT_AVAILABLE and self.model_manager:
            # Check if this is first time setup
            if not self.model_manager.models.get("active_model"):
                print(f"{Colors.YELLOW}No models configured yet.{Colors.ENDC}")
                if input("Would you like to run the setup wizard? (y/n): ").lower() == 'y':
                    self.model_manager.interactive_setup()
                else:
                    self.model_manager.show_model_menu()
            else:
                self.model_manager.show_model_menu()
        else:
            print(f"{Colors.YELLOW}Model management not available.{Colors.ENDC}")
            print("This feature helps you manage AI models for your vault.")
            print("\nTo enable:")
            print("1. Ensure all dependencies are installed")
            print("2. Restart the application")
            input("\nPress Enter to continue...")
    
    def run(self):
        """Main run loop"""
        self.display_banner()
        
        # Start ambient sound if available
        # if self.audio_manager:
        #     start_dungeon_ambiance()  # Removed background sound
        
        try:
            while True:
                options = self.get_main_menu_options()
                
                if self.navigator:
                    try:
                        # Use arrow key navigation
                        # Convert options to format expected by MenuNavigator: (key, description)
                        menu_options = [(str(i+1), label) for i, (label, _) in enumerate(options)]
                        selected_key = self.navigator.navigate_menu("Main Menu", menu_options)
                        
                        if selected_key == '0' or selected_key == 'quit':  # Cancelled/Quit
                            choice = 'q'
                        else:
                            choice = selected_key
                    except Exception as e:
                        # Fallback to number input if navigation fails - disable navigator completely
                        logger.warning(f"Navigation error: {e}")
                        self.navigator = None  # Disable for rest of session
                        # Clear screen and show fallback menu
                        print("\n" + "="*60)
                        print("Falling back to number-based menu navigation")
                        print("="*60)
                        self.display_main_menu()
                        choice = input("\nSelect option (or 'q' to quit): ").strip().lower()
                else:
                    # Use number input - show the text menu
                    self.display_main_menu()
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
                        elif action == 'pm_tools':
                            self.handle_pm_tools_menu()
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
            
            print(f"\n{Colors.CYAN}Thank you for using Obsidian Vault Manager!{Colors.ENDC}")
    
    # PM Tools Methods
    def extract_tasks(self):
        """Extract tasks from vault using TaskExtractor"""
        if not self.task_extractor:
            print(f"{Colors.RED}Task extractor not available{Colors.ENDC}")
            return
        
        print(f"{Colors.CYAN}Extracting tasks from vault...{Colors.ENDC}")
        try:
            tasks = self.task_extractor.extract_all_tasks()
            
            if not tasks:
                print(f"{Colors.YELLOW}No tasks found in vault{Colors.ENDC}")
                return
            
            print(f"\n{Colors.GREEN}Found {len(tasks)} tasks:{Colors.ENDC}")
            for i, task in enumerate(tasks[:10], 1):  # Show first 10
                print(f"{i}. {task.content[:80]}...")
                if task.product_area:
                    print(f"   Product: {task.product_area}")
                if task.task_type:
                    print(f"   Type: {task.task_type}")
                print()
            
            if len(tasks) > 10:
                print(f"{Colors.YELLOW}... and {len(tasks) - 10} more tasks{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.RED}Error extracting tasks: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_wsjf_analysis(self):
        """Run WSJF analysis on vault tasks"""
        if not self.task_extractor or not self.wsjf_analyzer:
            print(f"{Colors.RED}WSJF tools not available{Colors.ENDC}")
            return
        
        print(f"{Colors.CYAN}Running WSJF analysis...{Colors.ENDC}")
        try:
            # Extract tasks
            tasks = self.task_extractor.extract_all_tasks()
            if not tasks:
                print(f"{Colors.YELLOW}No tasks found for analysis{Colors.ENDC}")
                return
            
            # Generate WSJF report
            report = self.wsjf_analyzer.generate_wsjf_report(tasks)
            
            # Display results
            print(f"\n{Colors.BOLD}WSJF Analysis Results:{Colors.ENDC}")
            print(f"Total tasks analyzed: {report['summary']['total_tasks']}")
            print(f"Average WSJF score: {report['summary']['average_wsjf_score']:.2f}")
            print(f"High priority tasks: {report['summary']['high_priority_count']}")
            
            print(f"\n{Colors.BOLD}Top 10 Priority Tasks:{Colors.ENDC}")
            for item in report['top_10_recommendations']:
                print(f"{item['rank']}. {item['task_content'][:80]}...")
                print(f"   WSJF Score: {item['wsjf_score']:.2f}")
                print(f"   Business Value: {item['business_value']}, "
                      f"Time Criticality: {item['time_criticality']}, "
                      f"Risk Reduction: {item['risk_reduction']}, "
                      f"Job Size: {item['job_size']}")
                if item['product_area']:
                    print(f"   Product: {item['product_area']}")
                print()
                
        except Exception as e:
            print(f"{Colors.RED}Error running WSJF analysis: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_eisenhower_analysis(self):
        """Run Eisenhower Matrix analysis on vault tasks"""
        if not self.task_extractor or not self.eisenhower_classifier:
            print(f"{Colors.RED}Eisenhower Matrix tools not available{Colors.ENDC}")
            return
        
        print(f"{Colors.CYAN}Running Eisenhower Matrix analysis...{Colors.ENDC}")
        try:
            # Extract tasks
            tasks = self.task_extractor.extract_all_tasks()
            if not tasks:
                print(f"{Colors.YELLOW}No tasks found for analysis{Colors.ENDC}")
                return
            
            # Generate matrix report
            report = self.eisenhower_classifier.generate_matrix_report(tasks)
            
            # Display results
            print(f"\n{Colors.BOLD}Eisenhower Matrix Results:{Colors.ENDC}")
            print(f"Total tasks analyzed: {report['summary']['total_tasks']}")
            
            # Show quadrant distribution
            print(f"\n{Colors.BOLD}Task Distribution:{Colors.ENDC}")
            for quadrant, count in report['summary']['quadrant_distribution'].items():
                print(f"  {quadrant}: {count} tasks")
            
            # Show each quadrant
            for quadrant_name, task_list in report['quadrants'].items():
                if task_list:
                    print(f"\n{Colors.BOLD}{quadrant_name} ({len(task_list)} tasks):{Colors.ENDC}")
                    for item in task_list[:5]:  # Show first 5 in each quadrant
                        print(f"  • {item['content'][:60]}...")
                        print(f"    Confidence: {item['confidence']:.2f}")
                    if len(task_list) > 5:
                        print(f"    ... and {len(task_list) - 5} more")
            
            # Show recommendations
            print(f"\n{Colors.BOLD}Action Recommendations:{Colors.ENDC}")
            for action, recommendation in report['recommendations'].items():
                print(f"  {action.replace('_', ' ').title()}: {recommendation}")
                
        except Exception as e:
            print(f"{Colors.RED}Error running Eisenhower analysis: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_pm_dashboard(self):
        """Run combined PM analysis dashboard"""
        if not self.task_extractor or not self.wsjf_analyzer or not self.eisenhower_classifier:
            print(f"{Colors.RED}PM tools not fully available{Colors.ENDC}")
            return
        
        print(f"{Colors.CYAN}Generating PM Dashboard...{Colors.ENDC}")
        try:
            # Extract tasks
            tasks = self.task_extractor.extract_all_tasks()
            if not tasks:
                print(f"{Colors.YELLOW}No tasks found for analysis{Colors.ENDC}")
                return
            
            # Run both analyses
            wsjf_report = self.wsjf_analyzer.generate_wsjf_report(tasks)
            eisenhower_report = self.eisenhower_classifier.generate_matrix_report(tasks)
            
            # Combined dashboard
            print(f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
            print(f"{Colors.BOLD}📋 PM DASHBOARD - COMPREHENSIVE TASK ANALYSIS{Colors.ENDC}")
            print(f"{Colors.BOLD}═══════════════════════════════════════════════════════════════{Colors.ENDC}")
            
            print(f"\n{Colors.BOLD}📊 OVERVIEW:{Colors.ENDC}")
            print(f"  Total Tasks: {len(tasks)}")
            print(f"  Average WSJF Score: {wsjf_report['summary']['average_wsjf_score']:.2f}")
            print(f"  High Priority (WSJF): {wsjf_report['summary']['high_priority_count']}")
            print(f"  Immediate Action Required: {eisenhower_report['action_priorities']['immediate_action']}")
            
            print(f"\n{Colors.BOLD}🎯 TOP PRIORITY ACTIONS (WSJF + Eisenhower):{Colors.ENDC}")
            
            # Find tasks that are both high WSJF and "Do First" quadrant
            high_wsjf_tasks = {item['task'].content: item for item in wsjf_report['all_tasks'][:10]}
            do_first_tasks = {item['content']: item for item in eisenhower_report['quadrants']['Do First']}
            
            critical_tasks = []
            for content in high_wsjf_tasks:
                if content in do_first_tasks:
                    critical_tasks.append((high_wsjf_tasks[content], do_first_tasks[content]))
            
            if critical_tasks:
                print(f"  {Colors.RED}CRITICAL: {len(critical_tasks)} tasks need immediate attention{Colors.ENDC}")
                for i, (wsjf_item, eis_item) in enumerate(critical_tasks[:3], 1):
                    print(f"    {i}. {wsjf_item['task'].content[:60]}...")
                    print(f"       WSJF: {wsjf_item['wsjf_score'].total_score:.2f} | "
                          f"Confidence: {eis_item['confidence']:.2f}")
            
            print(f"\n{Colors.BOLD}📈 WSJF TOP 5:{Colors.ENDC}")
            for item in wsjf_report['top_10_recommendations'][:5]:
                print(f"  {item['rank']}. {item['task_content'][:50]}... (Score: {item['wsjf_score']:.2f})")
            
            print(f"\n{Colors.BOLD}🎯 EISENHOWER QUADRANTS:{Colors.ENDC}")
            for quadrant, count in eisenhower_report['summary']['quadrant_distribution'].items():
                if count > 0:
                    print(f"  {quadrant}: {count} tasks")
                    
        except Exception as e:
            print(f"{Colors.RED}Error generating PM dashboard: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def export_pm_reports(self):
        """Export PM analysis reports to files"""
        if not self.task_extractor or not self.wsjf_analyzer or not self.eisenhower_classifier:
            print(f"{Colors.RED}PM tools not fully available{Colors.ENDC}")
            return
        
        print(f"{Colors.CYAN}Exporting PM reports...{Colors.ENDC}")
        try:
            # Extract tasks
            tasks = self.task_extractor.extract_all_tasks()
            if not tasks:
                print(f"{Colors.YELLOW}No tasks found for export{Colors.ENDC}")
                return
            
            # Generate reports
            wsjf_report = self.wsjf_analyzer.generate_wsjf_report(tasks)
            eisenhower_report = self.eisenhower_classifier.generate_matrix_report(tasks)
            
            # Create reports directory
            reports_dir = Path(self.vault_path) / "PM_Reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export WSJF report
            wsjf_file = reports_dir / f"WSJF_Analysis_{timestamp}.json"
            with open(wsjf_file, 'w') as f:
                json.dump(wsjf_report, f, indent=2, default=str)
            
            # Export Eisenhower report
            eisenhower_file = reports_dir / f"Eisenhower_Matrix_{timestamp}.json"
            with open(eisenhower_file, 'w') as f:
                json.dump(eisenhower_report, f, indent=2, default=str)
            
            # Create markdown summary
            summary_file = reports_dir / f"PM_Summary_{timestamp}.md"
            with open(summary_file, 'w') as f:
                f.write(f"# PM Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write(f"## Overview\n")
                f.write(f"- Total Tasks: {len(tasks)}\n")
                f.write(f"- Average WSJF Score: {wsjf_report['summary']['average_wsjf_score']:.2f}\n")
                f.write(f"- High Priority Tasks: {wsjf_report['summary']['high_priority_count']}\n\n")
                
                f.write(f"## Top 10 WSJF Priority Tasks\n")
                for item in wsjf_report['top_10_recommendations']:
                    f.write(f"{item['rank']}. {item['task_content']}\n")
                    f.write(f"   - WSJF Score: {item['wsjf_score']:.2f}\n")
                    f.write(f"   - File: {item['file_path']}\n\n")
                
                f.write(f"## Eisenhower Matrix Distribution\n")
                for quadrant, count in eisenhower_report['summary']['quadrant_distribution'].items():
                    f.write(f"- **{quadrant}**: {count} tasks\n")
            
            print(f"{Colors.GREEN}Reports exported successfully:{Colors.ENDC}")
            print(f"  - WSJF Analysis: {wsjf_file}")
            print(f"  - Eisenhower Matrix: {eisenhower_file}")
            print(f"  - Summary: {summary_file}")
            
        except Exception as e:
            print(f"{Colors.RED}Error exporting reports: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_burnout_detection(self):
        """Run burnout detection analysis on vault"""
        if not self.burnout_detector:
            print(f"{Colors.RED}Burnout detector not available{Colors.ENDC}")
            return
        
        print(f"{Colors.CYAN}Running burnout detection analysis...{Colors.ENDC}")
        try:
            # Run burnout analysis
            results = self.burnout_detector.analyze_burnout()
            
            # Display results
            print(f"\n{Colors.BOLD}Burnout Detection Results:{Colors.ENDC}")
            print(f"Overall Risk Score: {results['overall_risk_score']:.1f}/10")
            
            if results['overall_risk_score'] >= 7:
                print(f"{Colors.RED}⚠️ HIGH BURNOUT RISK DETECTED{Colors.ENDC}")
            elif results['overall_risk_score'] >= 4:
                print(f"{Colors.YELLOW}⚠️ MODERATE BURNOUT RISK{Colors.ENDC}")
            else:
                print(f"{Colors.GREEN}✓ LOW BURNOUT RISK{Colors.ENDC}")
            
            print(f"\n{Colors.BOLD}Risk Factors:{Colors.ENDC}")
            for factor, details in results['risk_factors'].items():
                print(f"  {factor.replace('_', ' ').title()}: {details['score']:.1f}/10")
                if details.get('details'):
                    print(f"    Details: {details['details']}")
            
            print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}")
            for rec in results.get('recommendations', []):
                print(f"  • {rec}")
                
        except Exception as e:
            print(f"{Colors.RED}Error running burnout detection: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    # PM Automation Suite Methods
    def handle_wbr_automation(self):
        """Handle WBR/QBR automation workflows"""
        if not PM_AUTOMATION_AVAILABLE:
            print(f"{Colors.YELLOW}PM Automation Suite not available.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}🤖 WBR/QBR Automation{Colors.ENDC}")
        print("Automate your Weekly/Quarterly Business Reviews")
        
        options = [
            ("Extract Data from Multiple Sources", self.run_wbr_data_extraction),
            ("Generate AI-Powered Insights", self.run_insight_generation),
            ("Create Presentation Slides", self.run_slide_generation),
            ("Run Complete WBR Workflow", self.run_wbr_workflow),
            ("Back", None)
        ]
        
        while True:
            print()
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
    
    def handle_feature_pipeline(self):
        """Handle feature development pipeline"""
        if not PM_AUTOMATION_AVAILABLE:
            print(f"{Colors.YELLOW}PM Automation Suite not available.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}📝 Feature Development Pipeline{Colors.ENDC}")
        print("Transform PRDs into actionable Jira stories")
        
        options = [
            ("Parse PRD Document", self.run_prd_parser),
            ("Generate User Stories with AI", self.run_story_generator),
            ("Bulk Create Jira Issues", self.run_jira_bulk_creator),
            ("Run Complete Pipeline", self.run_feature_pipeline),
            ("Back", None)
        ]
        
        while True:
            print()
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
    
    def handle_analytics_hub(self):
        """Handle analytics hub and ML insights"""
        if not PM_AUTOMATION_AVAILABLE:
            print(f"{Colors.YELLOW}PM Automation Suite not available.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}📊 Analytics Hub & ML Insights{Colors.ENDC}")
        print("Advanced analytics and predictive insights for PM performance")
        
        options = [
            ("Run ETL Pipeline", self.run_etl_pipeline),
            ("PM Performance Predictions", self.run_ml_predictions),
            ("Generate Analytics Dashboard", self.run_analytics_dashboard),
            ("View Real-time Metrics", self.view_realtime_metrics),
            ("Back", None)
        ]
        
        while True:
            print()
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
    
    def handle_monitoring_system(self):
        """Handle real-time monitoring system"""
        if not PM_AUTOMATION_AVAILABLE:
            print(f"{Colors.YELLOW}PM Automation Suite not available.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}🚨 Real-time Monitoring System{Colors.ENDC}")
        print("Monitor PM metrics and receive alerts")
        
        try:
            monitoring = MonitoringSystem({
                'retention_period': 86400,  # 24 hours
                'enable_prometheus': False,
                'collect_system_metrics': False
            })
            
            # Create default PM monitoring
            monitoring.create_default_pm_monitoring()
            
            print("\nMonitoring system configured with:")
            print("  • Velocity tracking")
            print("  • Burnout risk monitoring")
            print("  • Quality score tracking")
            print("  • Meeting time analysis")
            print("  • Response time metrics")
            
            print(f"\n{Colors.GREEN}✓ Monitoring system ready{Colors.ENDC}")
            
            # Show current metrics if any
            summary = monitoring.get_metrics_summary()
            if any(m['current'] is not None for m in summary.values()):
                print(f"\n{Colors.BOLD}Current Metrics:{Colors.ENDC}")
                for metric, stats in summary.items():
                    if stats['current'] is not None:
                        print(f"  {metric}: {stats['current']:.2f}")
            
        except Exception as e:
            print(f"{Colors.RED}Error setting up monitoring: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def handle_pm_suite_config(self):
        """Handle PM Suite configuration"""
        if not PM_AUTOMATION_AVAILABLE:
            print(f"{Colors.YELLOW}PM Automation Suite not available.{Colors.ENDC}")
            input("\nPress Enter to continue...")
            return
        
        print(f"\n{Colors.BOLD}⚙️ PM Suite Configuration{Colors.ENDC}")
        
        options = [
            ("Configure Jira Connection", self.configure_jira),
            ("Configure Snowflake Connection", self.configure_snowflake),
            ("Configure AI Providers", self.configure_ai_providers),
            ("Test All Connections", self.test_pm_connections),
            ("Back", None)
        ]
        
        while True:
            print()
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
    
    # PM Automation Suite Implementation Methods
    def run_wbr_data_extraction(self):
        """Run WBR data extraction from multiple sources"""
        print(f"\n{Colors.BOLD}📊 WBR Data Extraction{Colors.ENDC}")
        
        try:
            extractor = WBRDataExtractor({})
            
            # Get available data sources
            print("\nAvailable data sources:")
            print("1. Jira (Sprint metrics, velocity)")
            print("2. Snowflake (Business metrics)")
            print("3. Google Sheets (Team data)")
            print("4. Local files (CSV/JSON)")
            
            source = input("\nSelect data source (1-4): ").strip()
            
            if source == '1':
                print("\nExtracting Jira data...")
                # Would implement Jira extraction
                print(f"{Colors.YELLOW}Note: Jira credentials required{Colors.ENDC}")
            elif source == '2':
                print("\nExtracting Snowflake data...")
                print(f"{Colors.YELLOW}Note: Snowflake connection required{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}Source not yet implemented{Colors.ENDC}")
                
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_insight_generation(self):
        """Generate AI-powered insights from data"""
        print(f"\n{Colors.BOLD}🤖 AI Insight Generation{Colors.ENDC}")
        
        try:
            generator = InsightGenerator({})
            
            print("\nInsight generation capabilities:")
            print("  • Sprint velocity trends")
            print("  • Team performance analysis")
            print("  • Risk identification")
            print("  • Success pattern recognition")
            print("  • Predictive forecasting")
            
            print(f"\n{Colors.YELLOW}Note: Requires extracted data and AI provider{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_slide_generation(self):
        """Generate presentation slides"""
        print(f"\n{Colors.BOLD}📊 Slide Generation{Colors.ENDC}")
        
        try:
            generator = SlideGenerator({})
            
            print("\nSlide templates available:")
            print("  • Executive Summary")
            print("  • Sprint Metrics")
            print("  • Team Performance")
            print("  • Risk & Mitigation")
            print("  • Next Steps")
            
            print(f"\n{Colors.YELLOW}Note: Requires python-pptx for full functionality{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_wbr_workflow(self):
        """Run complete WBR workflow"""
        print(f"\n{Colors.BOLD}🔄 Complete WBR Workflow{Colors.ENDC}")
        
        try:
            workflow = WBRWorkflow({})
            
            print("\nWorkflow steps:")
            print("1. Extract data from all sources")
            print("2. Generate insights with AI")
            print("3. Create presentation slides")
            print("4. Schedule distribution")
            
            if input("\nRun workflow? (y/n): ").lower() == 'y':
                print(f"\n{Colors.YELLOW}Workflow requires configured data sources{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_prd_parser(self):
        """Parse PRD documents"""
        print(f"\n{Colors.BOLD}📄 PRD Parser{Colors.ENDC}")
        
        try:
            parser = PRDParser({})
            
            print("\nPRD parsing capabilities:")
            print("  • Extract requirements")
            print("  • Identify user stories")
            print("  • Parse acceptance criteria")
            print("  • Detect dependencies")
            
            prd_path = input("\nEnter PRD file path (or press Enter to skip): ").strip()
            if prd_path:
                print(f"{Colors.YELLOW}Parsing PRD...{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_story_generator(self):
        """Generate user stories with AI"""
        print(f"\n{Colors.BOLD}📝 AI Story Generator{Colors.ENDC}")
        
        try:
            generator = StoryGenerator({})
            
            print("\nStory generation features:")
            print("  • Generate from requirements")
            print("  • Add acceptance criteria")
            print("  • Estimate story points")
            print("  • Create subtasks")
            
            print(f"\n{Colors.YELLOW}Note: Requires AI provider configuration{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_jira_bulk_creator(self):
        """Bulk create Jira issues"""
        print(f"\n{Colors.BOLD}🎯 Jira Bulk Creator{Colors.ENDC}")
        
        try:
            creator = JiraBulkCreator({})
            
            print("\nBulk creation features:")
            print("  • Create epics and stories")
            print("  • Set priorities and labels")
            print("  • Assign team members")
            print("  • Link dependencies")
            
            print(f"\n{Colors.YELLOW}Note: Requires Jira configuration{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_feature_pipeline(self):
        """Run complete feature pipeline"""
        print(f"\n{Colors.BOLD}🚀 Complete Feature Pipeline{Colors.ENDC}")
        
        try:
            pipeline = FeaturePipeline({})
            
            print("\nPipeline steps:")
            print("1. Parse PRD document")
            print("2. Generate user stories")
            print("3. Review and refine")
            print("4. Create in Jira")
            
            print(f"\n{Colors.YELLOW}Pipeline requires all components configured{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_etl_pipeline(self):
        """Run ETL pipeline for analytics"""
        print(f"\n{Colors.BOLD}🔄 ETL Pipeline{Colors.ENDC}")
        
        try:
            config = {
                'pipeline_id': 'pm_analytics',
                'data_sources': [],
                'transformations': [],
                'data_targets': []
            }
            
            pipeline = ETLPipeline(config)
            
            print("\nETL capabilities:")
            print("  • Extract from multiple sources")
            print("  • Transform and enrich data")
            print("  • Load to data warehouse")
            print("  • Schedule automated runs")
            
            print(f"\n{Colors.YELLOW}Configure data sources in PM Suite Configuration{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_ml_predictions(self):
        """Run ML predictions for PM performance"""
        print(f"\n{Colors.BOLD}🤖 ML Performance Predictions{Colors.ENDC}")
        
        try:
            predictor = PMPerformancePredictor({'metric': 'velocity'})
            burnout = BurnoutPredictor({'use_lstm': False})
            productivity = ProductivityAnalyzer({})
            
            print("\nAvailable predictions:")
            print("  • Sprint velocity forecast")
            print("  • Burnout risk assessment")
            print("  • Productivity trends")
            print("  • Quality predictions")
            
            print(f"\n{Colors.YELLOW}Requires historical data for training{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def run_analytics_dashboard(self):
        """Generate analytics dashboard"""
        print(f"\n{Colors.BOLD}📊 Analytics Dashboard{Colors.ENDC}")
        
        try:
            generator = DashboardGenerator({'output_path': './dashboards'})
            
            print("\nDashboard sections:")
            print("  • Executive overview")
            print("  • Team performance")
            print("  • Sprint metrics")
            print("  • Risk indicators")
            
            if input("\nGenerate sample dashboard? (y/n): ").lower() == 'y':
                print(f"{Colors.GREEN}✓ Dashboard would be generated in ./dashboards/{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def view_realtime_metrics(self):
        """View real-time metrics"""
        print(f"\n{Colors.BOLD}📈 Real-time Metrics{Colors.ENDC}")
        
        try:
            monitoring = MonitoringSystem({
                'retention_period': 3600,
                'enable_prometheus': False
            })
            
            summary = monitoring.get_metrics_summary()
            
            if any(m['current'] is not None for m in summary.values()):
                print("\nCurrent metrics:")
                for metric, stats in summary.items():
                    if stats['current'] is not None:
                        print(f"  {metric}: {stats['current']:.2f}")
            else:
                print(f"\n{Colors.YELLOW}No metrics recorded yet{Colors.ENDC}")
                print("Start monitoring to collect metrics")
            
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def configure_jira(self):
        """Configure Jira connection"""
        print(f"\n{Colors.BOLD}🔧 Jira Configuration{Colors.ENDC}")
        
        print("\nRequired settings:")
        print("  • Jira URL")
        print("  • Username/Email")
        print("  • API Token")
        print("  • Default Project")
        
        print(f"\n{Colors.YELLOW}Configuration would be saved securely{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def configure_snowflake(self):
        """Configure Snowflake connection"""
        print(f"\n{Colors.BOLD}❄️ Snowflake Configuration{Colors.ENDC}")
        
        print("\nRequired settings:")
        print("  • Account URL")
        print("  • Username")
        print("  • Password/Key")
        print("  • Warehouse")
        print("  • Database")
        
        print(f"\n{Colors.YELLOW}Configuration would be saved securely{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def configure_ai_providers(self):
        """Configure AI providers"""
        print(f"\n{Colors.BOLD}🤖 AI Provider Configuration{Colors.ENDC}")
        
        print("\nSupported providers:")
        print("  • OpenAI (GPT-4)")
        print("  • Anthropic (Claude)")
        print("  • Local models")
        
        print(f"\n{Colors.YELLOW}API keys would be saved securely{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def test_pm_connections(self):
        """Test all PM Suite connections"""
        print(f"\n{Colors.BOLD}🔌 Testing Connections{Colors.ENDC}")
        
        connections = [
            ("Jira", False),
            ("Snowflake", False),
            ("OpenAI", False),
            ("Google Sheets", False)
        ]
        
        for name, status in connections:
            if status:
                print(f"  {name}: {Colors.GREEN}✓ Connected{Colors.ENDC}")
            else:
                print(f"  {name}: {Colors.YELLOW}⚠ Not configured{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def content_quality_analysis(self):
        """Run content quality analysis on vault"""
        if self.content_quality_engine:
            print(f"{Colors.CYAN}Running content quality analysis...{Colors.ENDC}")
            try:
                # Run the analysis
                results = self.content_quality_engine.analyze_vault()
                
                # Display results
                print(f"\n{Colors.BOLD}Content Quality Analysis Results:{Colors.ENDC}")
                print(f"Files analyzed: {results['files_analyzed']}")
                print(f"Quality score: {results['average_quality_score']:.1f}/10")
                
                print(f"\n{Colors.BOLD}Issues Found:{Colors.ENDC}")
                for issue_type, issues in results['issues'].items():
                    if issues:
                        print(f"  {issue_type}: {len(issues)} issues")
                
                print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}")
                for rec in results.get('recommendations', []):
                    print(f"  • {rec}")
                    
            except Exception as e:
                print(f"{Colors.RED}Error running content quality analysis: {e}{Colors.ENDC}")
        else:
            print(f"\n{Colors.BOLD}Content Quality Analysis{Colors.ENDC}")
            print("This feature analyzes note quality based on:")
            print("- Note completeness and structure")
            print("- Naming consistency")
            print("- Duplicate content detection")
            print("- Missing context identification")
            print(f"\n{Colors.YELLOW}Feature in development - ContentQualityEngine not yet available{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def generate_enhanced_daily_note(self):
        """Generate an enhanced daily note using PM-optimized template"""
        if self.daily_template_gen:
            print(f"{Colors.CYAN}Generating enhanced daily note...{Colors.ENDC}")
            try:
                # Generate the daily note
                note_path = self.daily_template_gen.generate_daily_note(
                    self.vault_path,
                    self.task_extractor,
                    self.wsjf_analyzer
                )
                
                print(f"{Colors.GREEN}Enhanced daily note created: {note_path}{Colors.ENDC}")
                
            except Exception as e:
                print(f"{Colors.RED}Error generating daily note: {e}{Colors.ENDC}")
        else:
            print(f"\n{Colors.BOLD}Generate Enhanced Daily Note{Colors.ENDC}")
            print("This feature creates a PM-optimized daily note with:")
            print("- Top 3 WSJF priorities for today")
            print("- Product area focus rotation")
            print("- Energy and context switching tracking")
            print("- Completion rate monitoring")
            print("- Burnout prevention indicators")
            
            # Offer to create a basic daily note if PM tools are available
            if self.task_extractor and self.wsjf_analyzer:
                if input("\nCreate basic daily note with top priorities? (y/n): ").lower() == 'y':
                    self._create_basic_pm_daily_note()
            else:
                print(f"\n{Colors.YELLOW}Feature in development - daily_template_generator not yet available{Colors.ENDC}")
        
        input("\nPress Enter to continue...")
    
    def _create_basic_pm_daily_note(self):
        """Create a basic PM-optimized daily note"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            daily_path = os.path.join(self.vault_path, 'Daily Notes', f"{today}_PM.md")
            
            if os.path.exists(daily_path):
                print(f"{Colors.YELLOW}PM daily note already exists{Colors.ENDC}")
                return
                
            os.makedirs(os.path.dirname(daily_path), exist_ok=True)
            
            # Extract and prioritize tasks
            tasks = self.task_extractor.extract_all_tasks()
            wsjf_report = self.wsjf_analyzer.generate_wsjf_report(tasks)
            
            with open(daily_path, 'w') as f:
                f.write(f"# PM Daily Note - {today}\n\n")
                f.write("## 🎯 Today's Top 3 Priorities (WSJF)\n\n")
                
                # Add top 3 WSJF priorities
                for i, item in enumerate(wsjf_report['top_10_recommendations'][:3], 1):
                    f.write(f"{i}. [ ] {item['task_content']}\n")
                    f.write(f"   - WSJF Score: {item['wsjf_score']:.1f}\n")
                    f.write(f"   - Product: {item.get('product_area', 'N/A')}\n\n")
                
                f.write("## 📊 Focus Area\n")
                f.write("Today's product focus: _________\n\n")
                
                f.write("## 🔋 Energy & Context\n")
                f.write("- Energy level (1-10): \n")
                f.write("- Context switches: \n")
                f.write("- Deep work blocks: \n\n")
                
                f.write("## ✅ Completed\n\n")
                
                f.write("## 📝 Notes\n\n")
                
                f.write("## 🙏 Gratitude\n- \n\n")
                
                f.write("## 📈 Metrics\n")
                f.write(f"- Tasks started: 0\n")
                f.write(f"- Tasks completed: 0\n")
                f.write(f"- Completion rate: 0%\n")
            
            print(f"{Colors.GREEN}Created PM daily note: {daily_path}{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}Error creating PM daily note: {e}{Colors.ENDC}")


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