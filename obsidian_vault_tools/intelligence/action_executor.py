"""
Action Executor

Maps intents to concrete actions and executes them.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import logging

from .intent_detector import IntentType, DetectedIntent

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing an action"""
    success: bool
    data: Any
    message: str
    action_taken: str
    follow_up_actions: List[str] = None


class ActionExecutor:
    """Executes actions based on detected intents"""
    
    def __init__(self, vault_manager=None):
        self.vault_manager = vault_manager
        self.action_map = self._build_action_map()
        self._action_cache = {}
        
    def _build_action_map(self) -> Dict[IntentType, Callable]:
        """Build mapping of intents to action handlers"""
        return {
            IntentType.ANALYZE_TAGS: self._execute_tag_analysis,
            IntentType.ANALYZE_VAULT: self._execute_vault_analysis,
            IntentType.RESEARCH_TOPIC: self._execute_research,
            IntentType.ORGANIZE_FILES: self._execute_organization,
            IntentType.FIND_FILES: self._execute_file_search,
            IntentType.CLEANUP_TAGS: self._execute_tag_cleanup,
            IntentType.BACKUP_VAULT: self._execute_backup,
            IntentType.ANALYZE_LINKS: self._execute_link_analysis,
            IntentType.MERGE_DUPLICATES: self._execute_duplicate_merge,
            IntentType.HELP: self._execute_help,
        }
    
    async def execute_intent(self, intent: DetectedIntent) -> ActionResult:
        """Execute actions based on detected intent"""
        try:
            # Get the appropriate handler
            handler = self.action_map.get(intent.intent_type, self._execute_unknown)
            
            # Execute the action
            if asyncio.iscoroutinefunction(handler):
                result = await handler(intent)
            else:
                result = handler(intent)
                
            logger.info(f"Executed action for intent: {intent.intent_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing intent {intent.intent_type}: {e}")
            return ActionResult(
                success=False,
                data=None,
                message=f"Error executing action: {str(e)}",
                action_taken="error_handler"
            )
    
    def _execute_tag_analysis(self, intent: DetectedIntent) -> ActionResult:
        """Execute tag analysis action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            # Use the tag analyzer
            from ..analysis import TagAnalyzer
            analyzer = TagAnalyzer(self.vault_manager.current_vault)
            
            # Analyze all tags
            analysis = analyzer.analyze_all_tags()
            
            # Format results
            most_used = analysis.get('most_used_tags', [])
            total_tags = analysis.get('total_tags', 0)
            
            # Build response message
            if most_used:
                tag_list = "\n".join([
                    f"  {i+1}. #{tag}: {count} files" 
                    for i, (tag, count) in enumerate(most_used[:10])
                ])
                message = f"📊 Your most used tags:\n{tag_list}\n\nTotal unique tags: {total_tags}"
            else:
                message = "No tags found in your vault."
            
            return ActionResult(
                success=True,
                data=analysis,
                message=message,
                action_taken="analyze_all_tags",
                follow_up_actions=["cleanup_tags", "merge_similar_tags"]
            )
            
        except Exception as e:
            logger.error(f"Tag analysis error: {e}")
            return self._error_result(str(e), "analyze_all_tags")
    
    def _execute_vault_analysis(self, intent: DetectedIntent) -> ActionResult:
        """Execute vault analysis action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            # Use the vault analyzer
            from ..analysis import VaultAnalyzer
            analyzer = VaultAnalyzer(self.vault_manager.current_vault)
            
            # Get vault statistics
            structure = analyzer.analyze_vault_structure()
            health = analyzer.calculate_vault_health()
            
            # Format results
            message = f"""📊 Vault Overview:
  📄 Total files: {structure['total_files']}
  📁 Directories: {structure['total_directories']}
  🏷️  Health score: {health['score']}/100
  
File types: {', '.join(f"{ext}({count})" for ext, count in structure['file_types'].items())}"""
            
            return ActionResult(
                success=True,
                data={'structure': structure, 'health': health},
                message=message,
                action_taken="analyze_vault_complete",
                follow_up_actions=["organize_files", "cleanup_vault"]
            )
            
        except Exception as e:
            logger.error(f"Vault analysis error: {e}")
            return self._error_result(str(e), "analyze_vault")
    
    async def _execute_research(self, intent: DetectedIntent) -> ActionResult:
        """Execute research action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            topic = intent.entities.get('topic', intent.raw_input)
            
            # Use the enhanced research assistant
            from ..research_assistant import ResearchAssistant
            
            # Check if LLM is available
            llm_enabled = hasattr(self.vault_manager, 'llm_enabled') and self.vault_manager.llm_enabled
            
            research_assistant = ResearchAssistant(
                self.vault_manager.current_vault,
                llm_enabled=llm_enabled
            )
            
            # Perform comprehensive research
            print(f"\n🔍 Researching: {topic}")
            print("📚 Gathering information from multiple sources...")
            
            result = await research_assistant.research_topic(topic)
            
            if result['success']:
                message = f"""✅ Research completed for '{topic}'!

📄 Created: {os.path.basename(result['note_path'])}
📚 Sources found: {result['sources_found']}
🔗 Related notes: {result['related_notes']}

The research note includes:
- Overview and key concepts
- Technical details
- Applications and use cases
- Related vault content
- References and sources

Open the note to add your own findings and expand the research."""
                
                return ActionResult(
                    success=True,
                    data=result,
                    message=message,
                    action_taken="comprehensive_research",
                    follow_up_actions=["edit_note", "expand_research", "create_summary"]
                )
            else:
                return ActionResult(
                    success=False,
                    data=result,
                    message="Research failed. Please try again.",
                    action_taken="research_failed"
                )
            
        except Exception as e:
            logger.error(f"Research error: {e}")
            return self._error_result(str(e), "research_topic")
    
    def _execute_organization(self, intent: DetectedIntent) -> ActionResult:
        """Execute file organization action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            # Use the file organizer
            from ..organization import FileOrganizer
            organizer = FileOrganizer(self.vault_manager.current_vault)
            
            # Get organization suggestions
            suggestions = organizer.suggest_folder_structure()
            
            # Format top suggestions
            suggestion_list = []
            for file_path, suggestion in list(suggestions['suggestions'].items())[:5]:
                suggestion_list.append(
                    f"  • {file_path}\n    → {suggestion['suggested']} ({suggestion['reason']})"
                )
            
            message = f"""📁 Organization Analysis:
{suggestions['summary']}

Top suggestions:
{chr(10).join(suggestion_list)}

Use 'apply organization' to implement these changes."""
            
            return ActionResult(
                success=True,
                data=suggestions,
                message=message,
                action_taken="analyze_organization",
                follow_up_actions=["apply_organization", "review_suggestions"]
            )
            
        except Exception as e:
            logger.error(f"Organization error: {e}")
            return self._error_result(str(e), "organize_files")
    
    def _execute_file_search(self, intent: DetectedIntent) -> ActionResult:
        """Execute file search action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            # Extract search terms
            search_terms = intent.entities.get('topic', intent.raw_input).lower().split()
            
            # Search files
            import os
            found_files = []
            vault_path = self.vault_manager.current_vault
            
            for root, dirs, files in os.walk(vault_path):
                for file in files:
                    if file.endswith('.md'):
                        # Check filename
                        if any(term in file.lower() for term in search_terms):
                            found_files.append(os.path.relpath(os.path.join(root, file), vault_path))
                            continue
                            
                        # Check content
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                if any(term in content for term in search_terms):
                                    found_files.append(os.path.relpath(file_path, vault_path))
                        except:
                            continue
            
            if found_files:
                file_list = "\n".join([f"  • {f}" for f in found_files[:10]])
                message = f"🔍 Found {len(found_files)} files:\n{file_list}"
                if len(found_files) > 10:
                    message += f"\n  ... and {len(found_files) - 10} more"
            else:
                message = f"No files found containing: {' '.join(search_terms)}"
            
            return ActionResult(
                success=True,
                data={'files': found_files, 'search_terms': search_terms},
                message=message,
                action_taken="search_files",
                follow_up_actions=["open_file", "refine_search"]
            )
            
        except Exception as e:
            logger.error(f"File search error: {e}")
            return self._error_result(str(e), "find_files")
    
    def _execute_tag_cleanup(self, intent: DetectedIntent) -> ActionResult:
        """Execute tag cleanup action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            # Analyze tag issues
            from ..analysis import TagAnalyzer
            analyzer = TagAnalyzer(self.vault_manager.current_vault)
            
            # Simple tag cleanup suggestions
            all_tags = analyzer.analyze_all_tags()
            tag_counts = all_tags.get('tag_counts', {})
            
            # Find potential issues
            issues = {
                'single_use': [tag for tag, count in tag_counts.items() if count == 1],
                'similar': self._find_similar_tags(list(tag_counts.keys())),
                'formatting': [tag for tag in tag_counts.keys() if not tag.islower()],
            }
            
            message = f"""🏷️ Tag Cleanup Analysis:
  
Single-use tags: {len(issues['single_use'])}
Similar tags: {len(issues['similar'])} pairs
Formatting issues: {len(issues['formatting'])}

Suggested actions:
• Review single-use tags for removal
• Merge similar tags
• Standardize tag formatting"""
            
            return ActionResult(
                success=True,
                data=issues,
                message=message,
                action_taken="analyze_tag_issues",
                follow_up_actions=["merge_tags", "remove_unused_tags", "fix_formatting"]
            )
            
        except Exception as e:
            logger.error(f"Tag cleanup error: {e}")
            return self._error_result(str(e), "cleanup_tags")
    
    def _execute_backup(self, intent: DetectedIntent) -> ActionResult:
        """Execute backup action"""
        try:
            if not self.vault_manager:
                return self._no_vault_manager()
                
            # Create backup
            from ..backup import BackupManager
            backup_mgr = BackupManager()
            
            backup_path = backup_mgr.create_backup(
                self.vault_manager.current_vault,
                backup_type='full'
            )
            
            if backup_path:
                message = f"✅ Backup created successfully:\n{backup_path}"
                success = True
            else:
                message = "❌ Backup failed"
                success = False
            
            return ActionResult(
                success=success,
                data={'backup_path': backup_path},
                message=message,
                action_taken="create_backup",
                follow_up_actions=["verify_backup", "schedule_backups"]
            )
            
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return self._error_result(str(e), "backup_vault")
    
    def _execute_link_analysis(self, intent: DetectedIntent) -> ActionResult:
        """Execute link analysis"""
        # This would analyze internal links, find broken links, etc.
        return ActionResult(
            success=True,
            data={},
            message="Link analysis feature coming soon!",
            action_taken="link_analysis_placeholder"
        )
    
    def _execute_duplicate_merge(self, intent: DetectedIntent) -> ActionResult:
        """Execute duplicate merge"""
        # This would find and merge duplicate content
        return ActionResult(
            success=True,
            data={},
            message="Duplicate detection feature coming soon!",
            action_taken="duplicate_merge_placeholder"
        )
    
    def _execute_help(self, intent: DetectedIntent) -> ActionResult:
        """Show help information"""
        help_text = """🤖 Intelligence System Help

I understand natural language commands like:
• "show my most used tags" - Analyzes tag usage
• "analyze vault" - Shows vault statistics
• "research quantum computing" - Creates research note
• "organize files" - Suggests file organization
• "find notes about python" - Searches content
• "cleanup tags" - Analyzes tag issues
• "backup vault" - Creates backup

Just type what you want to do!"""
        
        return ActionResult(
            success=True,
            data={},
            message=help_text,
            action_taken="show_help"
        )
    
    def _execute_unknown(self, intent: DetectedIntent) -> ActionResult:
        """Handle unknown intents"""
        return ActionResult(
            success=False,
            data={},
            message=f"I didn't understand '{intent.raw_input}'. Type 'help' for examples.",
            action_taken="unknown_intent"
        )
    
    def _find_similar_tags(self, tags: List[str]) -> List[Tuple[str, str]]:
        """Find similar tags that might be duplicates"""
        similar_pairs = []
        
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                # Check for similarity
                if (tag1.lower() == tag2.lower() or
                    tag1 in tag2 or tag2 in tag1 or
                    self._levenshtein_distance(tag1.lower(), tag2.lower()) <= 2):
                    similar_pairs.append((tag1, tag2))
                    
        return similar_pairs[:5]  # Limit to top 5
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _no_vault_manager(self) -> ActionResult:
        """Return result when vault manager is not available"""
        return ActionResult(
            success=False,
            data=None,
            message="Vault manager not initialized. Please set up your vault first.",
            action_taken="error_no_vault"
        )
    
    def _error_result(self, error: str, action: str) -> ActionResult:
        """Create an error result"""
        return ActionResult(
            success=False,
            data=None,
            message=f"Error: {error}",
            action_taken=f"{action}_error"
        )