#!/usr/bin/env python3
"""
Natural Language Query System for Obsidian Vault Manager
Enables LLM-style querying of all vault management features
"""

import re
import difflib
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

@dataclass
class QueryResult:
    """Result of a natural language query"""
    confidence: float
    action: str
    function: Callable
    parameters: Dict
    description: str
    suggestions: List[str] = None

class NaturalLanguageProcessor:
    """Natural language processor for vault management commands"""
    
    def __init__(self, vault_manager):
        self.vault_manager = vault_manager
        self.command_patterns = self._build_command_patterns()
        self.synonyms = self._build_synonyms()
        
    def _build_synonyms(self) -> Dict[str, List[str]]:
        """Build synonym mappings for better query understanding"""
        return {
            'analyze': ['analyze', 'analyse', 'check', 'examine', 'inspect', 'review', 'scan', 'study'],
            'backup': ['backup', 'save', 'archive', 'preserve', 'copy', 'store'],
            'tags': ['tags', 'tag', 'labels', 'categories', 'keywords', 'metadata'],
            'files': ['files', 'documents', 'notes', 'content', 'markdown'],
            'organize': ['organize', 'structure', 'arrange', 'sort', 'order', 'tidy'],
            'fix': ['fix', 'repair', 'correct', 'resolve', 'clean', 'update'],
            'find': ['find', 'search', 'locate', 'discover', 'identify'],
            'create': ['create', 'make', 'generate', 'build', 'produce'],
            'merge': ['merge', 'combine', 'join', 'unite', 'consolidate'],
            'remove': ['remove', 'delete', 'eliminate', 'clean up', 'purge'],
            'research': ['research', 'investigate', 'explore', 'study', 'lookup'],
            'duplicate': ['duplicate', 'duplicates', 'copies', 'repeated', 'similar'],
            'structure': ['structure', 'folders', 'directories', 'organization'],
            'report': ['report', 'summary', 'analysis', 'statistics', 'stats'],
            'install': ['install', 'setup', 'configure', 'add'],
            'settings': ['settings', 'config', 'configuration', 'preferences', 'options']
        }
    
    def _build_command_patterns(self) -> Dict[str, Dict]:
        """Build comprehensive command patterns for all vault manager features"""
        return {
            # Vault Analysis
            'analyze_tags': {
                'patterns': [
                    r'(?:analyze|check|examine|review)\s+(?:my\s+)?(?:tags?|labels?)',
                    r'(?:tag|tags?)\s+(?:analysis|statistics|stats)',
                    r'(?:show|display|view)\s+(?:my\s+)?(?:tag|tags?)\s+(?:info|data)',
                    r'(?:what|which)\s+tags?\s+(?:do|are)',
                    r'count.*tags?',
                    r'analyze\s+tags?',
                    r'my\s+tags?'
                ],
                'function': 'analyze_vault_tags',
                'description': 'Analyze vault tags and show statistics',
                'menu_path': ['1', '1']
            },
            
            'analyze_structure': {
                'patterns': [
                    r'(?:analyze|check|examine)\s+(?:folder|directory|structure)',
                    r'(?:folder|directory)\s+(?:structure|analysis|stats)',
                    r'(?:show|display)\s+(?:folder|directory)\s+(?:info|structure)',
                    r'(?:how\s+many|count)\s+(?:folders?|directories)'
                ],
                'function': 'analyze_folder_structure',
                'description': 'Analyze folder structure and file distribution',
                'menu_path': ['1', '2']
            },
            
            'find_untagged': {
                'patterns': [
                    r'(?:find|search|locate)\s+(?:untagged|without\s+tags?)',
                    r'(?:files?|notes?)\s+(?:without|missing)\s+tags?',
                    r'(?:which|what)\s+(?:files?|notes?)\s+(?:have\s+no|lack)\s+tags?',
                    r'untagged\s+(?:files?|notes?|content)'
                ],
                'function': 'find_untagged_files',
                'description': 'Find files without tags',
                'menu_path': ['1', '3']
            },
            
            'generate_report': {
                'patterns': [
                    r'(?:generate|create|make)\s+(?:report|analysis)',
                    r'(?:export|save)\s+(?:analysis|data|stats)\s+(?:to\s+)?(?:json|file)',
                    r'(?:detailed|comprehensive)\s+(?:report|analysis)',
                    r'(?:save|export|output)\s+(?:tag|vault)\s+(?:data|info)'
                ],
                'function': 'generate_analysis_report',
                'description': 'Generate comprehensive analysis report',
                'menu_path': ['1', '4']
            },
            
            # Tag Management
            'preview_tag_issues': {
                'patterns': [
                    r'(?:preview|check|show)\s+(?:tag\s+)?issues?',
                    r'(?:what|which)\s+(?:tag\s+)?(?:problems?|issues?)',
                    r'(?:dry\s+run|preview)\s+(?:tag\s+)?(?:fix|repair)',
                    r'tag\s+(?:problems?|issues?)\s+(?:preview|check)'
                ],
                'function': 'preview_tag_issues',
                'description': 'Preview tag issues without making changes',
                'menu_path': ['2', '1']
            },
            
            'fix_all_tags': {
                'patterns': [
                    r'(?:fix|repair|correct)\s+(?:all\s+)?tags?',
                    r'(?:auto\s+)?fix\s+(?:tag\s+)?(?:issues?|problems?)',
                    r'(?:clean\s+up|tidy)\s+(?:all\s+)?tags?',
                    r'(?:repair|fix)\s+(?:everything|all)'
                ],
                'function': 'fix_all_tag_issues',
                'description': 'Fix all tag issues automatically',
                'menu_path': ['2', '2']
            },
            
            'fix_quoted_tags': {
                'patterns': [
                    r'(?:fix|repair)\s+(?:quoted|quote)\s+tags?',
                    r'(?:remove|fix)\s+(?:quotes?|quotation)\s+(?:from\s+)?tags?',
                    r'(?:quoted|quote)\s+tags?\s+(?:fix|repair|issue)',
                    r'tags?\s+(?:with|in)\s+quotes?'
                ],
                'function': 'fix_quoted_tags',
                'description': 'Fix quoted tags only',
                'menu_path': ['2', '3']
            },
            
            'merge_similar_tags': {
                'patterns': [
                    r'(?:merge|combine|join)\s+(?:similar|duplicate)\s+tags?',
                    r'(?:consolidate|unify)\s+tags?',
                    r'(?:similar|duplicate)\s+tags?\s+(?:merge|combine)',
                    r'(?:find|merge)\s+(?:tag\s+)?(?:duplicates?|similarities)'
                ],
                'function': 'merge_similar_tags',
                'description': 'Merge similar tags together',
                'menu_path': ['2', '4']
            },
            
            'remove_generic_tags': {
                'patterns': [
                    r'(?:remove|delete|clean)\s+(?:generic|common)\s+tags?',
                    r'(?:generic|useless|common)\s+tags?\s+(?:remove|delete)',
                    r'(?:clean\s+up|purge)\s+(?:generic|useless)\s+tags?',
                    r'(?:delete|remove)\s+(?:bad|useless)\s+tags?'
                ],
                'function': 'remove_generic_tags',
                'description': 'Remove generic or useless tags',
                'menu_path': ['2', '5']
            },
            
            'auto_tag_files': {
                'patterns': [
                    r'(?:auto\s+)?tag\s+(?:untagged\s+)?(?:files?|notes?)',
                    r'(?:automatically\s+)?(?:add|assign)\s+tags?',
                    r'(?:ai|smart)\s+(?:tagging|tags?)',
                    r'(?:tag|label)\s+(?:files?|notes?)\s+(?:automatically|with\s+ai)'
                ],
                'function': 'auto_tag_files',
                'description': 'Auto-tag untagged files with AI',
                'menu_path': ['2', '6']
            },
            
            # Backup Operations
            'incremental_backup': {
                'patterns': [
                    r'(?:incremental|quick)\s+backup',
                    r'backup\s+(?:changes|incrementally)',
                    r'(?:fast|quick)\s+backup',
                    r'(?:create|make)\s+(?:incremental|quick)\s+backup',
                    r'backup\s+(?:the\s+)?vault',
                    r'(?:create|make)\s+(?:a\s+)?backup'
                ],
                'function': 'create_incremental_backup',
                'description': 'Create incremental backup',
                'menu_path': ['3', '1']
            },
            
            'full_backup': {
                'patterns': [
                    r'(?:full|complete|comprehensive)\s+backup',
                    r'backup\s+(?:everything|all|complete)',
                    r'(?:create|make)\s+(?:full|complete)\s+backup',
                    r'(?:archive|backup)\s+(?:entire|whole)\s+vault'
                ],
                'function': 'create_full_backup',
                'description': 'Create full backup',
                'menu_path': ['3', '2']
            },
            
            'setup_kopia': {
                'patterns': [
                    r'(?:setup|install|configure)\s+kopia',
                    r'kopia\s+(?:setup|installation|config)',
                    r'(?:advanced|professional)\s+backup\s+(?:setup|system)',
                    r'(?:setup|configure)\s+(?:advanced|enterprise)\s+backup'
                ],
                'function': 'setup_kopia',
                'description': 'Setup Kopia for advanced backups',
                'menu_path': ['3', '3']
            },
            
            'view_backup_history': {
                'patterns': [
                    r'(?:view|show|display)\s+backup\s+(?:history|list)',
                    r'backup\s+(?:history|log|records?)',
                    r'(?:list|show)\s+(?:previous|old)\s+backups?',
                    r'(?:what|which)\s+backups?\s+(?:exist|available)'
                ],
                'function': 'view_backup_history',
                'description': 'View backup history',
                'menu_path': ['3', '4']
            },
            
            'restore_backup': {
                'patterns': [
                    r'(?:restore|recover)\s+(?:from\s+)?backup',
                    r'backup\s+(?:restore|recovery)',
                    r'(?:recover|restore)\s+(?:vault|files?)',
                    r'(?:rollback|revert)\s+(?:to\s+)?(?:backup|previous)'
                ],
                'function': 'restore_backup_menu',
                'description': 'Restore from backup',
                'menu_path': ['3', '5']
            },
            
            # V2 Features
            'ai_analysis': {
                'patterns': [
                    r'(?:ai|smart|intelligent)\s+(?:analysis|analyze)',
                    r'(?:analyze|examine)\s+(?:content\s+)?(?:with\s+)?ai',
                    r'(?:quality|content)\s+(?:analysis|check)\s+(?:with\s+)?ai',
                    r'(?:ai|artificial\s+intelligence)\s+(?:content|vault)\s+(?:analysis|review)'
                ],
                'function': 'ai_content_analysis',
                'description': 'AI-powered content analysis',
                'menu_path': ['4', '1']
            },
            
            'research_topic': {
                'patterns': [
                    r'(?:research|investigate|explore)\s+(?:topic|subject|\w+)',
                    r'(?:create|generate)\s+(?:research|notes?)\s+(?:about|on|for)',
                    r'(?:research|lookup|find)\s+(?:information|content)\s+(?:about|on)',
                    r'(?:ai\s+)?research\s+(?:assistant|help)',
                    r'research\s+\w+',
                    r'(?:research|investigate|explore)\s+\w+(?:\s+\w+)*'
                ],
                'function': 'research_topics',
                'description': 'Research topics and create notes',
                'menu_path': ['4', '2']
            },
            
            'smart_organize': {
                'patterns': [
                    r'(?:smart|ai|intelligent)\s+(?:organize|organization)',
                    r'(?:organize|structure)\s+(?:files?|vault)\s+(?:with\s+)?ai',
                    r'(?:auto|automatic)\s+(?:organization|organize)',
                    r'(?:ai|smart)\s+(?:file|content)\s+(?:organization|structuring)'
                ],
                'function': 'smart_file_organization',
                'description': 'Smart file organization with AI',
                'menu_path': ['4', '3']
            },
            
            'find_duplicates': {
                'patterns': [
                    r'(?:find|detect|locate)\s+(?:duplicates?|duplicate\s+content)',
                    r'(?:duplicate|similar)\s+(?:content|files?|notes?)',
                    r'(?:check|scan)\s+for\s+(?:duplicates?|similar\s+content)',
                    r'(?:duplicates?|copies)\s+(?:detection|finding)'
                ],
                'function': 'find_duplicate_content',
                'description': 'Find and merge duplicates',
                'menu_path': ['4', '4']
            },
            
            'advanced_analytics': {
                'patterns': [
                    r'(?:advanced|detailed)\s+(?:analytics|statistics|stats)',
                    r'(?:vault|content)\s+(?:analytics|metrics|insights)',
                    r'(?:generate|show)\s+(?:detailed|advanced)\s+(?:stats|analytics)',
                    r'(?:comprehensive|detailed)\s+(?:analysis|report)'
                ],
                'function': 'generate_advanced_analytics',
                'description': 'Generate advanced vault analytics',
                'menu_path': ['4', '5']
            },
            
            'comprehensive_curation': {
                'patterns': [
                    r'(?:comprehensive|complete|full)\s+(?:curation|cleanup)',
                    r'(?:curate|clean\s+up|improve)\s+(?:entire\s+)?vault',
                    r'(?:all|everything)\s+(?:improvements?|cleanup|curation)',
                    r'(?:complete|comprehensive)\s+vault\s+(?:maintenance|improvement)'
                ],
                'function': 'comprehensive_vault_curation',
                'description': 'Comprehensive vault curation',
                'menu_path': ['4', '6']
            },
            
            'configure_ai': {
                'patterns': [
                    r'(?:configure|setup|set)\s+(?:ai|artificial\s+intelligence)',
                    r'(?:ai|llm)\s+(?:settings|configuration|setup)',
                    r'(?:change|set|configure)\s+(?:ai\s+)?(?:provider|model)',
                    r'(?:openai|anthropic|ollama)\s+(?:setup|config)'
                ],
                'function': 'configure_ai_settings',
                'description': 'Configure AI settings',
                'menu_path': ['4', '7']
            },
            
            # Advanced Tools
            'install_v2': {
                'patterns': [
                    r'(?:install|setup|add)\s+(?:v2|version\s+2|librarian\s+v2)',
                    r'(?:obsidian\s+)?librarian\s+(?:v2\s+)?(?:install|setup)',
                    r'(?:upgrade|update)\s+to\s+v2',
                    r'(?:install|setup)\s+(?:advanced|new)\s+features?'
                ],
                'function': 'install_v2_features',
                'description': 'Install Obsidian Librarian v2',
                'menu_path': ['5', '1']
            },
            
            'check_updates': {
                'patterns': [
                    r'(?:check|look)\s+for\s+updates?',
                    r'(?:update|upgrade)\s+(?:available|check)',
                    r'(?:latest|new)\s+(?:version|update)',
                    r'(?:software|system)\s+updates?'
                ],
                'function': 'check_for_updates',
                'description': 'Check for updates',
                'menu_path': ['5', '2']
            },
            
            'performance_benchmark': {
                'patterns': [
                    r'(?:performance|speed)\s+(?:benchmark|test)',
                    r'(?:benchmark|test)\s+(?:performance|speed)',
                    r'(?:how\s+fast|speed)\s+(?:is|test)',
                    r'(?:performance|speed)\s+(?:analysis|metrics)'
                ],
                'function': 'run_performance_benchmark',
                'description': 'Run performance benchmarks',
                'menu_path': ['5', '3']
            },
            
            'debug_mode': {
                'patterns': [
                    r'(?:enable|turn\s+on|activate)\s+(?:debug|debugging)',
                    r'debug\s+(?:mode|on|enable)',
                    r'(?:troubleshoot|diagnose)\s+(?:problems?|issues?)',
                    r'(?:verbose|detailed)\s+(?:output|logging)'
                ],
                'function': 'enable_debug_mode',
                'description': 'Enable debug mode',
                'menu_path': ['5', '4']
            },
            
            'clean_cache': {
                'patterns': [
                    r'(?:clean|clear|remove)\s+(?:cache|temp|temporary)',
                    r'(?:cache|temporary)\s+(?:cleanup|clean|clear)',
                    r'(?:delete|remove)\s+(?:cached|temp)\s+files?',
                    r'(?:clean\s+up|clear)\s+(?:storage|disk\s+space)'
                ],
                'function': 'clean_cache_files',
                'description': 'Clean cache files',
                'menu_path': ['5', '5']
            },
            
            # Settings
            'change_vault': {
                'patterns': [
                    r'(?:change|set|switch)\s+(?:vault|directory)',
                    r'(?:vault\s+)?(?:location|path|directory)\s+(?:change|set)',
                    r'(?:different|new|another)\s+vault',
                    r'(?:switch|change)\s+to\s+(?:different|another)\s+vault'
                ],
                'function': 'change_vault_location',
                'description': 'Change vault location',
                'menu_path': ['6', '1']
            },
            
            'toggle_colors': {
                'patterns': [
                    r'(?:toggle|enable|disable)\s+(?:colors?|color)',
                    r'(?:colors?|color)\s+(?:on|off|toggle|enable|disable)',
                    r'(?:turn\s+)?(?:on|off)\s+(?:colors?|color\s+output)',
                    r'(?:colored|color)\s+(?:output|text|terminal)'
                ],
                'function': 'toggle_color_output',
                'description': 'Toggle color output',
                'menu_path': ['6', '2']
            },
            
            'backup_settings': {
                'patterns': [
                    r'(?:backup\s+)?settings?\s+(?:config|configure)',
                    r'(?:configure|set|change)\s+backup\s+settings?',
                    r'backup\s+(?:preferences|options|configuration)',
                    r'(?:auto\s+)?backup\s+(?:setup|config)'
                ],
                'function': 'configure_backup_settings',
                'description': 'Configure backup settings',
                'menu_path': ['6', '3']
            },
            
            'reset_settings': {
                'patterns': [
                    r'(?:reset|restore)\s+(?:settings?|defaults?)',
                    r'(?:default|factory)\s+(?:settings?|reset)',
                    r'(?:clear|reset)\s+(?:all\s+)?(?:settings?|config)',
                    r'(?:restore|return)\s+to\s+defaults?'
                ],
                'function': 'reset_to_defaults',
                'description': 'Reset settings to defaults',
                'menu_path': ['6', '4']
            },
            
            # General Help and Navigation
            'help': {
                'patterns': [
                    r'(?:help|assistance|guide)',
                    r'(?:how\s+to|what\s+can|what\s+does)',
                    r'(?:documentation|manual|instructions?)',
                    r'(?:show|display)\s+(?:help|options|commands?)'
                ],
                'function': 'show_help',
                'description': 'Show help and documentation',
                'menu_path': ['7']
            },
            
            'exit': {
                'patterns': [
                    r'(?:exit|quit|bye|goodbye)',
                    r'(?:close|end|finish|stop)',
                    r'(?:leave|get\s+out|done)',
                    r'(?:goodbye|farewell|see\s+you)'
                ],
                'function': 'exit_application',
                'description': 'Exit the application',
                'menu_path': ['0']
            }
        }
    
    def normalize_query(self, query: str) -> str:
        """Normalize the input query for better matching"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove common filler words
        filler_words = {'please', 'can', 'you', 'i', 'want', 'to', 'would', 'like', 'could', 'the', 'a', 'an'}
        words = query.split()
        words = [word for word in words if word not in filler_words]
        
        # Expand synonyms
        expanded_words = []
        for word in words:
            found_synonym = False
            for concept, synonyms in self.synonyms.items():
                if word in synonyms:
                    expanded_words.append(concept)
                    found_synonym = True
                    break
            if not found_synonym:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def calculate_confidence(self, query: str, pattern: str) -> float:
        """Calculate confidence score for a pattern match"""
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Base confidence from match
            confidence = 0.8
            
            # Bonus for exact word matches
            pattern_words = set(re.findall(r'\w+', pattern.replace('?:', '').replace('\\s+', ' ')))
            query_words = set(query.split())
            overlap = len(pattern_words.intersection(query_words))
            confidence += overlap * 0.05
            
            # Bonus for longer matches
            match_length = len(match.group(0))
            confidence += min(match_length / len(query), 0.2)
            
            return min(confidence, 1.0)
        return 0.0
    
    def find_best_match(self, query: str) -> Optional[QueryResult]:
        """Find the best matching command for a query"""
        normalized_query = self.normalize_query(query)
        best_match = None
        best_confidence = 0.0
        
        for command_name, command_info in self.command_patterns.items():
            for pattern in command_info['patterns']:
                confidence = self.calculate_confidence(normalized_query, pattern)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = QueryResult(
                        confidence=confidence,
                        action=command_name,
                        function=getattr(self.vault_manager, command_info['function'], None),
                        parameters={},
                        description=command_info['description'],
                        suggestions=[]
                    )
        
        # If confidence is too low, provide suggestions
        if best_confidence < 0.4:
            suggestions = self.get_suggestions(normalized_query)
            if best_match:
                best_match.suggestions = suggestions[:5]
            else:
                return QueryResult(
                    confidence=0.0,
                    action='unknown',
                    function=None,
                    parameters={},
                    description='No matching command found',
                    suggestions=suggestions[:5]
                )
        
        return best_match
    
    def get_suggestions(self, query: str) -> List[str]:
        """Get command suggestions based on partial matches"""
        suggestions = []
        query_words = set(query.split())
        
        for command_name, command_info in self.command_patterns.items():
            # Check if any pattern words match query words
            for pattern in command_info['patterns']:
                pattern_words = set(re.findall(r'\w+', pattern.replace('?:', '').replace('\\s+', ' ')))
                if pattern_words.intersection(query_words):
                    suggestions.append(command_info['description'])
                    break
        
        # Also suggest based on string similarity
        all_descriptions = [info['description'] for info in self.command_patterns.values()]
        similar = difflib.get_close_matches(query, all_descriptions, n=3, cutoff=0.3)
        suggestions.extend(similar)
        
        return list(set(suggestions))  # Remove duplicates
    
    def process_query(self, query: str) -> QueryResult:
        """Process a natural language query and return the result"""
        if not query.strip():
            return QueryResult(
                confidence=0.0,
                action='empty',
                function=None,
                parameters={},
                description='Empty query',
                suggestions=['Try: "analyze tags", "backup vault", "help"']
            )
        
        # Check for special parameter extraction
        parameters = self.extract_parameters(query)
        
        # Find best matching command
        result = self.find_best_match(query)
        if result:
            result.parameters.update(parameters)
        
        return result
    
    def extract_parameters(self, query: str) -> Dict:
        """Extract parameters from the query"""
        parameters = {}
        
        # Extract research topics
        research_match = re.search(r'(?:research|investigate|explore|lookup)\s+(?:about\s+|on\s+|for\s+)?"([^"]+)"', query, re.IGNORECASE)
        if not research_match:
            research_match = re.search(r'(?:research|investigate|explore|lookup)\s+(?:about\s+|on\s+|for\s+)?(\w+(?:\s+\w+){0,3})', query, re.IGNORECASE)
        if research_match:
            parameters['research_topic'] = research_match.group(1)
        
        # Extract vault paths
        path_match = re.search(r'(?:vault|path|directory)[\s:]+(["/][\w\s/.-]+)', query, re.IGNORECASE)
        if path_match:
            parameters['vault_path'] = path_match.group(1).strip('"')
        
        # Extract file patterns
        file_match = re.search(r'(?:files?|notes?)\s+(?:like|matching|with)\s+([*\w.-]+)', query, re.IGNORECASE)
        if file_match:
            parameters['file_pattern'] = file_match.group(1)
        
        return parameters
    
    def get_command_help(self) -> str:
        """Get help text for all available commands"""
        help_text = "\n🤖 Natural Language Commands Available:\n\n"
        
        categories = {
            'Vault Analysis': ['analyze_tags', 'analyze_structure', 'find_untagged', 'generate_report'],
            'Tag Management': ['preview_tag_issues', 'fix_all_tags', 'fix_quoted_tags', 'merge_similar_tags', 'remove_generic_tags', 'auto_tag_files'],
            'Backup Operations': ['incremental_backup', 'full_backup', 'setup_kopia', 'view_backup_history', 'restore_backup'],
            'AI Features': ['ai_analysis', 'research_topic', 'smart_organize', 'find_duplicates', 'advanced_analytics', 'comprehensive_curation', 'configure_ai'],
            'Advanced Tools': ['install_v2', 'check_updates', 'performance_benchmark', 'debug_mode', 'clean_cache'],
            'Settings': ['change_vault', 'toggle_colors', 'backup_settings', 'reset_settings'],
            'General': ['help', 'exit']
        }
        
        for category, commands in categories.items():
            help_text += f"📁 {category}:\n"
            for command in commands:
                if command in self.command_patterns:
                    desc = self.command_patterns[command]['description']
                    example = self.command_patterns[command]['patterns'][0]
                    # Clean up regex for display
                    example = re.sub(r'\(\?\:|\\s\+|[\[\](){}*+?|\\]', ' ', example)
                    example = ' '.join(example.split())
                    help_text += f"  • {desc}\n    Example: \"{example}\"\n"
            help_text += "\n"
        
        help_text += "💡 Tips:\n"
        help_text += "• Use natural language: \"analyze my tags\" or \"backup the vault\"\n"
        help_text += "• Be specific: \"research artificial intelligence\" or \"find duplicate files\"\n"
        help_text += "• Ask for help: \"what can you do?\" or \"show commands\"\n"
        help_text += "• Try variations: \"check tags\", \"examine tags\", \"look at tags\"\n"
        help_text += "• Ask questions: \"how many tags do I have?\" or \"what files need tags?\"\n"
        
        return help_text
    
    def get_autocomplete_suggestions(self, partial_query: str) -> List[str]:
        """Get autocomplete suggestions for partial queries"""
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Common starting phrases
        common_phrases = [
            "analyze my tags",
            "backup the vault",
            "find files without tags",
            "research about",
            "fix all tags",
            "merge similar tags",
            "create backup",
            "organize files",
            "check for duplicates",
            "show me help",
            "what can you do",
            "generate report",
            "clean cache",
            "install v2",
            "configure ai"
        ]
        
        # Find matching phrases
        for phrase in common_phrases:
            if phrase.startswith(partial_lower):
                suggestions.append(phrase)
        
        # Find matching words in patterns
        for command_info in self.command_patterns.values():
            for pattern in command_info['patterns']:
                # Extract meaningful words from pattern
                words = re.findall(r'\w+', pattern.replace('?:', ''))
                for word in words:
                    if word.lower().startswith(partial_lower):
                        suggestions.append(word)
        
        # Remove duplicates and limit
        return list(set(suggestions))[:8]
    
    def get_contextual_suggestions(self, query: str) -> List[str]:
        """Get contextual suggestions based on query intent"""
        suggestions = []
        query_lower = query.lower()
        
        # Contextual suggestions based on keywords
        if any(word in query_lower for word in ['tag', 'tags']):
            suggestions.extend([
                "analyze vault tags",
                "fix all tag issues", 
                "merge similar tags",
                "find files without tags",
                "auto-tag files with AI"
            ])
        
        if any(word in query_lower for word in ['backup', 'save', 'archive']):
            suggestions.extend([
                "create incremental backup",
                "create full backup",
                "view backup history",
                "restore from backup"
            ])
        
        if any(word in query_lower for word in ['research', 'investigate', 'explore']):
            suggestions.extend([
                "research artificial intelligence",
                "research machine learning",
                "research obsidian plugins"
            ])
        
        if any(word in query_lower for word in ['organize', 'structure', 'arrange']):
            suggestions.extend([
                "smart file organization",
                "analyze folder structure",
                "organize files with AI"
            ])
        
        if any(word in query_lower for word in ['duplicate', 'similar', 'copies']):
            suggestions.extend([
                "find duplicate content",
                "merge similar tags",
                "check for duplicate files"
            ])
        
        return suggestions[:5]