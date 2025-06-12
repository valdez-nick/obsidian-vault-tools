"""
Template Service for intelligent Templater integration and template application.
"""

import asyncio
import logging
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import structlog
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from dateutil.parser import parse as parse_date

from ..models import Note, TemplateApplication, TemplateRule, TemplateMatch
from ..vault import Vault
from ..ai.content_summarizer import ContentSummarizer

logger = structlog.get_logger(__name__)


class TemplateType(Enum):
    """Types of templates supported."""
    DAILY_NOTE = "daily_note"
    WEEKLY_NOTE = "weekly_note"  
    MONTHLY_NOTE = "monthly_note"
    PROJECT = "project"
    MEETING = "meeting"
    RESEARCH = "research"
    REFERENCE = "reference"
    TASK = "task"
    CONTACT = "contact"
    CUSTOM = "custom"


class TriggerType(Enum):
    """Types of template triggers."""
    FILE_CREATION = "file_creation"
    FILE_MODIFICATION = "file_modification"
    CONTENT_PATTERN = "content_pattern"
    METADATA_CONDITION = "metadata_condition"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class TemplateConfig:
    """Configuration for template operations."""
    # Template directories
    template_dirs: List[Path] = field(default_factory=lambda: [Path("Templates")])
    
    # Auto-application settings
    auto_apply: bool = True
    auto_apply_on_creation: bool = True
    auto_apply_on_modification: bool = False
    
    # Processing settings
    max_concurrent_applications: int = 5
    backup_before_apply: bool = True
    
    # Template matching
    fuzzy_matching: bool = True
    match_threshold: float = 0.7
    
    # Content analysis
    analyze_content_for_templates: bool = True
    suggest_templates: bool = True
    
    # Safety settings
    require_confirmation: bool = False
    dry_run_mode: bool = False


@dataclass
class TemplateContext:
    """Context for template rendering."""
    # Note information
    note: Optional[Note] = None
    note_path: Optional[Path] = None
    file_name: Optional[str] = None
    
    # Date/time context
    current_date: date = field(default_factory=date.today)
    current_datetime: datetime = field(default_factory=datetime.now)
    
    # Vault context
    vault_path: Optional[Path] = None
    
    # User-defined variables
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata from frontmatter
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamic content
    content_summary: Optional[str] = None
    content_topics: List[str] = field(default_factory=list)
    related_notes: List[str] = field(default_factory=list)


class TemplateService:
    """
    Intelligent template service for Obsidian notes.
    
    Provides:
    - Automatic template detection and application
    - Smart template suggestions based on content
    - Template rendering with dynamic context
    - Batch template operations
    - Integration with Templater plugin patterns
    """
    
    def __init__(
        self,
        vault: Vault,
        config: Optional[TemplateConfig] = None,
    ):
        self.vault = vault
        self.config = config or TemplateConfig()
        self.content_summarizer = ContentSummarizer()
        
        # Template storage
        self.templates: Dict[str, 'TemplateDefinition'] = {}
        self.template_rules: List[TemplateRule] = []
        
        # Jinja2 environment
        self.jinja_env = None
        
        # Compiled patterns for performance
        self._patterns = self._compile_patterns()
        
        # Load templates
        asyncio.create_task(self._load_templates())
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for template processing."""
        return {
            'templater_commands': re.compile(r'<%\s*([^%>]+)\s*%>'),
            'frontmatter': re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL),
            'template_variables': re.compile(r'\{\{([^}]+)\}\}'),
            'file_path_template': re.compile(r'\{\{([^}]+)\}\}'),
            'date_patterns': re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
        }
    
    async def _load_templates(self) -> None:
        """Load templates from configured directories."""
        logger.info("Loading templates", dirs=self.config.template_dirs)
        
        # Setup Jinja2 environment
        if self.config.template_dirs:
            loader = FileSystemLoader([str(d) for d in self.config.template_dirs])
            self.jinja_env = Environment(
                loader=loader,
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            
            # Add custom filters
            self.jinja_env.filters['dateformat'] = self._date_format_filter
            self.jinja_env.filters['slugify'] = self._slugify_filter
            self.jinja_env.filters['titlecase'] = self._title_case_filter
        
        # Load template definitions
        for template_dir in self.config.template_dirs:
            await self._load_templates_from_dir(template_dir)
        
        # Load template rules
        await self._load_template_rules()
        
        logger.info("Templates loaded", count=len(self.templates))
    
    async def apply_template_to_note(
        self,
        note_id: str,
        template_name: str,
        context: Optional[TemplateContext] = None,
        force: bool = False,
    ) -> TemplateApplication:
        """
        Apply a template to an existing note.
        
        Args:
            note_id: ID of the note to modify
            template_name: Name of the template to apply
            context: Additional context for rendering
            force: Apply even if note already has content
            
        Returns:
            Result of the template application
        """
        logger.info("Applying template", note_id=note_id, template=template_name)
        
        # Get the note
        note = await self.vault.get_note(note_id)
        if not note:
            raise ValueError(f"Note not found: {note_id}")
        
        # Get the template
        template_def = self.templates.get(template_name)
        if not template_def:
            raise ValueError(f"Template not found: {template_name}")
        
        # Check if note already has significant content
        if not force and len(note.content.strip()) > 100:
            return TemplateApplication(
                note_id=note_id,
                template_name=template_name,
                success=False,
                error="Note already has content (use force=True to override)",
                applied_at=datetime.utcnow(),
            )
        
        # Build rendering context
        render_context = await self._build_context(note, context)
        
        try:
            # Backup if configured
            if self.config.backup_before_apply:
                await self._backup_note(note)
            
            # Render template
            rendered_content = await self._render_template(template_def, render_context)
            
            # Apply to note
            if self.config.dry_run_mode:
                logger.info("Dry run mode - template would be applied", 
                           rendered_length=len(rendered_content))
                success = True
            else:
                # Update note content
                await self.vault.update_note(note_id, rendered_content)
                success = True
            
            return TemplateApplication(
                note_id=note_id,
                template_name=template_name,
                success=success,
                rendered_content=rendered_content if self.config.dry_run_mode else None,
                applied_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error("Template application failed", 
                        note_id=note_id, 
                        template=template_name, 
                        error=str(e))
            
            return TemplateApplication(
                note_id=note_id,
                template_name=template_name,
                success=False,
                error=str(e),
                applied_at=datetime.utcnow(),
            )
    
    async def suggest_templates_for_note(
        self,
        note_id: str,
        limit: int = 5,
    ) -> List[TemplateMatch]:
        """
        Suggest appropriate templates for a note based on content analysis.
        
        Args:
            note_id: ID of the note to analyze
            limit: Maximum number of suggestions
            
        Returns:
            List of template suggestions with confidence scores
        """
        note = await self.vault.get_note(note_id)
        if not note:
            return []
        
        logger.debug("Suggesting templates", note_id=note_id)
        
        suggestions = []
        
        for template_name, template_def in self.templates.items():
            confidence = await self._calculate_template_match_confidence(note, template_def)
            
            if confidence >= self.config.match_threshold:
                suggestions.append(TemplateMatch(
                    template_name=template_name,
                    confidence_score=confidence,
                    match_reasons=await self._get_match_reasons(note, template_def),
                    template_type=template_def.template_type,
                ))
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        return suggestions[:limit]
    
    async def auto_apply_templates(
        self,
        note_ids: Optional[List[str]] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
    ) -> List[TemplateApplication]:
        """
        Automatically apply templates based on rules and content analysis.
        
        Args:
            note_ids: Specific notes to process (default: all notes)
            trigger_type: What triggered this auto-application
            
        Returns:
            List of template applications
        """
        if not self.config.auto_apply:
            return []
        
        note_ids = note_ids or await self.vault.get_all_note_ids()
        logger.info("Auto-applying templates", count=len(note_ids), trigger=trigger_type)
        
        applications = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_applications)
        
        async def process_note(note_id: str) -> Optional[TemplateApplication]:
            async with semaphore:
                return await self._auto_apply_to_note(note_id, trigger_type)
        
        # Process notes concurrently
        tasks = [process_note(note_id) for note_id in note_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful applications
        for result in results:
            if isinstance(result, TemplateApplication) and result.success:
                applications.append(result)
            elif isinstance(result, Exception):
                logger.warning("Auto-application failed", error=str(result))
        
        logger.info("Auto-application completed", successful=len(applications))
        return applications
    
    async def create_note_from_template(
        self,
        template_name: str,
        file_path: Path,
        context: Optional[TemplateContext] = None,
    ) -> str:
        """
        Create a new note from a template.
        
        Args:
            template_name: Name of the template to use
            file_path: Path where the new note should be created
            context: Rendering context
            
        Returns:
            ID of the created note
        """
        template_def = self.templates.get(template_name)
        if not template_def:
            raise ValueError(f"Template not found: {template_name}")
        
        # Build context for new note
        if context is None:
            context = TemplateContext()
        
        context.note_path = file_path
        context.file_name = file_path.stem
        context.vault_path = self.vault.path
        
        # Render template
        content = await self._render_template(template_def, context)
        
        # Create the note
        note_id = await self.vault.create_note(file_path, content)
        
        logger.info("Created note from template", 
                   note_id=note_id, 
                   template=template_name, 
                   path=file_path)
        
        return note_id
    
    async def get_template_variables(self, template_name: str) -> List[str]:
        """Get list of variables used in a template."""
        template_def = self.templates.get(template_name)
        if not template_def:
            return []
        
        # Extract variables from template content
        variables = set()
        
        # Find Jinja2 variables
        for match in self._patterns['template_variables'].finditer(template_def.content):
            var_expr = match.group(1).strip()
            # Extract simple variable names (ignore complex expressions)
            if '.' not in var_expr and '|' not in var_expr and '(' not in var_expr:
                variables.add(var_expr)
        
        return sorted(list(variables))
    
    async def validate_template(self, template_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a template for syntax errors.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        template_def = self.templates.get(template_name)
        if not template_def:
            return False, [f"Template not found: {template_name}"]
        
        errors = []
        
        try:
            # Try to parse the template
            if self.jinja_env:
                self.jinja_env.from_string(template_def.content)
            
            # Check for required fields
            if not template_def.name:
                errors.append("Template name is required")
            
            if not template_def.content:
                errors.append("Template content is required")
            
        except Exception as e:
            errors.append(f"Template syntax error: {str(e)}")
        
        return len(errors) == 0, errors
    
    async def _build_context(
        self,
        note: Optional[Note] = None,
        additional_context: Optional[TemplateContext] = None,
    ) -> Dict[str, Any]:
        """Build complete rendering context."""
        context = {}
        
        # Date/time context
        now = datetime.now()
        today = date.today()
        
        context.update({
            'now': now,
            'today': today,
            'current_date': today,
            'current_datetime': now,
            'year': today.year,
            'month': today.month,
            'day': today.day,
            'weekday': today.strftime('%A'),
            'month_name': today.strftime('%B'),
        })
        
        # Note context
        if note:
            context.update({
                'note': note,
                'title': note.title,
                'content': note.content,
                'tags': note.tags,
                'path': str(note.path),
                'file_name': note.path.stem,
                'created': note.created_at,
                'modified': note.modified_at,
            })
            
            # Extract frontmatter metadata
            if note.frontmatter:
                context['metadata'] = note.frontmatter
                context.update(note.frontmatter)  # Add metadata directly to context
        
        # Vault context
        context.update({
            'vault_path': str(self.vault.path),
            'vault_name': self.vault.path.name,
        })
        
        # Additional context
        if additional_context:
            if additional_context.variables:
                context.update(additional_context.variables)
            
            if additional_context.metadata:
                context.update(additional_context.metadata)
            
            # Override dates if provided
            if additional_context.current_date:
                context['current_date'] = additional_context.current_date
                context['today'] = additional_context.current_date
        
        # Dynamic content analysis
        if note and self.config.analyze_content_for_templates:
            try:
                # Generate summary
                if len(note.content) > 200:
                    context['content_summary'] = await self.content_summarizer.summarize(
                        note.content, max_length=100
                    )
                
                # Extract topics (simplified)
                words = re.findall(r'\b\w+\b', note.content.lower())
                word_freq = {}
                for word in words:
                    if len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                context['content_topics'] = [word for word, _ in top_words]
                
            except Exception as e:
                logger.warning("Failed to analyze content for template context", error=str(e))
        
        return context
    
    async def _render_template(
        self,
        template_def: 'TemplateDefinition',
        context: Dict[str, Any],
    ) -> str:
        """Render a template with the given context."""
        if not self.jinja_env:
            # Simple variable substitution fallback
            content = template_def.content
            for key, value in context.items():
                content = content.replace(f'{{{{{key}}}}}', str(value))
            return content
        
        # Use Jinja2 for full template rendering
        template = self.jinja_env.from_string(template_def.content)
        return template.render(**context)
    
    async def _calculate_template_match_confidence(
        self,
        note: Note,
        template_def: 'TemplateDefinition',
    ) -> float:
        """Calculate how well a template matches a note."""
        confidence = 0.0
        
        # Check file path patterns
        if template_def.path_patterns:
            for pattern in template_def.path_patterns:
                if re.search(pattern, str(note.path), re.IGNORECASE):
                    confidence += 0.3
                    break
        
        # Check content patterns
        if template_def.content_patterns:
            for pattern in template_def.content_patterns:
                if re.search(pattern, note.content, re.IGNORECASE):
                    confidence += 0.2
        
        # Check metadata conditions
        if template_def.metadata_conditions and note.frontmatter:
            matches = 0
            for key, expected_value in template_def.metadata_conditions.items():
                if key in note.frontmatter:
                    actual_value = note.frontmatter[key]
                    if actual_value == expected_value:
                        matches += 1
            
            if matches > 0:
                confidence += 0.3 * (matches / len(template_def.metadata_conditions))
        
        # Check template type against note characteristics
        type_match = self._check_type_match(note, template_def.template_type)
        confidence += type_match * 0.2
        
        return min(confidence, 1.0)
    
    def _check_type_match(self, note: Note, template_type: TemplateType) -> float:
        """Check how well a note matches a template type."""
        path_str = str(note.path).lower()
        content = note.content.lower()
        
        if template_type == TemplateType.DAILY_NOTE:
            if 'daily' in path_str or re.search(r'\d{4}-\d{2}-\d{2}', path_str):
                return 1.0
        
        elif template_type == TemplateType.MEETING:
            if 'meeting' in path_str or 'meeting' in content:
                return 1.0
        
        elif template_type == TemplateType.PROJECT:
            if 'project' in path_str or 'project' in content:
                return 1.0
        
        elif template_type == TemplateType.RESEARCH:
            if 'research' in path_str or any(term in content for term in ['research', 'study', 'analysis']):
                return 1.0
        
        return 0.0
    
    async def _get_match_reasons(
        self,
        note: Note,
        template_def: 'TemplateDefinition',
    ) -> List[str]:
        """Get human-readable reasons why a template matches a note."""
        reasons = []
        
        if template_def.path_patterns:
            for pattern in template_def.path_patterns:
                if re.search(pattern, str(note.path), re.IGNORECASE):
                    reasons.append(f"File path matches pattern: {pattern}")
        
        if template_def.content_patterns:
            for pattern in template_def.content_patterns:
                if re.search(pattern, note.content, re.IGNORECASE):
                    reasons.append(f"Content matches pattern: {pattern}")
        
        type_match = self._check_type_match(note, template_def.template_type)
        if type_match > 0.5:
            reasons.append(f"Note type matches: {template_def.template_type.value}")
        
        return reasons
    
    async def _auto_apply_to_note(
        self,
        note_id: str,
        trigger_type: TriggerType,
    ) -> Optional[TemplateApplication]:
        """Auto-apply templates to a single note if rules match."""
        note = await self.vault.get_note(note_id)
        if not note:
            return None
        
        # Check if note is suitable for auto-application
        if len(note.content.strip()) > 200:  # Don't auto-apply to notes with content
            return None
        
        # Find matching rules
        matching_rules = []
        for rule in self.template_rules:
            if await self._rule_matches_note(rule, note, trigger_type):
                matching_rules.append(rule)
        
        if not matching_rules:
            return None
        
        # Apply the highest priority rule
        rule = max(matching_rules, key=lambda r: r.priority)
        return await self.apply_template_to_note(note_id, rule.template_name)
    
    async def _rule_matches_note(
        self,
        rule: TemplateRule,
        note: Note,
        trigger_type: TriggerType,
    ) -> bool:
        """Check if a template rule matches a note."""
        # Check trigger type
        if rule.trigger_type != trigger_type and rule.trigger_type != TriggerType.MANUAL:
            return False
        
        # Check conditions
        for condition in rule.conditions:
            if not await self._evaluate_condition(condition, note):
                return False
        
        return True
    
    async def _evaluate_condition(self, condition: Dict[str, Any], note: Note) -> bool:
        """Evaluate a rule condition against a note."""
        condition_type = condition.get('type')
        
        if condition_type == 'path_pattern':
            pattern = condition.get('pattern', '')
            return bool(re.search(pattern, str(note.path), re.IGNORECASE))
        
        elif condition_type == 'content_pattern':
            pattern = condition.get('pattern', '')
            return bool(re.search(pattern, note.content, re.IGNORECASE))
        
        elif condition_type == 'metadata':
            key = condition.get('key', '')
            value = condition.get('value')
            if note.frontmatter and key in note.frontmatter:
                return note.frontmatter[key] == value
        
        elif condition_type == 'file_size':
            operator = condition.get('operator', 'lt')
            threshold = condition.get('threshold', 0)
            size = len(note.content)
            
            if operator == 'lt':
                return size < threshold
            elif operator == 'gt':
                return size > threshold
            elif operator == 'eq':
                return size == threshold
        
        return False
    
    async def _load_templates_from_dir(self, template_dir: Path) -> None:
        """Load templates from a directory."""
        if not template_dir.exists():
            logger.warning("Template directory not found", dir=template_dir)
            return
        
        for template_file in template_dir.glob('*.md'):
            try:
                template_def = await self._parse_template_file(template_file)
                self.templates[template_def.name] = template_def
            except Exception as e:
                logger.warning("Failed to load template", file=template_file, error=str(e))
    
    async def _parse_template_file(self, template_file: Path) -> 'TemplateDefinition':
        """Parse a template file into a TemplateDefinition."""
        content = template_file.read_text(encoding='utf-8')
        
        # Extract frontmatter
        frontmatter = {}
        content_without_frontmatter = content
        
        frontmatter_match = self._patterns['frontmatter'].match(content)
        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            content_without_frontmatter = content[frontmatter_match.end():]
            
            # Parse YAML frontmatter (simplified)
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')
        
        # Create template definition
        template_type = TemplateType.CUSTOM
        if 'type' in frontmatter:
            try:
                template_type = TemplateType(frontmatter['type'])
            except ValueError:
                pass
        
        return TemplateDefinition(
            name=frontmatter.get('name', template_file.stem),
            content=content_without_frontmatter,
            template_type=template_type,
            description=frontmatter.get('description', ''),
            path_patterns=frontmatter.get('path_patterns', '').split(',') if frontmatter.get('path_patterns') else [],
            content_patterns=frontmatter.get('content_patterns', '').split(',') if frontmatter.get('content_patterns') else [],
            metadata_conditions=json.loads(frontmatter.get('metadata_conditions', '{}')),
            file_path=template_file,
        )
    
    async def _load_template_rules(self) -> None:
        """Load template application rules."""
        # This would load from a configuration file
        # For now, create some default rules
        
        self.template_rules = [
            TemplateRule(
                name="daily_note_auto",
                template_name="daily_note",
                trigger_type=TriggerType.FILE_CREATION,
                conditions=[
                    {
                        'type': 'path_pattern',
                        'pattern': r'\d{4}-\d{2}-\d{2}\.md$'
                    }
                ],
                priority=10,
            ),
            TemplateRule(
                name="meeting_note_auto",
                template_name="meeting",
                trigger_type=TriggerType.FILE_CREATION,
                conditions=[
                    {
                        'type': 'path_pattern',
                        'pattern': r'meeting|Meeting'
                    }
                ],
                priority=8,
            ),
        ]
    
    async def _backup_note(self, note: Note) -> None:
        """Create a backup of a note before applying template."""
        backup_path = note.path.with_suffix(f'.backup-{datetime.now().strftime("%Y%m%d-%H%M%S")}.md')
        backup_path.write_text(note.content, encoding='utf-8')
        logger.debug("Created note backup", backup=backup_path)
    
    # Jinja2 custom filters
    def _date_format_filter(self, date_obj: Union[datetime, date, str], format_string: str = '%Y-%m-%d') -> str:
        """Format a date object."""
        if isinstance(date_obj, str):
            try:
                date_obj = parse_date(date_obj)
            except Exception:
                return date_obj
        
        if isinstance(date_obj, (datetime, date)):
            return date_obj.strftime(format_string)
        
        return str(date_obj)
    
    def _slugify_filter(self, text: str) -> str:
        """Convert text to a URL-friendly slug."""
        # Simple slugify implementation
        import unicodedata
        
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
        text = re.sub(r'\s+', '-', text.strip())
        return text.lower()
    
    def _title_case_filter(self, text: str) -> str:
        """Convert text to title case."""
        return text.title()


@dataclass
class TemplateDefinition:
    """Definition of a template."""
    name: str
    content: str
    template_type: TemplateType
    description: str = ""
    
    # Matching criteria
    path_patterns: List[str] = field(default_factory=list)
    content_patterns: List[str] = field(default_factory=list)
    metadata_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Template metadata
    file_path: Optional[Path] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)