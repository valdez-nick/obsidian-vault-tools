#!/usr/bin/env python3
"""
Meeting Notes Organizer - AI-powered categorization of unstructured meeting notes.

Automatically organizes rapid meeting notes into proper template sections using
AI analysis and the user's QuickAdd templates, with smart backup management.
"""

import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# AI imports with graceful fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .smart_backup_manager import SmartBackupManager

logger = logging.getLogger(__name__)


class QuickAddTemplateParser:
    """Parser for QuickAdd plugin templates from Obsidian configuration."""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.quickadd_config_path = (
            self.vault_path / ".obsidian" / "plugins" / "quickadd" / "data.json"
        )
    
    def get_meeting_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract meeting templates from QuickAdd configuration.
        
        Returns:
            Dictionary with 'meeting' and 'one_on_one' template structures
        """
        if not self.quickadd_config_path.exists():
            logger.warning("QuickAdd configuration not found")
            return self._get_default_templates()
        
        try:
            with open(self.quickadd_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            templates = {}
            
            # Extract templates from macros
            for macro in config.get('macros', []):
                if 'meeting' in macro['name'].lower():
                    template_info = self._extract_template_from_macro(macro)
                    if '1on1' in macro['name'].lower():
                        templates['one_on_one'] = template_info
                    else:
                        templates['meeting'] = template_info
            
            # Fallback to defaults if not found
            if 'meeting' not in templates:
                templates['meeting'] = self._get_default_templates()['meeting']
            if 'one_on_one' not in templates:
                templates['one_on_one'] = self._get_default_templates()['one_on_one']
            
            return templates
            
        except Exception as e:
            logger.error(f"Error parsing QuickAdd config: {e}")
            return self._get_default_templates()
    
    def _extract_template_from_macro(self, macro: Dict[str, Any]) -> Dict[str, Any]:
        """Extract template structure from a QuickAdd macro."""
        template_info = {
            'sections': [],
            'format': '',
            'metadata_fields': []
        }
        
        # Look for capture choice in macro commands
        for command in macro.get('commands', []):
            if command.get('type') == 'NestedChoice':
                choice = command.get('choice', {})
                if choice.get('type') == 'Capture':
                    format_content = choice.get('format', {}).get('format', '')
                    template_info['format'] = format_content
                    template_info['sections'] = self._parse_sections_from_format(format_content)
                    break
        
        return template_info
    
    def _parse_sections_from_format(self, format_content: str) -> List[str]:
        """Parse section headers from template format."""
        sections = []
        
        # Find section headers (#### pattern)
        header_matches = re.findall(r'####\s*([^\\n]+)', format_content)
        for header in header_matches:
            # Clean up header text (remove emojis and extra spaces)
            clean_header = re.sub(r'[^\w\s&]', '', header).strip()
            if clean_header:
                sections.append(clean_header)
        
        return sections
    
    def _get_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get default template structures if QuickAdd config is unavailable."""
        return {
            'meeting': {
                'sections': [
                    'Agenda',
                    'Discussion', 
                    'Action Items',
                    'Decisions',
                    'Next Steps'
                ],
                'format': 'general_meeting',
                'metadata_fields': ['meeting-type', 'attendees', 'date']
            },
            'one_on_one': {
                'sections': [
                    'Topics to Discuss',
                    'Feedback & Recognition',
                    'Project or Initiative Updates', 
                    'Action Items',
                    'Note'
                ],
                'format': 'one_on_one',
                'metadata_fields': ['name', 'role', 'team', 'date']
            }
        }


class AIContentCategorizer:
    """AI-powered content categorization for meeting notes."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize available AI clients
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = openai.OpenAI()
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.anthropic_client = anthropic.Anthropic()
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Anthropic initialization failed: {e}")
    
    def categorize_content(
        self, 
        content: str, 
        template_sections: List[str],
        template_type: str = 'meeting'
    ) -> Dict[str, List[str]]:
        """
        Categorize unstructured content into template sections.
        
        Args:
            content: Unstructured meeting notes content
            template_sections: Available template sections
            template_type: Type of template ('meeting' or 'one_on_one')
            
        Returns:
            Dictionary mapping section names to categorized content
        """
        if self.openai_client:
            return self._categorize_with_openai(content, template_sections, template_type)
        elif self.anthropic_client:
            return self._categorize_with_anthropic(content, template_sections, template_type)
        else:
            return self._categorize_with_rules(content, template_sections, template_type)
    
    def _categorize_with_openai(
        self, 
        content: str, 
        template_sections: List[str],
        template_type: str
    ) -> Dict[str, List[str]]:
        """Categorize content using OpenAI."""
        try:
            prompt = self._build_categorization_prompt(content, template_sections, template_type)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at organizing meeting notes. Categorize the provided content into the specified sections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return self._parse_ai_response(response.choices[0].message.content, template_sections)
            
        except Exception as e:
            logger.error(f"OpenAI categorization failed: {e}")
            return self._categorize_with_rules(content, template_sections, template_type)
    
    def _categorize_with_anthropic(
        self, 
        content: str, 
        template_sections: List[str],
        template_type: str
    ) -> Dict[str, List[str]]:
        """Categorize content using Anthropic Claude."""
        try:
            prompt = self._build_categorization_prompt(content, template_sections, template_type)
            
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_ai_response(response.content[0].text, template_sections)
            
        except Exception as e:
            logger.error(f"Anthropic categorization failed: {e}")
            return self._categorize_with_rules(content, template_sections, template_type)
    
    def _build_categorization_prompt(
        self, 
        content: str, 
        template_sections: List[str],
        template_type: str
    ) -> str:
        """Build the prompt for AI categorization."""
        sections_list = '\n'.join([f"- {section}" for section in template_sections])
        
        context = ""
        if template_type == "one_on_one":
            context = """
This is content from a 1:1 meeting. Focus on:
- Personal development discussions
- Feedback and recognition
- Individual project updates
- Career conversations
"""
        else:
            context = """
This is content from a team/group meeting. Focus on:
- Project discussions and decisions
- Team coordination
- Broader organizational topics
- Action items for multiple people
"""
        
        return f"""Analyze the following unstructured meeting notes and categorize each piece of content into the most appropriate section.

{context}

Available sections:
{sections_list}

Instructions:
1. Read through all the content carefully
2. For each line or thought, determine which section it best fits
3. Some content might fit multiple sections - use your best judgment
4. If content doesn't fit any section clearly, put it in the most general section
5. Preserve the original wording as much as possible
6. Format your response as:

SECTION_NAME:
- content item 1
- content item 2

ANOTHER_SECTION:
- content item 3

Meeting notes to categorize:
```
{content}
```

Categorized content:"""
    
    def _parse_ai_response(self, response: str, template_sections: List[str]) -> Dict[str, List[str]]:
        """Parse AI response into categorized content."""
        categorized = {section: [] for section in template_sections}
        
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            section_found = False
            for section in template_sections:
                if line.upper().startswith(section.upper().replace(' ', '_')):
                    current_section = section
                    section_found = True
                    break
                elif section.upper() in line.upper():
                    current_section = section
                    section_found = True
                    break
            
            if section_found:
                continue
            
            # If we have a current section and this looks like content
            if current_section and (line.startswith('-') or line.startswith('•') or len(line) > 10):
                # Clean up the content
                content = line.lstrip('- •').strip()
                if content:
                    categorized[current_section].append(content)
        
        return categorized
    
    def _categorize_with_rules(
        self, 
        content: str, 
        template_sections: List[str],
        template_type: str
    ) -> Dict[str, List[str]]:
        """Fallback rule-based categorization."""
        categorized = {section: [] for section in template_sections}
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            section = self._classify_line_with_rules(line, template_sections, template_type)
            categorized[section].append(line)
        
        return categorized
    
    def _classify_line_with_rules(
        self, 
        line: str, 
        template_sections: List[str],
        template_type: str
    ) -> str:
        """Classify a single line using rules."""
        line_lower = line.lower()
        
        # Action item patterns
        action_keywords = ['todo', 'action', 'follow up', 'need to', 'should', 'will', 'assign']
        if any(keyword in line_lower for keyword in action_keywords) or line.startswith('- [ ]'):
            action_sections = [s for s in template_sections if 'action' in s.lower()]
            if action_sections:
                return action_sections[0]
        
        # Decision patterns
        decision_keywords = ['decided', 'agreed', 'resolved', 'conclusion']
        if any(keyword in line_lower for keyword in decision_keywords):
            decision_sections = [s for s in template_sections if 'decision' in s.lower()]
            if decision_sections:
                return decision_sections[0]
        
        # Feedback patterns (for 1:1s)
        feedback_keywords = ['feedback', 'recognition', 'praise', 'appreciate', 'good job']
        if template_type == 'one_on_one' and any(keyword in line_lower for keyword in feedback_keywords):
            feedback_sections = [s for s in template_sections if 'feedback' in s.lower() or 'recognition' in s.lower()]
            if feedback_sections:
                return feedback_sections[0]
        
        # Default to first section (usually most general)
        return template_sections[0] if template_sections else 'Notes'


class MeetingNotesOrganizer:
    """Main orchestrator for meeting notes organization."""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.backup_manager = SmartBackupManager(vault_path)
        self.template_parser = QuickAddTemplateParser(vault_path)
        self.ai_categorizer = AIContentCategorizer()
        self.current_session: Optional[str] = None
    
    def find_current_daily_note(self) -> Optional[Path]:
        """Find the current daily note file."""
        # Common daily note patterns
        today = datetime.now()
        patterns = [
            f"{today.strftime('%Y-%m-%d')}.md",
            f"{today.strftime('%Y%m%d')}.md", 
            f"{today.strftime('%m-%d-%Y')}.md",
        ]
        
        # Check common daily note locations
        locations = [
            self.vault_path / "Daily Notes",
            self.vault_path / "dailies",
            self.vault_path / "daily",
            self.vault_path
        ]
        
        for location in locations:
            if location.exists():
                for pattern in patterns:
                    daily_note = location / pattern
                    if daily_note.exists():
                        return daily_note
        
        return None
    
    def extract_meeting_content(self, file_content: str) -> Tuple[str, str, str]:
        """
        Extract unstructured meeting content from daily note.
        
        Returns:
            Tuple of (before_content, meeting_content, after_content)
        """
        # Look for meeting section markers
        meeting_patterns = [
            r'(#{1,4}\s*.*(?:meeting|1:?1|standup).*?)$(.*?)(?=#{1,4}|\Z)',
            r'(#{1,4}\s*.*(?:Meeting|1:1|Meeting).*?)$(.*?)(?=#{1,4}|\Z)'
        ]
        
        for pattern in meeting_patterns:
            match = re.search(pattern, file_content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if match:
                meeting_header = match.group(1)
                meeting_content = match.group(2).strip()
                
                # Split the content
                start_pos = match.start()
                end_pos = match.end()
                
                before_content = file_content[:start_pos]
                after_content = file_content[end_pos:]
                
                return before_content, meeting_content, after_content
        
        # If no clear meeting section found, assume everything after a certain point is meeting content
        lines = file_content.split('\n')
        meeting_start_idx = None
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['meeting', '1:1', 'standup', 'discussion']):
                meeting_start_idx = i
                break
        
        if meeting_start_idx is not None:
            before_lines = lines[:meeting_start_idx]
            meeting_lines = lines[meeting_start_idx:]
            
            return '\n'.join(before_lines), '\n'.join(meeting_lines), ''
        
        # Fallback: treat all content as meeting content
        return '', file_content, ''
    
    def organize_meeting_notes(
        self, 
        file_path: Optional[str] = None,
        template_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Main method to organize meeting notes.
        
        Args:
            file_path: Path to file to organize (defaults to current daily note)
            template_type: Type of template to use ('meeting', 'one_on_one', or 'auto')
            
        Returns:
            Dictionary with organization results and preview
        """
        try:
            # Find file to organize
            if file_path:
                target_file = Path(file_path)
            else:
                target_file = self.find_current_daily_note()
            
            if not target_file or not target_file.exists():
                return {
                    'success': False,
                    'error': 'No daily note found to organize',
                    'suggestions': ['Create a daily note first', 'Specify a file path']
                }
            
            # Read original content
            with open(target_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            if not original_content.strip():
                return {
                    'success': False,
                    'error': 'File is empty',
                    'suggestions': ['Add some meeting notes first']
                }
            
            # Create backup session
            self.current_session = self.backup_manager.create_session_backup(
                str(target_file), original_content
            )
            
            # Extract meeting content
            before_content, meeting_content, after_content = self.extract_meeting_content(original_content)
            
            if not meeting_content.strip():
                return {
                    'success': False,
                    'error': 'No meeting content found to organize',
                    'suggestions': ['Add meeting notes with clear section headers']
                }
            
            # Determine template type
            if template_type == 'auto':
                template_type = self._detect_template_type(meeting_content)
            
            # Get templates
            templates = self.template_parser.get_meeting_templates()
            template = templates.get(template_type, templates['meeting'])
            
            # Categorize content
            categorized = self.ai_categorizer.categorize_content(
                meeting_content, 
                template['sections'],
                template_type
            )
            
            # Generate organized content
            organized_content = self._generate_organized_content(
                before_content, categorized, after_content, template_type
            )
            
            # Generate preview
            preview = self._generate_preview(original_content, organized_content)
            
            return {
                'success': True,
                'session_id': self.current_session,
                'template_type': template_type,
                'categorized_content': categorized,
                'organized_content': organized_content,
                'preview': preview,
                'file_path': str(target_file),
                'can_undo': True
            }
            
        except Exception as e:
            logger.error(f"Error organizing meeting notes: {e}")
            return {
                'success': False,
                'error': f"Organization failed: {str(e)}",
                'session_id': self.current_session
            }
    
    def apply_organization(self, organized_content: str, file_path: str) -> bool:
        """Apply the organized content to the file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(organized_content)
            
            logger.info(f"Applied organization to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply organization: {e}")
            return False
    
    def undo_organization(self) -> bool:
        """Undo the last organization operation."""
        if not self.current_session:
            return False
        
        success = self.backup_manager.undo_from_memory(self.current_session)
        if success:
            self.backup_manager.complete_session(self.current_session, success=True)
            self.current_session = None
        
        return success
    
    def complete_organization(self, success: bool = True):
        """Complete the current organization session."""
        if self.current_session:
            self.backup_manager.complete_session(self.current_session, success)
            self.current_session = None
    
    def _detect_template_type(self, content: str) -> str:
        """Automatically detect whether this is a 1:1 or general meeting."""
        content_lower = content.lower()
        
        # 1:1 indicators
        one_on_one_keywords = ['1:1', 'one on one', 'feedback', 'career', 'personal', 'development']
        general_meeting_keywords = ['team', 'standup', 'project', 'group', 'everyone']
        
        one_on_one_score = sum(1 for keyword in one_on_one_keywords if keyword in content_lower)
        general_score = sum(1 for keyword in general_meeting_keywords if keyword in content_lower)
        
        return 'one_on_one' if one_on_one_score > general_score else 'meeting'
    
    def _generate_organized_content(
        self, 
        before_content: str, 
        categorized: Dict[str, List[str]], 
        after_content: str,
        template_type: str
    ) -> str:
        """Generate the final organized content."""
        # Build organized meeting section
        organized_meeting = []
        
        # Add appropriate header
        if template_type == 'one_on_one':
            organized_meeting.append("### 1:1 Meeting Notes")
        else:
            organized_meeting.append("### Meeting Notes")
        
        organized_meeting.append("")
        
        # Add each section with content
        for section, items in categorized.items():
            if items:  # Only add sections that have content
                organized_meeting.append(f"#### {section}")
                organized_meeting.append("")
                
                for item in items:
                    # Ensure proper list formatting
                    if not item.startswith(('- ', '* ', '• ')):
                        item = f"- {item}"
                    organized_meeting.append(item)
                
                organized_meeting.append("")
        
        # Combine all parts
        parts = []
        if before_content.strip():
            parts.append(before_content.rstrip())
        
        parts.append('\n'.join(organized_meeting))
        
        if after_content.strip():
            parts.append(after_content.strip())
        
        return '\n\n'.join(parts)
    
    def _generate_preview(self, original: str, organized: str) -> str:
        """Generate a preview showing before and after."""
        return f"""
BEFORE (Original):
{'-' * 50}
{original[:500]}{'...' if len(original) > 500 else ''}

AFTER (Organized):
{'-' * 50}
{organized[:500]}{'...' if len(organized) > 500 else ''}

Changes:
- Categorized content into proper sections
- Added section headers for better organization
- Maintained original content wording
"""
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current session."""
        if not self.current_session:
            return None
        
        return self.backup_manager.get_session_info(self.current_session)


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Test with sample content
    sample_content = """# Daily Note - 2024-01-15

## Morning Tasks
- Review project status
- Prepare for meeting

## Meeting with Sarah
We discussed the new feature rollout and had some good insights.
Action: I need to follow up with the engineering team about the timeline.
She gave positive feedback on the recent design changes.
Need to schedule another check-in next week.
Decision: We're moving forward with Option B for the user interface.
Sarah mentioned she appreciated the quick turnaround on the prototypes.

## Afternoon Tasks
- Update project documentation
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test vault structure
        vault_path = Path(temp_dir)
        daily_notes_dir = vault_path / "Daily Notes"
        daily_notes_dir.mkdir()
        
        # Create test daily note
        daily_note = daily_notes_dir / "2024-01-15.md"
        with open(daily_note, 'w') as f:
            f.write(sample_content)
        
        # Test organization
        organizer = MeetingNotesOrganizer(str(vault_path))
        result = organizer.organize_meeting_notes(str(daily_note), template_type='one_on_one')
        
        print("Organization Result:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Template Type: {result['template_type']}")
            print("\nCategorized Content:")
            for section, items in result['categorized_content'].items():
                if items:
                    print(f"  {section}: {len(items)} items")
            
            print(f"\nPreview:\n{result['preview']}")
        else:
            print(f"Error: {result['error']}")