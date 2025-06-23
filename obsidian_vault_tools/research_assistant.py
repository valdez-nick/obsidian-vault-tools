"""
Research Assistant

Provides comprehensive research capabilities without external dependencies.
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import urllib.parse
import urllib.request
import logging

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """Provides research capabilities using built-in tools"""
    
    def __init__(self, vault_path: str, llm_enabled: bool = False):
        self.vault_path = vault_path
        self.llm_enabled = llm_enabled
        self.research_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load research note templates"""
        return {
            'default': """# {topic}

*Created: {date}*

## Overview

{overview}

## Key Concepts

{concepts}

## Technical Details

{details}

## Applications & Use Cases

{applications}

## Related Topics

{related}

## Sources & References

{sources}

## Notes

{notes}

---
*Research compiled on {date}*
""",
            'technical': """# {topic} - Technical Research

*Created: {date}*

## Executive Summary

{overview}

## Technical Background

### Core Concepts
{concepts}

### Architecture & Implementation
{details}

## Code Examples

```
{code_examples}
```

## Best Practices

{best_practices}

## Common Issues & Solutions

{issues}

## Tools & Resources

{tools}

## References

{sources}

---
*Technical research compiled on {date}*
""",
            'academic': """# {topic} - Academic Research

*Created: {date}*

## Abstract

{overview}

## Introduction

{introduction}

## Literature Review

{literature}

## Methodology

{methodology}

## Key Findings

{findings}

## Discussion

{discussion}

## Conclusion

{conclusion}

## Bibliography

{sources}

---
*Academic research compiled on {date}*
"""
        }
    
    async def research_topic(self, topic: str, depth: str = 'standard') -> Dict[str, Any]:
        """Research a topic and create comprehensive notes"""
        logger.info(f"Researching topic: {topic}")
        
        research_data = {
            'topic': topic,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'sources': [],
            'content': {},
            'related_vault_content': []
        }
        
        # Search existing vault content
        research_data['related_vault_content'] = self._search_vault_for_topic(topic)
        
        # Gather information from various sources
        tasks = [
            self._search_wikipedia(topic),
            self._search_definitions(topic),
            self._generate_outline(topic),
        ]
        
        # If LLM is enabled, add AI-powered research
        if self.llm_enabled:
            tasks.append(self._llm_research(topic))
        
        # Gather all research
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Research task {i} failed: {result}")
                continue
                
            if isinstance(result, dict):
                research_data['content'].update(result)
                if 'sources' in result:
                    research_data['sources'].extend(result['sources'])
        
        # Generate research note
        note_content = self._generate_research_note(research_data, depth)
        
        # Save to vault
        note_path = self._save_research_note(topic, note_content)
        
        return {
            'success': True,
            'note_path': note_path,
            'topic': topic,
            'sources_found': len(research_data['sources']),
            'related_notes': len(research_data['related_vault_content']),
            'content_preview': note_content[:500] + '...' if len(note_content) > 500 else note_content
        }
    
    def _search_vault_for_topic(self, topic: str) -> List[Dict[str, str]]:
        """Search existing vault for related content"""
        related_content = []
        search_terms = topic.lower().split()
        
        try:
            for root, dirs, files in os.walk(self.vault_path):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Check for relevance
                            content_lower = content.lower()
                            relevance_score = sum(
                                content_lower.count(term) for term in search_terms
                            )
                            
                            if relevance_score > 0:
                                # Extract relevant snippet
                                snippet = self._extract_relevant_snippet(content, search_terms)
                                related_content.append({
                                    'file': os.path.relpath(file_path, self.vault_path),
                                    'relevance': relevance_score,
                                    'snippet': snippet
                                })
                        except Exception as e:
                            logger.error(f"Error reading {file_path}: {e}")
            
            # Sort by relevance
            related_content.sort(key=lambda x: x['relevance'], reverse=True)
            return related_content[:10]  # Top 10 related notes
            
        except Exception as e:
            logger.error(f"Error searching vault: {e}")
            return []
    
    def _extract_relevant_snippet(self, content: str, search_terms: List[str]) -> str:
        """Extract relevant snippet from content"""
        lines = content.split('\n')
        relevant_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in search_terms):
                # Get context (line before and after)
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                relevant_lines.extend(lines[start:end])
                
                if len(relevant_lines) > 5:
                    break
        
        return '\n'.join(relevant_lines[:5]) if relevant_lines else content[:200]
    
    async def _search_wikipedia(self, topic: str) -> Dict[str, Any]:
        """Search Wikipedia for topic information"""
        try:
            # Simple Wikipedia API search
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
            
            with urllib.request.urlopen(search_url, timeout=5) as response:
                data = json.loads(response.read().decode())
                
            if data.get('extract'):
                return {
                    'overview': data['extract'],
                    'sources': [f"Wikipedia: {data.get('content_urls', {}).get('desktop', {}).get('page', '')}"]
                }
                
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            
        return {}
    
    async def _search_definitions(self, topic: str) -> Dict[str, Any]:
        """Generate definitions and explanations"""
        # Generate basic structure based on topic analysis
        words = topic.split()
        
        definitions = {
            'concepts': f"""### {topic}

{topic} refers to...

Key aspects include:
- Primary characteristic 1
- Primary characteristic 2
- Primary characteristic 3

### Related Concepts
- Related concept 1
- Related concept 2
- Related concept 3""",
            
            'details': f"""### Technical Implementation

The implementation of {topic} typically involves:

1. **Component 1**: Description
2. **Component 2**: Description
3. **Component 3**: Description

### How It Works

[Detailed explanation to be added]""",
            
            'applications': f"""### Real-World Applications

{topic} is commonly used in:

1. **Field 1**: Application description
2. **Field 2**: Application description
3. **Field 3**: Application description

### Industry Use Cases

- Use case 1
- Use case 2
- Use case 3"""
        }
        
        return definitions
    
    async def _generate_outline(self, topic: str) -> Dict[str, Any]:
        """Generate a research outline"""
        return {
            'related': f"""- Related topic 1
- Related topic 2
- Related topic 3
- See also: [[Related Note]]""",
            
            'notes': f"""## Research Notes

### Key Takeaways
- 
- 
- 

### Questions for Further Research
- 
- 
- 

### Action Items
- [ ] Research subtopic 1
- [ ] Review related paper
- [ ] Implement example"""
        }
    
    async def _llm_research(self, topic: str) -> Dict[str, Any]:
        """Use LLM for enhanced research if available"""
        # This would integrate with the LLM system if enabled
        return {}
    
    def _generate_research_note(self, research_data: Dict[str, Any], depth: str) -> str:
        """Generate the final research note"""
        template_name = 'default'
        if depth == 'technical':
            template_name = 'technical'
        elif depth == 'academic':
            template_name = 'academic'
            
        template = self.research_templates[template_name]
        
        # Prepare content sections
        content = research_data['content']
        
        # Add related vault content section
        related_section = ""
        if research_data['related_vault_content']:
            related_section = "\n## Related Notes in Vault\n\n"
            for item in research_data['related_vault_content'][:5]:
                related_section += f"- [[{item['file']}]] (relevance: {item['relevance']})\n"
                related_section += f"  > {item['snippet'][:100]}...\n\n"
        
        # Fill template
        note_content = template.format(
            topic=research_data['topic'],
            date=research_data['date'],
            overview=content.get('overview', f"Research overview for {research_data['topic']}"),
            concepts=content.get('concepts', 'Key concepts to be researched'),
            details=content.get('details', 'Technical details to be added'),
            applications=content.get('applications', 'Applications and use cases'),
            related=content.get('related', 'Related topics') + related_section,
            sources='\n'.join(f"- {source}" for source in research_data['sources']) or '- Sources to be added',
            notes=content.get('notes', 'Additional notes'),
            # Additional fields for specific templates
            introduction=content.get('introduction', ''),
            literature=content.get('literature', ''),
            methodology=content.get('methodology', ''),
            findings=content.get('findings', ''),
            discussion=content.get('discussion', ''),
            conclusion=content.get('conclusion', ''),
            code_examples=content.get('code_examples', '# Example code here'),
            best_practices=content.get('best_practices', '- Best practice 1\n- Best practice 2'),
            issues=content.get('issues', '- Common issue 1\n- Common issue 2'),
            tools=content.get('tools', '- Tool 1\n- Tool 2')
        )
        
        return note_content
    
    def _save_research_note(self, topic: str, content: str) -> str:
        """Save research note to vault"""
        # Create research folder if it doesn't exist
        research_dir = os.path.join(self.vault_path, 'Research')
        os.makedirs(research_dir, exist_ok=True)
        
        # Generate filename
        safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_topic}_Research_{timestamp}.md"
        
        # Save file
        file_path = os.path.join(research_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Research note saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving research note: {e}")
            raise