"""
Vault organization tools - Tag fixing and file organization
"""

import os
import re
import shutil
from typing import List, Dict, Any
from pathlib import Path

class TagFixer:
    """Fixes and standardizes tags in vault files"""
    
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        
    def find_duplicate_tags(self) -> Dict[str, List[str]]:
        """Find potential duplicate tags (case variations, etc.)"""
        all_tags = {}
        
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    tags = self._extract_tags(file_path)
                    
                    for tag in tags:
                        lower_tag = tag.lower()
                        if lower_tag not in all_tags:
                            all_tags[lower_tag] = []
                        if tag not in all_tags[lower_tag]:
                            all_tags[lower_tag].append(tag)
        
        # Return only tags with variations
        return {k: v for k, v in all_tags.items() if len(v) > 1}
    
    def _extract_tags(self, file_path: str) -> List[str]:
        """Extract tags from a markdown file"""
        tags = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find hashtag-style tags
            hashtag_pattern = r'(?:^|\s)#([a-zA-Z0-9_-]+)'
            tags.extend(re.findall(hashtag_pattern, content))
                    
        except Exception:
            pass
            
        return tags
    
    def standardize_tags(self, tag_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Standardize tags based on provided mapping"""
        files_updated = 0
        total_replacements = 0
        
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    updated = self._update_file_tags(file_path, tag_mapping)
                    if updated:
                        files_updated += 1
                        total_replacements += updated
        
        return {
            "files_updated": files_updated,
            "total_replacements": total_replacements,
            "tag_mapping": tag_mapping
        }
    
    def _update_file_tags(self, file_path: str, tag_mapping: Dict[str, str]) -> int:
        """Update tags in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            replacements = 0
            
            for old_tag, new_tag in tag_mapping.items():
                # Replace hashtag-style tags
                pattern = r'(?:^|\s)#' + re.escape(old_tag) + r'(?=\s|$)'
                if re.search(pattern, content):
                    content = re.sub(pattern, f' #{new_tag}', content)
                    replacements += 1
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return replacements
            
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
        
        return 0


class FileOrganizer:
    """Organizes files within the vault"""
    
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        
    def find_orphaned_files(self) -> List[str]:
        """Find files that aren't linked to from anywhere"""
        all_files = set()
        linked_files = set()
        
        # Get all markdown files
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    all_files.add(file_path)
                    
                    # Find links in this file
                    linked_files.update(self._extract_links(file_path))
        
        # Convert relative links to absolute paths
        absolute_linked = set()
        for link in linked_files:
            abs_path = os.path.join(self.vault_path, link)
            if os.path.exists(abs_path):
                absolute_linked.add(abs_path)
        
        return list(all_files - absolute_linked)
    
    def _extract_links(self, file_path: str) -> List[str]:
        """Extract Obsidian-style links from a file"""
        links = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find [[wiki-style]] links
            wiki_pattern = r'\[\[([^\]]+)\]\]'
            matches = re.findall(wiki_pattern, content)
            
            for match in matches:
                # Handle links with aliases (link|alias)
                link_part = match.split('|')[0]
                if not link_part.endswith('.md'):
                    link_part += '.md'
                links.append(link_part)
                
        except Exception:
            pass
        
        return links
    
    def suggest_folder_structure(self) -> Dict[str, Any]:
        """Suggest folder organization based on tags and content"""
        file_analysis = {}
        
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.vault_path)
                    
                    analysis = {
                        "current_folder": os.path.dirname(relative_path),
                        "tags": self._extract_tags_simple(file_path),
                        "size": os.path.getsize(file_path),
                        "last_modified": os.path.getmtime(file_path)
                    }
                    
                    file_analysis[relative_path] = analysis
        
        # Generate suggestions based on tags
        suggestions = {}
        for file_path, analysis in file_analysis.items():
            if analysis["tags"]:
                primary_tag = analysis["tags"][0]  # Use first tag as primary
                suggested_folder = f"by-tag/{primary_tag}"
                
                if suggested_folder != analysis["current_folder"]:
                    suggestions[file_path] = {
                        "current": analysis["current_folder"],
                        "suggested": suggested_folder,
                        "reason": f"Based on primary tag: #{primary_tag}"
                    }
        
        return {
            "total_files": len(file_analysis),
            "suggestions": suggestions,
            "summary": f"Found {len(suggestions)} reorganization suggestions"
        }
    
    def _extract_tags_simple(self, file_path: str) -> List[str]:
        """Simple tag extraction"""
        tags = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            hashtag_pattern = r'(?:^|\s)#([a-zA-Z0-9_-]+)'
            tags = re.findall(hashtag_pattern, content)
                    
        except Exception:
            pass
            
        return tags