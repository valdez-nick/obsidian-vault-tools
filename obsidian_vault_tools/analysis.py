"""
Vault analysis tools - Tag analysis and vault health checking
"""

import os
import re
from typing import Dict, List, Any
from pathlib import Path

class TagAnalyzer:
    """Analyzes tags in vault files"""
    
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        
    def extract_tags(self, file_path: str) -> List[str]:
        """Extract tags from a markdown file"""
        tags = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find hashtag-style tags
            hashtag_pattern = r'(?:^|\s)#([a-zA-Z0-9_-]+)'
            tags.extend(re.findall(hashtag_pattern, content))
            
            # Find YAML frontmatter tags
            yaml_match = re.search(r'^---\s*\n(.*?)\n---', content, re.MULTILINE | re.DOTALL)
            if yaml_match:
                yaml_content = yaml_match.group(1)
                tag_match = re.search(r'tags:\s*\[(.*?)\]', yaml_content)
                if tag_match:
                    yaml_tags = [tag.strip().strip('"\'') for tag in tag_match.group(1).split(',')]
                    tags.extend(yaml_tags)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return list(set(tags))  # Remove duplicates
    
    def analyze_all_tags(self) -> Dict[str, Any]:
        """Analyze all tags in the vault"""
        tag_counts = {}
        file_count = 0
        
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    tags = self.extract_tags(file_path)
                    file_count += 1
                    
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_files": file_count,
            "total_tags": len(tag_counts),
            "tag_counts": tag_counts,
            "most_used_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }


class VaultAnalyzer:
    """Comprehensive vault analysis"""
    
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.tag_analyzer = TagAnalyzer(vault_path)
        
    def analyze_vault_structure(self) -> Dict[str, Any]:
        """Analyze overall vault structure"""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "file_types": {},
            "largest_files": [],
            "empty_files": []
        }
        
        for root, dirs, files in os.walk(self.vault_path):
            structure["total_directories"] += len(dirs)
            
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                structure["total_files"] += 1
                
                # Track file types
                ext = os.path.splitext(file)[1]
                structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
                
                # Track large files
                if file.endswith('.md') and file_size > 10000:  # Files larger than 10KB
                    structure["largest_files"].append({
                        "path": file_path,
                        "size": file_size
                    })
                
                # Track empty files
                if file_size == 0:
                    structure["empty_files"].append(file_path)
        
        # Sort largest files by size
        structure["largest_files"].sort(key=lambda x: x["size"], reverse=True)
        structure["largest_files"] = structure["largest_files"][:10]  # Top 10
        
        return structure
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        tag_analysis = self.tag_analyzer.analyze_all_tags()
        structure_analysis = self.analyze_vault_structure()
        
        return {
            "timestamp": str(Path(self.vault_path).stat().st_mtime),
            "vault_path": self.vault_path,
            "tags": tag_analysis,
            "structure": structure_analysis,
            "health_score": self._calculate_health_score(tag_analysis, structure_analysis)
        }
    
    def _calculate_health_score(self, tag_analysis: Dict, structure_analysis: Dict) -> str:
        """Calculate overall health score"""
        score = 0
        
        # Points for having files
        if structure_analysis["total_files"] > 0:
            score += 20
        
        # Points for having tags
        if tag_analysis["total_tags"] > 0:
            score += 20
        
        # Points for organization (multiple directories)
        if structure_analysis["total_directories"] > 1:
            score += 20
        
        # Points for active usage (large files indicate content)
        if len(structure_analysis["largest_files"]) > 0:
            score += 20
        
        # Deduct points for too many empty files
        if len(structure_analysis["empty_files"]) > structure_analysis["total_files"] * 0.1:
            score -= 10
        
        if score >= 70:
            return "Excellent"
        elif score >= 50:
            return "Good"
        elif score >= 30:
            return "Fair"
        else:
            return "Needs Attention"