#!/usr/bin/env python3
"""
Vault Query System
LLM-style querying and aggregation of vault information
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

class VaultQuerySystem:
    """
    LLM-style query system for Obsidian vault content aggregation
    """
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.cache = {}
        self.last_scan = None
        
    def scan_vault(self, force_rescan: bool = False) -> Dict[str, Any]:
        """
        Scan vault and build searchable index
        
        Args:
            force_rescan: Force a full rescan even if cache exists
            
        Returns:
            Dictionary containing vault statistics and index
        """
        if not force_rescan and self.cache and self.last_scan:
            # Check if cache is still fresh (within 1 hour)
            import time
            if time.time() - self.last_scan < 3600:
                return self.cache
        
        print("🔍 Scanning vault for content analysis...")
        
        vault_data = {
            'files': [],
            'tags': defaultdict(list),
            'links': defaultdict(list),
            'metadata': {},
            'content_index': {},
            'statistics': {},
            'file_types': defaultdict(int),
            'creation_dates': [],
            'word_count': 0,
            'total_files': 0
        }
        
        # Scan all markdown files
        md_files = list(self.vault_path.rglob("*.md"))
        vault_data['total_files'] = len(md_files)
        
        for file_path in md_files:
            try:
                # Skip if file is too large (>10MB)
                if file_path.stat().st_size > 10 * 1024 * 1024:
                    continue
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract file info
                relative_path = file_path.relative_to(self.vault_path)
                file_info = {
                    'path': str(relative_path),
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'word_count': len(content.split()),
                    'line_count': len(content.splitlines()),
                    'char_count': len(content)
                }
                
                # Extract tags
                tags = re.findall(r'#(\w+)', content)
                file_info['tags'] = tags
                for tag in tags:
                    vault_data['tags'][tag].append(str(relative_path))
                
                # Extract internal links
                links = re.findall(r'\[\[([^\]]+)\]\]', content)
                file_info['links'] = links
                for link in links:
                    vault_data['links'][link].append(str(relative_path))
                
                # Extract metadata blocks
                metadata_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL | re.MULTILINE)
                if metadata_match:
                    file_info['has_metadata'] = True
                    # Simple metadata extraction
                    metadata_lines = metadata_match.group(1).split('\n')
                    metadata = {}
                    for line in metadata_lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
                    file_info['metadata'] = metadata
                else:
                    file_info['has_metadata'] = False
                    file_info['metadata'] = {}
                
                # Extract headings
                headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
                file_info['headings'] = [(len(h[0]), h[1]) for h in headings]
                
                # Store content for searching (first 1000 chars)
                vault_data['content_index'][str(relative_path)] = content[:1000]
                
                vault_data['files'].append(file_info)
                vault_data['word_count'] += file_info['word_count']
                vault_data['creation_dates'].append(file_info['modified'])
                
                # File type classification
                if 'daily' in file_path.name.lower() or file_path.name.startswith('2'):
                    vault_data['file_types']['daily_notes'] += 1
                elif any(keyword in content.lower() for keyword in ['meeting', 'sync', '1:1']):
                    vault_data['file_types']['meetings'] += 1
                elif any(keyword in content.lower() for keyword in ['project', 'initiative', 'epic']):
                    vault_data['file_types']['projects'] += 1
                elif any(keyword in content.lower() for keyword in ['template', 'snippet']):
                    vault_data['file_types']['templates'] += 1
                else:
                    vault_data['file_types']['general'] += 1
                    
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        # Calculate statistics
        vault_data['statistics'] = {
            'total_files': len(vault_data['files']),
            'total_words': vault_data['word_count'],
            'total_tags': len(vault_data['tags']),
            'total_links': len(vault_data['links']),
            'avg_words_per_file': vault_data['word_count'] / max(len(vault_data['files']), 1),
            'most_common_tags': Counter(tag for tags in vault_data['tags'].keys()).most_common(10),
            'file_type_distribution': dict(vault_data['file_types'])
        }
        
        # Cache results
        self.cache = vault_data
        import time
        self.last_scan = time.time()
        
        print(f"✅ Scanned {vault_data['total_files']} files, {vault_data['word_count']:,} words")
        
        return vault_data
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process natural language query against vault content
        
        Args:
            query_text: Natural language query
            
        Returns:
            Query results with relevant files and aggregated information
        """
        vault_data = self.scan_vault()
        
        # Normalize query
        query_lower = query_text.lower()
        query_words = query_lower.split()
        
        results = {
            'query': query_text,
            'matching_files': [],
            'aggregated_info': {},
            'statistics': {},
            'recommendations': []
        }
        
        # Different query types
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            results.update(self._handle_count_query(query_lower, vault_data))
            
        elif any(word in query_lower for word in ['what', 'show me', 'find', 'search']):
            results.update(self._handle_search_query(query_words, vault_data))
            
        elif any(word in query_lower for word in ['when', 'recent', 'latest', 'oldest']):
            results.update(self._handle_time_query(query_lower, vault_data))
            
        elif any(word in query_lower for word in ['who', 'people', 'person', 'team']):
            results.update(self._handle_people_query(query_words, vault_data))
            
        elif any(word in query_lower for word in ['project', 'initiative', 'task']):
            results.update(self._handle_project_query(query_words, vault_data))
            
        else:
            # General content search
            results.update(self._handle_general_search(query_words, vault_data))
        
        return results
    
    def _handle_count_query(self, query: str, vault_data: Dict) -> Dict:
        """Handle counting queries"""
        results = {'query_type': 'count'}
        
        if 'files' in query or 'notes' in query:
            results['count'] = vault_data['statistics']['total_files']
            results['summary'] = f"Your vault contains {results['count']} files"
            
        elif 'words' in query:
            results['count'] = vault_data['statistics']['total_words']
            results['summary'] = f"Your vault contains {results['count']:,} words"
            
        elif 'tags' in query:
            results['count'] = vault_data['statistics']['total_tags']
            results['summary'] = f"Your vault contains {results['count']} unique tags"
            results['top_tags'] = list(vault_data['tags'].keys())[:10]
            
        elif 'links' in query:
            results['count'] = vault_data['statistics']['total_links']
            results['summary'] = f"Your vault contains {results['count']} internal links"
            
        elif 'meetings' in query:
            meeting_count = vault_data['statistics']['file_type_distribution'].get('meetings', 0)
            results['count'] = meeting_count
            results['summary'] = f"Your vault contains {meeting_count} meeting notes"
            
        return results
    
    def _handle_search_query(self, query_words: List[str], vault_data: Dict) -> Dict:
        """Handle search queries"""
        results = {'query_type': 'search', 'matching_files': []}
        
        search_terms = [word for word in query_words if len(word) > 2]
        
        for file_info in vault_data['files']:
            file_path = file_info['path']
            content = vault_data['content_index'].get(file_path, '')
            
            # Calculate relevance score
            score = 0
            matched_terms = []
            
            for term in search_terms:
                # Check file name
                if term in file_info['name'].lower():
                    score += 3
                    matched_terms.append(f"filename: {term}")
                
                # Check content
                if term in content.lower():
                    score += content.lower().count(term)
                    matched_terms.append(f"content: {term}")
                
                # Check tags
                if any(term in tag.lower() for tag in file_info['tags']):
                    score += 2
                    matched_terms.append(f"tag: {term}")
                
                # Check headings
                for level, heading in file_info['headings']:
                    if term in heading.lower():
                        score += 2
                        matched_terms.append(f"heading: {term}")
            
            if score > 0:
                results['matching_files'].append({
                    'path': file_path,
                    'name': file_info['name'],
                    'score': score,
                    'matched_terms': matched_terms,
                    'word_count': file_info['word_count'],
                    'modified': file_info['modified'].strftime('%Y-%m-%d')
                })
        
        # Sort by relevance score
        results['matching_files'].sort(key=lambda x: x['score'], reverse=True)
        results['summary'] = f"Found {len(results['matching_files'])} files matching your search"
        
        return results
    
    def _handle_time_query(self, query: str, vault_data: Dict) -> Dict:
        """Handle time-based queries"""
        results = {'query_type': 'time'}
        
        files_with_dates = [(f, f['modified']) for f in vault_data['files']]
        files_with_dates.sort(key=lambda x: x[1], reverse=True)
        
        if 'recent' in query or 'latest' in query:
            recent_files = files_with_dates[:10]
            results['files'] = [
                {
                    'path': f[0]['path'],
                    'name': f[0]['name'],
                    'modified': f[1].strftime('%Y-%m-%d %H:%M'),
                    'word_count': f[0]['word_count']
                }
                for f in recent_files
            ]
            results['summary'] = "10 most recently modified files"
            
        elif 'oldest' in query:
            old_files = files_with_dates[-10:]
            old_files.reverse()
            results['files'] = [
                {
                    'path': f[0]['path'],
                    'name': f[0]['name'],
                    'modified': f[1].strftime('%Y-%m-%d %H:%M'),
                    'word_count': f[0]['word_count']
                }
                for f in old_files
            ]
            results['summary'] = "10 oldest files in vault"
        
        return results
    
    def _handle_people_query(self, query_words: List[str], vault_data: Dict) -> Dict:
        """Handle people/team queries"""
        results = {'query_type': 'people', 'people_mentioned': defaultdict(list)}
        
        # Common name patterns
        name_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',  # First Last
            r'\b([A-Z][a-z]+)\b(?=\s+said|spoke|mentioned|reported)',  # Names before actions
        ]
        
        for file_info in vault_data['files']:
            content = vault_data['content_index'].get(file_info['path'], '')
            
            for pattern in name_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) > 3:  # Filter out short matches
                        results['people_mentioned'][match].append({
                            'file': file_info['name'],
                            'path': file_info['path']
                        })
        
        # Sort by frequency
        sorted_people = sorted(results['people_mentioned'].items(), 
                             key=lambda x: len(x[1]), reverse=True)
        
        results['top_people'] = sorted_people[:10]
        results['summary'] = f"Found {len(results['people_mentioned'])} people mentioned across files"
        
        return results
    
    def _handle_project_query(self, query_words: List[str], vault_data: Dict) -> Dict:
        """Handle project/initiative queries"""
        results = {'query_type': 'projects', 'projects': []}
        
        project_indicators = ['project', 'initiative', 'epic', 'feature', 'milestone']
        
        for file_info in vault_data['files']:
            content = vault_data['content_index'].get(file_info['path'], '')
            
            # Check if file seems project-related
            is_project = False
            
            # Check for project indicators in filename or content
            if any(indicator in file_info['name'].lower() for indicator in project_indicators):
                is_project = True
            elif any(indicator in content.lower() for indicator in project_indicators):
                is_project = True
            
            # Check for project-like tags
            project_tags = [tag for tag in file_info['tags'] 
                          if any(indicator in tag.lower() for indicator in project_indicators)]
            if project_tags:
                is_project = True
            
            if is_project:
                # Extract status information
                status = "Unknown"
                if any(word in content.lower() for word in ['completed', 'done', 'finished']):
                    status = "Completed"
                elif any(word in content.lower() for word in ['in progress', 'ongoing', 'active']):
                    status = "In Progress"
                elif any(word in content.lower() for word in ['planned', 'upcoming', 'future']):
                    status = "Planned"
                elif any(word in content.lower() for word in ['blocked', 'delayed', 'stalled']):
                    status = "Blocked"
                
                results['projects'].append({
                    'name': file_info['name'],
                    'path': file_info['path'],
                    'status': status,
                    'tags': file_info['tags'],
                    'word_count': file_info['word_count'],
                    'modified': file_info['modified'].strftime('%Y-%m-%d')
                })
        
        results['summary'] = f"Found {len(results['projects'])} project-related files"
        
        return results
    
    def _handle_general_search(self, query_words: List[str], vault_data: Dict) -> Dict:
        """Handle general content search"""
        return self._handle_search_query(query_words, vault_data)
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive vault summary report"""
        vault_data = self.scan_vault()
        
        report = []
        report.append("# 📊 Vault Analysis Report")
        report.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        
        # Overview
        stats = vault_data['statistics']
        report.append("## 📈 Overview")
        report.append(f"- **Total Files:** {stats['total_files']:,}")
        report.append(f"- **Total Words:** {stats['total_words']:,}")
        report.append(f"- **Average Words per File:** {stats['avg_words_per_file']:.0f}")
        report.append(f"- **Unique Tags:** {stats['total_tags']}")
        report.append(f"- **Internal Links:** {stats['total_links']}\n")
        
        # File Types
        report.append("## 📁 File Types")
        for file_type, count in stats['file_type_distribution'].items():
            percentage = (count / stats['total_files']) * 100
            report.append(f"- **{file_type.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)")
        report.append("")
        
        # Top Tags
        if vault_data['tags']:
            report.append("## 🏷️ Most Used Tags")
            sorted_tags = sorted(vault_data['tags'].items(), key=lambda x: len(x[1]), reverse=True)
            for tag, files in sorted_tags[:10]:
                report.append(f"- **#{tag}:** {len(files)} files")
            report.append("")
        
        # Recent Activity
        recent_files = sorted(vault_data['files'], key=lambda x: x['modified'], reverse=True)[:5]
        report.append("## 🕒 Recent Activity")
        for file_info in recent_files:
            report.append(f"- **{file_info['name']}** - {file_info['modified'].strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        vault_path = sys.argv[1]
    else:
        vault_path = "/Users/nvaldez/Documents/repos/Obsidian"  # Default path
    
    query_system = VaultQuerySystem(vault_path)
    
    # Test queries
    test_queries = [
        "How many files are in my vault?",
        "Show me recent meeting notes",
        "Find files about projects",
        "What are my most used tags?",
        "Show me files about payment protection"
    ]
    
    print("🔍 Testing Vault Query System")
    print("=" * 40)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = query_system.query(query)
        print(f"Type: {results.get('query_type', 'general')}")
        print(f"Summary: {results.get('summary', 'No summary available')}")
        
        if 'count' in results:
            print(f"Count: {results['count']}")
        
        if 'matching_files' in results and results['matching_files']:
            print(f"Top matches: {len(results['matching_files'][:3])}")
            for match in results['matching_files'][:3]:
                print(f"  - {match['name']} (score: {match['score']})")
    
    print("\n📋 Generating summary report...")
    summary = query_system.generate_summary_report()
    print(summary[:500] + "..." if len(summary) > 500 else summary)