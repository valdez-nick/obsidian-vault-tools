"""
Task Extractor - Extract and analyze incomplete tasks from Obsidian vault
Designed specifically for Nick's 435+ file PM workflow
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import logging

from ..security import validate_path, sanitize_filename

logger = logging.getLogger(__name__)


class Task:
    """Represents a single task with metadata"""
    
    def __init__(self, content: str, file_path: str, line_number: int, 
                 context: str = "", date_found: datetime = None):
        self.content = content.strip()
        self.file_path = file_path
        self.line_number = line_number
        self.context = context
        self.date_found = date_found or datetime.now()
        self.product_area = self._detect_product_area()
        self.task_type = self._classify_task_type()
        self.estimated_effort = self._estimate_effort()
        
    def _detect_product_area(self) -> str:
        """Detect which product area this task belongs to"""
        content_lower = self.content.lower()
        file_lower = self.file_path.lower()
        context_lower = self.context.lower()
        
        # Combined text for analysis
        all_text = f"{content_lower} {file_lower} {context_lower}"
        
        # DFP 2.0 / Device Fingerprinting keywords
        dfp_keywords = [
            'dfp', 'device fingerprinting', 'fingerprint', 'device id',
            'browser fingerprint', 'canvas fingerprint', 'webgl',
            'audio fingerprint', 'screen resolution', 'timezone',
            'device detection', 'bot detection'
        ]
        
        # Payment Protection keywords  
        payment_keywords = [
            'payment', 'transaction', 'fraud', 'chargeback', 'merchant',
            'payment protection', 'transaction monitoring', 'risk score',
            'payment fraud', 'checkout', 'billing', 'payment method'
        ]
        
        # Global Identity Intelligence keywords
        identity_keywords = [
            'identity', 'global identity', 'user identity', 'identity graph',
            'identity resolution', 'identity matching', 'cross-device',
            'user tracking', 'identity intelligence', 'coverage'
        ]
        
        # Score each area
        dfp_score = sum(1 for keyword in dfp_keywords if keyword in all_text)
        payment_score = sum(1 for keyword in payment_keywords if keyword in all_text)
        identity_score = sum(1 for keyword in identity_keywords if keyword in all_text)
        
        # Return highest scoring area
        if dfp_score >= payment_score and dfp_score >= identity_score:
            return "DFP 2.0"
        elif payment_score >= identity_score:
            return "Payment Protection"
        elif identity_score > 0:
            return "Global Identity Intelligence"
        else:
            return "General"
    
    def _classify_task_type(self) -> str:
        """Classify the type of task"""
        content_lower = self.content.lower()
        
        if any(word in content_lower for word in ['meet', 'call', 'sync', 'discuss']):
            return "Meeting/Communication"
        elif any(word in content_lower for word in ['review', 'analyze', 'research', 'investigate']):
            return "Analysis/Research"
        elif any(word in content_lower for word in ['write', 'document', 'draft', 'create']):
            return "Documentation"
        elif any(word in content_lower for word in ['plan', 'strategy', 'roadmap', 'okr']):
            return "Planning/Strategy"
        elif any(word in content_lower for word in ['follow up', 'respond', 'reply', 'email']):
            return "Follow-up/Communication"
        elif any(word in content_lower for word in ['implement', 'build', 'develop', 'code']):
            return "Implementation"
        else:
            return "Other"
    
    def _estimate_effort(self) -> str:
        """Estimate effort level based on task content"""
        content_lower = self.content.lower()
        
        # Quick wins (< 30 min)
        quick_indicators = ['quick', 'brief', 'short', 'email', 'reply', 'check', 'update']
        if any(word in content_lower for word in quick_indicators) or len(self.content) < 50:
            return "Quick (<30min)"
        
        # Deep work (> 2 hours)
        deep_indicators = ['strategy', 'analysis', 'research', 'document', 'plan', 'design', 'architecture']
        if any(word in content_lower for word in deep_indicators):
            return "Deep (>2hrs)"
        
        # Default to medium
        return "Medium (1-2hrs)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            'content': self.content,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'context': self.context,
            'date_found': self.date_found.isoformat(),
            'product_area': self.product_area,
            'task_type': self.task_type,
            'estimated_effort': self.estimated_effort
        }


class TaskExtractor:
    """Extract and analyze incomplete tasks from Obsidian vault"""
    
    def __init__(self, vault_path: str):
        self.vault_path = validate_path(vault_path)
        self.tasks = []
        self.task_patterns = [
            r'^[\s]*-[\s]*\[[\s]*\][\s]*(.+)$',  # - [ ] task
            r'^[\s]*\*[\s]*\[[\s]*\][\s]*(.+)$',  # * [ ] task  
            r'^[\s]*\d+\.[\s]*\[[\s]*\][\s]*(.+)$',  # 1. [ ] task
        ]
        
    def extract_all_tasks(self, file_extensions: List[str] = None) -> List[Task]:
        """Extract all incomplete tasks from vault"""
        if file_extensions is None:
            file_extensions = ['.md', '.txt']
        
        logger.info(f"Starting task extraction from {self.vault_path}")
        self.tasks = []
        
        # Walk through all files in vault
        for root, dirs, files in os.walk(self.vault_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        self._extract_tasks_from_file(file_path)
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Extracted {len(self.tasks)} incomplete tasks")
        return self.tasks
    
    def _extract_tasks_from_file(self, file_path: str) -> None:
        """Extract tasks from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Check each pattern for incomplete tasks
                for pattern in self.task_patterns:
                    match = re.match(pattern, line)
                    if match:
                        task_content = match.group(1).strip()
                        if task_content:  # Only add non-empty tasks
                            # Get surrounding context (2 lines before, 2 after)
                            context_lines = []
                            for i in range(max(0, line_num-3), min(len(lines), line_num+2)):
                                if i != line_num-1:  # Don't include the task line itself
                                    context_lines.append(lines[i].strip())
                            context = ' '.join(context_lines)
                            
                            task = Task(
                                content=task_content,
                                file_path=file_path,
                                line_number=line_num,
                                context=context
                            )
                            self.tasks.append(task)
                        break  # Only match first pattern per line
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    def get_tasks_by_product_area(self) -> Dict[str, List[Task]]:
        """Group tasks by product area"""
        tasks_by_area = defaultdict(list)
        for task in self.tasks:
            tasks_by_area[task.product_area].append(task)
        return dict(tasks_by_area)
    
    def get_tasks_by_type(self) -> Dict[str, List[Task]]:
        """Group tasks by type"""
        tasks_by_type = defaultdict(list)
        for task in self.tasks:
            tasks_by_type[task.task_type].append(task)
        return dict(tasks_by_type)
    
    def get_tasks_by_effort(self) -> Dict[str, List[Task]]:
        """Group tasks by estimated effort"""
        tasks_by_effort = defaultdict(list)
        for task in self.tasks:
            tasks_by_effort[task.estimated_effort].append(task)
        return dict(tasks_by_effort)
    
    def analyze_task_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in task data"""
        if not self.tasks:
            return {}
        
        # File distribution
        file_counts = Counter(task.file_path for task in self.tasks)
        
        # Product area distribution
        area_counts = Counter(task.product_area for task in self.tasks)
        
        # Task type distribution  
        type_counts = Counter(task.task_type for task in self.tasks)
        
        # Effort distribution
        effort_counts = Counter(task.estimated_effort for task in self.tasks)
        
        # Files with most tasks (potential overwhelm indicators)
        top_files = file_counts.most_common(10)
        
        return {
            'total_tasks': len(self.tasks),
            'files_with_tasks': len(file_counts),
            'avg_tasks_per_file': len(self.tasks) / len(file_counts) if file_counts else 0,
            'product_area_distribution': dict(area_counts),
            'task_type_distribution': dict(type_counts),
            'effort_distribution': dict(effort_counts),
            'top_files_by_task_count': top_files,
            'potential_overwhelm_files': [f for f, count in top_files if count > 10]
        }
    
    def detect_moved_tasks(self, days_to_check: int = 30) -> List[Dict[str, Any]]:
        """Detect tasks that appear to be moved between files"""
        # This is a simplified implementation - in practice you'd want to 
        # track tasks over time to detect actual movement patterns
        task_content_counts = Counter(task.content for task in self.tasks)
        potential_duplicates = [
            {
                'content': content,
                'count': count,
                'files': [task.file_path for task in self.tasks if task.content == content]
            }
            for content, count in task_content_counts.items() 
            if count > 1
        ]
        
        return potential_duplicates