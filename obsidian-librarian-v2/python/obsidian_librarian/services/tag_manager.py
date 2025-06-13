"""
Tag Management Core Service for intelligent tag analysis and operations.

Provides comprehensive tag management capabilities including:
- Tag extraction and analysis
- Similarity detection using string and semantic analysis
- Hierarchy optimization and structure building
- AI-powered auto-tagging suggestions
- Tag operations and batch processing
"""

import asyncio
import re
import hashlib
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import structlog
import numpy as np
from fuzzywuzzy import fuzz, process

from ..models.models import (
    Note, 
    NoteId,
    TagInfo,
    TagSuggestion,
    TagSimilarity,
    TagCluster,
    TagHierarchy,
    TagOperation,
    TagAnalysisResult,
    TagManagerConfig,
)
from ..vault import Vault
from ..ai.embeddings import EmbeddingService
from ..ai.content_analyzer import ContentAnalyzer
from .analysis import AnalysisService

logger = structlog.get_logger(__name__)


class TagAnalyzer:
    """Core tag analysis functionality."""
    
    def __init__(self, config: TagManagerConfig):
        self.config = config
        self._tag_cache: Dict[str, TagInfo] = {}
        self._pattern_cache: Dict[str, re.Pattern] = {}
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for tag extraction."""
        self._pattern_cache = {
            'frontmatter_tags': re.compile(r'^tags:\s*\[(.*?)\]', re.MULTILINE | re.IGNORECASE),
            'frontmatter_tags_list': re.compile(r'^tags:\s*\n((?:\s*-\s*.+\n?)+)', re.MULTILINE | re.IGNORECASE),
            'inline_tags': re.compile(r'(?:^|\s)#([a-zA-Z0-9/_-]+)(?:\s|$)', re.MULTILINE),
            'special_chars': re.compile(r'[^\w\s/-]'),
            'hierarchy_separator': re.compile(r'[/\\]'),
        }
    
    async def extract_tags_from_note(self, note: Note) -> List[str]:
        """Extract all tags from a note using configured methods."""
        all_tags = set()
        
        for method in self.config.tag_extraction_methods:
            if method == "frontmatter":
                tags = self._extract_frontmatter_tags(note.content)
                all_tags.update(tags)
            elif method == "inline":
                tags = self._extract_inline_tags(note.content)
                all_tags.update(tags)
            elif method == "content_based":
                tags = await self._extract_content_based_tags(note)
                all_tags.update(tags)
        
        return list(all_tags)
    
    def _extract_frontmatter_tags(self, content: str) -> List[str]:
        """Extract tags from YAML frontmatter."""
        tags = []
        
        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not frontmatter_match:
            return tags
        
        frontmatter = frontmatter_match.group(1)
        
        # Method 1: Array format - tags: [tag1, tag2]
        array_match = self._pattern_cache['frontmatter_tags'].search(frontmatter)
        if array_match:
            tag_string = array_match.group(1)
            # Parse comma-separated tags
            raw_tags = [tag.strip().strip('"\'') for tag in tag_string.split(',')]
            tags.extend([tag for tag in raw_tags if tag])
        
        # Method 2: List format - tags:\n  - tag1\n  - tag2
        list_match = self._pattern_cache['frontmatter_tags_list'].search(frontmatter)
        if list_match:
            tag_lines = list_match.group(1)
            list_tags = re.findall(r'^\s*-\s*(.+)$', tag_lines, re.MULTILINE)
            tags.extend([tag.strip().strip('"\'') for tag in list_tags])
        
        return self._normalize_tags(tags)
    
    def _extract_inline_tags(self, content: str) -> List[str]:
        """Extract inline #tags from content."""
        matches = self._pattern_cache['inline_tags'].findall(content)
        return self._normalize_tags(matches)
    
    async def _extract_content_based_tags(self, note: Note) -> List[str]:
        """Extract tags based on content analysis."""
        # This would use AI/ML to suggest tags based on content
        # For now, return empty list - will be implemented in AutoTagger
        return []
    
    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tag names according to configuration."""
        normalized = []
        
        for tag in tags:
            if not tag:
                continue
            
            # Convert to lowercase if configured
            if self.config.case_insensitive:
                tag = tag.lower()
            
            # Remove special characters if configured
            if self.config.normalize_special_chars:
                tag = self._pattern_cache['special_chars'].sub('', tag)
            
            # Clean up whitespace
            tag = tag.strip()
            
            if tag:
                normalized.append(tag)
        
        return normalized
    
    async def analyze_tag_usage(self, notes: List[Note]) -> Dict[str, TagInfo]:
        """Analyze tag usage patterns across notes."""
        tag_info = {}
        
        for note in notes:
            tags = await self.extract_tags_from_note(note)
            note_timestamp = note.modified_at
            
            for tag in tags:
                if tag not in tag_info:
                    tag_info[tag] = TagInfo(
                        name=tag,
                        normalized_name=self._normalize_tag_name(tag),
                        usage_count=0,
                        first_seen=note_timestamp,
                        last_used=note_timestamp,
                        notes=set()
                    )
                
                info = tag_info[tag]
                info.usage_count += 1
                info.notes.add(note.id)
                info.last_used = max(info.last_used, note_timestamp)
                info.first_seen = min(info.first_seen, note_timestamp)
        
        # Analyze hierarchies
        for tag_name, info in tag_info.items():
            info.parent_tags, info.child_tags, info.hierarchy_level = self._analyze_tag_hierarchy(
                tag_name, tag_info
            )
        
        return tag_info
    
    def _normalize_tag_name(self, tag: str) -> str:
        """Create a normalized version of tag name for comparison."""
        normalized = tag.lower()
        normalized = self._pattern_cache['special_chars'].sub('', normalized)
        normalized = re.sub(r'\s+', '-', normalized)
        return normalized
    
    def _analyze_tag_hierarchy(self, tag: str, all_tags: Dict[str, TagInfo]) -> Tuple[List[str], List[str], int]:
        """Analyze hierarchy relationships for a tag."""
        parents = []
        children = []
        level = 0
        
        # Check for hierarchical patterns (e.g., category/subcategory)
        if '/' in tag:
            parts = tag.split('/')
            level = len(parts) - 1
            
            # Find parent tags
            for i in range(len(parts) - 1):
                parent_tag = '/'.join(parts[:i+1])
                if parent_tag in all_tags:
                    parents.append(parent_tag)
        
        # Find child tags
        for other_tag in all_tags:
            if other_tag.startswith(tag + '/') and other_tag != tag:
                # Direct child only
                remaining = other_tag[len(tag) + 1:]
                if '/' not in remaining:
                    children.append(other_tag)
        
        return parents, children, level


class TagSimilarityDetector:
    """Detects similar tags using string similarity and semantic analysis."""
    
    def __init__(self, config: TagManagerConfig, embedding_service: Optional[EmbeddingService] = None):
        self.config = config
        self.embedding_service = embedding_service
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
    
    async def find_similar_tags(self, tags: List[str]) -> List[TagSimilarity]:
        """Find all similar tag pairs."""
        similarities = []
        
        for i, tag_a in enumerate(tags):
            for tag_b in tags[i+1:]:
                similarity = await self.calculate_similarity(tag_a, tag_b)
                if similarity:
                    similarities.append(similarity)
        
        return sorted(similarities, key=lambda x: x.similarity_score, reverse=True)
    
    async def calculate_similarity(self, tag_a: str, tag_b: str) -> Optional[TagSimilarity]:
        """Calculate similarity between two tags."""
        # Check cache first
        cache_key = tuple(sorted([tag_a, tag_b]))
        if cache_key in self._similarity_cache:
            score = self._similarity_cache[cache_key]
            return TagSimilarity(tag_a, tag_b, score, "cached")
        
        best_similarity = None
        best_score = 0.0
        
        # String similarity checks
        if self.config.enable_fuzzy_matching:
            fuzzy_score = self._calculate_fuzzy_similarity(tag_a, tag_b)
            if fuzzy_score > best_score:
                best_score = fuzzy_score
                best_similarity = TagSimilarity(tag_a, tag_b, fuzzy_score, "fuzzy")
        
        # Semantic similarity (if embedding service available)
        if self.config.enable_semantic_analysis and self.embedding_service:
            semantic_score = await self._calculate_semantic_similarity(tag_a, tag_b)
            if semantic_score > best_score:
                best_score = semantic_score
                best_similarity = TagSimilarity(tag_a, tag_b, semantic_score, "semantic")
        
        # Cache result
        self._similarity_cache[cache_key] = best_score
        
        # Return if above threshold
        if best_score >= self.config.fuzzy_similarity_threshold:
            return best_similarity
        
        return None
    
    def _calculate_fuzzy_similarity(self, tag_a: str, tag_b: str) -> float:
        """Calculate fuzzy string similarity."""
        # Normalize for comparison
        norm_a = tag_a.lower() if self.config.case_insensitive else tag_a
        norm_b = tag_b.lower() if self.config.case_insensitive else tag_b
        
        # Multiple fuzzy matching approaches
        scores = [
            fuzz.ratio(norm_a, norm_b) / 100.0,
            fuzz.partial_ratio(norm_a, norm_b) / 100.0,
            fuzz.token_sort_ratio(norm_a, norm_b) / 100.0,
            fuzz.token_set_ratio(norm_a, norm_b) / 100.0,
        ]
        
        # Use the highest score
        return max(scores)
    
    async def _calculate_semantic_similarity(self, tag_a: str, tag_b: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            # Generate embeddings for both tags
            embedding_a = await self.embedding_service.embed_text(tag_a)
            embedding_b = await self.embedding_service.embed_text(tag_b)
            
            # Calculate cosine similarity
            similarity = await self.embedding_service.compute_similarity(
                embedding_a, embedding_b, metric="cosine"
            )
            
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.warning("Semantic similarity calculation failed", 
                         tag_a=tag_a, tag_b=tag_b, error=str(e))
            return 0.0
    
    async def cluster_similar_tags(self, tags: List[str]) -> List[TagCluster]:
        """Cluster tags based on similarity."""
        similarities = await self.find_similar_tags(tags)
        
        # Build similarity graph
        graph = defaultdict(set)
        for sim in similarities:
            if sim.similarity_score >= self.config.fuzzy_similarity_threshold:
                graph[sim.tag_a].add(sim.tag_b)
                graph[sim.tag_b].add(sim.tag_a)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        cluster_id = 0
        
        for tag in tags:
            if tag not in visited:
                cluster_tags = self._dfs_cluster(tag, graph, visited)
                if len(cluster_tags) > 1:  # Only include multi-tag clusters
                    # Find representative tag (most commonly used)
                    representative = max(cluster_tags, key=lambda t: len(graph[t]))
                    
                    cluster = TagCluster(
                        cluster_id=f"cluster_{cluster_id}",
                        tags=cluster_tags,
                        cluster_type="similar",
                        confidence=0.8,  # TODO: Calculate based on similarities
                        representative_tag=representative,
                        suggested_merge=len(cluster_tags) <= 3
                    )
                    clusters.append(cluster)
                    cluster_id += 1
        
        return clusters
    
    def _dfs_cluster(self, start_tag: str, graph: Dict[str, Set[str]], visited: Set[str]) -> List[str]:
        """Depth-first search to find connected component."""
        cluster = []
        stack = [start_tag]
        
        while stack:
            tag = stack.pop()
            if tag not in visited:
                visited.add(tag)
                cluster.append(tag)
                stack.extend(graph[tag] - visited)
        
        return cluster


class TagHierarchyBuilder:
    """Builds and optimizes tag hierarchies."""
    
    def __init__(self, config: TagManagerConfig):
        self.config = config
    
    async def build_hierarchies(self, tag_info: Dict[str, TagInfo]) -> List[TagHierarchy]:
        """Build tag hierarchies from usage patterns."""
        hierarchies = []
        processed = set()
        
        # Find root tags (tags without parents)
        root_tags = [
            tag for tag, info in tag_info.items()
            if not info.parent_tags and tag not in processed
        ]
        
        for root_tag in root_tags:
            hierarchy = await self._build_hierarchy_tree(root_tag, tag_info, processed)
            if hierarchy:
                hierarchies.append(hierarchy)
        
        return hierarchies
    
    async def _build_hierarchy_tree(
        self, 
        root_tag: str, 
        tag_info: Dict[str, TagInfo], 
        processed: Set[str]
    ) -> Optional[TagHierarchy]:
        """Build a hierarchy tree starting from a root tag."""
        if root_tag in processed or root_tag not in tag_info:
            return None
        
        processed.add(root_tag)
        info = tag_info[root_tag]
        
        hierarchy = TagHierarchy(
            root_tag=root_tag,
            level=info.hierarchy_level,
            usage_count=info.usage_count,
            notes=info.notes.copy()
        )
        
        # Add children recursively
        for child_tag in info.child_tags:
            child_hierarchy = await self._build_hierarchy_tree(child_tag, tag_info, processed)
            if child_hierarchy:
                hierarchy.add_child(child_tag, child_hierarchy)
        
        return hierarchy
    
    async def suggest_hierarchy_optimizations(
        self, 
        hierarchies: List[TagHierarchy],
        tag_info: Dict[str, TagInfo]
    ) -> List[str]:
        """Suggest optimizations for tag hierarchies."""
        suggestions = []
        
        for hierarchy in hierarchies:
            # Check for deeply nested hierarchies
            max_depth = self._get_max_depth(hierarchy)
            if max_depth > self.config.max_hierarchy_depth:
                suggestions.append(
                    f"Hierarchy '{hierarchy.root_tag}' is too deep ({max_depth} levels). "
                    f"Consider flattening or reorganizing."
                )
            
            # Check for underused branches
            underused = self._find_underused_branches(hierarchy)
            for branch in underused:
                suggestions.append(
                    f"Tag '{branch}' in hierarchy '{hierarchy.root_tag}' has low usage. "
                    f"Consider removing or merging."
                )
            
            # Check for missing intermediate levels
            missing = self._find_missing_intermediate_levels(hierarchy, tag_info)
            for missing_level in missing:
                suggestions.append(
                    f"Consider creating intermediate tag '{missing_level}' in hierarchy '{hierarchy.root_tag}'"
                )
        
        return suggestions
    
    def _get_max_depth(self, hierarchy: TagHierarchy) -> int:
        """Get maximum depth of a hierarchy."""
        if not hierarchy.children:
            return hierarchy.level
        
        max_child_depth = max(
            self._get_max_depth(child) for child in hierarchy.children.values()
        )
        return max_child_depth
    
    def _find_underused_branches(self, hierarchy: TagHierarchy) -> List[str]:
        """Find branches with usage below threshold."""
        underused = []
        
        if hierarchy.usage_count < self.config.min_usage_threshold:
            underused.append(hierarchy.root_tag)
        
        for child in hierarchy.children.values():
            underused.extend(self._find_underused_branches(child))
        
        return underused
    
    def _find_missing_intermediate_levels(
        self, 
        hierarchy: TagHierarchy, 
        tag_info: Dict[str, TagInfo]
    ) -> List[str]:
        """Find potentially missing intermediate hierarchy levels."""
        missing = []
        
        # Look for common prefixes in child tags that might indicate missing levels
        if len(hierarchy.children) >= 3:
            child_names = list(hierarchy.children.keys())
            common_prefixes = self._find_common_prefixes(child_names)
            
            for prefix in common_prefixes:
                intermediate_tag = f"{hierarchy.root_tag}/{prefix}"
                if intermediate_tag not in tag_info:
                    missing.append(intermediate_tag)
        
        return missing
    
    def _find_common_prefixes(self, tags: List[str]) -> List[str]:
        """Find common prefixes in a list of tags."""
        prefixes = []
        
        for tag in tags:
            parts = tag.split('/')
            if len(parts) > 1:
                prefix = parts[-1].split('-')[0] if '-' in parts[-1] else parts[-1][:3]
                if len(prefix) >= 2:
                    prefixes.append(prefix)
        
        # Count occurrences and return frequent ones
        counter = Counter(prefixes)
        return [prefix for prefix, count in counter.items() if count >= 2]


class AutoTagger:
    """AI-powered automatic tag suggestion."""
    
    def __init__(
        self, 
        config: TagManagerConfig,
        embedding_service: Optional[EmbeddingService] = None,
        content_analyzer: Optional[ContentAnalyzer] = None
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.content_analyzer = content_analyzer
        self._tag_patterns: Dict[str, List[str]] = {}
    
    async def suggest_tags_for_note(
        self, 
        note: Note, 
        existing_tags: List[str],
        context_tags: Optional[Dict[str, TagInfo]] = None
    ) -> List[TagSuggestion]:
        """Suggest tags for a note based on its content."""
        suggestions = []
        
        # Content-based suggestions
        content_suggestions = await self._suggest_from_content(note, existing_tags)
        suggestions.extend(content_suggestions)
        
        # Pattern-based suggestions
        pattern_suggestions = await self._suggest_from_patterns(note, existing_tags)
        suggestions.extend(pattern_suggestions)
        
        # Semantic suggestions (based on similar notes)
        if self.embedding_service and context_tags:
            semantic_suggestions = await self._suggest_from_semantics(note, existing_tags, context_tags)
            suggestions.extend(semantic_suggestions)
        
        # AI-powered suggestions
        if self.config.enable_ai_suggestions and self.content_analyzer:
            ai_suggestions = await self._suggest_from_ai(note, existing_tags)
            suggestions.extend(ai_suggestions)
        
        # Filter and rank suggestions
        filtered_suggestions = self._filter_and_rank_suggestions(suggestions, existing_tags)
        
        return filtered_suggestions[:self.config.max_auto_tags_per_note]
    
    async def _suggest_from_content(self, note: Note, existing_tags: List[str]) -> List[TagSuggestion]:
        """Suggest tags based on content analysis."""
        suggestions = []
        content = note.content.lower()
        
        # Common content patterns
        patterns = {
            'meeting': ['meeting', 'agenda', 'minutes', 'discussion'],
            'project': ['project', 'milestone', 'deliverable', 'timeline'],
            'todo': ['todo', 'task', 'action item', '- [ ]'],
            'research': ['research', 'study', 'analysis', 'findings'],
            'documentation': ['documentation', 'guide', 'manual', 'instructions'],
            'review': ['review', 'feedback', 'assessment', 'evaluation'],
        }
        
        for tag, keywords in patterns.items():
            if tag not in existing_tags:
                matches = sum(1 for keyword in keywords if keyword in content)
                if matches >= 2:  # At least 2 keyword matches
                    confidence = min(0.9, matches / len(keywords))
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        confidence=confidence,
                        reason=f"Content contains {matches} related keywords",
                        source="content",
                        context=f"Keywords: {', '.join(keywords[:3])}"
                    ))
        
        return suggestions
    
    async def _suggest_from_patterns(self, note: Note, existing_tags: List[str]) -> List[TagSuggestion]:
        """Suggest tags based on filename and structure patterns."""
        suggestions = []
        
        # File path patterns
        path_str = str(note.path).lower()
        
        path_patterns = {
            'daily-note': r'(\d{4}-\d{2}-\d{2}|daily)',
            'weekly-note': r'(week|weekly)',
            'template': r'(template|templ)',
            'archive': r'(archive|archived)',
            'draft': r'(draft|wip|work-in-progress)',
        }
        
        for tag, pattern in path_patterns.items():
            if tag not in existing_tags and re.search(pattern, path_str):
                suggestions.append(TagSuggestion(
                    tag=tag,
                    confidence=0.8,
                    reason=f"Filename matches pattern: {pattern}",
                    source="pattern",
                    context=f"Path: {note.path.name}"
                ))
        
        # Content structure patterns
        content = note.content
        
        structure_patterns = {
            'checklist': (r'^\s*- \[[ x]\]', 0.7),
            'numbered-list': (r'^\s*\d+\.', 0.6),
            'code': (r'```[\s\S]*?```', 0.8),
            'table': (r'\|.*\|', 0.7),
        }
        
        for tag, (pattern, base_confidence) in structure_patterns.items():
            if tag not in existing_tags:
                matches = len(re.findall(pattern, content, re.MULTILINE))
                if matches >= 3:  # Significant presence
                    confidence = min(0.9, base_confidence + (matches / 20))
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        confidence=confidence,
                        reason=f"Contains {matches} instances of pattern",
                        source="pattern",
                        context=f"Pattern: {pattern[:20]}..."
                    ))
        
        return suggestions
    
    async def _suggest_from_semantics(
        self, 
        note: Note, 
        existing_tags: List[str],
        context_tags: Dict[str, TagInfo]
    ) -> List[TagSuggestion]:
        """Suggest tags based on semantic similarity to other notes."""
        suggestions = []
        
        try:
            # Get embedding for current note content
            note_embedding = await self.embedding_service.embed_text(note.content[:1000])
            
            # Compare with tag embeddings
            tag_similarities = []
            for tag_name, tag_info in context_tags.items():
                if tag_name not in existing_tags and tag_info.usage_count >= 2:
                    # Create representative text for the tag
                    tag_text = f"{tag_name} {' '.join(tag_info.similar_tags[:3])}"
                    tag_embedding = await self.embedding_service.embed_text(tag_text)
                    
                    similarity = await self.embedding_service.compute_similarity(
                        note_embedding, tag_embedding
                    )
                    
                    if similarity >= self.config.semantic_similarity_threshold:
                        tag_similarities.append((tag_name, similarity, tag_info))
            
            # Create suggestions from top similarities
            tag_similarities.sort(key=lambda x: x[1], reverse=True)
            
            for tag_name, similarity, tag_info in tag_similarities[:5]:
                suggestions.append(TagSuggestion(
                    tag=tag_name,
                    confidence=similarity,
                    reason=f"Semantically similar (score: {similarity:.2f})",
                    source="semantic",
                    context=f"Used in {tag_info.usage_count} notes"
                ))
        
        except Exception as e:
            logger.warning("Semantic tag suggestion failed", error=str(e))
        
        return suggestions
    
    async def _suggest_from_ai(self, note: Note, existing_tags: List[str]) -> List[TagSuggestion]:
        """Use AI to suggest tags based on content understanding."""
        suggestions = []
        
        try:
            if self.content_analyzer:
                # Use content analyzer to understand the note
                analysis = await self.content_analyzer.analyze_content(
                    note.content,
                    analysis_type="tag_suggestion"
                )
                
                # Extract suggested tags from analysis
                if "suggested_tags" in analysis:
                    for tag_data in analysis["suggested_tags"]:
                        if isinstance(tag_data, dict):
                            tag = tag_data.get("tag", "")
                            confidence = tag_data.get("confidence", 0.5)
                            reason = tag_data.get("reason", "AI suggested")
                        else:
                            tag = str(tag_data)
                            confidence = 0.6
                            reason = "AI suggested"
                        
                        if tag and tag not in existing_tags:
                            suggestions.append(TagSuggestion(
                                tag=tag,
                                confidence=confidence,
                                reason=reason,
                                source="ai"
                            ))
        
        except Exception as e:
            logger.warning("AI tag suggestion failed", error=str(e))
        
        return suggestions
    
    def _filter_and_rank_suggestions(
        self, 
        suggestions: List[TagSuggestion], 
        existing_tags: List[str]
    ) -> List[TagSuggestion]:
        """Filter and rank tag suggestions."""
        # Remove duplicates and existing tags
        seen_tags = set(existing_tags)
        filtered = []
        
        for suggestion in suggestions:
            if suggestion.tag not in seen_tags:
                seen_tags.add(suggestion.tag)
                filtered.append(suggestion)
        
        # Filter by confidence threshold
        filtered = [
            s for s in filtered 
            if s.confidence >= self.config.auto_tag_confidence_threshold
        ]
        
        # Sort by confidence (descending)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered


class TagOperations:
    """Handles tag manipulation operations."""
    
    def __init__(self, config: TagManagerConfig, vault: Vault):
        self.config = config
        self.vault = vault
    
    async def apply_tag_operation(self, operation: TagOperation) -> bool:
        """Apply a single tag operation."""
        try:
            note = await self.vault.get_note(operation.note_id)
            if not note:
                logger.error("Note not found for tag operation", note_id=operation.note_id)
                return False
            
            if operation.operation_type == "add":
                return await self._add_tag_to_note(note, operation.new_tag)
            elif operation.operation_type == "remove":
                return await self._remove_tag_from_note(note, operation.old_tag)
            elif operation.operation_type == "rename":
                return await self._rename_tag_in_note(note, operation.old_tag, operation.new_tag)
            else:
                logger.error("Unknown tag operation type", operation_type=operation.operation_type)
                return False
        
        except Exception as e:
            logger.error("Tag operation failed", operation=operation, error=str(e))
            return False
    
    async def apply_batch_operations(self, operations: List[TagOperation]) -> Dict[str, Any]:
        """Apply multiple tag operations in batch."""
        results = {
            "total": len(operations),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Group operations by note for efficiency
        operations_by_note = defaultdict(list)
        for op in operations:
            operations_by_note[op.note_id].append(op)
        
        # Process operations for each note
        for note_id, note_operations in operations_by_note.items():
            try:
                note = await self.vault.get_note(note_id)
                if not note:
                    results["failed"] += len(note_operations)
                    results["errors"].append(f"Note not found: {note_id}")
                    continue
                
                # Apply all operations to this note
                modified_content = note.content
                success_count = 0
                
                for operation in note_operations:
                    if operation.operation_type == "add":
                        modified_content = self._add_tag_to_content(modified_content, operation.new_tag)
                        success_count += 1
                    elif operation.operation_type == "remove":
                        modified_content = self._remove_tag_from_content(modified_content, operation.old_tag)
                        success_count += 1
                    elif operation.operation_type == "rename":
                        modified_content = self._rename_tag_in_content(
                            modified_content, operation.old_tag, operation.new_tag
                        )
                        success_count += 1
                
                # Save modified note
                if modified_content != note.content:
                    await self.vault.update_note_content(note_id, modified_content)
                
                results["successful"] += success_count
                
            except Exception as e:
                results["failed"] += len(note_operations)
                results["errors"].append(f"Error processing note {note_id}: {str(e)}")
        
        return results
    
    async def _add_tag_to_note(self, note: Note, tag: str) -> bool:
        """Add a tag to a note."""
        modified_content = self._add_tag_to_content(note.content, tag)
        if modified_content != note.content:
            await self.vault.update_note_content(note.id, modified_content)
            return True
        return False
    
    async def _remove_tag_from_note(self, note: Note, tag: str) -> bool:
        """Remove a tag from a note."""
        modified_content = self._remove_tag_from_content(note.content, tag)
        if modified_content != note.content:
            await self.vault.update_note_content(note.id, modified_content)
            return True
        return False
    
    async def _rename_tag_in_note(self, note: Note, old_tag: str, new_tag: str) -> bool:
        """Rename a tag in a note."""
        modified_content = self._rename_tag_in_content(note.content, old_tag, new_tag)
        if modified_content != note.content:
            await self.vault.update_note_content(note.id, modified_content)
            return True
        return False
    
    def _add_tag_to_content(self, content: str, tag: str) -> str:
        """Add a tag to note content."""
        # Try to add to frontmatter first
        frontmatter_match = re.match(r'^(---\s*\n.*?\n---\s*\n)', content, re.DOTALL)
        
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            rest_content = content[len(frontmatter):]
            
            # Check if tags already exist in frontmatter
            if re.search(r'^tags:\s*', frontmatter, re.MULTILINE):
                # Add to existing tags
                modified_fm = re.sub(
                    r'^(tags:\s*\[)(.*?)(\])$',
                    lambda m: f"{m.group(1)}{m.group(2)}, {tag}{m.group(3)}" if m.group(2).strip() else f"{m.group(1)}{tag}{m.group(3)}",
                    frontmatter,
                    flags=re.MULTILINE
                )
                return modified_fm + rest_content
            else:
                # Add tags field to frontmatter
                modified_fm = frontmatter.replace('---\n', f'---\ntags: [{tag}]\n', 1)
                return modified_fm + rest_content
        else:
            # No frontmatter, add at the beginning
            new_frontmatter = f"---\ntags: [{tag}]\n---\n\n"
            return new_frontmatter + content
    
    def _remove_tag_from_content(self, content: str, tag: str) -> str:
        """Remove a tag from note content."""
        # Remove from frontmatter
        content = re.sub(
            rf'(tags:\s*\[)([^]]*?){re.escape(tag)}([^]]*?)(\])',
            lambda m: f"{m.group(1)}{m.group(2).rstrip(', ')}{m.group(3).lstrip(', ')}{m.group(4)}",
            content,
            flags=re.MULTILINE
        )
        
        # Remove inline tags
        content = re.sub(rf'(?:^|\s)#{re.escape(tag)}(?:\s|$)', ' ', content, flags=re.MULTILINE)
        
        # Clean up extra spaces
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content
    
    def _rename_tag_in_content(self, content: str, old_tag: str, new_tag: str) -> str:
        """Rename a tag in note content."""
        # Rename in frontmatter
        content = re.sub(
            rf'(tags:\s*\[[^]]*?){re.escape(old_tag)}([^]]*?\])',
            rf'\1{new_tag}\2',
            content,
            flags=re.MULTILINE
        )
        
        # Rename inline tags
        content = re.sub(
            rf'((?:^|\s))#{re.escape(old_tag)}(?=\s|$)',
            rf'\1#{new_tag}',
            content,
            flags=re.MULTILINE
        )
        
        return content


class TagManagerService:
    """
    Main tag management service that orchestrates all tag operations.
    
    Provides comprehensive tag management capabilities including analysis,
    similarity detection, hierarchy building, auto-tagging, and operations.
    """
    
    def __init__(
        self,
        vault: Vault,
        config: Optional[TagManagerConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        content_analyzer: Optional[ContentAnalyzer] = None,
        analysis_service: Optional[AnalysisService] = None,
    ):
        self.vault = vault
        self.config = config or TagManagerConfig()
        self.embedding_service = embedding_service
        self.content_analyzer = content_analyzer
        self.analysis_service = analysis_service
        
        # Initialize components
        self.analyzer = TagAnalyzer(self.config)
        self.similarity_detector = TagSimilarityDetector(self.config, embedding_service)
        self.hierarchy_builder = TagHierarchyBuilder(self.config)
        self.auto_tagger = AutoTagger(self.config, embedding_service, content_analyzer)
        self.operations = TagOperations(self.config, vault)
        
        # Caches
        self._tag_info_cache: Optional[Dict[str, TagInfo]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = 3600  # 1 hour
    
    async def analyze_tags(self, note_ids: Optional[List[str]] = None) -> TagAnalysisResult:
        """
        Perform comprehensive tag analysis across specified notes.
        
        Args:
            note_ids: Specific notes to analyze (default: all notes)
            
        Returns:
            Comprehensive tag analysis result
        """
        logger.info("Starting comprehensive tag analysis")
        
        # Get notes to analyze
        if note_ids is None:
            note_ids = await self.vault.get_all_note_ids()
        
        notes = []
        for note_id in note_ids:
            note = await self.vault.get_note(note_id)
            if note:
                notes.append(note)
        
        # Analyze tag usage
        tag_info = await self.analyzer.analyze_tag_usage(notes)
        
        # Find similar tags
        all_tags = list(tag_info.keys())
        similar_clusters = await self.similarity_detector.cluster_similar_tags(all_tags)
        
        # Build hierarchies
        hierarchies = await self.hierarchy_builder.build_hierarchies(tag_info)
        
        # Find orphaned tags (unused or rarely used)
        orphaned_tags = [
            tag for tag, info in tag_info.items()
            if info.usage_count < self.config.min_usage_threshold
        ]
        
        # Generate statistics
        usage_stats = self._calculate_usage_statistics(tag_info)
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(tag_info, similar_clusters)
        
        # Generate optimization suggestions
        optimization_suggestions = await self._generate_optimization_suggestions(
            tag_info, similar_clusters, hierarchies
        )
        
        result = TagAnalysisResult(
            total_tags=len(all_tags),
            unique_tags=len(set(info.normalized_name for info in tag_info.values())),
            orphaned_tags=orphaned_tags,
            duplicate_candidates=similar_clusters,
            hierarchy_suggestions=hierarchies,
            usage_statistics=usage_stats,
            quality_issues=quality_issues,
            optimization_suggestions=optimization_suggestions
        )
        
        logger.info("Tag analysis completed", 
                   total_tags=result.total_tags,
                   duplicates=len(result.duplicate_candidates),
                   orphaned=len(result.orphaned_tags))
        
        return result
    
    async def suggest_tags(
        self, 
        note_id: str, 
        max_suggestions: int = 10
    ) -> List[TagSuggestion]:
        """
        Suggest tags for a specific note.
        
        Args:
            note_id: ID of the note to suggest tags for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of tag suggestions with confidence scores
        """
        note = await self.vault.get_note(note_id)
        if not note:
            raise ValueError(f"Note not found: {note_id}")
        
        # Get existing tags
        existing_tags = await self.analyzer.extract_tags_from_note(note)
        
        # Get tag context (for semantic suggestions)
        tag_info = await self._get_tag_info_cached()
        
        # Generate suggestions
        suggestions = await self.auto_tagger.suggest_tags_for_note(
            note, existing_tags, tag_info
        )
        
        return suggestions[:max_suggestions]
    
    async def find_similar_tags(
        self, 
        tags: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[TagSimilarity]:
        """
        Find similar tags using fuzzy matching and semantic analysis.
        
        Args:
            tags: Specific tags to check (default: all tags)
            threshold: Similarity threshold (default: config threshold)
            
        Returns:
            List of tag similarities above threshold
        """
        if tags is None:
            tag_info = await self._get_tag_info_cached()
            tags = list(tag_info.keys())
        
        if threshold is not None:
            original_threshold = self.config.fuzzy_similarity_threshold
            self.config.fuzzy_similarity_threshold = threshold
        
        try:
            similarities = await self.similarity_detector.find_similar_tags(tags)
            return similarities
        finally:
            if threshold is not None:
                self.config.fuzzy_similarity_threshold = original_threshold
    
    async def optimize_tag_hierarchy(self) -> List[str]:
        """
        Analyze and optimize tag hierarchies.
        
        Returns:
            List of optimization suggestions
        """
        tag_info = await self._get_tag_info_cached()
        hierarchies = await self.hierarchy_builder.build_hierarchies(tag_info)
        
        suggestions = await self.hierarchy_builder.suggest_hierarchy_optimizations(
            hierarchies, tag_info
        )
        
        return suggestions
    
    async def batch_apply_tags(self, operations: List[TagOperation]) -> Dict[str, Any]:
        """
        Apply multiple tag operations in batch.
        
        Args:
            operations: List of tag operations to perform
            
        Returns:
            Batch operation results with success/failure counts
        """
        logger.info("Applying batch tag operations", count=len(operations))
        
        results = await self.operations.apply_batch_operations(operations)
        
        # Invalidate cache after modifications
        self._invalidate_cache()
        
        logger.info("Batch tag operations completed", 
                   successful=results["successful"],
                   failed=results["failed"])
        
        return results
    
    async def get_tag_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tag usage statistics."""
        tag_info = await self._get_tag_info_cached()
        return self._calculate_usage_statistics(tag_info)
    
    async def _get_tag_info_cached(self) -> Dict[str, TagInfo]:
        """Get tag info with caching."""
        now = datetime.utcnow()
        
        # Check if cache is valid
        if (self._tag_info_cache is not None and 
            self._cache_timestamp is not None and
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._tag_info_cache
        
        # Rebuild cache
        logger.debug("Rebuilding tag info cache")
        
        all_note_ids = await self.vault.get_all_note_ids()
        notes = []
        
        for note_id in all_note_ids:
            note = await self.vault.get_note(note_id)
            if note:
                notes.append(note)
        
        self._tag_info_cache = await self.analyzer.analyze_tag_usage(notes)
        self._cache_timestamp = now
        
        return self._tag_info_cache
    
    def _invalidate_cache(self):
        """Invalidate the tag info cache."""
        self._tag_info_cache = None
        self._cache_timestamp = None
    
    def _calculate_usage_statistics(self, tag_info: Dict[str, TagInfo]) -> Dict[str, Any]:
        """Calculate comprehensive tag usage statistics."""
        if not tag_info:
            return {}
        
        usage_counts = [info.usage_count for info in tag_info.values()]
        hierarchical_tags = [info for info in tag_info.values() if info.is_hierarchical]
        
        stats = {
            "total_tags": len(tag_info),
            "total_usage": sum(usage_counts),
            "average_usage": sum(usage_counts) / len(usage_counts),
            "median_usage": sorted(usage_counts)[len(usage_counts) // 2],
            "max_usage": max(usage_counts),
            "min_usage": min(usage_counts),
            
            # Hierarchy stats
            "hierarchical_tags": len(hierarchical_tags),
            "hierarchy_percentage": len(hierarchical_tags) / len(tag_info) * 100,
            "max_hierarchy_depth": max((info.hierarchy_level for info in tag_info.values()), default=0),
            
            # Distribution
            "usage_distribution": {
                "unused": len([c for c in usage_counts if c == 0]),
                "rare (1-2)": len([c for c in usage_counts if 1 <= c <= 2]),
                "occasional (3-10)": len([c for c in usage_counts if 3 <= c <= 10]),
                "frequent (11-50)": len([c for c in usage_counts if 11 <= c <= 50]),
                "very_frequent (50+)": len([c for c in usage_counts if c > 50]),
            },
            
            # Top tags
            "most_used_tags": sorted(
                [(name, info.usage_count) for name, info in tag_info.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
        }
        
        return stats
    
    def _identify_quality_issues(
        self, 
        tag_info: Dict[str, TagInfo], 
        similar_clusters: List[TagCluster]
    ) -> List[str]:
        """Identify quality issues in tag usage."""
        issues = []
        
        # Check for potential duplicates
        duplicate_clusters = [c for c in similar_clusters if c.suggested_merge]
        if duplicate_clusters:
            issues.append(f"Found {len(duplicate_clusters)} potential duplicate tag groups")
        
        # Check for orphaned tags
        orphaned_count = len([
            info for info in tag_info.values() 
            if info.usage_count < self.config.min_usage_threshold
        ])
        if orphaned_count > 0:
            issues.append(f"Found {orphaned_count} rarely used tags")
        
        # Check for inconsistent naming
        naming_issues = 0
        for info in tag_info.values():
            if ' ' in info.name or info.name != info.name.lower():
                naming_issues += 1
        
        if naming_issues > 0:
            issues.append(f"Found {naming_issues} tags with naming inconsistencies")
        
        # Check for overly deep hierarchies
        deep_hierarchies = [
            info for info in tag_info.values()
            if info.hierarchy_level > self.config.max_hierarchy_depth
        ]
        if deep_hierarchies:
            issues.append(f"Found {len(deep_hierarchies)} tags in overly deep hierarchies")
        
        return issues
    
    async def _generate_optimization_suggestions(
        self,
        tag_info: Dict[str, TagInfo],
        similar_clusters: List[TagCluster],
        hierarchies: List[TagHierarchy]
    ) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Merge suggestions from similar clusters
        merge_candidates = [c for c in similar_clusters if c.suggested_merge]
        if merge_candidates:
            suggestions.append(
                f"Consider merging {len(merge_candidates)} groups of similar tags to reduce redundancy"
            )
        
        # Cleanup suggestions for orphaned tags
        orphaned_count = len([
            info for info in tag_info.values()
            if info.usage_count < self.config.min_usage_threshold
        ])
        if orphaned_count > 0:
            suggestions.append(f"Consider removing {orphaned_count} rarely used tags")
        
        # Hierarchy optimization suggestions
        hierarchy_suggestions = await self.hierarchy_builder.suggest_hierarchy_optimizations(
            hierarchies, tag_info
        )
        suggestions.extend(hierarchy_suggestions)
        
        # Naming consistency suggestions
        naming_issues = [
            info.name for info in tag_info.values()
            if ' ' in info.name or info.name != info.name.lower()
        ]
        if naming_issues:
            suggestions.append(
                f"Consider standardizing naming for {len(naming_issues)} tags "
                f"(use lowercase, replace spaces with hyphens)"
            )
        
        return suggestions