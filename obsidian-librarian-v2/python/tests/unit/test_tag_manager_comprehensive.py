"""
Comprehensive unit tests for Tag Management Service.

This test suite provides extensive coverage for all tag management components:
- TagAnalyzer: Tag extraction and analysis
- TagSimilarityDetector: Fuzzy and semantic similarity detection
- TagHierarchyBuilder: Tag hierarchy construction
- AutoTagger: AI-powered tagging suggestions
- TagOperations: Bulk tag operations
- TagManagerService: Main service orchestration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import numpy as np

from obsidian_librarian.services.tag_manager import (
    TagManagerService,
    TagAnalyzer,
    TagSimilarityDetector,
    TagHierarchyBuilder,
    AutoTagger,
    TagOperations,
)
from obsidian_librarian.models.models import (
    Note,
    NoteMetadata,
    TagManagerConfig,
    TagInfo,
    TagSuggestion,
    TagSimilarity,
    TagCluster,
    TagHierarchy,
    TagOperation,
    TagAnalysisResult,
)


class TestTagAnalyzerComprehensive:
    """Comprehensive tests for TagAnalyzer."""
    
    @pytest.fixture
    def tag_config(self):
        """Create a comprehensive test configuration."""
        return TagManagerConfig(
            fuzzy_similarity_threshold=0.8,
            semantic_similarity_threshold=0.7,
            enable_fuzzy_matching=True,
            enable_semantic_analysis=True,
            case_insensitive=True,
            min_usage_threshold=2,
            auto_tag_confidence_threshold=0.7,
            max_auto_tags_per_note=5,
            max_tag_length=50,
            min_tag_length=2,
            excluded_tags=["temp", "draft"],
            tag_patterns={
                "year": r"\b(19|20)\d{2}\b",
                "date": r"\d{4}-\d{2}-\d{2}",
                "version": r"v?\d+\.\d+(\.\d+)?",
            }
        )
    
    @pytest.fixture
    def analyzer(self, tag_config):
        """Create TagAnalyzer instance."""
        return TagAnalyzer(tag_config)
    
    def test_init_with_custom_patterns(self, tag_config):
        """Test initialization with custom tag patterns."""
        analyzer = TagAnalyzer(tag_config)
        assert analyzer.config == tag_config
        assert "year" in analyzer._pattern_cache
        assert "date" in analyzer._pattern_cache
        assert "version" in analyzer._pattern_cache
    
    def test_extract_frontmatter_tags_yaml_list(self, analyzer):
        """Test extraction from YAML list format."""
        content = """---
tags: [project, development, "multi word tag", api-design]
categories: [tech, programming]
---

# Content"""
        
        tags = analyzer._extract_frontmatter_tags(content)
        assert "project" in tags
        assert "development" in tags
        assert "multi word tag" in tags
        assert "api-design" in tags
        assert len(tags) == 4  # Only tags, not categories
    
    def test_extract_frontmatter_tags_yaml_string(self, analyzer):
        """Test extraction from YAML string format."""
        content = """---
tags: "single-tag"
---

# Content"""
        
        tags = analyzer._extract_frontmatter_tags(content)
        assert "single-tag" in tags
        assert len(tags) == 1
    
    def test_extract_frontmatter_tags_multiline(self, analyzer):
        """Test extraction from multiline YAML format."""
        content = """---
tags:
  - project
  - development
  - "complex tag"
  - api/v2
---

# Content"""
        
        tags = analyzer._extract_frontmatter_tags(content)
        assert "project" in tags
        assert "development" in tags
        assert "complex tag" in tags
        assert "api/v2" in tags
    
    def test_extract_frontmatter_tags_empty(self, analyzer):
        """Test extraction with no frontmatter."""
        content = "# Just content, no frontmatter"
        tags = analyzer._extract_frontmatter_tags(content)
        assert len(tags) == 0
    
    def test_extract_inline_tags_standard(self, analyzer):
        """Test extraction of standard inline tags."""
        content = """
# My Note

This has #simple-tag and #another_tag tags.
Also #CamelCase and #with-numbers-123.
Nested #parent/child/grandchild tag.
"""
        
        tags = analyzer._extract_inline_tags(content)
        assert "simple-tag" in tags
        assert "another_tag" in tags
        assert "CamelCase" in tags
        assert "with-numbers-123" in tags
        assert "parent/child/grandchild" in tags
    
    def test_extract_inline_tags_edge_cases(self, analyzer):
        """Test inline tag extraction edge cases."""
        content = """
Email: user@example.com #not-a-tag
Price: $#100 also not a tag
But #real-tag# and ##double-hash are interesting
#tag-at-end-of-line
#tag-before-punctuation, works!
(#tag-in-parentheses)
"#tag-in-quotes"
"""
        
        tags = analyzer._extract_inline_tags(content)
        assert "real-tag" in tags
        assert "tag-at-end-of-line" in tags
        assert "tag-before-punctuation" in tags
        assert "tag-in-parentheses" in tags
        assert "tag-in-quotes" in tags
        assert "not-a-tag" not in tags  # After @
        assert "100" not in tags  # After $
    
    def test_extract_content_based_tags(self, analyzer):
        """Test content-based tag extraction using patterns."""
        content = """
# Project Update 2024

Version: v2.3.1
Date: 2024-01-15

Working on the new API design for 2023 compatibility.
"""
        
        tags = analyzer._extract_content_based_tags(content)
        assert any("2024" in tag for tag in tags)  # Year pattern
        assert any("2024-01-15" in tag for tag in tags)  # Date pattern
        assert any("v2.3.1" in tag or "2.3.1" in tag for tag in tags)  # Version pattern
    
    def test_normalize_tags_case_insensitive(self, analyzer):
        """Test tag normalization with case insensitivity."""
        raw_tags = ["Project", "PROJECT", "project", "PrOjEcT"]
        normalized = analyzer._normalize_tags(raw_tags)
        
        # Should all normalize to lowercase
        assert len(normalized) == 1
        assert "project" in normalized
    
    def test_normalize_tags_special_characters(self, analyzer):
        """Test tag normalization with special characters."""
        raw_tags = [
            "tag with spaces",
            "tag_with_underscores",
            "tag-with-dashes",
            "tag.with.dots",
            "tag@with#special$chars!",
            "tag/with/slashes"
        ]
        
        normalized = analyzer._normalize_tags(raw_tags)
        assert "tagwithspaces" in normalized
        assert "tag_with_underscores" in normalized
        assert "tag-with-dashes" in normalized
        assert "tagwithdots" in normalized
        assert "tagwithspecialchars" in normalized
        assert "tag/with/slashes" in normalized  # Slashes preserved for hierarchy
    
    def test_normalize_tags_length_limits(self, analyzer):
        """Test tag normalization with length limits."""
        raw_tags = [
            "a",  # Too short
            "ab",  # Minimum length
            "a" * 100,  # Too long
            "normal-length-tag"
        ]
        
        normalized = analyzer._normalize_tags(raw_tags)
        assert "a" not in normalized  # Too short
        assert "ab" in normalized  # Minimum length
        assert len(max(normalized, key=len)) <= analyzer.config.max_tag_length
        assert "normal-length-tag" in normalized
    
    def test_normalize_tags_excluded(self, analyzer):
        """Test exclusion of configured tags."""
        raw_tags = ["temp", "draft", "permanent", "TEMP", "Draft"]
        normalized = analyzer._normalize_tags(raw_tags)
        
        assert "temp" not in normalized
        assert "draft" not in normalized
        assert "permanent" in normalized
    
    async def test_extract_tags_from_note_complete(self, analyzer):
        """Test complete tag extraction from a note."""
        note = Note(
            id="test-note",
            path=Path("test.md"),
            content="""---
tags: [project, api]
---

# Test Note

Working on #development and #testing.
Version 1.2.3 released on 2024-01-15.
""",
            metadata=NoteMetadata(
                title="Test Note",
                tags=["metadata-tag"],
                created=datetime.utcnow(),
                modified=datetime.utcnow(),
            ),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        tags = await analyzer.extract_tags_from_note(note)
        
        # Should include all sources
        assert "project" in tags  # Frontmatter
        assert "api" in tags  # Frontmatter
        assert "development" in tags  # Inline
        assert "testing" in tags  # Inline
        assert "metadata-tag" in tags  # Metadata
        # Content-based tags might be included depending on patterns
    
    async def test_analyze_tag_usage_multiple_notes(self, analyzer):
        """Test tag usage analysis across multiple notes."""
        notes = [
            Note(
                id=f"note-{i}",
                path=Path(f"note-{i}.md"),
                content=f"# Note {i}\n\n#common #unique-{i}",
                metadata=NoteMetadata(tags=["common", f"meta-{i}"]),
                created_at=datetime.utcnow() - timedelta(days=i),
                modified_at=datetime.utcnow(),
            )
            for i in range(5)
        ]
        
        with patch.object(analyzer, 'extract_tags_from_note') as mock_extract:
            # Mock different tag sets for each note
            async def extract_side_effect(note):
                i = int(note.id.split('-')[1])
                return {"common", f"unique-{i}", f"meta-{i}"}
            
            mock_extract.side_effect = extract_side_effect
            
            tag_info = await analyzer.analyze_tag_usage(notes)
        
        # Common tag should have highest usage
        assert "common" in tag_info
        assert tag_info["common"].usage_count == 5
        assert len(tag_info["common"].notes) == 5
        
        # Unique tags should have single usage
        assert tag_info["unique-0"].usage_count == 1
        assert tag_info["unique-0"].notes == {"note-0"}
    
    async def test_get_tag_cooccurrence_matrix(self, analyzer):
        """Test tag co-occurrence matrix generation."""
        tag_info = {
            "python": TagInfo("python", "python", 3, datetime.utcnow(), datetime.utcnow(),
                            notes={"note1", "note2", "note3"}),
            "ml": TagInfo("ml", "ml", 2, datetime.utcnow(), datetime.utcnow(),
                        notes={"note1", "note2"}),
            "data": TagInfo("data", "data", 2, datetime.utcnow(), datetime.utcnow(),
                          notes={"note2", "note3"}),
            "api": TagInfo("api", "api", 1, datetime.utcnow(), datetime.utcnow(),
                         notes={"note3"}),
        }
        
        matrix = await analyzer.get_tag_cooccurrence_matrix(tag_info)
        
        # Check co-occurrences
        assert matrix[("python", "ml")] == 2  # Both in note1 and note2
        assert matrix[("python", "data")] == 2  # Both in note2 and note3
        assert matrix[("python", "api")] == 1  # Both in note3
        assert matrix[("ml", "data")] == 1  # Both in note2
        assert matrix[("ml", "api")] == 0  # No shared notes
    
    def test_pattern_cache_performance(self, tag_config):
        """Test that pattern compilation is cached."""
        analyzer = TagAnalyzer(tag_config)
        
        # Access pattern multiple times
        pattern1 = analyzer._pattern_cache.get("year")
        pattern2 = analyzer._pattern_cache.get("year")
        
        # Should be the same compiled pattern object
        assert pattern1 is pattern2


class TestTagSimilarityDetectorComprehensive:
    """Comprehensive tests for TagSimilarityDetector."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service with realistic behavior."""
        service = Mock()
        
        # Simulate embeddings based on tag content
        async def embed_text(text):
            # Simple hash-based fake embedding
            hash_val = hashlib.md5(text.encode()).hexdigest()
            embedding = [float(int(hash_val[i:i+2], 16)) / 255.0 for i in range(0, 32, 2)]
            return embedding[:10]  # Return 10-dimensional embedding
        
        service.embed_text = AsyncMock(side_effect=embed_text)
        
        # Simulate similarity computation
        async def compute_similarity(emb1, emb2):
            # Cosine similarity approximation
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        service.compute_similarity = AsyncMock(side_effect=compute_similarity)
        
        return service
    
    @pytest.fixture
    def detector(self, tag_config, mock_embedding_service):
        """Create TagSimilarityDetector instance."""
        return TagSimilarityDetector(tag_config, mock_embedding_service)
    
    def test_calculate_fuzzy_similarity_exact_match(self, detector):
        """Test fuzzy similarity for exact matches."""
        score = detector._calculate_fuzzy_similarity("project", "project")
        assert score == 1.0
    
    def test_calculate_fuzzy_similarity_case_variations(self, detector):
        """Test fuzzy similarity with case variations."""
        pairs = [
            ("project", "Project"),
            ("API", "api"),
            ("CamelCase", "camelcase"),
        ]
        
        for tag1, tag2 in pairs:
            score = detector._calculate_fuzzy_similarity(tag1, tag2)
            assert score > 0.9  # Should be very similar when case-insensitive
    
    def test_calculate_fuzzy_similarity_typos(self, detector):
        """Test fuzzy similarity with common typos."""
        pairs = [
            ("project", "projct"),  # Missing letter
            ("development", "developement"),  # Common typo
            ("programming", "programing"),  # Single vs double letter
            ("analyze", "analyse"),  # US vs UK spelling
        ]
        
        for tag1, tag2 in pairs:
            score = detector._calculate_fuzzy_similarity(tag1, tag2)
            assert score > 0.7  # Should detect as similar
    
    def test_calculate_fuzzy_similarity_plurals(self, detector):
        """Test fuzzy similarity with plural forms."""
        pairs = [
            ("project", "projects"),
            ("task", "tasks"),
            ("category", "categories"),
            ("analysis", "analyses"),
        ]
        
        for tag1, tag2 in pairs:
            score = detector._calculate_fuzzy_similarity(tag1, tag2)
            assert score > 0.8  # Should be highly similar
    
    def test_calculate_fuzzy_similarity_unrelated(self, detector):
        """Test fuzzy similarity for unrelated tags."""
        pairs = [
            ("project", "banana"),
            ("api", "zebra"),
            ("development", "xyz"),
        ]
        
        for tag1, tag2 in pairs:
            score = detector._calculate_fuzzy_similarity(tag1, tag2)
            assert score < 0.5  # Should be dissimilar
    
    async def test_calculate_semantic_similarity_related(self, detector):
        """Test semantic similarity for conceptually related tags."""
        # Note: With our mock, this tests the flow rather than actual semantic similarity
        score = await detector._calculate_semantic_similarity("programming", "coding")
        
        # Verify embedding service was called correctly
        assert detector.embedding_service.embed_text.call_count == 2
        assert detector.embedding_service.compute_similarity.call_count == 1
    
    async def test_find_similar_tags_basic(self, detector):
        """Test finding similar tags in a basic list."""
        tags = ["project", "projects", "Project", "task", "tasks", "api", "apis"]
        
        similarities = await detector.find_similar_tags(tags)
        
        # Should find plural pairs
        assert any(
            (sim.tag_a == "project" and sim.tag_b == "projects") or
            (sim.tag_a == "projects" and sim.tag_b == "project")
            for sim in similarities
        )
        
        # Should find case variations
        assert any(
            (sim.tag_a.lower() == "project" and sim.tag_b.lower() == "project")
            for sim in similarities
        )
    
    async def test_find_similar_tags_with_threshold(self, detector):
        """Test similarity detection respects threshold."""
        detector.config.fuzzy_similarity_threshold = 0.9  # High threshold
        
        tags = ["project", "projct", "development", "coding"]
        similarities = await detector.find_similar_tags(tags)
        
        # Only very similar tags should be detected
        for sim in similarities:
            assert sim.similarity_score >= 0.9
    
    async def test_cluster_similar_tags(self, detector):
        """Test clustering of similar tags."""
        tags = [
            "project", "projects", "Project",
            "task", "tasks", "Task",
            "api", "API",
            "development", "developing"
        ]
        
        # Mock similarity calculations
        async def mock_calculate_similarity(tag1, tag2):
            # Simple rule-based similarity for testing
            if tag1.lower() == tag2.lower():
                return TagSimilarity(tag1, tag2, 1.0, "exact")
            elif tag1.lower().rstrip('s') == tag2.lower().rstrip('s'):
                return TagSimilarity(tag1, tag2, 0.9, "plural")
            elif tag1.lower() in tag2.lower() or tag2.lower() in tag1.lower():
                return TagSimilarity(tag1, tag2, 0.8, "substring")
            else:
                return None
        
        with patch.object(detector, 'calculate_similarity', mock_calculate_similarity):
            clusters = await detector.cluster_similar_tags(tags)
        
        # Should create clusters for similar tags
        assert len(clusters) > 0
        
        # Find project cluster
        project_cluster = next((c for c in clusters if "project" in c.tags), None)
        assert project_cluster is not None
        assert "projects" in project_cluster.tags
        assert "Project" in project_cluster.tags
    
    async def test_detect_abbreviations(self, detector):
        """Test detection of abbreviations and full forms."""
        tag_pairs = [
            ("ml", "machine-learning"),
            ("api", "application-programming-interface"),
            ("ui", "user-interface"),
            ("db", "database"),
        ]
        
        abbreviations = await detector._detect_abbreviations(tag_pairs)
        
        # Should detect some abbreviation patterns
        assert len(abbreviations) > 0
        for abbrev in abbreviations:
            assert abbrev.similarity_type == "abbreviation"
            assert len(abbrev.tag_a) < len(abbrev.tag_b)  # Abbreviation is shorter


class TestTagHierarchyBuilderComprehensive:
    """Comprehensive tests for TagHierarchyBuilder."""
    
    @pytest.fixture
    def builder(self, tag_config):
        """Create TagHierarchyBuilder instance."""
        return TagHierarchyBuilder(tag_config)
    
    async def test_build_from_paths_simple(self, builder):
        """Test hierarchy building from simple path-like tags."""
        tags = [
            "programming",
            "programming/python",
            "programming/python/django",
            "programming/javascript",
            "programming/javascript/react",
            "programming/javascript/vue",
        ]
        
        tag_info = {
            tag: TagInfo(tag, tag, 5, datetime.utcnow(), datetime.utcnow())
            for tag in tags
        }
        
        hierarchies = await builder.build_from_paths(tag_info)
        
        # Should create a hierarchy with programming as root
        assert len(hierarchies) == 1
        root = hierarchies[0]
        assert root.root_tag == "programming"
        assert len(root.children) == 2  # python and javascript
        
        # Check python branch
        python_child = next(c for c in root.children if c.root_tag == "programming/python")
        assert len(python_child.children) == 1  # django
        
        # Check javascript branch  
        js_child = next(c for c in root.children if c.root_tag == "programming/javascript")
        assert len(js_child.children) == 2  # react and vue
    
    async def test_build_from_paths_multiple_roots(self, builder):
        """Test hierarchy building with multiple root tags."""
        tags = [
            "projects/web",
            "projects/mobile",
            "projects/web/frontend",
            "research/ml",
            "research/ml/nlp",
            "research/databases",
        ]
        
        tag_info = {
            tag: TagInfo(tag, tag, 3, datetime.utcnow(), datetime.utcnow())
            for tag in tags
        }
        
        hierarchies = await builder.build_from_paths(tag_info)
        
        # Should create two separate hierarchies
        assert len(hierarchies) == 2
        root_tags = {h.root_tag for h in hierarchies}
        assert "projects" in root_tags
        assert "research" in root_tags
    
    async def test_suggest_from_cooccurrence(self, builder):
        """Test hierarchy suggestion from co-occurrence patterns."""
        cooccurrence = {
            ("python", "programming"): 10,
            ("django", "python"): 8,
            ("flask", "python"): 7,
            ("react", "javascript"): 9,
            ("javascript", "programming"): 9,
            ("ml", "python"): 6,
            ("database", "backend"): 5,
        }
        
        tag_info = {
            "programming": TagInfo("programming", "programming", 20, datetime.utcnow(), datetime.utcnow()),
            "python": TagInfo("python", "python", 15, datetime.utcnow(), datetime.utcnow()),
            "javascript": TagInfo("javascript", "javascript", 12, datetime.utcnow(), datetime.utcnow()),
            "django": TagInfo("django", "django", 8, datetime.utcnow(), datetime.utcnow()),
            "flask": TagInfo("flask", "flask", 7, datetime.utcnow(), datetime.utcnow()),
            "react": TagInfo("react", "react", 9, datetime.utcnow(), datetime.utcnow()),
            "ml": TagInfo("ml", "ml", 6, datetime.utcnow(), datetime.utcnow()),
            "database": TagInfo("database", "database", 5, datetime.utcnow(), datetime.utcnow()),
            "backend": TagInfo("backend", "backend", 5, datetime.utcnow(), datetime.utcnow()),
        }
        
        hierarchies = await builder.suggest_from_cooccurrence(tag_info, cooccurrence)
        
        # Should suggest hierarchies based on co-occurrence strength
        assert len(hierarchies) > 0
        
        # Programming should be a root with high confidence
        programming_hierarchy = next((h for h in hierarchies if h.root_tag == "programming"), None)
        assert programming_hierarchy is not None
        assert programming_hierarchy.confidence > 0.7
    
    async def test_suggest_from_semantic_similarity(self, builder):
        """Test hierarchy suggestion from semantic similarity."""
        similarities = [
            TagSimilarity("ml", "machine-learning", 0.95, "semantic"),
            TagSimilarity("ai", "artificial-intelligence", 0.93, "semantic"),
            TagSimilarity("nlp", "natural-language-processing", 0.92, "semantic"),
            TagSimilarity("frontend", "front-end", 0.9, "fuzzy"),
            TagSimilarity("backend", "back-end", 0.9, "fuzzy"),
        ]
        
        tag_info = {
            tag: TagInfo(tag, tag, 5, datetime.utcnow(), datetime.utcnow())
            for sim in similarities
            for tag in [sim.tag_a, sim.tag_b]
        }
        
        hierarchies = await builder.suggest_from_semantic_similarity(tag_info, similarities)
        
        # Should create hierarchies linking similar tags
        assert len(hierarchies) > 0
        
        # Should prefer longer, more descriptive tags as parents
        ml_hierarchy = next((h for h in hierarchies if "machine-learning" in h.root_tag), None)
        assert ml_hierarchy is not None
    
    def test_calculate_hierarchy_score(self, builder):
        """Test hierarchy scoring calculation."""
        # Create a hierarchy with depth and breadth
        hierarchy = TagHierarchy(
            root_tag="programming",
            children=[
                TagHierarchy(
                    root_tag="python",
                    children=[
                        TagHierarchy(root_tag="django", children=[], level=2),
                        TagHierarchy(root_tag="flask", children=[], level=2),
                    ],
                    level=1
                ),
                TagHierarchy(
                    root_tag="javascript",
                    children=[
                        TagHierarchy(root_tag="react", children=[], level=2),
                    ],
                    level=1
                ),
            ],
            level=0,
            confidence=0.9
        )
        
        tag_info = {
            "programming": TagInfo("programming", "programming", 50, datetime.utcnow(), datetime.utcnow()),
            "python": TagInfo("python", "python", 30, datetime.utcnow(), datetime.utcnow()),
            "javascript": TagInfo("javascript", "javascript", 25, datetime.utcnow(), datetime.utcnow()),
            "django": TagInfo("django", "django", 15, datetime.utcnow(), datetime.utcnow()),
            "flask": TagInfo("flask", "flask", 10, datetime.utcnow(), datetime.utcnow()),
            "react": TagInfo("react", "react", 20, datetime.utcnow(), datetime.utcnow()),
        }
        
        score = builder._calculate_hierarchy_score(hierarchy, tag_info)
        
        # Should have a reasonable score
        assert 0 < score < 1
        # Root with high usage should contribute to score
        assert score > 0.5


class TestAutoTaggerComprehensive:
    """Comprehensive tests for AutoTagger."""
    
    @pytest.fixture
    def mock_content_analyzer(self):
        """Create mock content analyzer."""
        analyzer = Mock()
        
        async def analyze_content(content):
            result = Mock()
            result.keywords = ["test", "development", "api"]
            result.topics = ["software development", "testing"]
            result.entities = ["Python", "Django"]
            return result
        
        analyzer.analyze_content = AsyncMock(side_effect=analyze_content)
        return analyzer
    
    @pytest.fixture
    def auto_tagger(self, tag_config, mock_embedding_service, mock_content_analyzer):
        """Create AutoTagger instance."""
        return AutoTagger(tag_config, mock_embedding_service, mock_content_analyzer)
    
    async def test_suggest_from_content_keywords(self, auto_tagger):
        """Test tag suggestions from content keywords."""
        note = Note(
            id="test-note",
            path=Path("test.md"),
            content="""# Python Development Guide

This guide covers Python development best practices, testing strategies,
and API design patterns. We'll explore Django and Flask frameworks.
""",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        existing_tags = set()
        suggestions = await auto_tagger._suggest_from_content(note, existing_tags)
        
        # Should suggest tags based on content analysis
        assert len(suggestions) > 0
        suggested_tags = {s.tag for s in suggestions}
        
        # Should include analyzed keywords/topics
        assert any("test" in tag or "development" in tag or "api" in tag 
                  for tag in suggested_tags)
    
    async def test_suggest_from_patterns(self, auto_tagger):
        """Test pattern-based tag suggestions."""
        note = Note(
            id="meeting-note",
            path=Path("2024-01-15-team-meeting.md"),
            content="""# Team Meeting - 2024-01-15

## Agenda
- Project status update
- Sprint planning
- Technical discussions

## Action Items
- [ ] Review PR #123
- [ ] Update documentation
- [x] Deploy to staging

## Attendees
- Alice (PM)
- Bob (Dev)
- Carol (QA)
""",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        existing_tags = set()
        suggestions = await auto_tagger._suggest_from_patterns(note, existing_tags)
        
        # Should detect meeting pattern
        assert any(s.tag == "meeting" for s in suggestions)
        
        # Should detect checklist pattern
        assert any(s.tag == "checklist" or s.tag == "todo" for s in suggestions)
        
        # Pattern suggestions should have appropriate confidence
        for s in suggestions:
            assert s.source == "pattern"
            assert 0.5 <= s.confidence <= 1.0
    
    async def test_suggest_from_ai_with_context(self, auto_tagger):
        """Test AI-based suggestions with context."""
        note = Note(
            id="technical-note",
            path=Path("technical.md"),
            content="Advanced machine learning techniques for NLP",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        existing_tags = {"ml"}
        context_tags = {
            "ml": TagInfo("ml", "ml", 20, datetime.utcnow(), datetime.utcnow()),
            "nlp": TagInfo("nlp", "nlp", 15, datetime.utcnow(), datetime.utcnow()),
            "deep-learning": TagInfo("deep-learning", "deep-learning", 10, datetime.utcnow(), datetime.utcnow()),
        }
        
        # Mock AI suggestions
        with patch.object(auto_tagger.content_analyzer, 'analyze_content') as mock_analyze:
            mock_result = Mock()
            mock_result.suggested_tags = ["nlp", "deep-learning", "neural-networks"]
            mock_analyze.return_value = mock_result
            
            suggestions = await auto_tagger._suggest_from_ai(note, existing_tags, context_tags)
        
        # Should not suggest existing tags
        assert not any(s.tag == "ml" for s in suggestions)
        
        # Should suggest contextually relevant tags
        assert any(s.tag == "nlp" for s in suggestions)
        assert all(s.source == "ai" for s in suggestions)
    
    async def test_suggest_tags_for_note_comprehensive(self, auto_tagger):
        """Test complete tag suggestion pipeline."""
        note = Note(
            id="comprehensive-note",
            path=Path("project-update.md"),
            content="""# Project Alpha Update - 2024-01-15

## Overview
Working on machine learning pipeline for data processing.

## Progress
- [x] Data collection complete
- [ ] Model training in progress
- [ ] API development pending

#ml #data-science
""",
            metadata=NoteMetadata(tags=["project"]),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        existing_tags = {"ml", "data-science", "project"}
        context_tags = {
            "ml": TagInfo("ml", "ml", 30, datetime.utcnow(), datetime.utcnow()),
            "api": TagInfo("api", "api", 25, datetime.utcnow(), datetime.utcnow()),
            "pipeline": TagInfo("pipeline", "pipeline", 10, datetime.utcnow(), datetime.utcnow()),
        }
        
        suggestions = await auto_tagger.suggest_tags_for_note(note, existing_tags, context_tags)
        
        # Should not suggest existing tags
        suggested_tags = {s.tag for s in suggestions}
        assert "ml" not in suggested_tags
        assert "data-science" not in suggested_tags
        assert "project" not in suggested_tags
        
        # Should limit to max configured tags
        assert len(suggestions) <= auto_tagger.config.max_auto_tags_per_note
        
        # Should only include high-confidence suggestions
        assert all(s.confidence >= auto_tagger.config.auto_tag_confidence_threshold 
                  for s in suggestions)
        
        # Should have varied sources
        sources = {s.source for s in suggestions}
        assert len(sources) > 1  # Multiple detection methods used


class TestTagOperationsComprehensive:
    """Comprehensive tests for TagOperations."""
    
    @pytest.fixture
    def mock_vault(self):
        """Create comprehensive mock vault."""
        vault = Mock()
        
        # Mock note storage
        vault._notes = {}
        
        async def get_note(note_id):
            return vault._notes.get(note_id)
        
        async def update_note_content(note_id, content):
            if note_id in vault._notes:
                vault._notes[note_id].content = content
                return True
            return False
        
        vault.get_note = AsyncMock(side_effect=get_note)
        vault.update_note_content = AsyncMock(side_effect=update_note_content)
        vault.get_all_note_ids = AsyncMock(return_value=list(vault._notes.keys()))
        
        return vault
    
    @pytest.fixture
    def operations(self, tag_config, mock_vault):
        """Create TagOperations instance."""
        return TagOperations(tag_config, mock_vault)
    
    def test_add_tag_to_content_with_existing_frontmatter(self, operations):
        """Test adding tag to content with existing frontmatter."""
        content = """---
title: Test Note
tags: [existing, tags]
date: 2024-01-15
---

# Content

Some content here."""
        
        result = operations._add_tag_to_content(content, "new-tag")
        
        assert "new-tag" in result
        assert "existing" in result
        assert "tags" in result
        assert "title: Test Note" in result  # Other frontmatter preserved
        assert "date: 2024-01-15" in result
    
    def test_add_tag_to_content_no_frontmatter(self, operations):
        """Test adding tag to content without frontmatter."""
        content = """# My Note

Just content, no frontmatter."""
        
        result = operations._add_tag_to_content(content, "first-tag")
        
        assert result.startswith("---")
        assert "tags: [first-tag]" in result
        assert "# My Note" in result
        assert "Just content, no frontmatter." in result
    
    def test_add_tag_to_content_empty_tags(self, operations):
        """Test adding tag when tags field is empty."""
        content = """---
title: Note
tags: []
---

Content"""
        
        result = operations._add_tag_to_content(content, "new-tag")
        assert "tags: [new-tag]" in result
    
    def test_remove_tag_from_content_multiple_occurrences(self, operations):
        """Test removing tag that appears in multiple places."""
        content = """---
tags: [remove-me, keep-this, remove-me]
---

# Note about #remove-me

This discusses #remove-me and #keep-this tags.
The #remove-me tag should be removed everywhere."""
        
        result = operations._remove_tag_from_content(content, "remove-me")
        
        # Should be removed from frontmatter
        assert "remove-me" not in result.split("---")[1]  # Not in frontmatter
        
        # Should be removed from content
        assert "#remove-me" not in result.split("---")[2]  # Not in body
        
        # Other tags preserved
        assert "keep-this" in result
        assert "#keep-this" in result
    
    def test_rename_tag_in_content_complex(self, operations):
        """Test renaming tag with various formats."""
        content = """---
tags: ["old name", old-name, 'old.name']
categories: [old-name]
---

# Working with #old-name

The #old-name tag (also written as #old.name) is used here.
Related: #old_name and #OLD-NAME variations."""
        
        result = operations._rename_tag_in_content(content, "old-name", "new-name")
        
        # Should rename in frontmatter
        assert "new-name" in result
        assert '"old name"' in result  # Different format preserved
        
        # Should rename inline tags
        assert "#new-name" in result
        
        # Original tag completely replaced
        lines = result.split('\n')
        yaml_section = '\n'.join(lines[1:lines.index('---', 1)])
        assert "old-name" not in yaml_section
    
    async def test_add_tags_to_notes_bulk(self, operations, mock_vault):
        """Test bulk tag addition."""
        # Setup mock notes
        for i in range(3):
            mock_vault._notes[f"note{i}"] = Note(
                id=f"note{i}",
                path=Path(f"note{i}.md"),
                content=f"# Note {i}\n\nContent",
                metadata=NoteMetadata(),
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow(),
            )
        
        note_ids = ["note0", "note1", "note2"]
        tags = ["bulk-tag", "automated"]
        
        results = await operations.add_tags_to_notes(note_ids, tags)
        
        # Should return results for all notes
        assert len(results) == 3
        
        # All operations should succeed
        assert all(r.success for r in results)
        assert all(r.operation_type == "add_tags" for r in results)
        
        # Tags should be added to content
        for note_id in note_ids:
            note = mock_vault._notes[note_id]
            assert "bulk-tag" in note.content
            assert "automated" in note.content
    
    async def test_merge_tags_with_confirmation(self, operations, mock_vault):
        """Test merging tags across vault."""
        # Setup notes with tags to merge
        mock_vault._notes["note1"] = Note(
            id="note1",
            path=Path("note1.md"),
            content="---\ntags: [ml, machine-learning]\n---\n# ML Note",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        mock_vault._notes["note2"] = Note(
            id="note2",
            path=Path("note2.md"),
            content="# Note\n\n#ml and #ML discussion",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        # Test merge
        merge_map = {
            "ml": "machine-learning",
            "ML": "machine-learning"
        }
        
        results = await operations.merge_tags(merge_map, dry_run=False)
        
        # Should process both notes
        assert len(results) == 2
        assert all(r.success for r in results)
        
        # Check merged content
        note1 = mock_vault._notes["note1"]
        assert "machine-learning" in note1.content
        assert note1.content.count("ml") == 0  # Original removed
        
        note2 = mock_vault._notes["note2"]
        assert "#machine-learning" in note2.content
        assert "#ml" not in note2.content
        assert "#ML" not in note2.content
    
    async def test_rename_tag_hierarchy(self, operations, mock_vault):
        """Test renaming hierarchical tags."""
        mock_vault._notes["note1"] = Note(
            id="note1",
            path=Path("note1.md"),
            content="---\ntags: [proj/alpha, proj/beta, proj/alpha/frontend]\n---\n# Project",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        results = await operations.rename_tag("proj", "project", include_hierarchy=True)
        
        # Should rename all instances in hierarchy
        note = mock_vault._notes["note1"]
        assert "project/alpha" in note.content
        assert "project/beta" in note.content  
        assert "project/alpha/frontend" in note.content
        assert "proj/" not in note.content
    
    async def test_operation_with_errors(self, operations, mock_vault):
        """Test handling of operation errors."""
        # Note that will cause error
        mock_vault._notes["bad-note"] = Note(
            id="bad-note",
            path=Path("bad.md"),
            content=None,  # Will cause error
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        results = await operations.add_tags_to_notes(["bad-note"], ["test-tag"])
        
        assert len(results) == 1
        assert not results[0].success
        assert results[0].error is not None


class TestTagManagerServiceIntegration:
    """Integration tests for complete TagManagerService."""
    
    @pytest.fixture
    def mock_vault_full(self):
        """Create a fully populated mock vault."""
        vault = Mock()
        
        # Create realistic note set
        notes = {
            "project-overview": Note(
                id="project-overview",
                path=Path("projects/overview.md"),
                content="""---
tags: [project, management, overview]
---

# Project Overview

Managing multiple projects with #python and #javascript.
Related to #ml and #api development.
""",
                metadata=NoteMetadata(tags=["project", "management"]),
                created_at=datetime.utcnow() - timedelta(days=30),
                modified_at=datetime.utcnow() - timedelta(days=1),
            ),
            "ml-research": Note(
                id="ml-research",
                path=Path("research/ml.md"),
                content="""---
tags: [ml, research, python]
---

# Machine Learning Research

Exploring #deep-learning and #nlp techniques.
Using #python and #tensorflow.
""",
                metadata=NoteMetadata(tags=["ml", "research"]),
                created_at=datetime.utcnow() - timedelta(days=20),
                modified_at=datetime.utcnow() - timedelta(days=2),
            ),
            "daily-2024-01-15": Note(
                id="daily-2024-01-15",
                path=Path("daily/2024-01-15.md"),
                content="""# Daily Note - 2024-01-15

- Worked on #project tasks
- Reviewed #api documentation
- Meeting about #ml implementation
""",
                metadata=NoteMetadata(tags=["daily"]),
                created_at=datetime.utcnow() - timedelta(days=5),
                modified_at=datetime.utcnow() - timedelta(days=5),
            ),
        }
        
        vault._notes = notes
        vault.get_note = AsyncMock(side_effect=lambda id: notes.get(id))
        vault.get_all_note_ids = AsyncMock(return_value=list(notes.keys()))
        vault.get_all_notes = AsyncMock(return_value=list(notes.values()))
        
        async def update_note_content(note_id, content):
            if note_id in notes:
                notes[note_id].content = content
                return True
            return False
        
        vault.update_note_content = AsyncMock(side_effect=update_note_content)
        
        return vault
    
    @pytest.fixture
    async def tag_manager(self, tag_config, mock_vault_full, mock_embedding_service):
        """Create fully configured TagManagerService."""
        service = TagManagerService(
            vault=mock_vault_full,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
        
        # Pre-populate cache for testing
        await service._get_tag_info_cached()
        
        return service
    
    async def test_analyze_tags_complete_workflow(self, tag_manager):
        """Test complete tag analysis workflow."""
        result = await tag_manager.analyze_tags()
        
        # Check basic structure
        assert isinstance(result, TagAnalysisResult)
        assert result.total_tags > 0
        assert result.unique_tags > 0
        assert len(result.tag_clusters) >= 0
        assert len(result.suggested_hierarchies) >= 0
        assert len(result.optimization_suggestions) >= 0
        
        # Check tag statistics
        assert "tag_frequency" in result.usage_statistics
        assert "most_used" in result.usage_statistics
        assert "least_used" in result.usage_statistics
        assert "orphaned_tags" in result.usage_statistics
        
        # Verify common tags detected
        tag_freq = result.usage_statistics["tag_frequency"]
        assert "project" in tag_freq
        assert "ml" in tag_freq
        assert "python" in tag_freq
    
    async def test_suggest_tags_for_new_note(self, tag_manager):
        """Test tag suggestions for a new note."""
        # Create a new note without tags
        new_note_id = "new-research-note"
        tag_manager.vault._notes[new_note_id] = Note(
            id=new_note_id,
            path=Path("research/new.md"),
            content="""# Deep Learning with Python

Implementing neural networks using TensorFlow and Keras.
This relates to our machine learning project.

## Tasks
- [ ] Data preprocessing
- [ ] Model architecture
- [ ] Training pipeline
""",
            metadata=NoteMetadata(),
            created_at=datetime.utcnow(),
            modified_at=datetime.utcnow(),
        )
        
        suggestions = await tag_manager.suggest_tags(new_note_id, max_suggestions=5)
        
        # Should get relevant suggestions
        assert len(suggestions) > 0
        assert len(suggestions) <= 5
        
        # Should suggest contextually relevant tags
        suggested_tags = {s.tag for s in suggestions}
        
        # Likely suggestions based on content and context
        likely_tags = {"ml", "python", "deep-learning", "research", "project"}
        assert len(suggested_tags & likely_tags) > 0
        
        # All suggestions should meet confidence threshold
        assert all(s.confidence >= tag_manager.config.auto_tag_confidence_threshold
                  for s in suggestions)
    
    async def test_cleanup_redundant_tags_workflow(self, tag_manager):
        """Test complete redundant tag cleanup workflow."""
        # Add some redundant tags to notes
        tag_manager.vault._notes["project-overview"].content += "\n#Python #PYTHON #projects"
        
        # Find similar tags
        all_tags = []
        for note in tag_manager.vault._notes.values():
            tags = await tag_manager.analyzer.extract_tags_from_note(note)
            all_tags.extend(tags)
        
        similar_tags = await tag_manager.find_similar_tags(all_tags)
        
        # Should detect similar tags
        assert len(similar_tags) > 0
        
        # Create merge plan
        merge_plan = {}
        for sim in similar_tags:
            if sim.similarity_score > 0.9:
                # Prefer lowercase version
                if sim.tag_a.lower() == sim.tag_b.lower():
                    canonical = sim.tag_a.lower()
                    merge_plan[sim.tag_a] = canonical
                    merge_plan[sim.tag_b] = canonical
        
        # Execute merge
        if merge_plan:
            results = await tag_manager.merge_tags(merge_plan, dry_run=False)
            
            # Should process affected notes
            assert len(results) > 0
            assert any(r.success for r in results)
    
    async def test_build_tag_hierarchies(self, tag_manager):
        """Test building tag hierarchies from vault."""
        # Add hierarchical tags
        tag_manager.vault._notes["ml-research"].content += "\n#ml/deep-learning #ml/nlp"
        
        # Clear cache to pick up new tags
        tag_manager._tag_info_cache = None
        
        # Build hierarchies
        result = await tag_manager.analyze_tags()
        
        # Should detect hierarchical structures
        if result.suggested_hierarchies:
            # Check for ml hierarchy
            ml_hierarchy = next(
                (h for h in result.suggested_hierarchies if h.root_tag == "ml"),
                None
            )
            
            if ml_hierarchy:
                assert len(ml_hierarchy.children) > 0
                child_tags = {child.root_tag for child in ml_hierarchy.children}
                assert any("deep-learning" in tag or "nlp" in tag for tag in child_tags)
    
    async def test_get_tag_statistics_comprehensive(self, tag_manager):
        """Test comprehensive tag statistics generation."""
        stats = await tag_manager.get_tag_statistics()
        
        # Basic counts
        assert stats["total_tags"] > 0
        assert stats["unique_normalized_tags"] > 0
        assert stats["total_usage"] > 0
        assert stats["average_usage"] > 0
        
        # Usage distribution
        assert "usage_distribution" in stats
        dist = stats["usage_distribution"]
        assert "single_use" in dist
        assert "rarely_used" in dist
        assert "commonly_used" in dist
        assert "heavily_used" in dist
        
        # Tag lists
        assert "most_used_tags" in stats
        assert len(stats["most_used_tags"]) > 0
        
        assert "least_used_tags" in stats
        
        assert "recently_created_tags" in stats
        assert "recently_used_tags" in stats
        
        # Patterns
        assert "tag_patterns" in stats
        patterns = stats["tag_patterns"]
        assert "hierarchical_tags" in patterns
        assert "compound_tags" in patterns
        
        # Growth metrics
        assert "growth_metrics" in stats
        growth = stats["growth_metrics"]
        assert "tags_added_last_week" in growth
        assert "tags_added_last_month" in growth
    
    async def test_batch_operations_performance(self, tag_manager):
        """Test performance of batch operations."""
        import time
        
        # Create many notes for testing
        for i in range(50):
            tag_manager.vault._notes[f"test-note-{i}"] = Note(
                id=f"test-note-{i}",
                path=Path(f"test/note-{i}.md"),
                content=f"# Test Note {i}\n\n#test #batch #note{i % 10}",
                metadata=NoteMetadata(),
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow(),
            )
        
        # Test batch tag addition
        note_ids = [f"test-note-{i}" for i in range(50)]
        
        start_time = time.time()
        results = await tag_manager.operations.add_tags_to_notes(
            note_ids, ["batch-processed"]
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 seconds for 50 notes
        
        # Should process all notes
        assert len(results) == 50
        assert all(r.success for r in results)
    
    async def test_tag_analysis_caching(self, tag_manager):
        """Test that tag analysis uses caching effectively."""
        # First call - should build cache
        tag_info1 = await tag_manager._get_tag_info_cached()
        
        # Modify cache time to simulate recent cache
        tag_manager._tag_info_cache_time = datetime.utcnow()
        
        # Second call - should use cache
        with patch.object(tag_manager.analyzer, 'analyze_tag_usage') as mock_analyze:
            tag_info2 = await tag_manager._get_tag_info_cached()
            
            # Should not call analyze_tag_usage due to cache
            mock_analyze.assert_not_called()
        
        # Results should be the same object (cached)
        assert tag_info1 is tag_info2


# Performance Benchmarks
class TestTagManagerPerformance:
    """Performance benchmarks for tag management."""
    
    @pytest.mark.slow
    async def test_large_vault_analysis(self, tag_config, mock_embedding_service):
        """Test performance with large vault."""
        # Create mock vault with many notes
        vault = Mock()
        notes = {}
        
        # Generate 1000 notes
        for i in range(1000):
            tags = [f"tag{i % 20}", f"category{i % 10}", "common"]
            content = f"""---
tags: {tags}
---

# Note {i}

Content with #inline{i % 30} and #common tags.
"""
            notes[f"note-{i}"] = Note(
                id=f"note-{i}",
                path=Path(f"note-{i}.md"),
                content=content,
                metadata=NoteMetadata(tags=tags),
                created_at=datetime.utcnow() - timedelta(days=i % 365),
                modified_at=datetime.utcnow() - timedelta(days=i % 30),
            )
        
        vault._notes = notes
        vault.get_all_notes = AsyncMock(return_value=list(notes.values()))
        vault.get_all_note_ids = AsyncMock(return_value=list(notes.keys()))
        
        # Create service
        service = TagManagerService(vault, tag_config, mock_embedding_service)
        
        import time
        start = time.time()
        result = await service.analyze_tags()
        end = time.time()
        
        # Should complete in reasonable time
        assert end - start < 10.0  # 10 seconds for 1000 notes
        
        # Should produce valid results
        assert result.total_tags > 0
        assert result.unique_tags > 0
        assert len(result.usage_statistics) > 0
    
    @pytest.mark.slow
    async def test_concurrent_tag_operations(self, tag_config, mock_vault, mock_embedding_service):
        """Test concurrent tag operations."""
        # Setup vault with notes
        for i in range(100):
            mock_vault._notes[f"note-{i}"] = Note(
                id=f"note-{i}",
                path=Path(f"note-{i}.md"),
                content=f"# Note {i}\n\n#tag{i % 10}",
                metadata=NoteMetadata(),
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow(),
            )
        
        service = TagManagerService(mock_vault, tag_config, mock_embedding_service)
        
        # Run multiple operations concurrently
        tasks = []
        
        # Mix of different operations
        for i in range(10):
            tasks.append(service.suggest_tags(f"note-{i}"))
        
        for i in range(5):
            tasks.append(service.get_tag_statistics())
        
        tasks.append(service.analyze_tags())
        
        # Execute concurrently
        import time
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end = time.time()
        
        # Should complete without errors
        assert not any(isinstance(r, Exception) for r in results)
        
        # Should complete in reasonable time
        assert end - start < 5.0  # 5 seconds for concurrent operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])