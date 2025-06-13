"""
Unit tests for Tag Management Service.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

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
)


@pytest.fixture
def sample_note():
    """Create a sample note for testing."""
    metadata = NoteMetadata(
        title="Test Note",
        tags=["test", "sample"],
        created=datetime.utcnow(),
        modified=datetime.utcnow(),
    )
    
    content = """---
tags: [project, meeting, 2024]
---

# Test Note

This is a test note with #inline-tag and some content.

- [ ] Task with #todo tag
- [x] Completed task

## Meeting Notes

Discussion about #api-platform and #payments/fraud-detection.
"""
    
    return Note(
        id="test-note-001",
        path=Path("test-note.md"),
        content=content,
        metadata=metadata,
        created_at=datetime.utcnow(),
        modified_at=datetime.utcnow(),
        size_bytes=len(content),
        word_count=50,
    )


@pytest.fixture
def mock_vault():
    """Create a mock vault for testing."""
    vault = Mock()
    vault.get_note = AsyncMock()
    vault.get_all_note_ids = AsyncMock(return_value=["note1", "note2", "note3"])
    vault.update_note_content = AsyncMock()
    return vault


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = Mock()
    service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    service.compute_similarity = AsyncMock(return_value=0.8)
    return service


@pytest.fixture
def tag_config():
    """Create a test tag manager configuration."""
    return TagManagerConfig(
        fuzzy_similarity_threshold=0.8,
        semantic_similarity_threshold=0.7,
        enable_fuzzy_matching=True,
        enable_semantic_analysis=True,
        case_insensitive=True,
        min_usage_threshold=2,
        auto_tag_confidence_threshold=0.7,
        max_auto_tags_per_note=5,
    )


class TestTagAnalyzer:
    """Test the TagAnalyzer component."""
    
    def test_init(self, tag_config):
        """Test TagAnalyzer initialization."""
        analyzer = TagAnalyzer(tag_config)
        assert analyzer.config == tag_config
        assert len(analyzer._pattern_cache) > 0
        assert 'frontmatter_tags' in analyzer._pattern_cache
    
    async def test_extract_frontmatter_tags(self, tag_config, sample_note):
        """Test frontmatter tag extraction."""
        analyzer = TagAnalyzer(tag_config)
        tags = analyzer._extract_frontmatter_tags(sample_note.content)
        
        assert "project" in tags
        assert "meeting" in tags
        assert "2024" in tags
        assert len(tags) == 3
    
    async def test_extract_inline_tags(self, tag_config, sample_note):
        """Test inline tag extraction."""
        analyzer = TagAnalyzer(tag_config)
        tags = analyzer._extract_inline_tags(sample_note.content)
        
        assert "inline-tag" in tags
        assert "todo" in tags
        assert "api-platform" in tags
        assert "payments/fraud-detection" in tags
    
    async def test_extract_tags_from_note(self, tag_config, sample_note):
        """Test complete tag extraction from note."""
        analyzer = TagAnalyzer(tag_config)
        
        with patch.object(analyzer, '_extract_content_based_tags', 
                         return_value=[]):
            tags = await analyzer.extract_tags_from_note(sample_note)
        
        # Should contain both frontmatter and inline tags
        assert "project" in tags
        assert "inline-tag" in tags
        assert "api-platform" in tags
        assert len(tags) > 0
    
    async def test_analyze_tag_usage(self, tag_config, sample_note):
        """Test tag usage analysis across notes."""
        analyzer = TagAnalyzer(tag_config)
        
        # Create multiple notes with overlapping tags
        notes = [sample_note]
        
        with patch.object(analyzer, 'extract_tags_from_note',
                         return_value=["project", "meeting", "api"]):
            tag_info = await analyzer.analyze_tag_usage(notes)
        
        assert "project" in tag_info
        assert tag_info["project"].usage_count == 1
        assert sample_note.id in tag_info["project"].notes
    
    def test_normalize_tags(self, tag_config):
        """Test tag normalization."""
        analyzer = TagAnalyzer(tag_config)
        
        raw_tags = ["Project", "API-Platform", "Special@Chars!", "  spaced  "]
        normalized = analyzer._normalize_tags(raw_tags)
        
        # Should be lowercase and cleaned
        assert "project" in normalized
        assert "api-platform" in normalized
        assert "specialchars" in normalized
        assert "spaced" in normalized


class TestTagSimilarityDetector:
    """Test the TagSimilarityDetector component."""
    
    def test_init(self, tag_config, mock_embedding_service):
        """Test TagSimilarityDetector initialization."""
        detector = TagSimilarityDetector(tag_config, mock_embedding_service)
        assert detector.config == tag_config
        assert detector.embedding_service == mock_embedding_service
    
    def test_calculate_fuzzy_similarity(self, tag_config):
        """Test fuzzy string similarity calculation."""
        detector = TagSimilarityDetector(tag_config)
        
        # Test similar tags
        score = detector._calculate_fuzzy_similarity("project", "projects")
        assert score > 0.8
        
        # Test dissimilar tags
        score = detector._calculate_fuzzy_similarity("project", "meeting")
        assert score < 0.5
        
        # Test identical tags
        score = detector._calculate_fuzzy_similarity("api", "api")
        assert score == 1.0
    
    async def test_calculate_semantic_similarity(self, tag_config, mock_embedding_service):
        """Test semantic similarity calculation."""
        detector = TagSimilarityDetector(tag_config, mock_embedding_service)
        
        score = await detector._calculate_semantic_similarity("project", "task")
        assert score == 0.8  # Mocked return value
        
        # Verify embedding service was called
        mock_embedding_service.embed_text.assert_called()
        mock_embedding_service.compute_similarity.assert_called_once()
    
    async def test_calculate_similarity(self, tag_config, mock_embedding_service):
        """Test overall similarity calculation."""
        detector = TagSimilarityDetector(tag_config, mock_embedding_service)
        
        similarity = await detector.calculate_similarity("project", "projects")
        
        assert similarity is not None
        assert similarity.tag_a == "project"
        assert similarity.tag_b == "projects" 
        assert similarity.similarity_score > 0.8
        assert similarity.similarity_type in ["fuzzy", "semantic"]
    
    async def test_find_similar_tags(self, tag_config):
        """Test finding similar tags in a list."""
        detector = TagSimilarityDetector(tag_config)
        
        tags = ["project", "projects", "meeting", "meetings", "api", "api-platform"]
        
        with patch.object(detector, 'calculate_similarity') as mock_calc:
            # Mock some similarities above threshold
            mock_calc.side_effect = [
                TagSimilarity("project", "projects", 0.9, "fuzzy"),
                TagSimilarity("meeting", "meetings", 0.9, "fuzzy"),
                None,  # Below threshold
                None,  # Below threshold
                None,  # Below threshold
            ]
            
            similarities = await detector.find_similar_tags(tags)
        
        assert len(similarities) == 2
        assert all(sim.similarity_score >= 0.8 for sim in similarities)


class TestAutoTagger:
    """Test the AutoTagger component."""
    
    def test_init(self, tag_config, mock_embedding_service):
        """Test AutoTagger initialization."""
        tagger = AutoTagger(tag_config, mock_embedding_service)
        assert tagger.config == tag_config
        assert tagger.embedding_service == mock_embedding_service
    
    async def test_suggest_from_content(self, tag_config, sample_note):
        """Test content-based tag suggestions."""
        tagger = AutoTagger(tag_config)
        
        suggestions = await tagger._suggest_from_content(sample_note, [])
        
        # Should find meeting-related content
        meeting_suggestions = [s for s in suggestions if s.tag == "meeting"]
        assert len(meeting_suggestions) > 0
        assert meeting_suggestions[0].source == "content"
    
    async def test_suggest_from_patterns(self, tag_config, sample_note):
        """Test pattern-based tag suggestions."""
        tagger = AutoTagger(tag_config)
        
        suggestions = await tagger._suggest_from_patterns(sample_note, [])
        
        # Should detect checklist pattern
        checklist_suggestions = [s for s in suggestions if s.tag == "checklist"]
        assert len(checklist_suggestions) > 0
        assert checklist_suggestions[0].source == "pattern"
    
    async def test_suggest_tags_for_note(self, tag_config, sample_note, mock_embedding_service):
        """Test complete tag suggestion for a note."""
        tagger = AutoTagger(tag_config, mock_embedding_service)
        
        existing_tags = ["existing"]
        context_tags = {
            "project": TagInfo(
                name="project",
                normalized_name="project", 
                usage_count=5,
                first_seen=datetime.utcnow(),
                last_used=datetime.utcnow()
            )
        }
        
        with patch.object(tagger, '_suggest_from_ai', return_value=[]):
            suggestions = await tagger.suggest_tags_for_note(
                sample_note, existing_tags, context_tags
            )
        
        assert len(suggestions) > 0
        assert all(s.confidence >= tag_config.auto_tag_confidence_threshold 
                  for s in suggestions)
        assert all(s.tag not in existing_tags for s in suggestions)


class TestTagOperations:
    """Test the TagOperations component."""
    
    def test_init(self, tag_config, mock_vault):
        """Test TagOperations initialization."""
        operations = TagOperations(tag_config, mock_vault)
        assert operations.config == tag_config
        assert operations.vault == mock_vault
    
    def test_add_tag_to_content(self, tag_config, mock_vault):
        """Test adding tag to note content."""
        operations = TagOperations(tag_config, mock_vault)
        
        # Test adding to existing frontmatter
        content_with_fm = """---
tags: [existing]
---

Content here."""
        
        result = operations._add_tag_to_content(content_with_fm, "new-tag")
        assert "new-tag" in result
        assert "existing" in result
        
        # Test adding to content without frontmatter
        content_no_fm = "Just some content"
        result = operations._add_tag_to_content(content_no_fm, "new-tag")
        assert "---" in result
        assert "tags: [new-tag]" in result
    
    def test_remove_tag_from_content(self, tag_config, mock_vault):
        """Test removing tag from note content."""
        operations = TagOperations(tag_config, mock_vault)
        
        content = """---
tags: [keep, remove, also-keep]
---

Content with #remove and #keep tags."""
        
        result = operations._remove_tag_from_content(content, "remove")
        assert "remove" not in result
        assert "keep" in result
        assert "also-keep" in result
    
    def test_rename_tag_in_content(self, tag_config, mock_vault):
        """Test renaming tag in note content."""
        operations = TagOperations(tag_config, mock_vault)
        
        content = """---
tags: [old-name, other]
---

Content with #old-name tag."""
        
        result = operations._rename_tag_in_content(content, "old-name", "new-name")
        assert "old-name" not in result
        assert "new-name" in result
        assert "other" in result


class TestTagManagerService:
    """Test the main TagManagerService."""
    
    def test_init(self, tag_config, mock_vault, mock_embedding_service):
        """Test TagManagerService initialization."""
        service = TagManagerService(
            vault=mock_vault,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
        
        assert service.vault == mock_vault
        assert service.config == tag_config
        assert service.embedding_service == mock_embedding_service
        assert isinstance(service.analyzer, TagAnalyzer)
        assert isinstance(service.similarity_detector, TagSimilarityDetector)
        assert isinstance(service.auto_tagger, AutoTagger)
        assert isinstance(service.operations, TagOperations)
    
    async def test_suggest_tags(self, tag_config, mock_vault, mock_embedding_service, sample_note):
        """Test tag suggestions for a note."""
        service = TagManagerService(
            vault=mock_vault,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
        
        mock_vault.get_note.return_value = sample_note
        
        with patch.object(service.analyzer, 'extract_tags_from_note',
                         return_value=["existing"]):
            with patch.object(service.auto_tagger, 'suggest_tags_for_note',
                             return_value=[
                                 TagSuggestion("suggested1", 0.9, "reason", "ai"),
                                 TagSuggestion("suggested2", 0.8, "reason", "content"),
                             ]):
                suggestions = await service.suggest_tags("test-note-001", max_suggestions=5)
        
        assert len(suggestions) == 2
        assert suggestions[0].tag == "suggested1"
        assert suggestions[0].confidence == 0.9
    
    async def test_find_similar_tags(self, tag_config, mock_vault, mock_embedding_service):
        """Test finding similar tags."""
        service = TagManagerService(
            vault=mock_vault,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
        
        tags = ["project", "projects", "meeting"]
        
        with patch.object(service.similarity_detector, 'find_similar_tags',
                         return_value=[
                             TagSimilarity("project", "projects", 0.9, "fuzzy")
                         ]):
            similarities = await service.find_similar_tags(tags)
        
        assert len(similarities) == 1
        assert similarities[0].similarity_score == 0.9
    
    async def test_get_tag_statistics(self, tag_config, mock_vault, mock_embedding_service):
        """Test getting tag statistics."""
        service = TagManagerService(
            vault=mock_vault,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
        
        # Mock the cached tag info
        mock_tag_info = {
            "project": TagInfo("project", "project", 5, datetime.utcnow(), datetime.utcnow()),
            "meeting": TagInfo("meeting", "meeting", 3, datetime.utcnow(), datetime.utcnow()),
            "api": TagInfo("api", "api", 1, datetime.utcnow(), datetime.utcnow()),
        }
        
        with patch.object(service, '_get_tag_info_cached', return_value=mock_tag_info):
            stats = await service.get_tag_statistics()
        
        assert stats["total_tags"] == 3
        assert stats["total_usage"] == 9
        assert stats["average_usage"] == 3.0
        assert "usage_distribution" in stats
        assert "most_used_tags" in stats


@pytest.mark.integration
class TestTagManagerIntegration:
    """Integration tests for tag manager components."""
    
    async def test_full_tag_analysis_workflow(self, tag_config, mock_vault, mock_embedding_service):
        """Test complete tag analysis workflow."""
        service = TagManagerService(
            vault=mock_vault,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
        
        # Mock multiple notes with various tags
        notes = [
            Note(
                id=f"note-{i}",
                path=Path(f"note-{i}.md"),
                content=f"---\ntags: [tag{i}, common, test]\n---\nContent {i}",
                metadata=NoteMetadata(tags=[f"tag{i}", "common"]),
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow(),
            )
            for i in range(3)
        ]
        
        mock_vault.get_note.side_effect = lambda note_id: next(
            (note for note in notes if note.id == note_id), None
        )
        mock_vault.get_all_note_ids.return_value = [note.id for note in notes]
        
        # Mock similarity detection
        with patch.object(service.similarity_detector, 'cluster_similar_tags',
                         return_value=[]):
            with patch.object(service.hierarchy_builder, 'build_hierarchies',
                             return_value=[]):
                result = await service.analyze_tags()
        
        assert result.total_tags > 0
        assert isinstance(result.usage_statistics, dict)
        assert isinstance(result.optimization_suggestions, list)
        assert result.analysis_timestamp is not None


if __name__ == "__main__":
    pytest.main([__file__])