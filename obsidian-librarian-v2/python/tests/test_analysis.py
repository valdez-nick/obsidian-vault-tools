"""
Unit tests for analysis service.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

import numpy as np

from obsidian_librarian.services.analysis import (
    AnalysisService, AnalysisConfig, ContentMetrics, AnalysisResult,
    ContentSimilarity, DuplicateCluster
)
from obsidian_librarian.models import Note, NoteMetadata


@pytest.fixture
def analysis_config():
    """Create test analysis configuration."""
    return AnalysisConfig(
        exact_duplicate_threshold=0.98,
        near_duplicate_threshold=0.85,
        similar_content_threshold=0.7,
        min_content_length=50,
        batch_size=10,
        enable_quality_scoring=True,
    )


@pytest.fixture
def mock_vault():
    """Create a mock vault."""
    vault = Mock()
    vault.path = "/test/vault"
    vault.get_note = AsyncMock()
    vault.get_all_note_ids = AsyncMock(return_value=["note1", "note2", "note3"])
    return vault


@pytest.fixture
def sample_notes():
    """Create sample notes for testing."""
    notes = []
    
    # Note 1: Well-structured note
    note1 = Note(
        id="note1",
        path="/test/note1.md",
        content="""---
title: Machine Learning Basics
tags: [ml, ai, tutorial]
---

# Machine Learning Basics

## Introduction

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Key Concepts

- **Supervised Learning**: Learning from labeled data
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through interaction

## Examples

Here's a simple example of linear regression:

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

## Conclusion

Machine learning is transforming how we build intelligent systems.

[[Deep Learning]] [[Neural Networks]]
""",
        metadata=NoteMetadata(
            title="Machine Learning Basics",
            tags=["ml", "ai", "tutorial"],
        ),
        word_count=100,
        size_bytes=500,
    )
    
    # Note 2: Similar to Note 1 (near duplicate)
    note2 = Note(
        id="note2",
        path="/test/note2.md",
        content="""# Machine Learning Fundamentals

## Overview

Machine learning is an AI technique that allows systems to learn from data.

## Main Types

- Supervised Learning: Uses labeled data
- Unsupervised Learning: Discovers patterns
- Reinforcement Learning: Learns from feedback

## Code Example

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
```

Machine learning is revolutionizing technology.

[[Deep Learning]]
""",
        metadata=NoteMetadata(title="Machine Learning Fundamentals"),
        word_count=80,
        size_bytes=400,
    )
    
    # Note 3: Different topic
    note3 = Note(
        id="note3",
        path="/test/note3.md",
        content="""# Project Management Best Practices

- [ ] Define clear objectives
- [x] Create timeline
- [ ] Assign responsibilities

Managing projects effectively requires planning and communication.
""",
        metadata=NoteMetadata(tags=["project", "management"]),
        word_count=30,
        size_bytes=200,
    )
    
    notes.extend([note1, note2, note3])
    return notes


@pytest.mark.asyncio
async def test_analysis_service_initialization(mock_vault, analysis_config):
    """Test analysis service initialization."""
    service = AnalysisService(mock_vault, analysis_config)
    
    assert service.vault == mock_vault
    assert service.config == analysis_config
    assert service.content_summarizer is not None
    assert len(service._content_cache) == 0
    assert len(service._similarity_cache) == 0


@pytest.mark.asyncio
async def test_extract_content_metrics(mock_vault, analysis_config, sample_notes):
    """Test content metrics extraction."""
    service = AnalysisService(mock_vault, analysis_config)
    
    note = sample_notes[0]  # Well-structured note
    metrics = await service._extract_content_metrics(note)
    
    assert isinstance(metrics, ContentMetrics)
    assert metrics.word_count > 0
    assert metrics.heading_count == 5  # H1 + 4 H2s
    assert metrics.link_count == 2  # Two wiki links
    assert metrics.code_block_count == 1
    assert metrics.has_frontmatter == True
    assert metrics.has_headings == True
    assert metrics.has_links == True
    assert metrics.heading_hierarchy_score > 0.8  # Good hierarchy


@pytest.mark.asyncio
async def test_quality_score_calculation(mock_vault, analysis_config, sample_notes):
    """Test quality score calculation."""
    service = AnalysisService(mock_vault, analysis_config)
    
    # Test well-structured note
    note1 = sample_notes[0]
    metrics1 = await service._extract_content_metrics(note1)
    score1 = service._calculate_quality_score(metrics1)
    assert score1 > 0.7  # Should have high quality score
    
    # Test poorly structured note
    note3 = sample_notes[2]
    metrics3 = await service._extract_content_metrics(note3)
    score3 = service._calculate_quality_score(metrics3)
    assert score3 < score1  # Should have lower score


@pytest.mark.asyncio
async def test_analyze_note(mock_vault, analysis_config, sample_notes):
    """Test analyzing a single note."""
    service = AnalysisService(mock_vault, analysis_config)
    
    note = sample_notes[0]
    mock_vault.get_note.return_value = note
    
    # Mock similar notes
    with pytest.mock.patch.object(service, 'find_similar_notes', new_callable=AsyncMock) as mock_similar:
        mock_similar.return_value = [
            ContentSimilarity(
                note_a="note1",
                note_b="note2",
                similarity_score=0.85,
                similarity_type="content",
            )
        ]
        
        result = await service.analyze_note("note1")
        
        assert isinstance(result, AnalysisResult)
        assert result.note_id == "note1"
        assert result.quality_score > 0.7
        assert len(result.similar_notes) == 1
        assert len(result.topics) > 0
        assert len(result.recommendations) >= 0


@pytest.mark.asyncio
async def test_find_duplicates(mock_vault, analysis_config, sample_notes):
    """Test duplicate detection."""
    service = AnalysisService(mock_vault, analysis_config)
    
    # Mock vault to return sample notes
    async def get_note_side_effect(note_id):
        return next((n for n in sample_notes if n.id == note_id), None)
    
    mock_vault.get_note.side_effect = get_note_side_effect
    
    # Mock feature matrix building
    with pytest.mock.patch.object(service, '_build_feature_matrix', new_callable=AsyncMock):
        # Create mock similarity matrix showing notes 1 and 2 are similar
        service._note_ids = ["note1", "note2", "note3"]
        service._feature_matrix = np.array([
            [1.0, 0.0, 0.0],  # note1 vector
            [0.9, 0.1, 0.0],  # note2 vector (similar to note1)
            [0.0, 0.0, 1.0],  # note3 vector (different)
        ])
        
        clusters = await service.find_duplicates(["note1", "note2", "note3"])
        
        assert len(clusters) >= 1
        assert isinstance(clusters[0], DuplicateCluster)
        
        # Should find notes 1 and 2 as near duplicates
        cluster = clusters[0]
        assert "note1" in cluster.note_ids
        assert "note2" in cluster.note_ids
        assert cluster.cluster_type in ["near_duplicate", "exact_duplicate"]


@pytest.mark.asyncio
async def test_batch_analyze(mock_vault, analysis_config, sample_notes):
    """Test batch analysis of multiple notes."""
    service = AnalysisService(mock_vault, analysis_config)
    
    # Mock get_note to return sample notes
    async def get_note_side_effect(note_id):
        return next((n for n in sample_notes if n.id == note_id), None)
    
    mock_vault.get_note.side_effect = get_note_side_effect
    
    # Mock similar notes to avoid complex calculations
    with pytest.mock.patch.object(service, 'find_similar_notes', new_callable=AsyncMock) as mock_similar:
        mock_similar.return_value = []
        
        results = []
        async for result in service.batch_analyze(["note1", "note2", "note3"]):
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert results[0].note_id == "note1"
        assert results[1].note_id == "note2"
        assert results[2].note_id == "note3"


@pytest.mark.asyncio
async def test_content_statistics(mock_vault, analysis_config, sample_notes):
    """Test vault content statistics generation."""
    service = AnalysisService(mock_vault, analysis_config)
    
    # Mock get_note to return sample notes
    async def get_note_side_effect(note_id):
        return next((n for n in sample_notes if n.id == note_id), None)
    
    mock_vault.get_note.side_effect = get_note_side_effect
    mock_vault.get_all_note_ids.return_value = ["note1", "note2", "note3"]
    
    stats = await service.get_content_statistics()
    
    assert stats['total_notes'] == 3
    assert stats['total_words'] > 0
    assert stats['total_links'] > 0
    assert 'content_distribution' in stats


@pytest.mark.asyncio
async def test_recommendations_generation(mock_vault, analysis_config):
    """Test recommendation generation for notes."""
    service = AnalysisService(mock_vault, analysis_config)
    
    # Create a note with issues
    note = Note(
        id="problematic",
        path="/test/problematic.md",
        content="Short note without structure",
        metadata=NoteMetadata(),
        word_count=5,
        size_bytes=30,
    )
    
    metrics = await service._extract_content_metrics(note)
    recommendations = service._generate_recommendations(note, metrics, [])
    
    assert len(recommendations) > 0
    assert any("expanding" in r for r in recommendations)  # Should recommend expansion
    assert any("heading" in r for r in recommendations)  # Should recommend headings
    assert any("link" in r for r in recommendations)  # Should recommend links


@pytest.mark.asyncio
async def test_topic_extraction(mock_vault, analysis_config):
    """Test topic extraction from content."""
    service = AnalysisService(mock_vault, analysis_config)
    
    content = """
    Machine learning algorithms are transforming artificial intelligence.
    Deep learning models use neural networks for pattern recognition.
    Python programming is essential for data science and machine learning.
    """
    
    topics = service._extract_topics(content)
    
    assert len(topics) > 0
    assert "machine" in topics or "learning" in topics
    assert len(topics) <= 10  # Should limit topics


def test_heading_hierarchy_score():
    """Test heading hierarchy scoring."""
    service = AnalysisService(Mock(), AnalysisConfig())
    
    # Good hierarchy
    good_content = """
# Title
## Section 1
### Subsection 1.1
## Section 2
### Subsection 2.1
### Subsection 2.2
"""
    score = service._calculate_heading_hierarchy_score(good_content)
    assert score > 0.9  # Should be nearly perfect
    
    # Bad hierarchy (jumping levels)
    bad_content = """
# Title
#### Deep subsection (skipping H2 and H3)
## Section
##### Very deep (skipping H3 and H4)
"""
    score = service._calculate_heading_hierarchy_score(bad_content)
    assert score < 0.5  # Should be low due to violations