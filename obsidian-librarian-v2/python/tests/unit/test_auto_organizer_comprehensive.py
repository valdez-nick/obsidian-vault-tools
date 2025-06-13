"""
Comprehensive unit tests for Directory Organization Service.

This test suite provides extensive coverage for all auto-organization components:
- ContentClassifier: Content analysis and classification
- DirectoryRouter: Smart file routing and placement
- OrganizationLearner: Pattern learning from user feedback
- RuleEngine: Rule-based organization logic
- FileWatcher: Real-time file monitoring
- AutoOrganizer: Main service orchestration
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import json

from obsidian_librarian.services.auto_organizer import (
    AutoOrganizer,
    ContentClassifier,
    DirectoryRouter,
    OrganizationLearner,
    RuleEngine,
    FileWatcher,
    ClassificationResult,
    OrganizationRule,
    UserFeedback,
    ContentFeatures,
    ClassificationConfidence,
    OrganizationAction,
    DirectoryStructure,
    LearnedPattern,
)
from obsidian_librarian.models import Note, NoteMetadata, LibrarianConfig
from obsidian_librarian.vault import Vault
from obsidian_librarian.ai import ContentAnalyzer, EmbeddingService


class TestContentClassifierComprehensive:
    """Comprehensive tests for ContentClassifier."""
    
    @pytest.fixture
    def mock_config(self):
        """Create comprehensive test configuration."""
        return LibrarianConfig(
            vault_path=Path("/test/vault"),
            enable_ai_features=True,
            auto_organization={
                "enabled": True,
                "confidence_threshold": 0.7,
                "require_confirmation": False,
                "excluded_paths": ["Templates", "Archive", ".obsidian"],
                "date_patterns": {
                    "daily": r"\d{4}-\d{2}-\d{2}",
                    "weekly": r"\d{4}-W\d{2}",
                    "monthly": r"\d{4}-\d{2}",
                },
                "content_patterns": {
                    "meeting": ["meeting", "agenda", "minutes", "attendees"],
                    "project": ["project", "milestone", "deliverable", "timeline"],
                    "research": ["research", "study", "analysis", "findings"],
                },
            }
        )
    
    @pytest.fixture
    def mock_content_analyzer(self):
        """Create mock content analyzer with realistic behavior."""
        analyzer = Mock(spec=ContentAnalyzer)
        
        async def analyze(content):
            result = Mock()
            # Simulate topic detection based on content
            if "meeting" in content.lower():
                result.topics = ["meetings", "collaboration"]
            elif "project" in content.lower():
                result.topics = ["project management", "planning"]
            elif "research" in content.lower():
                result.topics = ["research", "analysis"]
            else:
                result.topics = ["general", "notes"]
            
            # Extract keywords
            words = content.lower().split()
            result.keywords = list(set(w for w in words if len(w) > 4))[:10]
            
            # Detect entities
            result.entities = []
            if "TODO" in content or "todo" in content:
                result.entities.append("task-list")
            if any(name in content for name in ["Alice", "Bob", "Carol"]):
                result.entities.extend(["person-mention"])
                
            return result
        
        analyzer.analyze_content = AsyncMock(side_effect=analyze)
        return analyzer
    
    @pytest.fixture
    def classifier(self, mock_vault_session, mock_content_analyzer, mock_embedding_service, mock_config):
        """Create ContentClassifier instance."""
        return ContentClassifier(
            mock_vault_session,
            mock_content_analyzer,
            mock_embedding_service,
            mock_config
        )
    
    async def test_extract_features_comprehensive(self, classifier):
        """Test comprehensive feature extraction from notes."""
        note = Note(
            path=Path("complex-note.md"),
            content="""---
title: Complex Note Example
tags: [project, important]
created: 2024-01-15
type: meeting
attendees: [Alice, Bob, Carol]
---

# Team Meeting - Project Alpha

## Agenda
1. Project status update
2. Technical decisions
3. Next steps

## Discussion Points

### Current Status
- Development is **on track**
- 3 features completed
- 2 bugs fixed

### Technical Decisions
We decided to use [[Python]] for the backend and [[React]] for frontend.
See also: [[Architecture Decisions]]

### Action Items
- [ ] Alice: Review code changes
- [x] Bob: Update documentation  
- [ ] Carol: Schedule next meeting
- [ ] Team: Complete sprint tasks

## Code Examples

```python
def process_data(items):
    return [transform(item) for item in items]
```

```javascript
const Component = () => {
    return <div>Hello</div>;
};
```

## Resources
- External link: [Documentation](https://example.com)
- Internal link: [[Project Resources]]
- Image: ![[architecture.png]]

---
*Meeting duration: 1 hour*
""",
            metadata=NoteMetadata(
                created_time=datetime(2024, 1, 15),
                modified_time=datetime(2024, 1, 16),
                tags=["project", "important"],
                frontmatter={
                    "type": "meeting",
                    "attendees": ["Alice", "Bob", "Carol"]
                }
            )
        )
        
        features = await classifier.extract_features(note)
        
        # Basic metrics
        assert features.word_count > 100
        assert features.line_count > 20
        assert features.header_count >= 6  # Multiple levels of headers
        
        # List detection
        assert features.list_count >= 2  # Ordered and unordered lists
        assert features.checklist_count == 4  # Task items
        assert features.completed_tasks == 1
        assert features.task_completion_ratio == 0.25
        
        # Link analysis
        assert features.link_count >= 5  # Wiki links
        assert features.external_link_count >= 1
        assert "Python" in features.linked_notes
        assert "React" in features.linked_notes
        
        # Code detection
        assert features.code_block_count == 2
        assert "python" in features.code_languages
        assert "javascript" in features.code_languages
        
        # Metadata
        assert features.has_frontmatter
        assert "project" in features.tags
        assert features.frontmatter["type"] == "meeting"
        assert len(features.frontmatter["attendees"]) == 3
        
        # Date features
        assert features.creation_date == datetime(2024, 1, 15)
        assert features.is_daily_note is False  # Doesn't match daily pattern
        
        # Content analysis
        assert "meetings" in features.topics or "collaboration" in features.topics
        assert len(features.keywords) > 0
        assert "task-list" in features.entities
        assert "person-mention" in features.entities
    
    async def test_classify_daily_note_patterns(self, classifier):
        """Test classification of various daily note patterns."""
        patterns = [
            ("2024-01-15.md", "Daily Notes/2024/01"),
            ("2024-01-15 Monday.md", "Daily Notes/2024/01"),
            ("Daily Note 2024-01-15.md", "Daily Notes/2024/01"),
            ("01-15-2024.md", "Daily Notes/2024/01"),
            ("2024.01.15.md", "Daily Notes/2024/01"),
        ]
        
        for filename, expected_path in patterns:
            note = Note(
                path=Path(filename),
                content=f"# {filename}\n\nDaily thoughts and tasks.",
                metadata=NoteMetadata(
                    created_time=datetime(2024, 1, 15),
                    modified_time=datetime(2024, 1, 15)
                )
            )
            
            result = await classifier.classify_content(note)
            
            assert result.confidence >= ClassificationConfidence.HIGH
            assert "Daily Notes" in str(result.suggested_path)
            assert result.action == OrganizationAction.MOVE
            assert "daily note pattern" in result.reasoning.lower()
    
    async def test_classify_meeting_notes_patterns(self, classifier):
        """Test classification of meeting notes."""
        meeting_contents = [
            """# Team Standup
## Attendees
- Alice
- Bob

## Agenda
1. Status updates
2. Blockers""",
            
            """# Client Meeting - Project X
Date: 2024-01-15
Participants: Team A, Client B

## Discussion Points
- Requirements review
- Timeline adjustments""",
            
            """Meeting Minutes - Board Meeting
Date: January 15, 2024

Present: All board members

Decisions:
1. Approved budget
2. New hire approved"""
        ]
        
        for i, content in enumerate(meeting_contents):
            note = Note(
                path=Path(f"meeting-{i}.md"),
                content=content,
                metadata=NoteMetadata(
                    created_time=datetime.utcnow(),
                    modified_time=datetime.utcnow()
                )
            )
            
            result = await classifier.classify_content(note)
            
            assert result.confidence >= ClassificationConfidence.MEDIUM
            assert "meeting" in str(result.suggested_path).lower() or \
                   "meeting" in result.reasoning.lower()
            assert result.action in [OrganizationAction.MOVE, OrganizationAction.SUGGEST]
    
    async def test_classify_project_documentation(self, classifier):
        """Test classification of project-related notes."""
        note = Note(
            path=Path("new-feature-spec.md"),
            content="""# New Feature Specification

## Project: Alpha
## Sprint: 23

### Requirements
- Feature must support real-time updates
- Must integrate with existing API
- Performance target: <100ms response

### Milestones
1. Design complete: Jan 20
2. Implementation: Feb 1  
3. Testing: Feb 15
4. Release: Mar 1

### Deliverables
- [ ] Technical design doc
- [ ] API specifications
- [ ] Test plan
- [ ] User documentation""",
            metadata=NoteMetadata(
                tags=["project", "specification"],
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        
        result = await classifier.classify_content(note)
        
        assert result.confidence >= ClassificationConfidence.HIGH
        assert "project" in str(result.suggested_path).lower()
        assert "project documentation" in result.reasoning.lower() or \
               "project" in result.reasoning.lower()
        assert result.metadata.get("detected_project") == "Alpha"
        assert result.metadata.get("has_milestones") is True
    
    async def test_classify_research_notes(self, classifier):
        """Test classification of research notes."""
        note = Note(
            path=Path("ml-research.md"),
            content="""# Machine Learning Research

## Abstract
This study investigates the application of neural networks...

## Literature Review
Recent papers have shown [[Smith2023]] and [[Jones2024]]...

## Methodology
We employed a quantitative analysis using...

## Findings
1. Model accuracy improved by 15%
2. Training time reduced by 30%

## References
- Smith et al. (2023)
- Jones et al. (2024)""",
            metadata=NoteMetadata(
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        
        result = await classifier.classify_content(note)
        
        assert "research" in str(result.suggested_path).lower()
        assert result.confidence >= ClassificationConfidence.MEDIUM
        assert "research" in result.reasoning.lower()
    
    async def test_classify_template_detection(self, classifier):
        """Test that templates are not moved."""
        template_locations = [
            Path("Templates/daily.md"),
            Path("templates/meeting.md"),
            Path("_templates/project.md"),
        ]
        
        for path in template_locations:
            note = Note(
                path=path,
                content="# {{title}}\n\nDate: {{date}}\n\n## Notes\n{{content}}",
                metadata=NoteMetadata(
                    created_time=datetime.utcnow(),
                    modified_time=datetime.utcnow()
                )
            )
            
            result = await classifier.classify_content(note)
            
            assert result.action == OrganizationAction.IGNORE
            assert result.confidence == ClassificationConfidence.HIGH
            assert "template" in result.reasoning.lower()
    
    async def test_classify_archived_content(self, classifier):
        """Test that archived content is not moved."""
        archive_paths = [
            Path("Archive/old-note.md"),
            Path("archive/2023/note.md"),
            Path("Archived Projects/project.md"),
        ]
        
        for path in archive_paths:
            note = Note(
                path=path,
                content="# Archived Content",
                metadata=NoteMetadata(
                    created_time=datetime.utcnow() - timedelta(days=365),
                    modified_time=datetime.utcnow() - timedelta(days=300)
                )
            )
            
            result = await classifier.classify_content(note)
            
            assert result.action == OrganizationAction.IGNORE
            assert "archive" in result.reasoning.lower()
    
    async def test_classify_confidence_levels(self, classifier):
        """Test different confidence level scenarios."""
        # High confidence - clear pattern
        daily_note = Note(
            path=Path("2024-01-15.md"),
            content="# Daily Note\n\nToday's tasks...",
            metadata=NoteMetadata(
                created_time=datetime(2024, 1, 15),
                modified_time=datetime(2024, 1, 15)
            )
        )
        result = await classifier.classify_content(daily_note)
        assert result.confidence == ClassificationConfidence.HIGH
        
        # Medium confidence - some indicators
        maybe_meeting = Note(
            path=Path("notes.md"),
            content="Discussed with team about project timeline",
            metadata=NoteMetadata(
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        result = await classifier.classify_content(maybe_meeting)
        assert result.confidence in [ClassificationConfidence.MEDIUM, ClassificationConfidence.LOW]
        
        # Low confidence - ambiguous content
        ambiguous = Note(
            path=Path("random.md"),
            content="Some random thoughts",
            metadata=NoteMetadata(
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        result = await classifier.classify_content(ambiguous)
        assert result.confidence == ClassificationConfidence.LOW
    
    async def test_extract_project_metadata(self, classifier):
        """Test extraction of project-specific metadata."""
        note = Note(
            path=Path("project-update.md"),
            content="""# Project: DataPipeline v2.0

Sprint: 15
Status: In Progress
Owner: Alice
Due Date: 2024-02-01

## Updates
- Completed data ingestion module
- API integration 70% complete""",
            metadata=NoteMetadata(
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        
        result = await classifier.classify_content(note)
        
        assert result.metadata.get("detected_project") == "DataPipeline v2.0"
        assert result.metadata.get("sprint") == "15"
        assert result.metadata.get("status") == "In Progress"
        assert result.metadata.get("owner") == "Alice"


class TestDirectoryRouterComprehensive:
    """Comprehensive tests for DirectoryRouter."""
    
    @pytest.fixture
    def router(self, mock_vault_session, mock_config):
        """Create DirectoryRouter instance."""
        return DirectoryRouter(mock_vault_session, mock_config)
    
    async def test_route_file_basic_scenarios(self, router):
        """Test basic file routing scenarios."""
        scenarios = [
            # (note_path, classification, expected_contains)
            (
                Path("meeting.md"),
                ClassificationResult(
                    suggested_path=Path("Meetings/2024/01/meeting.md"),
                    confidence=ClassificationConfidence.HIGH,
                    reasoning="Meeting pattern detected",
                    action=OrganizationAction.MOVE,
                    score=0.9
                ),
                "Meetings"
            ),
            (
                Path("project-spec.md"),
                ClassificationResult(
                    suggested_path=Path("Projects/Active/project-spec.md"),
                    confidence=ClassificationConfidence.HIGH,
                    reasoning="Project documentation",
                    action=OrganizationAction.MOVE,
                    score=0.85
                ),
                "Projects"
            ),
        ]
        
        for note_path, classification, expected in scenarios:
            note = Note(
                path=note_path,
                content="# Test",
                metadata=NoteMetadata(
                    created_time=datetime.utcnow(),
                    modified_time=datetime.utcnow()
                )
            )
            
            result = await router.route_file(note, classification)
            
            assert isinstance(result, Path)
            assert expected in str(result)
            assert result.name == note_path.name
    
    async def test_apply_routing_rules(self, router):
        """Test application of routing rules."""
        classification = ClassificationResult(
            suggested_path=Path("Projects/project.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Project note",
            action=OrganizationAction.MOVE,
            score=0.8,
            metadata={
                "detected_project": "Alpha",
                "has_milestones": True,
                "sprint": "23"
            }
        )
        
        result = await router._apply_routing_rules(classification)
        
        # Should apply project-specific routing
        assert "Alpha" in str(result) or "Sprint" in str(result) or \
               result == classification.suggested_path
    
    async def test_validate_target_path(self, router):
        """Test target path validation."""
        # Valid paths
        valid_paths = [
            Path("Notes/note.md"),
            Path("Projects/2024/project.md"),
            Path("Research/ML/paper.md"),
        ]
        
        for path in valid_paths:
            is_valid, reason = await router._validate_target_path(path)
            assert is_valid
            assert reason == ""
        
        # Invalid paths
        invalid_scenarios = [
            (Path(".obsidian/config.md"), "system directory"),
            (Path("../outside/note.md"), "outside vault"),
            (Path("/absolute/path/note.md"), "absolute path"),
            (Path("Templates/template.md"), "excluded directory"),
        ]
        
        for path, expected_reason in invalid_scenarios:
            is_valid, reason = await router._validate_target_path(path)
            assert not is_valid
            assert expected_reason in reason.lower()
    
    async def test_resolve_naming_conflicts_sequence(self, router, temp_vault):
        """Test sequential conflict resolution."""
        # Create conflicting files
        target_dir = temp_vault / "Projects"
        target_dir.mkdir(exist_ok=True)
        
        base_path = Path("Projects/note.md")
        (temp_vault / base_path).write_text("Original")
        (temp_vault / "Projects/note_1.md").write_text("First conflict")
        (temp_vault / "Projects/note_2.md").write_text("Second conflict")
        
        router.vault_session.vault_path = temp_vault
        
        # Should find next available number
        result = await router._resolve_naming_conflicts(base_path)
        
        assert result == Path("Projects/note_3.md")
    
    async def test_create_directory_structure(self, router, temp_vault):
        """Test directory structure creation."""
        router.vault_session.vault_path = temp_vault
        
        deep_path = Path("Research/2024/ML/Papers/note.md")
        
        success = await router._create_directory_structure(deep_path)
        
        assert success
        assert (temp_vault / "Research/2024/ML/Papers").exists()
        assert (temp_vault / "Research/2024/ML/Papers").is_dir()
    
    async def test_route_with_metadata_enrichment(self, router):
        """Test routing with metadata-based path enrichment."""
        note = Note(
            path=Path("untitled.md"),
            content="# Q1 Planning Meeting\n\nProject: DataViz",
            metadata=NoteMetadata(
                created_time=datetime(2024, 1, 15, 10, 30),
                modified_time=datetime(2024, 1, 15, 11, 45),
                tags=["meeting", "planning", "q1"]
            )
        )
        
        classification = ClassificationResult(
            suggested_path=Path("Meetings/untitled.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Meeting detected",
            action=OrganizationAction.MOVE,
            score=0.85,
            metadata={
                "meeting_type": "planning",
                "detected_project": "DataViz",
                "quarter": "Q1"
            }
        )
        
        result = await router.route_file(note, classification)
        
        # Should create more specific path based on metadata
        assert "Meetings" in str(result)
        # May include date, project, or other metadata in path
        assert result != Path("Meetings/untitled.md")  # Should be more specific


class TestOrganizationLearnerComprehensive:
    """Comprehensive tests for OrganizationLearner."""
    
    @pytest.fixture
    def learner(self, mock_config):
        """Create OrganizationLearner instance."""
        return OrganizationLearner(mock_config)
    
    async def test_record_feedback_types(self, learner):
        """Test recording different types of feedback."""
        feedback_types = [
            UserFeedback(
                original_path=Path("note.md"),
                suggested_path=Path("Projects/note.md"),
                actual_path=Path("Projects/note.md"),
                accepted=True,
                timestamp=datetime.utcnow(),
                feedback_type="approval",
                confidence_score=0.9
            ),
            UserFeedback(
                original_path=Path("meeting.md"),
                suggested_path=Path("Meetings/meeting.md"),
                actual_path=Path("TeamMeetings/meeting.md"),
                accepted=True,
                timestamp=datetime.utcnow(),
                feedback_type="correction",
                confidence_score=0.7
            ),
            UserFeedback(
                original_path=Path("random.md"),
                suggested_path=Path("Misc/random.md"),
                actual_path=Path("random.md"),
                accepted=False,
                timestamp=datetime.utcnow(),
                feedback_type="rejection",
                confidence_score=0.5
            ),
        ]
        
        for feedback in feedback_types:
            await learner.record_feedback(feedback)
        
        assert len(learner.feedback_history) == 3
        assert learner.approval_count == 1
        assert learner.correction_count == 1
        assert learner.rejection_count == 1
    
    async def test_learn_patterns_from_corrections(self, learner):
        """Test pattern learning from user corrections."""
        # Record multiple similar corrections
        corrections = [
            UserFeedback(
                original_path=Path(f"standup-{i}.md"),
                suggested_path=Path(f"Meetings/standup-{i}.md"),
                actual_path=Path(f"Team/Standups/standup-{i}.md"),
                accepted=True,
                timestamp=datetime.utcnow(),
                feedback_type="correction"
            )
            for i in range(5)
        ]
        
        for feedback in corrections:
            await learner.record_feedback(feedback)
        
        # Should learn pattern
        pattern = await learner.get_learned_pattern("standup-new.md")
        
        assert pattern is not None
        assert pattern.pattern_type == "correction"
        assert pattern.confidence > 0.7
        assert "Team/Standups" in pattern.target_template
    
    async def test_adjust_confidence_scores(self, learner):
        """Test confidence score adjustments based on feedback."""
        # High confidence but rejected
        rejection = UserFeedback(
            original_path=Path("doc.md"),
            suggested_path=Path("Documents/doc.md"),
            actual_path=Path("doc.md"),
            accepted=False,
            timestamp=datetime.utcnow(),
            feedback_type="rejection",
            confidence_score=0.9
        )
        
        await learner.record_feedback(rejection)
        adjustments = await learner.get_confidence_adjustments()
        
        # Should reduce confidence for similar suggestions
        assert "Documents" in adjustments
        assert adjustments["Documents"] < 1.0  # Penalty applied
    
    async def test_pattern_expiration(self, learner):
        """Test that old patterns expire."""
        # Add old pattern
        old_feedback = UserFeedback(
            original_path=Path("old.md"),
            suggested_path=Path("Old/old.md"),
            actual_path=Path("Archive/old.md"),
            accepted=True,
            timestamp=datetime.utcnow() - timedelta(days=100),
            feedback_type="correction"
        )
        
        await learner.record_feedback(old_feedback)
        
        # Add recent pattern
        recent_feedback = UserFeedback(
            original_path=Path("new.md"),
            suggested_path=Path("New/new.md"),
            actual_path=Path("Current/new.md"),
            accepted=True,
            timestamp=datetime.utcnow(),
            feedback_type="correction"
        )
        
        await learner.record_feedback(recent_feedback)
        
        # Clean old patterns
        await learner.cleanup_old_patterns(days=30)
        
        # Old pattern should have reduced influence
        old_pattern = await learner.get_learned_pattern("old-similar.md")
        new_pattern = await learner.get_learned_pattern("new-similar.md")
        
        if old_pattern and new_pattern:
            assert new_pattern.confidence > old_pattern.confidence
    
    async def test_export_import_learning_data(self, learner, tmp_path):
        """Test exporting and importing learned patterns."""
        # Create some feedback
        feedback_items = [
            UserFeedback(
                original_path=Path(f"note{i}.md"),
                suggested_path=Path(f"Cat1/note{i}.md"),
                actual_path=Path(f"Cat2/note{i}.md"),
                accepted=True,
                timestamp=datetime.utcnow(),
                feedback_type="correction"
            )
            for i in range(3)
        ]
        
        for feedback in feedback_items:
            await learner.record_feedback(feedback)
        
        # Export data
        export_file = tmp_path / "learning_data.json"
        await learner.export_learning_data(export_file)
        
        assert export_file.exists()
        
        # Create new learner and import
        new_learner = OrganizationLearner(learner.config)
        await new_learner.import_learning_data(export_file)
        
        # Should have same patterns
        assert len(new_learner.feedback_history) == len(learner.feedback_history)
        assert new_learner.learned_patterns == learner.learned_patterns
    
    async def test_get_statistics(self, learner):
        """Test learning statistics generation."""
        # Add varied feedback
        for i in range(10):
            feedback = UserFeedback(
                original_path=Path(f"file{i}.md"),
                suggested_path=Path(f"Suggested/file{i}.md"),
                actual_path=Path(f"Actual/file{i}.md") if i % 2 == 0 else Path(f"Suggested/file{i}.md"),
                accepted=i % 3 != 0,
                timestamp=datetime.utcnow() - timedelta(hours=i),
                feedback_type="approval" if i % 3 == 0 else "correction"
            )
            await learner.record_feedback(feedback)
        
        stats = await learner.get_statistics()
        
        assert stats["total_feedback"] == 10
        assert stats["approval_rate"] > 0
        assert stats["correction_rate"] > 0
        assert "most_corrected_paths" in stats
        assert "learning_trend" in stats


class TestRuleEngineComprehensive:
    """Comprehensive tests for RuleEngine."""
    
    @pytest.fixture
    def rule_engine(self, mock_config):
        """Create RuleEngine instance."""
        return RuleEngine(mock_config)
    
    def test_built_in_rules_comprehensive(self, rule_engine):
        """Test all built-in rules are loaded correctly."""
        rule_names = [rule.name for rule in rule_engine.built_in_rules]
        
        # Essential built-in rules
        assert "daily_notes_by_date" in rule_names
        assert "meeting_notes_by_meeting" in rule_names
        assert "project_documentation" in rule_names
        assert "research_papers" in rule_names
        assert "templates_ignore" in rule_names
        assert "archive_ignore" in rule_names
        
        # Check rule properties
        daily_rule = next(r for r in rule_engine.built_in_rules if r.name == "daily_notes_by_date")
        assert daily_rule.priority >= 8  # High priority
        assert daily_rule.action == OrganizationAction.MOVE
        assert "filename_pattern" in daily_rule.conditions
    
    async def test_evaluate_conditions_complex(self, rule_engine):
        """Test complex condition evaluation."""
        rule = OrganizationRule(
            name="complex_rule",
            conditions={
                "filename_pattern": r"^project-.*\.md$",
                "has_tag": "project",
                "min_word_count": 100,
                "has_frontmatter_key": "status",
                "created_within_days": 30,
                "contains_text": ["milestone", "deliverable"],
                "has_checklist": True
            },
            action=OrganizationAction.MOVE,
            target_pattern="Projects/Active/{filename}",
            priority=5
        )
        
        note = Note(
            path=Path("project-alpha.md"),
            content="x" * 500,  # Enough words
            metadata=NoteMetadata(
                created_time=datetime.utcnow() - timedelta(days=10),
                modified_time=datetime.utcnow(),
                tags=["project"],
                frontmatter={"status": "active"}
            )
        )
        
        features = ContentFeatures(
            word_count=150,
            tags={"project"},
            frontmatter={"status": "active"},
            creation_date=datetime.utcnow() - timedelta(days=10),
            checklist_count=5,
            content_snippet="Project with milestone and deliverable tracking"
        )
        
        matches = await rule_engine._evaluate_conditions(
            rule.conditions, note, features
        )
        
        assert matches  # All conditions should match
    
    async def test_pattern_matching_variations(self, rule_engine):
        """Test various pattern matching scenarios."""
        patterns = [
            # (pattern, filename, should_match)
            (r"^\d{4}-\d{2}-\d{2}\.md$", "2024-01-15.md", True),
            (r"^\d{4}-\d{2}-\d{2}\.md$", "2024-1-15.md", False),
            (r"^meeting-.*\.md$", "meeting-notes.md", True),
            (r"^meeting-.*\.md$", "team-meeting.md", False),
            (r".*\.(png|jpg|jpeg|gif)$", "image.png", True),
            (r".*\.(png|jpg|jpeg|gif)$", "document.pdf", False),
        ]
        
        for pattern, filename, expected in patterns:
            rule = OrganizationRule(
                name="test",
                conditions={"filename_pattern": pattern},
                action=OrganizationAction.MOVE,
                target_pattern="Test/{filename}"
            )
            
            note = Note(
                path=Path(filename),
                content="",
                metadata=NoteMetadata(
                    created_time=datetime.utcnow(),
                    modified_time=datetime.utcnow()
                )
            )
            
            features = ContentFeatures()
            
            matches = await rule_engine._evaluate_conditions(
                rule.conditions, note, features
            )
            
            assert matches == expected
    
    async def test_rule_priority_ordering(self, rule_engine):
        """Test that rules are evaluated in priority order."""
        # Add custom rules with different priorities
        high_priority_rule = OrganizationRule(
            name="high_priority",
            conditions={"filename_pattern": r".*\.md$"},
            action=OrganizationAction.MOVE,
            target_pattern="HighPriority/{filename}",
            priority=10
        )
        
        low_priority_rule = OrganizationRule(
            name="low_priority",
            conditions={"filename_pattern": r".*\.md$"},
            action=OrganizationAction.MOVE,
            target_pattern="LowPriority/{filename}",
            priority=1
        )
        
        rule_engine.add_custom_rule(high_priority_rule)
        rule_engine.add_custom_rule(low_priority_rule)
        
        note = Note(
            path=Path("test.md"),
            content="Test",
            metadata=NoteMetadata(
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        
        features = ContentFeatures()
        
        results = await rule_engine.evaluate_rules(note, features)
        
        # High priority rule should come first
        assert len(results) >= 2
        high_priority_index = next(i for i, r in enumerate(results) 
                                  if "HighPriority" in str(r.suggested_path))
        low_priority_index = next(i for i, r in enumerate(results) 
                                 if "LowPriority" in str(r.suggested_path))
        assert high_priority_index < low_priority_index
    
    async def test_target_pattern_formatting(self, rule_engine):
        """Test target pattern variable substitution."""
        rule = OrganizationRule(
            name="format_test",
            conditions={"filename_pattern": r".*"},
            action=OrganizationAction.MOVE,
            target_pattern="Projects/{year}/{month}/{project}/{filename}",
            priority=5
        )
        
        note = Note(
            path=Path("feature-spec.md"),
            content="Project: Alpha",
            metadata=NoteMetadata(
                created_time=datetime(2024, 1, 15),
                modified_time=datetime(2024, 1, 15),
                frontmatter={"project": "Alpha"}
            )
        )
        
        features = ContentFeatures(
            creation_date=datetime(2024, 1, 15),
            frontmatter={"project": "Alpha"}
        )
        
        formatted = await rule_engine._format_target_path(
            rule.target_pattern, note, features
        )
        
        assert formatted == Path("Projects/2024/01/Alpha/feature-spec.md")
    
    def test_rule_validation(self, rule_engine):
        """Test rule validation logic."""
        # Valid rule
        valid_rule = OrganizationRule(
            name="valid_rule",
            conditions={"filename_pattern": r".*\.md$"},
            action=OrganizationAction.MOVE,
            target_pattern="Valid/{filename}",
            priority=5
        )
        
        is_valid, errors = rule_engine.validate_rule(valid_rule)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid rules
        invalid_rules = [
            # Missing required fields
            OrganizationRule(
                name="",
                conditions={},
                action=OrganizationAction.MOVE,
                target_pattern="Test/{filename}"
            ),
            # Invalid regex pattern
            OrganizationRule(
                name="bad_regex",
                conditions={"filename_pattern": r"["},
                action=OrganizationAction.MOVE,
                target_pattern="Test/{filename}"
            ),
            # Invalid target pattern
            OrganizationRule(
                name="bad_target",
                conditions={"filename_pattern": r".*"},
                action=OrganizationAction.MOVE,
                target_pattern="../../../etc/passwd"
            ),
        ]
        
        for rule in invalid_rules:
            is_valid, errors = rule_engine.validate_rule(rule)
            assert not is_valid
            assert len(errors) > 0
    
    async def test_conditional_rules(self, rule_engine):
        """Test rules with multiple conditions."""
        # Rule that only applies to recent project files with specific tags
        conditional_rule = OrganizationRule(
            name="recent_project_files",
            conditions={
                "filename_pattern": r"^(?!daily-).*\.md$",  # Not daily notes
                "has_tag": "project",
                "created_within_days": 7,
                "min_word_count": 50,
                "NOT_has_tag": "archive"  # Exclusion condition
            },
            action=OrganizationAction.MOVE,
            target_pattern="Projects/Recent/{filename}",
            priority=7
        )
        
        rule_engine.add_custom_rule(conditional_rule)
        
        # Note that matches all conditions
        matching_note = Note(
            path=Path("new-project.md"),
            content="x" * 100,
            metadata=NoteMetadata(
                created_time=datetime.utcnow() - timedelta(days=3),
                modified_time=datetime.utcnow(),
                tags=["project", "important"]
            )
        )
        
        matching_features = ContentFeatures(
            word_count=100,
            tags={"project", "important"},
            creation_date=datetime.utcnow() - timedelta(days=3)
        )
        
        # Note that fails one condition
        non_matching_note = Note(
            path=Path("old-project.md"),
            content="x" * 100,
            metadata=NoteMetadata(
                created_time=datetime.utcnow() - timedelta(days=30),
                modified_time=datetime.utcnow(),
                tags=["project", "archive"]
            )
        )
        
        non_matching_features = ContentFeatures(
            word_count=100,
            tags={"project", "archive"},
            creation_date=datetime.utcnow() - timedelta(days=30)
        )
        
        # Evaluate rules
        matching_results = await rule_engine.evaluate_rules(matching_note, matching_features)
        non_matching_results = await rule_engine.evaluate_rules(non_matching_note, non_matching_features)
        
        # Should match for recent note without archive tag
        assert any("Projects/Recent" in str(r.suggested_path) for r in matching_results)
        
        # Should not match for old note or archived note
        assert not any("Projects/Recent" in str(r.suggested_path) for r in non_matching_results)


class TestFileWatcherComprehensive:
    """Comprehensive tests for FileWatcher."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for FileWatcher."""
        return {
            "classifier": Mock(spec=ContentClassifier),
            "router": Mock(spec=DirectoryRouter),
            "learner": Mock(spec=OrganizationLearner),
            "rule_engine": Mock(spec=RuleEngine)
        }
    
    @pytest.fixture
    def file_watcher(self, mock_vault_session, mock_components, mock_config):
        """Create FileWatcher instance."""
        return FileWatcher(
            mock_vault_session,
            mock_components["classifier"],
            mock_components["router"],
            mock_components["learner"],
            mock_components["rule_engine"],
            mock_config
        )
    
    async def test_watch_lifecycle(self, file_watcher):
        """Test file watcher start/stop lifecycle."""
        # Mock the watch methods
        file_watcher._watch_vault = AsyncMock()
        file_watcher._process_batch = AsyncMock()
        
        # Should not be watching initially
        assert not file_watcher.is_watching
        
        # Start watching
        await file_watcher.start_watching()
        assert file_watcher.is_watching
        
        # Stop watching
        await file_watcher.stop_watching()
        assert not file_watcher.is_watching
        
        # Ensure tasks were cancelled
        assert file_watcher._watch_task is None
        assert file_watcher._process_task is None
    
    async def test_file_event_handling(self, file_watcher, temp_vault):
        """Test handling of file system events."""
        file_watcher.vault_session.vault_path = temp_vault
        
        # Create a test file
        test_file = temp_vault / "new-note.md"
        test_file.write_text("# New Note")
        
        # Simulate file event
        await file_watcher._handle_file_event("created", test_file)
        
        # Should be queued for processing
        assert Path("new-note.md") in file_watcher.pending_files
        
        # Modify file
        test_file.write_text("# Modified Note")
        await file_watcher._handle_file_event("modified", test_file)
        
        # Should update queue
        assert Path("new-note.md") in file_watcher.pending_files
    
    async def test_batch_processing(self, file_watcher, mock_components):
        """Test batch processing of pending files."""
        # Setup pending files
        file_watcher.pending_files = {
            Path("note1.md"): datetime.utcnow() - timedelta(seconds=5),
            Path("note2.md"): datetime.utcnow() - timedelta(seconds=3),
            Path("note3.md"): datetime.utcnow() - timedelta(seconds=1),
        }
        
        # Mock processing
        async def mock_process(path):
            return ClassificationResult(
                suggested_path=Path(f"Processed/{path}"),
                confidence=ClassificationConfidence.HIGH,
                reasoning="Test",
                action=OrganizationAction.MOVE,
                score=0.9
            )
        
        file_watcher._process_file = AsyncMock(side_effect=mock_process)
        
        # Process batch
        await file_watcher._process_pending_files()
        
        # Should process files meeting delay threshold
        assert file_watcher._process_file.call_count >= 2
        
        # Recent file might still be pending
        assert len(file_watcher.pending_files) <= 1
    
    async def test_process_file_complete_workflow(self, file_watcher, mock_components):
        """Test complete file processing workflow."""
        test_path = Path("test-note.md")
        
        # Setup mocks
        mock_note = Note(
            path=test_path,
            content="# Test Note",
            metadata=NoteMetadata(
                created_time=datetime.utcnow(),
                modified_time=datetime.utcnow()
            )
        )
        
        file_watcher.vault_session.load_note = AsyncMock(return_value=mock_note)
        
        classification = ClassificationResult(
            suggested_path=Path("Projects/test-note.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Project content detected",
            action=OrganizationAction.MOVE,
            score=0.85
        )
        
        mock_components["classifier"].classify_content = AsyncMock(return_value=classification)
        mock_components["router"].route_file = AsyncMock(return_value=Path("Projects/test-note.md"))
        file_watcher.vault_session.move_note = AsyncMock(return_value=True)
        
        # Process file
        result = await file_watcher._process_file(test_path)
        
        # Verify workflow
        assert result == classification
        file_watcher.vault_session.load_note.assert_called_once_with(test_path)
        mock_components["classifier"].classify_content.assert_called_once_with(mock_note)
        
        # Should record feedback
        if not file_watcher.dry_run_mode:
            mock_components["learner"].record_feedback.assert_called_once()
    
    async def test_ignore_patterns(self, file_watcher):
        """Test file ignore patterns."""
        ignore_patterns = [
            Path(".obsidian/workspace.json"),
            Path(".git/config"),
            Path(".DS_Store"),
            Path("node_modules/package.json"),
            Path("test.tmp"),
            Path("~$document.docx"),
        ]
        
        for path in ignore_patterns:
            should_process = await file_watcher._should_process_file(path)
            assert not should_process
        
        # Valid files should be processed
        valid_files = [
            Path("note.md"),
            Path("projects/project.md"),
            Path("daily/2024-01-15.md"),
        ]
        
        for path in valid_files:
            should_process = await file_watcher._should_process_file(path)
            assert should_process
    
    async def test_error_handling(self, file_watcher, mock_components):
        """Test error handling during file processing."""
        test_path = Path("error-note.md")
        
        # Setup error scenario
        file_watcher.vault_session.load_note = AsyncMock(
            side_effect=Exception("Failed to load note")
        )
        
        # Should handle error gracefully
        result = await file_watcher._process_file(test_path)
        
        assert result is None
        assert test_path not in file_watcher.pending_files
        assert file_watcher.error_count > 0
    
    async def test_configuration_changes(self, file_watcher):
        """Test dynamic configuration changes."""
        # Test auto-organization toggle
        file_watcher.enable_auto_organization(True)
        assert file_watcher.auto_organize_enabled
        
        file_watcher.enable_auto_organization(False)
        assert not file_watcher.auto_organize_enabled
        
        # Test dry run mode
        file_watcher.set_dry_run_mode(True)
        assert file_watcher.dry_run_mode
        
        # Test processing delay
        file_watcher.set_processing_delay(10)
        assert file_watcher.processing_delay == 10
        
        # Test batch size
        file_watcher.set_batch_size(20)
        assert file_watcher.batch_size == 20


class TestAutoOrganizerIntegration:
    """Integration tests for complete AutoOrganizer system."""
    
    @pytest.fixture
    async def full_organizer(self, temp_vault, mock_content_analyzer, 
                            mock_embedding_service, mock_query_processor):
        """Create fully configured AutoOrganizer."""
        config = LibrarianConfig(
            vault_path=temp_vault,
            enable_ai_features=True,
            auto_organization={
                "enabled": True,
                "confidence_threshold": 0.7,
                "require_confirmation": False
            }
        )
        
        vault_session = Mock(spec=Vault)
        vault_session.vault_path = temp_vault
        
        # Create realistic vault structure
        for dir_name in ["Projects", "Meetings", "Daily Notes", "Research", "Archive", "Templates"]:
            (temp_vault / dir_name).mkdir(exist_ok=True)
        
        # Add some existing notes
        existing_notes = {
            "Projects/project-alpha.md": "# Project Alpha\n\nActive project",
            "Meetings/2024-01-10-standup.md": "# Daily Standup\n\nTeam sync",
            "Daily Notes/2024-01-14.md": "# Sunday\n\nDaily thoughts",
            "Templates/meeting.md": "# {{title}}\n\nDate: {{date}}",
        }
        
        for path, content in existing_notes.items():
            file_path = temp_vault / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Setup vault session methods
        async def load_note(path):
            full_path = temp_vault / path
            if full_path.exists():
                return Note(
                    path=path,
                    content=full_path.read_text(),
                    metadata=NoteMetadata(
                        created_time=datetime.utcnow(),
                        modified_time=datetime.utcnow()
                    )
                )
            return None
        
        async def move_note(old_path, new_path):
            old_full = temp_vault / old_path
            new_full = temp_vault / new_path
            new_full.parent.mkdir(parents=True, exist_ok=True)
            if old_full.exists():
                shutil.move(str(old_full), str(new_full))
                return True
            return False
        
        vault_session.load_note = AsyncMock(side_effect=load_note)
        vault_session.move_note = AsyncMock(side_effect=move_note)
        vault_session.get_all_notes = AsyncMock()
        
        organizer = AutoOrganizer(
            vault_session,
            mock_content_analyzer,
            mock_embedding_service,
            mock_query_processor,
            config
        )
        
        return organizer
    
    async def test_organize_single_file_scenarios(self, full_organizer, temp_vault):
        """Test organizing individual files in various scenarios."""
        test_files = [
            ("2024-01-15.md", "# Monday\n\nDaily note content", "Daily Notes"),
            ("team-meeting.md", "# Team Meeting\n\n## Agenda\n- Updates\n\n## Action Items\n- [ ] Task", "Meetings"),
            ("ml-research.md", "# ML Research\n\n## Abstract\nStudy findings...", "Research"),
            ("project-beta.md", "# Project Beta\n\nMilestones:\n- [ ] Phase 1", "Projects"),
        ]
        
        for filename, content, expected_dir in test_files:
            # Create test file
            test_file = temp_vault / filename
            test_file.write_text(content)
            
            # Organize
            result = await full_organizer.organize_file(Path(filename))
            
            assert result is not None
            assert result.confidence >= ClassificationConfidence.MEDIUM
            assert expected_dir in str(result.suggested_path)
            
            # Verify file was moved (or would be moved in dry run)
            if result.action == OrganizationAction.MOVE:
                if not full_organizer.config.auto_organization.get("require_confirmation"):
                    assert not test_file.exists()  # Original gone
                    assert (temp_vault / result.suggested_path).exists()  # Moved to new location
    
    async def test_organize_vault_complete(self, full_organizer, temp_vault):
        """Test organizing entire vault."""
        # Add unorganized notes
        unorganized = [
            ("random-meeting.md", "# Random Meeting\n\nDiscussion points..."),
            ("2024-01-16.md", "# Tuesday\n\nTasks for today"),
            ("research-notes.md", "# Research\n\nFindings from study"),
            ("old-project.md", "# Old Project\n\nArchived content"),
        ]
        
        for filename, content in unorganized:
            (temp_vault / filename).write_text(content)
        
        # Organize vault
        results = await full_organizer.organize_vault(dry_run=True)
        
        assert results["processed"] >= len(unorganized)
        assert results["organized"] > 0
        assert len(results["suggestions"]) > 0
        
        # Check suggestions quality
        for suggestion in results["suggestions"]:
            assert "original_path" in suggestion
            assert "suggested_path" in suggestion
            assert "confidence" in suggestion
            assert "reasoning" in suggestion
    
    async def test_learning_integration(self, full_organizer, temp_vault):
        """Test learning from user feedback integration."""
        # Create a note
        test_file = temp_vault / "ambiguous-note.md"
        test_file.write_text("# Planning Session\n\nDiscussing project timeline")
        
        # First organization attempt
        result1 = await full_organizer.organize_file(Path("ambiguous-note.md"))
        
        # Simulate user correction
        if result1 and result1.suggested_path != Path("Projects/Planning/ambiguous-note.md"):
            feedback = UserFeedback(
                original_path=Path("ambiguous-note.md"),
                suggested_path=result1.suggested_path,
                actual_path=Path("Projects/Planning/ambiguous-note.md"),
                accepted=True,
                timestamp=datetime.utcnow(),
                feedback_type="correction"
            )
            
            await full_organizer.add_feedback(feedback)
        
        # Create similar note
        similar_file = temp_vault / "planning-session-2.md"
        similar_file.write_text("# Another Planning Session\n\nProject planning")
        
        # Should learn from previous feedback
        result2 = await full_organizer.organize_file(Path("planning-session-2.md"))
        
        if result2:
            # Should suggest similar organization
            assert "Projects" in str(result2.suggested_path) or \
                   "Planning" in str(result2.suggested_path)
    
    async def test_custom_rules_integration(self, full_organizer, temp_vault):
        """Test custom rules integration."""
        # Add custom rule for specific project
        custom_rule = OrganizationRule(
            name="project_alpha_rule",
            conditions={
                "filename_pattern": r"alpha-.*\.md",
                "contains_text": ["Project Alpha", "Alpha Team"]
            },
            action=OrganizationAction.MOVE,
            target_pattern="Projects/Alpha/{year}/{filename}",
            priority=9
        )
        
        full_organizer.add_custom_rule(custom_rule)
        
        # Create matching file
        alpha_file = temp_vault / "alpha-update.md"
        alpha_file.write_text("# Project Alpha Update\n\nAlpha Team progress report")
        
        # Organize
        result = await full_organizer.organize_file(Path("alpha-update.md"))
        
        assert result is not None
        assert "Projects/Alpha" in str(result.suggested_path)
        assert str(datetime.utcnow().year) in str(result.suggested_path)
    
    async def test_concurrent_organization(self, full_organizer, temp_vault):
        """Test concurrent file organization."""
        # Create multiple files
        files = []
        for i in range(10):
            filename = f"concurrent-{i}.md"
            content = f"# Note {i}\n\nContent for testing"
            (temp_vault / filename).write_text(content)
            files.append(Path(filename))
        
        # Organize concurrently
        tasks = [full_organizer.organize_file(f) for f in files]
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == 10
        assert all(r is not None for r in results)
    
    async def test_error_recovery(self, full_organizer, temp_vault):
        """Test error handling and recovery."""
        # Create problematic scenarios
        
        # 1. File that disappears during processing
        disappearing_file = temp_vault / "disappearing.md"
        disappearing_file.write_text("# Will be deleted")
        
        # Mock to delete file during processing
        original_load = full_organizer.vault_session.load_note
        
        async def load_with_delete(path):
            if path == Path("disappearing.md"):
                disappearing_file.unlink()  # Delete file
                return None
            return await original_load(path)
        
        full_organizer.vault_session.load_note = load_with_delete
        
        # Should handle gracefully
        result = await full_organizer.organize_file(Path("disappearing.md"))
        assert result is None
        
        # Restore original
        full_organizer.vault_session.load_note = original_load
        
        # 2. Invalid file content
        invalid_file = temp_vault / "invalid.md"
        invalid_file.write_text("")  # Empty file
        
        result = await full_organizer.organize_file(Path("invalid.md"))
        # Should still provide some classification
        assert result is not None
    
    async def test_statistics_tracking(self, full_organizer):
        """Test statistics and metrics tracking."""
        stats = await full_organizer.get_statistics()
        
        assert "total_files_processed" in stats
        assert "total_files_organized" in stats
        assert "confidence_distribution" in stats
        assert "common_patterns" in stats
        assert "learning_metrics" in stats
        
        # If learner has data
        if stats["learning_metrics"]["total_feedback"] > 0:
            assert stats["learning_metrics"]["approval_rate"] >= 0
            assert stats["learning_metrics"]["correction_rate"] >= 0


class TestPerformanceBenchmarks:
    """Performance benchmarks for auto-organization."""
    
    @pytest.mark.slow
    async def test_large_vault_organization(self, temp_vault):
        """Test performance with large vault."""
        # Create mock services
        config = LibrarianConfig(
            vault_path=temp_vault,
            auto_organization={"enabled": True}
        )
        
        vault_session = Mock(spec=Vault)
        vault_session.vault_path = temp_vault
        
        # Create many files
        file_count = 1000
        for i in range(file_count):
            category = ["meeting", "project", "daily", "research"][i % 4]
            filename = f"{category}-note-{i}.md"
            content = f"# {category.title()} Note {i}\n\nContent for {category}"
            (temp_vault / filename).write_text(content)
        
        # Mock services for speed
        mock_analyzer = Mock(spec=ContentAnalyzer)
        mock_analyzer.analyze_content = AsyncMock(return_value=Mock(
            topics=["general"],
            keywords=["test"],
            entities=[]
        ))
        
        mock_embedding = Mock(spec=EmbeddingService)
        mock_query = Mock()
        
        organizer = AutoOrganizer(
            vault_session,
            mock_analyzer,
            mock_embedding,
            mock_query,
            config
        )
        
        # Time the organization
        import time
        start = time.time()
        
        # Process a subset to keep test reasonable
        files = list(temp_vault.glob("*.md"))[:100]
        results = []
        for file in files:
            result = await organizer.organize_file(file.relative_to(temp_vault))
            results.append(result)
        
        end = time.time()
        
        # Should complete in reasonable time
        assert end - start < 30  # 30 seconds for 100 files
        assert len(results) == 100
        assert sum(1 for r in results if r is not None) > 80  # Most classified
    
    @pytest.mark.slow
    async def test_concurrent_classification_performance(self):
        """Test performance of concurrent classification."""
        # Create minimal test setup
        config = LibrarianConfig()
        
        classifier = ContentClassifier(
            Mock(),  # vault
            Mock(spec=ContentAnalyzer),  # analyzer
            Mock(),  # embedding
            config
        )
        
        # Mock heavy operations
        classifier.content_analyzer.analyze_content = AsyncMock(
            return_value=Mock(topics=[], keywords=[], entities=[])
        )
        
        # Create test notes
        notes = [
            Note(
                path=Path(f"note-{i}.md"),
                content=f"# Note {i}\n" + "x" * 1000,
                metadata=NoteMetadata(
                    created_time=datetime.utcnow(),
                    modified_time=datetime.utcnow()
                )
            )
            for i in range(50)
        ]
        
        # Time concurrent classification
        import time
        start = time.time()
        
        tasks = [classifier.classify_content(note) for note in notes]
        results = await asyncio.gather(*tasks)
        
        end = time.time()
        
        # Should scale well
        assert end - start < 10  # 10 seconds for 50 concurrent classifications
        assert len(results) == 50
        assert all(isinstance(r, ClassificationResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])