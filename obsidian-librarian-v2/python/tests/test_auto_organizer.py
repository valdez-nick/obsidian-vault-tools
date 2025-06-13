"""
Tests for the auto-organization service.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
)
from obsidian_librarian.models import Note, NoteMetadata, LibrarianConfig
from obsidian_librarian.vault import Vault
from obsidian_librarian.ai import ContentAnalyzer, EmbeddingService, QueryProcessor


@pytest.fixture
def temp_vault():
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir) / "test_vault"
        vault_path.mkdir()
        
        # Create some test files
        (vault_path / "2024-01-15.md").write_text("# Daily Note\nThis is a daily note.")
        (vault_path / "meeting-notes.md").write_text("# Meeting\nAction items:\n- [ ] Task 1")
        (vault_path / "project-alpha.md").write_text("# Project Alpha\nProject documentation")
        
        yield vault_path


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return LibrarianConfig(
        enable_file_watching=True,
        auto_apply_templates=True,
        enable_ai_features=True
    )


@pytest.fixture
def mock_vault_session(temp_vault):
    """Create a mock vault session."""
    vault = MagicMock(spec=Vault)
    vault.path = temp_vault
    
    async def mock_get_note(note_id):
        # Try to find a note by ID (simplified)
        for md_file in temp_vault.rglob("*.md"):
            if md_file.stem == note_id or str(md_file.relative_to(temp_vault)) == note_id:
                content = md_file.read_text()
                return Note(
                    path=md_file.relative_to(temp_vault),
                    content=content,
                    metadata=NoteMetadata(
                        created_time=datetime.now(),
                        modified_time=datetime.now(),
                        tags=[],
                        frontmatter={}
                    )
                )
        return None
    
    vault.get_note = AsyncMock(side_effect=mock_get_note)
    return vault


@pytest.fixture
def mock_content_analyzer():
    """Create a mock content analyzer."""
    analyzer = MagicMock(spec=ContentAnalyzer)
    
    async def mock_analyze(content):
        analysis = MagicMock()
        analysis.topics = ["productivity", "notes"]
        analysis.entities = ["Obsidian", "Markdown"]
        analysis.keywords = ["note", "organization", "knowledge"]
        return analysis
    
    analyzer.analyze_content = AsyncMock(side_effect=mock_analyze)
    return analyzer


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock(spec=EmbeddingService)
    service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return service


@pytest.fixture
def mock_query_processor():
    """Create a mock query processor."""
    return MagicMock(spec=QueryProcessor)


class TestContentClassifier:
    """Test the ContentClassifier class."""
    
    @pytest.fixture
    def classifier(self, mock_vault_session, mock_content_analyzer, mock_embedding_service, mock_config):
        return ContentClassifier(
            mock_vault_session, 
            mock_content_analyzer, 
            mock_embedding_service, 
            mock_config
        )
    
    @pytest.mark.asyncio
    async def test_extract_features_basic(self, classifier):
        """Test basic feature extraction."""
        note = Note(
            path=Path("test.md"),
            content="# Title\nThis is content with [[link]] and - list item",
            metadata=NoteMetadata(
                created_time=datetime.now(),
                modified_time=datetime.now(),
                tags=["#productivity"],
                frontmatter={"type": "note"}
            )
        )
        
        features = await classifier.extract_features(note)
        
        assert features.word_count > 0
        assert features.header_count == 1
        assert features.list_count == 1
        assert features.link_count == 1
        assert "#productivity" in features.tags
        assert features.frontmatter["type"] == "note"
    
    @pytest.mark.asyncio
    async def test_classify_daily_note_pattern(self, classifier):
        """Test classification of daily note patterns."""
        note = Note(
            path=Path("2024-01-15.md"),
            content="# Daily Note\nToday's tasks and thoughts",
            metadata=NoteMetadata(
                created_time=datetime.now(),
                modified_time=datetime.now()
            )
        )
        
        result = await classifier.classify_content(note)
        
        assert result.confidence in [ClassificationConfidence.MEDIUM, ClassificationConfidence.HIGH]
        assert "Daily Notes" in str(result.suggested_path)
    
    @pytest.mark.asyncio
    async def test_classify_meeting_notes(self, classifier):
        """Test classification of meeting notes."""
        note = Note(
            path=Path("team-meeting.md"),
            content="# Team Meeting\nAgenda:\n1. Project updates\n\nAction items:\n- [ ] Review PRs",
            metadata=NoteMetadata(
                created_time=datetime.now(),
                modified_time=datetime.now()
            )
        )
        
        result = await classifier.classify_content(note)
        
        assert result.confidence != ClassificationConfidence.LOW
        # Should detect meeting-related content
        assert "meeting" in result.reasoning.lower() or "Meeting" in str(result.suggested_path)


class TestDirectoryRouter:
    """Test the DirectoryRouter class."""
    
    @pytest.fixture
    def router(self, mock_vault_session, mock_config):
        return DirectoryRouter(mock_vault_session, mock_config)
    
    @pytest.mark.asyncio
    async def test_route_file_basic(self, router):
        """Test basic file routing."""
        note = Note(
            path=Path("test.md"),
            content="Test content",
            metadata=NoteMetadata(
                created_time=datetime.now(),
                modified_time=datetime.now()
            )
        )
        
        classification = ClassificationResult(
            suggested_path=Path("Knowledge Base/test.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Test classification",
            action=OrganizationAction.MOVE,
            score=0.8
        )
        
        result = await router.route_file(note, classification)
        
        assert isinstance(result, Path)
        assert result.name == "test.md"
    
    @pytest.mark.asyncio
    async def test_resolve_naming_conflicts(self, router, temp_vault):
        """Test naming conflict resolution."""
        # Create a file that would conflict
        existing_file = temp_vault / "Knowledge Base" / "test.md"
        existing_file.parent.mkdir(parents=True, exist_ok=True)
        existing_file.write_text("Existing content")
        
        target_path = Path("Knowledge Base/test.md")
        result = await router._resolve_naming_conflicts(target_path)
        
        # Should suggest an alternative name
        assert result != target_path
        assert result.parent == target_path.parent
        assert result.suffix == target_path.suffix


class TestOrganizationLearner:
    """Test the OrganizationLearner class."""
    
    @pytest.fixture
    def learner(self, mock_config):
        return OrganizationLearner(mock_config)
    
    @pytest.mark.asyncio
    async def test_record_positive_feedback(self, learner):
        """Test recording positive feedback."""
        feedback = UserFeedback(
            original_path=Path("unsorted/test.md"),
            suggested_path=Path("suggested/test.md"),
            actual_path=Path("correct/test.md"),
            accepted=True,
            timestamp=datetime.now(),
            feedback_type="approval"
        )
        
        await learner.record_feedback(feedback)
        
        assert len(learner.feedback_history) == 1
        assert learner.feedback_history[0] == feedback
    
    @pytest.mark.asyncio
    async def test_learned_patterns_creation(self, learner):
        """Test that patterns are learned from feedback."""
        feedback = UserFeedback(
            original_path=Path("unsorted/meeting-notes.md"),
            suggested_path=Path("wrong/meeting-notes.md"),
            actual_path=Path("Meetings/meeting-notes.md"),
            accepted=True,
            timestamp=datetime.now(),
            feedback_type="correction"
        )
        
        await learner.record_feedback(feedback)
        
        # Check that a pattern was learned
        assert len(learner.learned_patterns) > 0


class TestRuleEngine:
    """Test the RuleEngine class."""
    
    @pytest.fixture
    def rule_engine(self, mock_config):
        return RuleEngine(mock_config)
    
    def test_built_in_rules_loaded(self, rule_engine):
        """Test that built-in rules are loaded."""
        assert len(rule_engine.built_in_rules) > 0
        
        # Check for specific built-in rules
        rule_names = [rule.name for rule in rule_engine.built_in_rules]
        assert "daily_notes_by_date" in rule_names
        assert "meeting_notes_by_meeting" in rule_names
    
    @pytest.mark.asyncio
    async def test_daily_note_rule_matching(self, rule_engine):
        """Test daily note rule matching."""
        note = Note(
            path=Path("2024-01-15.md"),
            content="Daily note content",
            metadata=NoteMetadata(
                created_time=datetime(2024, 1, 15),
                modified_time=datetime(2024, 1, 15)
            )
        )
        
        features = ContentFeatures(
            creation_date=datetime(2024, 1, 15),
            modification_date=datetime(2024, 1, 15)
        )
        
        results = await rule_engine.evaluate_rules(note, features)
        
        assert len(results) > 0
        # Should match daily note rule
        daily_results = [r for r in results if "Daily Notes" in str(r.suggested_path)]
        assert len(daily_results) > 0
    
    def test_add_custom_rule(self, rule_engine):
        """Test adding custom rules."""
        custom_rule = OrganizationRule(
            name="custom_test_rule",
            conditions={"filename_pattern": r"test.*"},
            action=OrganizationAction.MOVE,
            target_pattern="Tests/{filename}",
            priority=5
        )
        
        initial_count = len(rule_engine.custom_rules)
        rule_engine.add_custom_rule(custom_rule)
        
        assert len(rule_engine.custom_rules) == initial_count + 1
        assert rule_engine.custom_rules[-1] == custom_rule
    
    def test_remove_custom_rule(self, rule_engine):
        """Test removing custom rules."""
        custom_rule = OrganizationRule(
            name="custom_test_rule",
            conditions={"filename_pattern": r"test.*"},
            action=OrganizationAction.MOVE,
            target_pattern="Tests/{filename}",
            priority=5
        )
        
        rule_engine.add_custom_rule(custom_rule)
        initial_count = len(rule_engine.custom_rules)
        
        removed = rule_engine.remove_custom_rule("custom_test_rule")
        
        assert removed is True
        assert len(rule_engine.custom_rules) == initial_count - 1


class TestFileWatcher:
    """Test the FileWatcher class."""
    
    @pytest.fixture
    def file_watcher(self, mock_vault_session, mock_config):
        # Create all required mocks
        classifier = MagicMock()
        router = MagicMock()
        learner = MagicMock()
        rule_engine = MagicMock()
        
        return FileWatcher(
            mock_vault_session,
            classifier,
            router,
            learner,
            rule_engine,
            mock_config
        )
    
    @pytest.mark.asyncio
    async def test_start_stop_watching(self, file_watcher):
        """Test starting and stopping file watching."""
        assert not file_watcher.is_watching
        
        # Mock the _watch_vault method to prevent actual file watching
        with patch.object(file_watcher, '_watch_vault', new_callable=AsyncMock):
            with patch.object(file_watcher, '_process_batch', new_callable=AsyncMock):
                await file_watcher.start_watching()
                assert file_watcher.is_watching
                
                await file_watcher.stop_watching()
                assert not file_watcher.is_watching
    
    def test_configuration_methods(self, file_watcher):
        """Test configuration methods."""
        # Test auto-organization toggle
        file_watcher.enable_auto_organization(True)
        assert file_watcher.auto_organize_enabled is True
        
        file_watcher.enable_auto_organization(False)
        assert file_watcher.auto_organize_enabled is False
        
        # Test dry run mode
        file_watcher.set_dry_run_mode(True)
        assert file_watcher.dry_run_mode is True
        
        file_watcher.set_dry_run_mode(False)
        assert file_watcher.dry_run_mode is False
    
    @pytest.mark.asyncio
    async def test_queue_file_for_processing(self, file_watcher, temp_vault):
        """Test file queuing mechanism."""
        test_file = temp_vault / "test.md"
        test_file.write_text("Test content")
        
        relative_path = test_file.relative_to(temp_vault)
        await file_watcher._queue_file_for_processing(test_file)
        
        assert relative_path in file_watcher.pending_files


class TestAutoOrganizer:
    """Test the main AutoOrganizer class."""
    
    @pytest.fixture
    def auto_organizer(self, mock_vault_session, mock_content_analyzer, 
                      mock_embedding_service, mock_query_processor, mock_config):
        return AutoOrganizer(
            mock_vault_session,
            mock_content_analyzer,
            mock_embedding_service,
            mock_query_processor,
            mock_config
        )
    
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, auto_organizer):
        """Test starting and stopping the service."""
        assert not auto_organizer.is_running
        
        # Mock the file watcher methods
        auto_organizer.file_watcher.start_watching = AsyncMock()
        auto_organizer.file_watcher.stop_watching = AsyncMock()
        
        await auto_organizer.start()
        assert auto_organizer.is_running
        auto_organizer.file_watcher.start_watching.assert_called_once()
        
        await auto_organizer.stop()
        assert not auto_organizer.is_running
        auto_organizer.file_watcher.stop_watching.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_organize_single_file(self, auto_organizer, mock_vault_session):
        """Test organizing a single file."""
        # Mock the classification process
        auto_organizer.classifier.classify_content = AsyncMock(return_value=ClassificationResult(
            suggested_path=Path("Knowledge Base/test.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Test classification",
            action=OrganizationAction.MOVE,
            score=0.9
        ))
        
        auto_organizer.router.route_file = AsyncMock(return_value=Path("Knowledge Base/test.md"))
        
        result = await auto_organizer.organize_file(Path("test.md"))
        
        assert isinstance(result, ClassificationResult)
        assert result.confidence == ClassificationConfidence.HIGH
        assert result.action == OrganizationAction.MOVE
    
    @pytest.mark.asyncio
    async def test_organize_vault_dry_run(self, auto_organizer, temp_vault):
        """Test organizing entire vault in dry run mode."""
        # Mock the organize_file method
        auto_organizer.organize_file = AsyncMock(return_value=ClassificationResult(
            suggested_path=Path("Knowledge Base/test.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Test classification",
            action=OrganizationAction.MOVE,
            score=0.9
        ))
        
        # Mock vault_session.vault_path
        auto_organizer.vault_session.vault_path = temp_vault
        
        results = await auto_organizer.organize_vault(dry_run=True)
        
        assert "processed" in results
        assert "organized" in results
        assert "suggestions" in results
        assert results["processed"] > 0
    
    def test_add_custom_rule(self, auto_organizer):
        """Test adding custom rules."""
        rule = OrganizationRule(
            name="test_rule",
            conditions={"filename_pattern": r"test.*"},
            action=OrganizationAction.MOVE,
            target_pattern="Tests/{filename}",
            priority=5
        )
        
        auto_organizer.add_custom_rule(rule)
        
        # Verify rule was added to rule engine
        assert rule in auto_organizer.rule_engine.custom_rules
    
    def test_configure_auto_organization(self, auto_organizer):
        """Test configuration of auto-organization."""
        auto_organizer.configure_auto_organization(
            enabled=True,
            dry_run=False,
            require_confirmation=False
        )
        
        assert auto_organizer.file_watcher.auto_organize_enabled is True
        assert auto_organizer.file_watcher.dry_run_mode is False
        assert auto_organizer.file_watcher.require_user_confirmation is False


class TestDataClasses:
    """Test data classes and enums."""
    
    def test_classification_result_creation(self):
        """Test ClassificationResult creation."""
        result = ClassificationResult(
            suggested_path=Path("test/path.md"),
            confidence=ClassificationConfidence.HIGH,
            reasoning="Test reasoning",
            action=OrganizationAction.MOVE,
            score=0.85
        )
        
        assert result.suggested_path == Path("test/path.md")
        assert result.confidence == ClassificationConfidence.HIGH
        assert result.reasoning == "Test reasoning"
        assert result.action == OrganizationAction.MOVE
        assert result.score == 0.85
    
    def test_organization_rule_creation(self):
        """Test OrganizationRule creation."""
        rule = OrganizationRule(
            name="test_rule",
            conditions={"pattern": "test"},
            action=OrganizationAction.MOVE,
            target_pattern="target/{filename}",
            priority=5
        )
        
        assert rule.name == "test_rule"
        assert rule.conditions["pattern"] == "test"
        assert rule.action == OrganizationAction.MOVE
        assert rule.target_pattern == "target/{filename}"
        assert rule.priority == 5
        assert rule.enabled is True
        assert rule.usage_count == 0
    
    def test_user_feedback_creation(self):
        """Test UserFeedback creation."""
        feedback = UserFeedback(
            original_path=Path("original.md"),
            suggested_path=Path("suggested.md"),
            actual_path=Path("actual.md"),
            accepted=True,
            timestamp=datetime.now(),
            feedback_type="approval"
        )
        
        assert feedback.original_path == Path("original.md")
        assert feedback.suggested_path == Path("suggested.md")
        assert feedback.actual_path == Path("actual.md")
        assert feedback.accepted is True
        assert feedback.feedback_type == "approval"
    
    def test_content_features_creation(self):
        """Test ContentFeatures creation."""
        features = ContentFeatures(
            word_count=100,
            header_count=3,
            tags={"#productivity", "#notes"},
            topics=["organization", "knowledge management"]
        )
        
        assert features.word_count == 100
        assert features.header_count == 3
        assert "#productivity" in features.tags
        assert "#notes" in features.tags
        assert "organization" in features.topics
        assert "knowledge management" in features.topics


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the auto-organization system."""
    
    @pytest.fixture
    def full_auto_organizer(self, temp_vault, mock_content_analyzer, 
                           mock_embedding_service, mock_query_processor):
        """Create a fully configured auto-organizer for integration testing."""
        config = LibrarianConfig(
            vault_path=temp_vault,
            features={"auto_organization": True}
        )
        
        vault_session = MagicMock(spec=Vault)
        vault_session.vault_path = temp_vault
        
        async def mock_load_note(path):
            full_path = temp_vault / path
            if full_path.exists():
                content = full_path.read_text()
                return Note(
                    path=path,
                    content=content,
                    metadata=NoteMetadata(
                        created_time=datetime.now(),
                        modified_time=datetime.now(),
                        tags=[],
                        frontmatter={}
                    )
                )
            return None
        
        vault_session.load_note = AsyncMock(side_effect=mock_load_note)
        
        return AutoOrganizer(
            vault_session,
            mock_content_analyzer,
            mock_embedding_service,
            mock_query_processor,
            config
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_organization(self, full_auto_organizer, temp_vault):
        """Test end-to-end organization workflow."""
        # Create test files
        daily_note = temp_vault / "2024-01-15.md"
        daily_note.write_text("# Daily Note\nToday's tasks and reflections")
        
        meeting_note = temp_vault / "team-standup.md"
        meeting_note.write_text("# Team Standup\nAgenda:\n- Sprint review\n\nAction items:\n- [ ] Update docs")
        
        # Test organization
        daily_result = await full_auto_organizer.organize_file(Path("2024-01-15.md"))
        meeting_result = await full_auto_organizer.organize_file(Path("team-standup.md"))
        
        # Verify classifications
        assert daily_result.confidence != ClassificationConfidence.LOW
        assert meeting_result.confidence != ClassificationConfidence.LOW
        
        # Daily note should be organized by date
        assert "Daily Notes" in str(daily_result.suggested_path) or "2024" in str(daily_result.suggested_path)
        
        # Meeting note should be organized appropriately  
        assert meeting_result.action != OrganizationAction.IGNORE
    
    @pytest.mark.asyncio
    async def test_learning_from_feedback(self, full_auto_organizer):
        """Test that the system learns from user feedback."""
        # Create initial feedback
        feedback = UserFeedback(
            original_path=Path("unsorted/project-notes.md"),
            suggested_path=Path("wrong/project-notes.md"),
            actual_path=Path("Projects/Alpha/project-notes.md"),
            accepted=True,
            timestamp=datetime.now(),
            feedback_type="correction"
        )
        
        # Record feedback
        await full_auto_organizer.add_feedback(feedback)
        
        # Verify feedback was recorded
        assert len(full_auto_organizer.learner.feedback_history) == 1
        
        # The learning system should adapt (specific behavior depends on implementation)
        assert feedback in full_auto_organizer.learner.feedback_history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])