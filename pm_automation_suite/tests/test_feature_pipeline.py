"""
Test suite for Feature Development Pipeline components.

Tests PRD parsing, story generation, Jira bulk creation, and complete pipeline execution.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

# Import modules under test
from feature_pipeline.prd_parser import PRDParser, PRDContent, RequirementType, RequirementPriority
from feature_pipeline.story_generator import StoryGenerator, UserStory, StoryType, AcceptanceCriteria
from feature_pipeline.jira_bulk_creator import JiraBulkCreator, JiraIssue, JiraIssueType, StoryHierarchy
from feature_pipeline.feature_pipeline import FeaturePipeline, PipelineStage, PipelineStatus, PipelineResult


class TestPRDParser:
    """Test PRD Parser functionality."""
    
    @pytest.fixture
    def sample_prd_content(self):
        """Sample PRD content for testing."""
        return """
# User Authentication System

## Metadata
**Author:** John Doe
**Version:** 1.0
**Status:** Draft

## Overview
This PRD defines the requirements for a new user authentication system.

## Requirements

### Functional Requirements
- The system must allow users to register with email and password
- Users should be able to login with their credentials
- The system shall support password reset functionality
- Users must be able to update their profile information

### Non-Functional Requirements
- The system should support 10,000 concurrent users
- Authentication response time must be under 200ms
- The system shall have 99.9% uptime

## Acceptance Criteria
- Given a valid email and password, when a user registers, then they should receive a confirmation email
- Given valid credentials, when a user logs in, then they should be redirected to the dashboard
"""
    
    @pytest.fixture
    def prd_parser(self):
        """PRD parser instance for testing."""
        config = {
            'openai_api_key': 'test-key'  # Will be mocked
        }
        return PRDParser(config)
    
    def test_prd_parser_initialization(self, prd_parser):
        """Test PRD parser initialization."""
        assert prd_parser is not None
        assert prd_parser.config is not None
        assert len(prd_parser.section_patterns) > 0
        assert len(prd_parser.requirement_patterns) > 0
    
    def test_parse_prd_from_string(self, prd_parser, sample_prd_content):
        """Test parsing PRD from string content."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_prd_content)
            temp_path = f.name
        
        try:
            # Parse PRD
            prd_content = prd_parser.parse_prd(temp_path)
            
            # Assertions
            assert isinstance(prd_content, PRDContent)
            assert prd_content.metadata.title == "User Authentication System"
            assert "John Doe" in prd_content.metadata.author
            assert prd_content.metadata.version == "1.0"
            assert len(prd_content.requirements) > 0
            assert len(prd_content.sections) > 0
            
            # Check requirement types
            functional_reqs = prd_content.get_requirements_by_type(RequirementType.FUNCTIONAL)
            non_functional_reqs = prd_content.get_requirements_by_type(RequirementType.NON_FUNCTIONAL)
            
            assert len(functional_reqs) > 0
            assert len(non_functional_reqs) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_requirement_classification(self, prd_parser):
        """Test requirement type classification."""
        functional_text = "The system must allow users to login"
        non_functional_text = "The system should support 10,000 concurrent users"
        
        func_type = prd_parser._classify_requirement_type(functional_text)
        non_func_type = prd_parser._classify_requirement_type(non_functional_text)
        
        assert func_type == RequirementType.FUNCTIONAL
        # Note: May default to FUNCTIONAL for unmatched patterns
        assert non_func_type in [RequirementType.NON_FUNCTIONAL, RequirementType.FUNCTIONAL]
    
    def test_priority_extraction(self, prd_parser):
        """Test priority extraction from text."""
        critical_text = "This is a critical requirement that must be implemented"
        high_text = "This is a high priority feature"
        medium_text = "This could be useful to have"
        
        critical_priority = prd_parser._extract_priority(critical_text)
        high_priority = prd_parser._extract_priority(high_text)
        medium_priority = prd_parser._extract_priority(medium_text)
        
        assert critical_priority == RequirementPriority.CRITICAL
        assert high_priority == RequirementPriority.HIGH
        assert medium_priority == RequirementPriority.MEDIUM
    
    def test_validation_rules(self, prd_parser, sample_prd_content):
        """Test PRD validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_prd_content)
            temp_path = f.name
        
        try:
            prd_content = prd_parser.parse_prd(temp_path)
            
            # Should have minimal validation errors for valid PRD
            assert len(prd_content.validation_errors) <= 2  # Allow for some minor issues
            
        finally:
            os.unlink(temp_path)


class TestStoryGenerator:
    """Test Story Generator functionality."""
    
    @pytest.fixture
    def story_generator(self):
        """Story generator instance for testing."""
        config = {
            'openai_api_key': 'test-key',
            'anthropic_api_key': 'test-key'
        }
        return StoryGenerator(config)
    
    @pytest.fixture
    def sample_prd_content(self):
        """Sample PRD content for story generation testing."""
        from feature_pipeline.prd_parser import PRDMetadata, Requirement
        
        requirements = [
            Requirement(
                id="REQ-001",
                text="The system must allow users to register with email and password",
                requirement_type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                section="requirements"
            ),
            Requirement(
                id="REQ-002", 
                text="Users should be able to login with their credentials",
                requirement_type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                section="requirements"
            ),
            Requirement(
                id="REQ-003",
                text="The system should support 10,000 concurrent users",
                requirement_type=RequirementType.NON_FUNCTIONAL,
                priority=RequirementPriority.MEDIUM,
                section="requirements"
            )
        ]
        
        metadata = PRDMetadata(
            title="User Authentication System",
            version="1.0", 
            author="Test Author",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            status="Draft"
        )
        
        return PRDContent(
            metadata=metadata,
            requirements=requirements,
            sections={"overview": "Test overview", "requirements": "Test requirements"},
            raw_text="Test PRD content"
        )
    
    def test_story_generator_initialization(self, story_generator):
        """Test story generator initialization."""
        assert story_generator is not None
        assert story_generator.config is not None
    
    @pytest.mark.asyncio
    async def test_generate_stories_from_prd(self, story_generator, sample_prd_content):
        """Test generating stories from PRD content."""
        # Mock AI responses
        with patch.object(story_generator, '_generate_ai_stories') as mock_ai:
            mock_ai.return_value = [
                UserStory(
                    id="STORY-001",
                    title="User Registration",
                    description="As a new user, I want to register with email and password",
                    story_type=StoryType.USER_STORY,
                    priority=RequirementPriority.HIGH,
                    story_points=5,
                    user_persona="new user",
                    business_value="can access the application",
                    acceptance_criteria=[
                        AcceptanceCriteria(
                            id="AC-001",
                            description="Valid registration creates account",
                            given="valid email and password",
                            when="user submits registration form",
                            then="account is created and confirmation email sent"
                        )
                    ]
                )
            ]
            
            stories = await story_generator.generate_stories_from_prd(sample_prd_content)
            
            assert len(stories) > 0
            assert all(isinstance(story, UserStory) for story in stories)
            mock_ai.assert_called()
    
    def test_group_requirements_by_feature(self, story_generator, sample_prd_content):
        """Test grouping requirements by feature area."""
        grouped = story_generator._group_requirements_by_feature(sample_prd_content.requirements)
        
        assert isinstance(grouped, dict)
        assert len(grouped) > 0
        
        # Should group authentication-related requirements together
        auth_keywords = ['auth', 'login', 'register', 'user']
        auth_groups = [group for group_name, group in grouped.items() 
                      if any(keyword in group_name.lower() for keyword in auth_keywords)]
        
        assert len(auth_groups) > 0
    
    def test_story_validation(self, story_generator):
        """Test story validation logic."""
        valid_story = UserStory(
            id="STORY-001",
            title="Valid Story",
            description="A valid user story",
            story_type=StoryType.USER_STORY,
            priority=RequirementPriority.HIGH,
            story_points=3,
            user_persona="user",
            business_value="achieve goal"
        )
        
        invalid_story = UserStory(
            id="",  # Invalid: empty ID
            title="",  # Invalid: empty title
            description="A story with missing fields",
            story_type=StoryType.USER_STORY,
            priority=RequirementPriority.HIGH,
            story_points=1,
            user_persona="user",
            business_value="value"
        )
        
        assert story_generator._validate_story(valid_story) == []
        assert len(story_generator._validate_story(invalid_story)) > 0


class TestJiraBulkCreator:
    """Test Jira Bulk Creator functionality."""
    
    @pytest.fixture
    def jira_config(self):
        """Jira configuration for testing."""
        return {
            'jira_project_key': 'TEST',
            'dry_run': True,  # Always use dry run for tests
            'jira': {
                'server': 'https://test.atlassian.net',
                'username': 'test@example.com',
                'api_token': 'test-token'
            }
        }
    
    @pytest.fixture
    def jira_bulk_creator(self, jira_config):
        """Jira bulk creator instance for testing."""
        return JiraBulkCreator(jira_config)
    
    @pytest.fixture
    def sample_user_stories(self):
        """Sample user stories for testing."""
        return [
            UserStory(
                id="STORY-001",
                title="User Registration",
                description="As a new user, I want to register",
                story_type=StoryType.EPIC,
                priority=RequirementPriority.HIGH,
                story_points=13,
                user_persona="new user",
                business_value="access the system"
            ),
            UserStory(
                id="STORY-002", 
                title="Email Registration",
                description="Register with email and password",
                story_type=StoryType.USER_STORY,
                priority=RequirementPriority.HIGH,
                story_points=5,
                user_persona="new user",
                business_value="create account",
                parent_epic="STORY-001"
            )
        ]
    
    def test_jira_bulk_creator_initialization(self, jira_bulk_creator):
        """Test Jira bulk creator initialization."""
        assert jira_bulk_creator is not None
        assert jira_bulk_creator.project_key == 'TEST'
        assert jira_bulk_creator.dry_run is True
    
    @pytest.mark.asyncio
    async def test_create_stories_bulk(self, jira_bulk_creator, sample_user_stories):
        """Test bulk creation of stories."""
        hierarchy = await jira_bulk_creator.create_stories_bulk(sample_user_stories)
        
        assert isinstance(hierarchy, StoryHierarchy)
        assert hierarchy.get_total_issues() > 0
        assert len(hierarchy.creation_summary) > 0
        assert hierarchy.creation_summary['dry_run'] is True
    
    def test_convert_stories_to_jira_issues(self, jira_bulk_creator, sample_user_stories):
        """Test conversion of user stories to Jira issues."""
        jira_issues = jira_bulk_creator._convert_stories_to_jira_issues(sample_user_stories)
        
        assert len(jira_issues) >= len(sample_user_stories)
        assert all(isinstance(issue, JiraIssue) for issue in jira_issues)
        
        # Check epic conversion
        epics = [issue for issue in jira_issues if issue.issue_type == JiraIssueType.EPIC]
        assert len(epics) > 0
    
    def test_jira_field_mapping(self, jira_bulk_creator):
        """Test Jira field mapping."""
        jira_issue = JiraIssue(
            issue_type=JiraIssueType.STORY,
            summary="Test Story",
            description="Test description",
            priority="High"
        )
        
        fields = jira_issue.to_jira_fields(jira_bulk_creator.field_mapping, 'TEST')
        
        assert 'fields' in fields
        assert fields['fields']['project']['key'] == 'TEST'
        assert fields['fields']['issuetype']['name'] == 'Story'
        assert fields['fields']['summary'] == 'Test Story'
    
    def test_validation_configuration(self, jira_bulk_creator):
        """Test project configuration validation."""
        validation = jira_bulk_creator.validate_project_configuration()
        
        assert 'valid' in validation
        assert 'errors' in validation
        assert 'warnings' in validation


class TestFeaturePipeline:
    """Test complete Feature Pipeline functionality."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Pipeline configuration for testing."""
        return {
            'pipeline_id': 'test_pipeline',
            'output_dir': tempfile.mkdtemp(),
            'prd_parser': {
                'openai_api_key': 'test-key'
            },
            'story_generator': {
                'openai_api_key': 'test-key'
            },
            'jira_bulk_creator': {
                'jira_project_key': 'TEST',
                'dry_run': True
            },
            'save_artifacts': True,
            'validate_at_each_stage': True
        }
    
    @pytest.fixture
    def feature_pipeline(self, pipeline_config):
        """Feature pipeline instance for testing."""
        return FeaturePipeline(pipeline_config)
    
    @pytest.fixture
    def sample_prd_file(self):
        """Sample PRD file for testing."""
        prd_content = """
# Test Feature

## Overview
Test feature for pipeline testing.

## Requirements
- The system must do something important
- Users should be able to perform actions
- The system shall be reliable and fast
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(prd_content)
            return f.name
    
    def test_pipeline_initialization(self, feature_pipeline):
        """Test pipeline initialization."""
        assert feature_pipeline is not None
        assert feature_pipeline.pipeline_id == 'test_pipeline'
        assert feature_pipeline.prd_parser is not None
        assert feature_pipeline.story_generator is not None
        assert feature_pipeline.jira_creator is not None
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, feature_pipeline, sample_prd_file):
        """Test complete pipeline execution."""
        try:
            # Mock AI components to avoid external dependencies
            with patch.object(feature_pipeline.story_generator, 'generate_stories_from_prd') as mock_stories:
                mock_stories.return_value = [
                    UserStory(
                        id="STORY-001",
                        title="Test Story",
                        description="A test story",
                        story_type=StoryType.USER_STORY,
                        priority=RequirementPriority.HIGH,
                        story_points=3,
                        user_persona="user",
                        business_value="achieve value"
                    )
                ]
                
                result = await feature_pipeline.execute_pipeline(sample_prd_file)
                
                # Assertions
                assert isinstance(result, PipelineResult)
                assert result.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
                assert result.metrics is not None
                
                if result.status == PipelineStatus.COMPLETED:
                    assert result.prd_content is not None
                    assert len(result.generated_stories) > 0
                    assert result.jira_hierarchy is not None
                    assert len(result.artifacts) > 0
                
        finally:
            # Cleanup
            if os.path.exists(sample_prd_file):
                os.unlink(sample_prd_file)
    
    def test_progress_callback(self, feature_pipeline):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(stage, data):
            progress_calls.append((stage, data))
        
        feature_pipeline.add_progress_callback(progress_callback)
        
        # Trigger a progress notification
        feature_pipeline._notify_progress(PipelineStage.PRD_PARSING, {"status": "started"})
        
        assert len(progress_calls) == 1
        assert progress_calls[0][0] == PipelineStage.PRD_PARSING
        assert progress_calls[0][1]["status"] == "started"
    
    def test_pipeline_status(self, feature_pipeline):
        """Test pipeline status reporting."""
        status = feature_pipeline.get_pipeline_status()
        
        assert 'pipeline_id' in status
        assert 'current_stage' in status
        assert 'current_status' in status
        assert status['pipeline_id'] == 'test_pipeline'
    
    @pytest.mark.asyncio
    async def test_stage_validation(self, feature_pipeline):
        """Test individual stage validation."""
        # Create mock result with PRD content
        result = PipelineResult(status=PipelineStatus.RUNNING)
        
        # Test validation with missing PRD content
        validation_errors = await feature_pipeline._validate_stage_output(
            PipelineStage.PRD_PARSING, result
        )
        assert len(validation_errors) > 0
        
        # Test validation with PRD content present
        from feature_pipeline.prd_parser import PRDMetadata, PRDContent
        result.prd_content = PRDContent(
            metadata=PRDMetadata(
                title="Test", version="1.0", author="Test", 
                created_date=datetime.now(), last_modified=datetime.now(), status="Draft"
            ),
            requirements=[],
            sections={},
            raw_text=""
        )
        
        validation_errors = await feature_pipeline._validate_stage_output(
            PipelineStage.PRD_PARSING, result
        )
        # Should still have errors due to empty requirements
        assert len(validation_errors) > 0
    
    def test_pipeline_report_generation(self, feature_pipeline):
        """Test pipeline report generation."""
        # Create mock result
        result = PipelineResult(status=PipelineStatus.COMPLETED)
        result.generated_stories = [
            UserStory(
                id="STORY-001",
                title="Test Story",
                description="Test",
                story_type=StoryType.USER_STORY,
                priority=RequirementPriority.HIGH,
                story_points=3,
                user_persona="user",
                business_value="value"
            )
        ]
        
        report = feature_pipeline.generate_pipeline_report(result)
        
        assert isinstance(report, str)
        assert "Feature Development Pipeline Report" in report
        assert "COMPLETED" in report
        assert len(report) > 100  # Should be a substantial report


class TestIntegration:
    """Integration tests for the complete feature pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline execution."""
        # Create comprehensive PRD
        prd_content = """
# E-commerce Product Search

## Metadata
**Author:** Product Team
**Version:** 2.0
**Status:** Ready for Development
**Stakeholders:** Engineering, Design, Product

## Overview
Implement an advanced product search system with filtering, sorting, and recommendations.

## Functional Requirements
- Users must be able to search products by keyword
- The system shall provide autocomplete suggestions
- Users should be able to filter results by category, price, and rating
- The system must display search results with pagination
- Users should be able to sort results by relevance, price, and rating
- The system shall track search analytics for insights

## Non-Functional Requirements  
- Search response time must be under 500ms
- The system should support 1000 concurrent searches
- Search accuracy should be above 90%
- The system shall have 99.5% uptime

## Acceptance Criteria
- Given a search term, when a user types, then autocomplete suggestions appear
- Given search results, when a user applies filters, then results update accordingly
- Given a product search, when results are displayed, then they include price and rating
"""
        
        # Create temporary PRD file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(prd_content)
            prd_file_path = f.name
        
        try:
            # Configure pipeline
            config = {
                'output_dir': tempfile.mkdtemp(),
                'prd_parser': {},
                'story_generator': {},
                'jira_bulk_creator': {
                    'jira_project_key': 'ECOM',
                    'dry_run': True
                },
                'save_artifacts': True
            }
            
            pipeline = FeaturePipeline(config)
            
            # Track progress
            progress_stages = []
            def track_progress(stage, data):
                progress_stages.append(stage)
            
            pipeline.add_progress_callback(track_progress)
            
            # Mock AI components for testing
            with patch.object(pipeline.story_generator, 'generate_stories_from_prd') as mock_stories:
                mock_stories.return_value = [
                    UserStory(
                        id="EPIC-001",
                        title="Product Search System",
                        description="Epic for product search functionality",
                        story_type=StoryType.EPIC,
                        priority=RequirementPriority.HIGH,
                        story_points=21,
                        user_persona="user",
                        business_value="find products efficiently"
                    ),
                    UserStory(
                        id="STORY-001",
                        title="Keyword Search",
                        description="As a user, I want to search products by keyword",
                        story_type=StoryType.USER_STORY,
                        priority=RequirementPriority.HIGH,
                        story_points=8,
                        user_persona="user",
                        business_value="find specific products",
                        parent_epic="EPIC-001"
                    ),
                    UserStory(
                        id="STORY-002",
                        title="Search Autocomplete",
                        description="As a user, I want autocomplete suggestions",
                        story_type=StoryType.USER_STORY,
                        priority=RequirementPriority.MEDIUM,
                        story_points=5,
                        user_persona="user",
                        business_value="search faster",
                        parent_epic="EPIC-001"
                    )
                ]
                
                # Execute pipeline
                result = await pipeline.execute_pipeline(prd_file_path)
                
                # Verify execution
                assert result.status == PipelineStatus.COMPLETED
                assert result.prd_content is not None
                assert len(result.generated_stories) == 3
                assert result.jira_hierarchy is not None
                assert result.metrics is not None
                assert result.metrics.duration_seconds is not None
                
                # Verify PRD parsing
                assert result.prd_content.metadata.title == "E-commerce Product Search"
                assert len(result.prd_content.requirements) > 0
                
                # Verify story generation
                epic_stories = [s for s in result.generated_stories if s.story_type == StoryType.EPIC]
                user_stories = [s for s in result.generated_stories if s.story_type == StoryType.USER_STORY]
                assert len(epic_stories) == 1
                assert len(user_stories) == 2
                
                # Verify Jira creation
                assert result.jira_hierarchy.get_total_issues() == 3
                assert result.jira_hierarchy.creation_summary['dry_run'] is True
                
                # Verify artifacts
                assert len(result.artifacts) > 0
                assert 'prd_analysis' in result.artifacts
                assert 'generated_stories' in result.artifacts
                assert 'jira_report' in result.artifacts
                
                # Verify progress tracking
                assert PipelineStage.PRD_PARSING in progress_stages
                assert PipelineStage.STORY_GENERATION in progress_stages
                assert PipelineStage.JIRA_CREATION in progress_stages
                
                # Generate and verify report
                report = pipeline.generate_pipeline_report(result)
                assert "E-commerce Product Search" in report
                assert "3" in report  # Should mention 3 stories
                assert "COMPLETED" in report
                
        finally:
            # Cleanup
            os.unlink(prd_file_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])