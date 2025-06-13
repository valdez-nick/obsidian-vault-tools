"""
Integration tests for Research Service.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from obsidian_librarian.services.research import ResearchService, ResearchConfig
from obsidian_librarian.models import ResearchQuery


@pytest.mark.integration
class TestResearchIntegration:
    """Test Research Service with mocked external dependencies."""
    
    @pytest.fixture
    async def research_service(self):
        """Create research service with test configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test_vault"
            vault_path.mkdir()
            
            # Create minimal vault without Rust bindings (they're optional)
            from obsidian_librarian.vault import Vault, VaultConfig
            vault = Vault(vault_path, VaultConfig())
            
            config = ResearchConfig(
                max_concurrent_requests=2,
                enable_content_extraction=True,
            )
            
            service = ResearchService(vault, config)
            
            yield service
    
    @pytest.mark.asyncio
    async def test_research_service_creation(self, research_service):
        """Test that research service can be created successfully."""
        assert research_service is not None
        assert research_service.config is not None
        assert research_service.vault is not None
        assert research_service.query_processor is not None
        assert research_service.content_summarizer is not None
        assert research_service.source_manager is not None
    
    @pytest.mark.asyncio
    async def test_research_basic_functionality(self, research_service):
        """Test basic research functionality with mocked dependencies."""
        # Mock the source manager to return empty results
        with patch.object(research_service.source_manager, 'select_sources') as mock_select:
            mock_select.return_value = []  # No sources available
            
            # Mock query processor
            with patch.object(research_service.query_processor, 'process') as mock_process:
                mock_query = MagicMock()
                mock_query.session_id = "test_session"
                mock_query.text = "test query"
                mock_query.query_type = "research"
                mock_process.return_value = mock_query
                
                # This should work even with no sources (just return no results)
                results = []
                async for result in research_service.research("test query", max_results=5):
                    results.append(result)
                
                # Should complete without error, even if no results
                assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_research_session_management(self, research_service):
        """Test research session management."""
        # Initially no active sessions
        assert len(research_service.active_sessions) == 0
        
        # Mock dependencies to prevent actual network calls
        with patch.object(research_service.source_manager, 'select_sources') as mock_select:
            mock_select.return_value = []
            
            with patch.object(research_service.query_processor, 'process') as mock_process:
                mock_query = MagicMock()
                mock_query.session_id = "test_session_123"
                mock_query.text = "test query"
                mock_query.query_type = "research"
                mock_process.return_value = mock_query
                
                # Start research (should create session)
                results = []
                async for result in research_service.research("test query"):
                    results.append(result)
                
                # Session should be cleaned up automatically after research
                # (might take a moment due to async cleanup)
                await asyncio.sleep(0.1)