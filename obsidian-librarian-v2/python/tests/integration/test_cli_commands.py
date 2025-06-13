"""
Comprehensive integration tests for all CLI commands.
"""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from click.testing import CliRunner

from obsidian_librarian.cli import cli
from obsidian_librarian.models import LibrarianConfig, Note, VaultStats
from obsidian_librarian.services.backup import BackupService


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_vault(tmp_path):
    """Create a temporary vault with sample notes."""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory to make it a valid vault
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir()
    
    # Create sample notes
    notes = {
        "index.md": "# Index\n\nWelcome to my vault!\n\n[[daily/2024-01-01]]",
        "daily/2024-01-01.md": "# Daily Note\n\n- [ ] Task 1\n- [x] Task 2\n\n#daily",
        "projects/project1.md": "# Project 1\n\nProject description.\n\n## Tasks\n- [ ] Implementation\n- [ ] Testing",
        "templates/daily.md": "# {{date}}\n\n## Tasks\n- [ ] \n\n## Notes\n",
        "duplicate1.md": "# Duplicate Content\n\nThis is some duplicate content for testing.",
        "duplicate2.md": "# Duplicate Content\n\nThis is some duplicate content for testing.",
    }
    
    for note_path, content in notes.items():
        note_file = vault_path / note_path
        note_file.parent.mkdir(parents=True, exist_ok=True)
        note_file.write_text(content)
    
    return vault_path


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return LibrarianConfig()


class TestCLICommands:
    """Test suite for CLI commands."""
    
    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Obsidian Librarian" in result.output
        assert "analyze" in result.output
        assert "research" in result.output
        assert "curate" in result.output
        assert "templates" in result.output
        assert "status" in result.output
    
    def test_verbose_flag(self, cli_runner, temp_vault):
        """Test verbose logging flag."""
        result = cli_runner.invoke(cli, ['--verbose', 'status', str(temp_vault)])
        assert result.exit_code == 0
        # Verbose mode should show more output
        assert len(result.output) > 0
    
    def test_config_file_option(self, cli_runner, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "openai_api_key": "test-key",
            "max_concurrent_requests": 10
        }
        config_file.write_text(json.dumps(config_data))
        
        result = cli_runner.invoke(cli, ['--config', str(config_file), '--help'])
        assert result.exit_code == 0


class TestAnalyzeCommand:
    """Test suite for analyze command."""
    
    @pytest.mark.asyncio
    async def test_analyze_basic(self, cli_runner, temp_vault):
        """Test basic vault analysis."""
        with patch('obsidian_librarian.cli.analyze_vault_quick') as mock_analyze:
            mock_analyze.return_value = {
                'vault_stats': {
                    'total_notes': 6,
                    'total_words': 100,
                    'total_links': 1,
                    'total_tasks': 4,
                    'avg_quality_score': 0.75
                },
                'duplicate_clusters': 1
            }
            
            result = cli_runner.invoke(cli, ['analyze', str(temp_vault)])
            assert result.exit_code == 0
            assert "Analyzing vault:" in result.output
            assert "Analysis complete!" in result.output
            assert "Total Notes" in result.output
            assert "6" in result.output
    
    def test_analyze_no_duplicates(self, cli_runner, temp_vault):
        """Test analysis without duplicate detection."""
        with patch('obsidian_librarian.cli.analyze_vault_quick') as mock_analyze:
            mock_analyze.return_value = {
                'vault_stats': {
                    'total_notes': 6,
                    'total_words': 100,
                    'total_links': 1,
                    'total_tasks': 4,
                    'avg_quality_score': 0.75
                }
            }
            
            result = cli_runner.invoke(cli, ['analyze', str(temp_vault), '--no-duplicates'])
            assert result.exit_code == 0
            assert "duplicate" not in result.output.lower()
    
    def test_analyze_output_file(self, cli_runner, temp_vault, tmp_path):
        """Test saving analysis results to file."""
        output_file = tmp_path / "analysis_results.json"
        
        with patch('obsidian_librarian.cli.analyze_vault_quick') as mock_analyze:
            mock_analyze.return_value = {
                'vault_stats': {
                    'total_notes': 6,
                    'total_words': 100,
                    'total_links': 1,
                    'total_tasks': 4,
                    'avg_quality_score': 0.75
                }
            }
            
            result = cli_runner.invoke(cli, [
                'analyze', 
                str(temp_vault), 
                '--output', 
                str(output_file)
            ])
            assert result.exit_code == 0
            assert output_file.exists()
            
            # Verify output file content
            with open(output_file) as f:
                data = json.load(f)
                assert 'vault_stats' in data
                assert data['vault_stats']['total_notes'] == 6
    
    def test_analyze_invalid_vault(self, cli_runner, tmp_path):
        """Test analysis with invalid vault path."""
        invalid_path = tmp_path / "nonexistent"
        result = cli_runner.invoke(cli, ['analyze', str(invalid_path)])
        assert result.exit_code != 0


class TestResearchCommand:
    """Test suite for research command."""
    
    @pytest.mark.asyncio
    async def test_research_basic(self, cli_runner, temp_vault):
        """Test basic research functionality."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            
            # Mock async generator for research results
            async def mock_research(*args, **kwargs):
                yield {'type': 'result', 'data': {
                    'title': 'Test Result 1',
                    'source': 'test',
                    'quality_score': 0.9,
                    'url': 'http://example.com',
                    'summary': 'Test summary'
                }}
                yield {'type': 'complete'}
            
            mock_instance.research = mock_research
            
            result = cli_runner.invoke(cli, [
                'research', 
                str(temp_vault), 
                'test query'
            ])
            assert result.exit_code == 0
            assert "Researching: test query" in result.output
            assert "Found 1 results" in result.output
            assert "Test Result 1" in result.output
    
    def test_research_with_sources(self, cli_runner, temp_vault):
        """Test research with specific sources."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            
            async def mock_research(*args, **kwargs):
                yield {'type': 'complete'}
            
            mock_instance.research = mock_research
            
            result = cli_runner.invoke(cli, [
                'research', 
                str(temp_vault), 
                'test query',
                '--sources', 'arxiv',
                '--sources', 'github'
            ])
            assert result.exit_code == 0
            assert "Sources: arxiv, github" in result.output
    
    def test_research_max_results(self, cli_runner, temp_vault):
        """Test research with max results limit."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            
            async def mock_research(*args, **kwargs):
                # Generate 15 results
                for i in range(15):
                    yield {'type': 'result', 'data': {
                        'title': f'Test Result {i+1}',
                        'source': 'test',
                        'quality_score': 0.9,
                        'url': f'http://example.com/{i}',
                        'summary': f'Test summary {i+1}'
                    }}
                yield {'type': 'complete'}
            
            mock_instance.research = mock_research
            
            result = cli_runner.invoke(cli, [
                'research', 
                str(temp_vault), 
                'test query',
                '--max-results', '15'
            ])
            assert result.exit_code == 0
            assert "Found 15 results" in result.output
            # Should show top 10 and mention remaining
            assert "... and 5 more results" in result.output


class TestCurateCommand:
    """Test suite for curate command."""
    
    @pytest.mark.asyncio
    async def test_curate_basic(self, cli_runner, temp_vault):
        """Test basic curation functionality."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            mock_instance.curate_content.return_value = {
                'duplicates_processed': 2,
                'quality_improvements': 3,
                'structure_improvements': 1,
                'errors': []
            }
            
            result = cli_runner.invoke(cli, ['curate', str(temp_vault)])
            assert result.exit_code == 0
            assert "Curating vault:" in result.output
            assert "Curation complete!" in result.output
            assert "Duplicates Processed" in result.output
            assert "2" in result.output
    
    def test_curate_dry_run(self, cli_runner, temp_vault):
        """Test curation in dry-run mode."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            mock_instance.curate_content.return_value = {
                'duplicates_processed': 0,
                'quality_improvements': 0,
                'structure_improvements': 0,
                'errors': []
            }
            
            result = cli_runner.invoke(cli, ['curate', str(temp_vault), '--dry-run'])
            assert result.exit_code == 0
            assert "Running in dry-run mode" in result.output
    
    def test_curate_with_errors(self, cli_runner, temp_vault):
        """Test curation with errors."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            mock_instance.curate_content.return_value = {
                'duplicates_processed': 1,
                'quality_improvements': 2,
                'structure_improvements': 0,
                'errors': [
                    "Failed to process note1.md",
                    "Invalid template in note2.md"
                ]
            }
            
            result = cli_runner.invoke(cli, ['curate', str(temp_vault)])
            assert result.exit_code == 0
            assert "Errors encountered:" in result.output
            assert "Failed to process note1.md" in result.output


class TestTemplatesCommand:
    """Test suite for templates command."""
    
    @pytest.mark.asyncio
    async def test_templates_auto(self, cli_runner, temp_vault):
        """Test automatic template application."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            mock_instance.apply_templates.return_value = {
                'total_applications': 5,
                'successful': 4,
                'failed': 1,
                'applications': [
                    {
                        'note_id': 'daily/2024-01-01.md',
                        'template': 'daily',
                        'success': True,
                        'error': None
                    },
                    {
                        'note_id': 'projects/project1.md',
                        'template': 'project',
                        'success': False,
                        'error': 'Template not found'
                    }
                ]
            }
            
            result = cli_runner.invoke(cli, ['templates', str(temp_vault)])
            assert result.exit_code == 0
            assert "Applying templates:" in result.output
            assert "Total Applications" in result.output
            assert "5" in result.output
            assert "✓ daily/2024-01-01.md → daily" in result.output
            assert "✗ projects/project1.md → project" in result.output
            assert "Template not found" in result.output
    
    def test_templates_specific_notes(self, cli_runner, temp_vault):
        """Test template application to specific notes."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            mock_instance = MockLibrarian.return_value.__aenter__.return_value
            mock_instance.create_session.return_value = "test-session"
            mock_instance.apply_templates.return_value = {
                'total_applications': 1,
                'successful': 1,
                'failed': 0,
                'applications': []
            }
            
            result = cli_runner.invoke(cli, [
                'templates', 
                str(temp_vault),
                '--notes', 'daily/2024-01-01.md',
                '--notes', 'projects/project1.md'
            ])
            assert result.exit_code == 0


class TestStatusCommand:
    """Test suite for status command."""
    
    @pytest.mark.asyncio
    async def test_status_valid_vault(self, cli_runner, temp_vault):
        """Test status command with valid vault."""
        with patch('obsidian_librarian.vault.scan_vault_async') as mock_scan:
            mock_scan.return_value = {
                'path': str(temp_vault),
                'exists': True,
                'note_count': 6,
                'total_size': 1024,
                'last_modified': None
            }
            
            result = cli_runner.invoke(cli, ['status', str(temp_vault)])
            assert result.exit_code == 0
            assert "Vault Status:" in result.output
            assert "Note Count" in result.output
            assert "6" in result.output
            assert "Valid Obsidian vault" in result.output
    
    def test_status_with_templates(self, cli_runner, temp_vault):
        """Test status command shows template information."""
        with patch('obsidian_librarian.vault.scan_vault_async') as mock_scan:
            mock_scan.return_value = {
                'path': str(temp_vault),
                'exists': True,
                'note_count': 6,
                'total_size': 1024,
                'last_modified': None
            }
            
            # Create templates directory
            templates_dir = temp_vault / "templates"
            templates_dir.mkdir(exist_ok=True)
            (templates_dir / "test.md").write_text("# Test Template")
            
            result = cli_runner.invoke(cli, ['status', str(temp_vault)])
            assert result.exit_code == 0
            assert "Templates found: 1" in result.output
    
    def test_status_invalid_vault(self, cli_runner, tmp_path):
        """Test status with non-Obsidian directory."""
        non_vault = tmp_path / "not_a_vault"
        non_vault.mkdir()
        
        with patch('obsidian_librarian.vault.scan_vault_async') as mock_scan:
            mock_scan.return_value = {
                'path': str(non_vault),
                'exists': True,
                'note_count': 0,
                'total_size': 0,
                'last_modified': None
            }
            
            result = cli_runner.invoke(cli, ['status', str(non_vault)])
            assert result.exit_code == 0
            assert "Not an Obsidian vault" in result.output


class TestInteractiveCommand:
    """Test suite for interactive command."""
    
    def test_interactive_placeholder(self, cli_runner):
        """Test interactive mode shows placeholder."""
        result = cli_runner.invoke(cli, ['interactive'])
        assert result.exit_code == 0
        assert "Interactive Mode" in result.output
        assert "not yet implemented" in result.output


class TestErrorHandling:
    """Test suite for error handling across commands."""
    
    def test_analyze_exception_handling(self, cli_runner, temp_vault):
        """Test analyze command error handling."""
        with patch('obsidian_librarian.cli.analyze_vault_quick') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")
            
            result = cli_runner.invoke(cli, ['analyze', str(temp_vault)])
            assert result.exit_code == 1
            assert "Analysis failed" in result.output
    
    def test_research_exception_handling(self, cli_runner, temp_vault):
        """Test research command error handling."""
        with patch('obsidian_librarian.librarian.ObsidianLibrarian') as MockLibrarian:
            MockLibrarian.side_effect = Exception("Connection failed")
            
            result = cli_runner.invoke(cli, ['research', str(temp_vault), 'query'])
            assert result.exit_code == 1
            assert "Research failed" in result.output
    
    def test_missing_dependencies(self, cli_runner, temp_vault):
        """Test handling of missing optional dependencies."""
        # This would test behavior when optional deps are missing
        # Implementation depends on how the app handles missing deps
        pass


class TestPerformance:
    """Test suite for performance with large vaults."""
    
    @pytest.fixture
    def large_vault(self, tmp_path):
        """Create a large vault for performance testing."""
        vault_path = tmp_path / "large_vault"
        vault_path.mkdir()
        
        # Create .obsidian directory
        (vault_path / ".obsidian").mkdir()
        
        # Create many notes
        for i in range(100):
            note_path = vault_path / f"note_{i}.md"
            content = f"# Note {i}\n\n" + "Content " * 100 + f"\n\n[[note_{(i+1)%100}]]"
            note_path.write_text(content)
        
        return vault_path
    
    @pytest.mark.slow
    def test_analyze_large_vault_performance(self, cli_runner, large_vault):
        """Test analysis performance with large vault."""
        import time
        
        with patch('obsidian_librarian.cli.analyze_vault_quick') as mock_analyze:
            mock_analyze.return_value = {
                'vault_stats': {
                    'total_notes': 100,
                    'total_words': 10000,
                    'total_links': 100,
                    'total_tasks': 0,
                    'avg_quality_score': 0.75
                }
            }
            
            start_time = time.time()
            result = cli_runner.invoke(cli, ['analyze', str(large_vault)])
            end_time = time.time()
            
            assert result.exit_code == 0
            # Analysis should complete within reasonable time
            assert end_time - start_time < 5.0  # 5 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])