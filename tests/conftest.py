"""
Pytest configuration and shared fixtures for all tests
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_vault(tmp_path):
    """Create a temporary test vault"""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()
    
    # Make it a valid Obsidian vault
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir()
    
    # Create sample files
    sample_files = {
        "note1.md": "# Note 1\n\nContent with #tag1 and #tag2",
        "note2.md": "# Note 2\n\nContent with #tag2 and #tag3",
        "untagged.md": "# Untagged Note\n\nThis note has no tags",
        "Daily Notes/2024-01-01.md": "# Daily Note\n\n- [ ] Task 1\n- [x] Task 2",
        "Projects/project1.md": "# Project 1\n\n#project #active",
    }
    
    for file_path, content in sample_files.items():
        full_path = vault_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    yield str(vault_path)
    
    # Cleanup is automatic with tmp_path


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock optional dependencies for testing"""
    # Mock pygame for audio tests
    class MockPygame:
        class mixer:
            @staticmethod
            def init():
                pass
            
            @staticmethod
            def quit():
                pass
            
            class Sound:
                def __init__(self, path):
                    self.path = path
                
                def play(self):
                    pass
    
    # Mock MCP if needed
    class MockMCP:
        pass
    
    return {
        'pygame': MockPygame,
        'mcp': MockMCP
    }


@pytest.fixture(autouse=True)
def suppress_audio(monkeypatch):
    """Automatically suppress audio in all tests"""
    monkeypatch.setenv("DISABLE_AUDIO", "1")


@pytest.fixture
def cli_runner():
    """Create a CLI test runner"""
    from click.testing import CliRunner
    return CliRunner()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )