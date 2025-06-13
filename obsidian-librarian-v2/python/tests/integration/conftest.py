"""
Shared fixtures and configuration for integration tests.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's automatically cleaned up."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses."""
    mock = MagicMock()
    
    # Mock embedding response
    mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    
    # Mock completion response  
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="Mock AI response")
        )]
    )
    
    return mock


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-456")
    monkeypatch.setenv("OBSIDIAN_LIBRARIAN_ENV", "test")


@pytest.fixture
def sample_notes() -> Dict[str, str]:
    """Sample note content for testing."""
    return {
        "index.md": """# My Vault Index

Welcome to my knowledge base!

## Areas
- [[projects/active]]
- [[areas/learning]]
- [[references/books]]

## Recent Notes
- [[daily/2024-01-15]]
- [[daily/2024-01-14]]

Tags: #index #organization
""",
        
        "daily/2024-01-15.md": """# 2024-01-15

## Tasks
- [x] Morning review
- [ ] Write project proposal
- [ ] Review PRs

## Notes
Had a productive meeting with the team about [[projects/new-feature]].

## Reflections
Good progress on the project today.

Tags: #daily #journal
""",
        
        "projects/new-feature.md": """# New Feature Development

## Overview
Developing a new feature for the application.

## Requirements
- User authentication
- Data persistence  
- API integration

## Timeline
- Start: 2024-01-01
- MVP: 2024-02-01
- Launch: 2024-03-01

## Team
- [[team/john]] - Backend
- [[team/jane]] - Frontend
- [[team/bob]] - Design

Tags: #project #active #development
""",
        
        "references/books/clean-code.md": """# Clean Code by Robert C. Martin

## Key Concepts
1. Meaningful names
2. Functions should do one thing
3. Comments are a failure to express in code
4. Formatting matters
5. Error handling is important

## Quotes
> "Clean code reads like well-written prose."

## Application
- Apply to [[projects/new-feature]]
- Share with [[team/john]]

Tags: #book #reference #programming #clean-code
""",
        
        "templates/daily.md": """# {{date}}

## Tasks
- [ ] 

## Schedule
{{schedule}}

## Notes


## Reflections


Tags: #daily #journal
""",
        
        "templates/project.md": """# {{title}}

## Overview
{{description}}

## Goals
- [ ] 

## Timeline
- Start: {{start_date}}
- End: {{end_date}}

## Resources
- 

## Team
- 

Tags: #project {{tags}}
"""
    }


@pytest.fixture
async def initialized_vault(temp_dir, sample_notes):
    """Create an initialized vault with sample content."""
    vault_path = temp_dir / "test_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir()
    
    # Create sample notes
    for note_path, content in sample_notes.items():
        file_path = vault_path / note_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    return vault_path


@pytest.fixture
def mock_async_client():
    """Mock async HTTP client."""
    client = AsyncMock()
    
    # Mock successful responses
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"success": True})
    response.text = AsyncMock(return_value="Mock response")
    
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    
    return client


@pytest.fixture
def performance_monitoring():
    """Fixture for performance monitoring in tests."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            if self.start_time is None:
                raise ValueError("Monitor not started")
            
            duration = time.time() - self.start_time
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - self.start_memory
            
            return {
                "duration": duration,
                "memory_used": memory_used,
                "start_memory": self.start_memory,
                "end_memory": end_memory
            }
    
    return PerformanceMonitor()


# Markers for different test categories
pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.requires_api = pytest.mark.requires_api


# Skip tests that require external services in CI
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require external API access"
    )


# Async test utilities
async def wait_for_condition(condition_func, timeout=5, interval=0.1):
    """Wait for a condition to become true."""
    start_time = asyncio.get_event_loop().time()
    
    while True:
        if await condition_func():
            return True
        
        if asyncio.get_event_loop().time() - start_time > timeout:
            return False
        
        await asyncio.sleep(interval)


async def create_test_notes(vault_path: Path, count: int) -> List[Path]:
    """Create multiple test notes quickly."""
    notes = []
    
    for i in range(count):
        note_path = vault_path / f"test_note_{i}.md"
        content = f"""# Test Note {i}

This is test note number {i}.

Links: [[test_note_{(i+1) % count}]]

Tags: #test #note{i % 10}
"""
        note_path.write_text(content)
        notes.append(note_path)
    
    return notes


# Test data generators
def generate_markdown_content(size: str = "small") -> str:
    """Generate markdown content of various sizes."""
    if size == "small":
        return """# Small Note

A brief note with minimal content.

Tags: #small #test
"""
    elif size == "medium":
        return """# Medium Note

## Introduction
This is a medium-sized note with more content.

## Main Points
1. First point with some explanation
2. Second point with details
3. Third point with examples

## Conclusion
Summary of the main ideas.

## References
- [[reference1]]
- [[reference2]]

Tags: #medium #test #example
""" + "\n".join([f"Additional paragraph {i}." for i in range(10)])
    
    elif size == "large":
        sections = []
        for i in range(20):
            sections.append(f"""
## Section {i}

This is section {i} with substantial content. It includes multiple paragraphs 
and various markdown elements.

### Subsection {i}.1
Details about this subsection with **bold** and *italic* text.

### Subsection {i}.2  
- List item 1
- List item 2
- List item 3

### Code Example
```python
def function_{i}():
    return "Example {i}"
```
""")
        
        return "# Large Note\n\n" + "\n".join(sections) + "\n\nTags: #large #test #comprehensive"
    
    else:
        raise ValueError(f"Unknown size: {size}")


# Utility functions for tests
def assert_file_contains(file_path: Path, expected_content: List[str]):
    """Assert that a file contains all expected content."""
    content = file_path.read_text()
    for expected in expected_content:
        assert expected in content, f"Expected '{expected}' not found in {file_path}"


def assert_file_not_contains(file_path: Path, unexpected_content: List[str]):
    """Assert that a file does not contain unexpected content."""
    content = file_path.read_text()
    for unexpected in unexpected_content:
        assert unexpected not in content, f"Unexpected '{unexpected}' found in {file_path}"