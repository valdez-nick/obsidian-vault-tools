"""
Pytest configuration for Obsidian Librarian tests.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add the package to Python path for testing
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_vault(tmp_path):
    """Create a temporary vault directory with basic structure."""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()
    
    # Create basic structure
    (vault_path / ".obsidian").mkdir()
    (vault_path / "Templates").mkdir()
    (vault_path / "Notes").mkdir()
    
    # Create a few test notes
    (vault_path / "note1.md").write_text("# Note 1\n\nContent for note 1")
    (vault_path / "note2.md").write_text("# Note 2\n\nContent with [[note1]] link")
    (vault_path / "Templates" / "daily.md").write_text("# {{date}}\n\n## Tasks\n- [ ] ")
    
    return vault_path


@pytest.fixture
def sample_note_content():
    """Provide sample note content for testing."""
    return """---
title: Sample Note
tags: [test, sample]
created: 2024-01-01
---

# Sample Note

This is a sample note for testing.

## Section 1

Some content here with a [[wiki link]] and a [regular link](https://example.com).

## Tasks

- [ ] Task 1
- [x] Completed task
- [ ] Task 3 #todo

## Code

```python
def hello():
    print("Hello, world!")
```
"""