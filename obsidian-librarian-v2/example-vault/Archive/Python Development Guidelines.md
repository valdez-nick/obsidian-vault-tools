---
title: Python Development Guidelines
tags: [python, development, guidelines]
created: 2023-12-15
status: archived
---

# Python Development Guidelines

> Note: This document contains similar information to [[Python Best Practices]]. Consider merging.

## Coding Standards

### Following PEP 8

- Indentation: 4 spaces (never tabs)
- Line length: Maximum 79 characters
- Naming conventions:
  - Functions: `lower_case_with_underscores`
  - Classes: `CapitalizedWords`
  - Constants: `ALL_CAPS_WITH_UNDERSCORES`

### Type Annotations

Use type hints to improve code readability:

```python
from typing import List, Dict, Optional

def process_items(data: List[str], settings: Optional[Dict] = None) -> bool:
    """Process a list of items with optional settings."""
    settings = settings or {}
    # Implementation here
    return True
```

## Asynchronous Programming

### AsyncIO Best Practices

1. **Proper async/await usage**
   ```python
   async def get_data(endpoint: str) -> dict:
       async with aiohttp.ClientSession() as session:
           async with session.get(endpoint) as resp:
               return await resp.json()
   ```

2. **Avoid blocking calls**
   - Use async versions of I/O operations
   - Run CPU-bound tasks in thread pool

## Testing Approach

### Unit Testing with Pytest

- Follow Test-Driven Development (TDD)
- Maintain high code coverage (>80%)
- Use fixtures for test setup

Example test:
```python
import pytest

class TestFeature:
    @pytest.fixture
    def setup(self):
        return {"config": "value"}
    
    def test_feature(self, setup):
        assert setup["config"] == "value"
```

## Performance Considerations

1. Profile first, optimize second
2. Use appropriate data structures
3. Cache expensive operations
4. Consider generator expressions for memory efficiency

## Environment Setup

### Virtual Environments

```bash
# Create virtual environment
python -m venv env

# Activate
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### Managing Dependencies

- Use requirements.txt or modern pyproject.toml
- Pin versions for reproducibility
- Keep dependencies updated

## Code Documentation

### Writing Good Docstrings

```python
def compute_mean(values: List[float]) -> float:
    """Compute arithmetic mean of values.
    
    Args:
        values: List of numeric values.
        
    Returns:
        Mean of the input values.
        
    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Empty list provided")
    return sum(values) / len(values)
```

## See Also

- [[Testing Best Practices]]
- [[Async Programming Guide]]
- [[Performance Optimization]]

---
*This is an older version of our Python guidelines. See [[Python Best Practices]] for the latest version.*