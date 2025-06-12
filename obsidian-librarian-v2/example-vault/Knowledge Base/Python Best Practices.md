---
title: Python Best Practices
tags: [python, programming, best-practices]
created: 2024-01-08
---

# Python Best Practices

## Code Style

### PEP 8 Compliance

- Use 4 spaces for indentation
- Limit lines to 79 characters
- Use descriptive variable names
- Follow naming conventions:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants

### Type Hints

Always use type hints for better code clarity:

```python
from typing import List, Optional, Dict

def process_data(items: List[str], config: Optional[Dict[str, Any]] = None) -> bool:
    """Process a list of items with optional configuration."""
    if config is None:
        config = {}
    # Processing logic here
    return True
```

## Async Programming

### Best Practices for AsyncIO

1. **Use async/await properly**
   ```python
   async def fetch_data(url: str) -> dict:
       async with aiohttp.ClientSession() as session:
           async with session.get(url) as response:
               return await response.json()
   ```

2. **Avoid blocking operations**
   - Use `asyncio.create_task()` for concurrent operations
   - Never use blocking I/O in async functions

3. **Error handling**
   ```python
   try:
       result = await some_async_operation()
   except asyncio.TimeoutError:
       logger.error("Operation timed out")
   ```

## Testing

### Unit Testing

- Write tests first (TDD)
- Use pytest for testing
- Aim for >80% code coverage

### Test Structure

```python
import pytest
from mymodule import MyClass

class TestMyClass:
    @pytest.fixture
    def instance(self):
        return MyClass()
    
    def test_initialization(self, instance):
        assert instance.value == 0
    
    @pytest.mark.asyncio
    async def test_async_method(self, instance):
        result = await instance.async_method()
        assert result is not None
```

## Performance

### Optimization Tips

1. Profile before optimizing
2. Use built-in data structures efficiently
3. Consider using `functools.lru_cache` for expensive computations
4. Use generators for large datasets

## Dependencies

### Virtual Environments

Always use virtual environments:

```bash
python -m venv venv
source venv/bin/activate  # On Unix
venv\Scripts\activate     # On Windows
```

### Dependency Management

- Use `requirements.txt` or `pyproject.toml`
- Pin versions for production
- Regular security updates

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def calculate_average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers to average.
        
    Returns:
        The arithmetic mean of the input numbers.
        
    Raises:
        ValueError: If the list is empty.
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
```

## Related Notes

- [[Python Async Patterns]]
- [[Testing Strategies]]
- [[Code Review Checklist]]