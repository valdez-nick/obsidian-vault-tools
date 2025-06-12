# Contributing to Obsidian Librarian

Thank you for your interest in contributing to Obsidian Librarian! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8+
- Rust 1.70+
- Git
- Make (optional but recommended)

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/obsidian-librarian.git
   cd obsidian-librarian
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   make dev
   # Or manually:
   pip install maturin
   cd python && maturin develop --release
   pip install -e .[dev]
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

### Branch Naming

- Feature branches: `feature/description`
- Bugfix branches: `fix/description`
- Documentation: `docs/description`
- Performance: `perf/description`

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(research): add support for custom research sources
fix(vault): handle special characters in note paths
docs(api): update vault API documentation
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run Python tests only
cd python && pytest

# Run Rust tests only
cd rust-core && cargo test

# Run specific test file
pytest tests/test_vault.py

# Run with coverage
pytest --cov=obsidian_librarian
```

### Writing Tests

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names
- Include both positive and negative test cases

Example test:
```python
import pytest
from obsidian_librarian import Vault

@pytest.mark.asyncio
async def test_vault_initialization():
    """Test that vault initializes correctly."""
    vault = Vault("/path/to/vault")
    await vault.initialize()
    
    assert vault.path.exists()
    stats = await vault.get_stats()
    assert stats.total_notes >= 0
```

## Submitting Changes

### Pull Request Process

1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Submit a pull request with a clear description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
```

## Style Guidelines

### Python Style

We use Black for formatting and follow PEP 8:

```bash
# Format code
black obsidian_librarian/

# Check style
flake8 obsidian_librarian/

# Type checking
mypy obsidian_librarian/
```

### Rust Style

We use rustfmt and clippy:

```bash
# Format code
cargo fmt

# Lint code
cargo clippy -- -D warnings
```

### Documentation

- Use clear, concise language
- Include code examples
- Update docstrings for API changes
- Keep README.md up to date

## Architecture Guidelines

### Python Code

- Use type hints everywhere
- Prefer async/await for I/O operations
- Keep functions focused and small
- Use descriptive variable names

### Rust Code

- Minimize unsafe code
- Document public APIs
- Use Result<T, E> for error handling
- Write unit tests for all modules

## Performance Considerations

- Profile before optimizing
- Consider memory usage for large vaults
- Use appropriate data structures
- Benchmark significant changes

## Community

### Getting Help

- GitHub Discussions: Ask questions and share ideas
- Discord: Real-time chat with contributors
- Issues: Report bugs or request features

### Code Reviews

All submissions require review before merging. We look for:

- Correctness
- Test coverage
- Documentation
- Performance impact
- Code style

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Publish to PyPI and crates.io

## Recognition

Contributors are recognized in:
- CHANGELOG.md
- GitHub contributors page
- Release notes

Thank you for contributing to Obsidian Librarian! 🎉