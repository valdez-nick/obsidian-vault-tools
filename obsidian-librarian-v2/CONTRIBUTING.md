# Contributing to Obsidian Librarian v2

Welcome! We're excited that you want to contribute to Obsidian Librarian v2. This guide will help you get started with development and contribution workflows.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Rust 1.70+ (optional but recommended)
- Git
- An Obsidian vault for testing

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/valdez-nick/obsidian-librarian-v2
   cd obsidian-librarian-v2
   ```

2. **Set up development environment**:
   ```bash
   make dev  # Installs all dependencies and sets up pre-commit hooks
   ```

3. **Verify installation**:
   ```bash
   make test  # Run all tests
   make lint  # Run linting and formatting checks
   ```

### Development Commands

```bash
# Build everything (Rust + Python)
make build

# Run all tests
make test

# Development setup
make dev

# Code formatting
make format

# Linting
make lint

# Full pipeline
make all
```

## 🏗️ Architecture Overview

### Hybrid Python-Rust Design
- **Rust core** (`rust-core/`) handles performance-critical operations
- **Python layer** (`python/`) provides AI/ML integration and service orchestration
- **Graceful degradation** when Rust bindings are unavailable

### Key Components
- **CLI Layer** (`python/obsidian_librarian/cli/`) - User interface and command handling
- **Services** (`python/obsidian_librarian/services/`) - Core business logic
- **Models** (`python/obsidian_librarian/models/`) - Data structures and schemas
- **Database Layer** (`python/obsidian_librarian/database/`) - Multi-database abstraction
- **AI Pipeline** (`python/obsidian_librarian/ai/`) - LLM and embedding services

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Development Guidelines

#### Python Code Style
- Use **Black** for formatting: `black obsidian_librarian tests`
- Use **isort** for import sorting: `isort obsidian_librarian tests`
- Follow **PEP 8** and use **mypy** for type checking
- Add docstrings for all public functions and classes

#### Rust Code Style
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Follow Rust naming conventions

#### Testing Requirements
- **Unit tests** for all new functions and classes
- **Integration tests** for service interactions
- **E2E tests** for complete workflows
- Target **80%+ code coverage**

### 3. Commit Messages

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
feat(tags): add AI-powered tag suggestions
fix(cli): resolve Click to Typer conversion issues
docs(api): update service API documentation
```

## 🧪 Testing Strategy

### Test Categories
- **Unit Tests** (`tests/unit/`) - Test individual components
- **Integration Tests** (`tests/integration/`) - Test service interactions
- **E2E Tests** (`tests/e2e/`) - Test complete workflows

### Running Tests

```bash
# Python tests
cd python
python -m pytest

# Rust tests
cd rust-core
cargo test

# All tests
make test

# Specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/e2e/

# With coverage
python -m pytest --cov=obsidian_librarian
```

### Writing Tests

Example test structure:
```python
import pytest
from obsidian_librarian.services.tag_manager import TagManagerService

@pytest.mark.asyncio
async def test_tag_analysis():
    """Test tag analysis functionality."""
    service = TagManagerService(vault_path=test_vault_path)
    result = await service.analyze_tags()
    
    assert result.total_tags > 0
    assert len(result.duplicate_clusters) >= 0
```

### Test Data
- Use `example-vault/` for test data
- Create minimal test vaults in `tests/fixtures/`
- Mock external dependencies (APIs, databases)

## 📝 Adding New Features

### 1. CLI Commands
Add new commands in `python/obsidian_librarian/cli/`:

```python
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def my_command(
    vault_path: Path = typer.Argument(..., help="Path to Obsidian vault"),
    option: bool = typer.Option(False, "--option", help="Example option")
) -> None:
    """Description of your command."""
    # Implementation here
```

### 2. Services
Create services in `python/obsidian_librarian/services/`:

```python
from typing import List, Optional
from ..models.models import YourModel

class YourService:
    """Service for handling specific functionality."""
    
    async def __init__(self, vault_path: Path):
        self.vault_path = vault_path
    
    async def your_method(self) -> List[YourModel]:
        """Method description."""
        # Implementation here
```

### 3. Models
Add data models in `python/obsidian_librarian/models/models.py`:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class YourModel:
    """Description of your model."""
    field1: str
    field2: Optional[int] = None
```

## 🤖 Multi-Agent Development

For complex features, we use parallel development with git worktrees:

### 1. Create Worktree
```bash
git worktree add -b feature-branch worktrees/feature-name
cd worktrees/feature-name
```

### 2. Develop in Parallel
Each agent/developer works in their own worktree, allowing:
- Parallel feature development
- Independent testing
- Isolated dependencies

### 3. Integration
```bash
# Switch back to main repo
cd ../../
git merge feature-branch
```

## 🔄 Pull Request Process

### 1. Before Submitting
- [ ] All tests pass: `make test`
- [ ] Code is formatted: `make format`
- [ ] No linting errors: `make lint`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)

### 2. PR Template
When creating a PR, include:

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### 3. Review Process
- All PRs require at least one review
- Automated tests must pass
- Code coverage should not decrease significantly

## 🐛 Reporting Issues

### Bug Reports
Include:
- Python version
- Rust version (if applicable)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error logs (if any)

### Feature Requests
Include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## 🏷️ Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow
1. Update `CHANGELOG.md`
2. Create release PR
3. Tag release: `git tag v0.x.y`
4. GitHub Actions builds and publishes to PyPI

## 🆘 Getting Help

- **Documentation**: Check README.md and docstrings first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Code**: Review existing code for patterns and examples

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Obsidian Librarian v2! 🚀