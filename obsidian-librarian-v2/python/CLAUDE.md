# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status & Critical Issues

**⚠️ CURRENT BLOCKERS FOR v0.1.0 RELEASE:**
- CLI entry point is broken: `pyproject.toml` references `obsidian_librarian.cli.main:app` but `cli/main.py` doesn't exist
- Many CLI commands referenced in README are not implemented
- Rust-Python integration needs completion and testing
- Database layer has incomplete implementations for optional dependencies

**📋 See DEVELOPMENT_PLAN.md for complete roadmap to v0.1.0 release**

## Development Commands

### Repository-Wide Commands (use Makefile at project root)
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

### Building and Installation
```bash
# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Build Rust components (if Rust is available)
maturin develop

# Standard installation from PyPI
pip install obsidian-librarian
```

### Testing
```bash
# Run all tests with coverage
python -m pytest

# Run specific test categories
python -m pytest -m unit
python -m pytest -m integration  
python -m pytest -m e2e

# Run single test file
python -m pytest tests/test_ai_components.py

# Run with specific marker and verbose output
python -m pytest -m "not slow" -v

# Generate coverage report
python -m pytest --cov-report=html
```

### Code Quality
```bash
# Format code
black obsidian_librarian tests
isort obsidian_librarian tests

# Lint code
flake8 obsidian_librarian
mypy obsidian_librarian

# Run pre-commit hooks
pre-commit run --all-files
```

### CLI Usage During Development
```bash
# ⚠️ WARNING: CLI currently broken - needs cli/main.py implementation
# These commands are from README but not yet implemented:

# Initialize a test vault
obsidian-librarian init /path/to/test/vault

# Test core functionality  
obsidian-librarian stats
obsidian-librarian duplicates --threshold 0.9
obsidian-librarian research "machine learning" --max-results 5
```

### Debugging and Development
```bash
# Test with example vault
cd /path/to/repo/example-vault
python -m obsidian_librarian.cli stats

# Test Python API directly (bypasses CLI)
cd python && python -c "
import asyncio
from obsidian_librarian.librarian import ObsidianLibrarian
from obsidian_librarian.models import LibrarianConfig
from pathlib import Path

async def test():
    config = LibrarianConfig()
    librarian = ObsidianLibrarian(config)
    await librarian.initialize()
    session_id = await librarian.create_session(Path('../example-vault'))
    status = await librarian.get_session_status(session_id)
    print(f'Vault stats: {status}')
    await librarian.close()

asyncio.run(test())
"
```

## Architecture Overview

### Hybrid Python-Rust Design
The codebase implements a **hybrid architecture** where:
- **Rust core** (`_core` module) handles performance-critical operations (file I/O, parsing, search)
- **Python layer** provides AI/ML integration, service orchestration, and async patterns
- **Graceful degradation** when Rust bindings are unavailable (development/testing mode)

### Session-Based Architecture
The main `ObsidianLibrarian` class uses a session pattern:
- Each vault gets its own `LibrarianSession` with isolated services and state
- Multiple concurrent vaults supported through `VaultManager`
- Sessions coordinate between services: Research, Analysis, Template, and Database layers

### Service Layer Pattern
Three main services with clear separation of concerns:
- **ResearchService**: Web research, query processing, content extraction
- **AnalysisService**: Content quality scoring, duplicate detection, similarity analysis  
- **TemplateService**: Intelligent template matching and application

### Multi-Database Layer with Optional Dependencies
The database layer gracefully handles missing dependencies:
- **AnalyticsDB** (DuckDB): Vault statistics and usage patterns
- **VectorDB** (Qdrant): Embeddings and similarity search
- **CacheDB** (Redis with SQLite fallback): Session caching
- System continues to function even when optional databases are unavailable

### AI Pipeline Components
Modular AI components with provider abstraction:
- **QueryProcessor**: Natural language query understanding
- **ContentSummarizer**: Intelligent summarization with multiple strategies
- **LanguageModelService**: LLM integration (OpenAI, Anthropic, local models)
- **EmbeddingService**: Multiple embedding provider support

## Key Implementation Patterns

### Optional Dependency Handling
The codebase extensively uses graceful degradation:
```python
try:
    import optional_dependency
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False
    # Provide fallback or skip functionality
```

### Async/Await Throughout
All I/O operations are async with proper resource management:
- Context managers for database connections
- ThreadPoolExecutor for CPU-bound Rust operations
- Async generators for streaming results

### Event-Driven File Monitoring
The vault layer uses watchdog for real-time file system monitoring:
- Debounced events to prevent excessive processing
- Callback system for pluggable event handlers
- Thread-safe event processing

### Configuration-Driven Design
Multiple configuration layers allow extensive customization:
- `LibrarianConfig`: Global system settings
- Service-specific configs for each component
- Database configuration with fallback options

## Testing Considerations

### Test Categories
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test service interactions and database integration
- **E2E tests**: Test complete workflows with mocked external dependencies

### Handling Optional Dependencies
Tests gracefully skip functionality when optional dependencies are missing:
- DuckDB/Qdrant tests skip when databases unavailable
- Rust functionality tests skip when bindings missing
- AI features may use mocked providers

### Async Test Patterns
All tests use `pytest-asyncio` with proper async fixture management:
```python
@pytest.mark.asyncio
async def test_async_functionality():
    async with service:
        result = await service.some_async_method()
```

## Important Development Notes

### Rust Bindings Development
When working on Rust components:
- Rust bindings are optional for development but required for full functionality
- Use `maturin develop` to rebuild Python bindings after Rust changes
- Python fallbacks exist for core operations when Rust unavailable
- **Current Status**: Rust code exists in `/rust-core/` but Python integration needs work

### Database Migrations
The migration system automatically handles database schema changes:
- Located in `database/migrations.py`
- Supports multiple database backends
- Gracefully handles missing optional databases

### Logging and Observability
Uses `structlog` for structured logging throughout:
- Context-aware logging with request/session IDs
- Performance monitoring for long-running operations
- Error tracking with full context preservation

## Development Priorities (see DEVELOPMENT_PLAN.md)

### Phase 1: Critical Fixes for v0.1.0 (1-2 weeks)
1. **Fix CLI entry point**: Create `cli/main.py` with proper Typer CLI implementation
2. **Complete Rust-Python integration**: Test `maturin develop` workflow 
3. **Database layer fixes**: Ensure graceful degradation works properly
4. **Basic functionality**: Get core research/analysis workflows working

### Phase 2: Feature Completion (2-3 weeks) 
1. **Research Assistant MVP**: Complete web scraping and content organization
2. **Analysis Service**: Semantic duplicate detection with embeddings
3. **Template Service**: Intelligent template matching and application
4. **Performance optimization**: Benchmarking and optimization

### Phase 3: Production Ready (1-2 weeks)
1. **PyPI distribution**: GitHub Actions CI/CD for automated releases
2. **Documentation**: User guides and API documentation  
3. **Web UI**: FastAPI server with React/Vue dashboard
4. **Examples and demos**: Comprehensive example vault content

## Repository Structure Notes
- **Project root**: Contains Makefile, DEVELOPMENT_PLAN.md, example-vault/
- **`/python/`**: Main Python package, tests, documentation
- **`/rust-core/`**: Rust components (librarian-core, librarian-search, librarian-web, python-bindings)
- **`/example-vault/`**: Test Obsidian vault with sample content for development