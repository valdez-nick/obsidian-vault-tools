# Integration Tests

Comprehensive integration tests for the Obsidian Librarian CLI and core functionality.

## Test Structure

### test_cli_commands.py
Tests all CLI commands including:
- Basic command functionality (`analyze`, `research`, `curate`, `templates`, `status`)
- Command options and flags
- Error handling and edge cases
- Output formatting
- Configuration loading

### test_backup_restore.py
Tests backup and restore functionality:
- Full and incremental backups
- Compression formats (tar.gz, zip, etc.)
- Backup verification and checksums
- Restore with conflict handling
- Backup encryption
- Retention policies
- Scheduled backups

### test_research_workflow.py
Tests the complete research workflow:
- Multi-source research (ArXiv, GitHub, web)
- Result filtering and deduplication
- Automatic note organization
- Template application
- Research caching
- Export formats (BibTeX, Markdown)

### test_analysis_workflow.py
Tests vault analysis capabilities:
- Comprehensive vault statistics
- Quality analysis and scoring
- Duplicate detection
- Link analysis and broken link detection
- Task extraction and analysis
- Tag analysis
- Performance with large vaults
- Report generation (HTML, Markdown, JSON)

## Running Tests

### Run all integration tests:
```bash
pytest tests/integration/ -v
```

### Run specific test file:
```bash
pytest tests/integration/test_cli_commands.py -v
```

### Run with coverage:
```bash
pytest tests/integration/ --cov=obsidian_librarian --cov-report=html
```

### Run excluding slow tests:
```bash
pytest tests/integration/ -m "not slow"
```

### Run only tests requiring API access:
```bash
pytest tests/integration/ -m "requires_api"
```

## Test Markers

- `@pytest.mark.slow` - Tests that take longer to run (e.g., large vault tests)
- `@pytest.mark.integration` - All integration tests
- `@pytest.mark.requires_api` - Tests requiring external API access
- `@pytest.mark.asyncio` - Async tests

## Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_vault` - Creates a temporary vault with sample notes
- `mock_ai_service` - Mocked AI service for testing
- `cli_runner` - Click CLI test runner
- `performance_monitoring` - Performance monitoring utilities

## Environment Variables

Set these for full integration testing:
```bash
export OBSIDIAN_LIBRARIAN_TEST_MODE=true
export OPENAI_API_KEY=your-test-key  # For API tests
```

## Performance Benchmarks

Performance targets for large vaults (500+ notes):
- Full analysis: < 10 seconds
- Incremental analysis: < 2 seconds
- Memory usage: < 500MB
- Backup creation: < 5 seconds

## Test Data

Sample vault structures are created dynamically for each test to ensure isolation.
See individual test files for specific vault configurations.

## Debugging

Enable verbose output:
```bash
pytest tests/integration/ -v -s
```

Run with debugging:
```bash
pytest tests/integration/ --pdb
```

## Known Issues

- Some tests may be skipped if optional dependencies are not installed
- API-dependent tests require mock setup or actual API keys
- Large vault tests are marked as slow and skipped by default in CI