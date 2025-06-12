# Changelog

All notable changes to Obsidian Librarian will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Obsidian Librarian v2
- High-performance Rust core with Python bindings
- Intelligent vault organization by content, tags, dates, or links
- Duplicate detection with configurable similarity thresholds
- Research assistant with web scraping and auto-organization
- Template management with Templater integration
- Git integration for automatic backups
- AI-powered features using OpenAI/Anthropic
- Comprehensive CLI interface
- Python API for programmatic access
- Multi-database support (planned for DuckDB, Qdrant, Redis)
- File watching with debounced updates
- Async operations throughout
- Extensive test coverage
- Performance benchmarks
- User documentation and tutorials

### Technical Details
- Rust core using Tokio for async operations
- PyO3 bindings for Python integration
- Memory-mapped file operations for large vaults
- SIMD-accelerated text processing
- Concurrent web scraping with rate limiting
- Vector embeddings for semantic search
- Graph-based link analysis

## [0.1.0] - TBD

- Initial public release

[Unreleased]: https://github.com/obsidian-librarian/obsidian-librarian/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/obsidian-librarian/obsidian-librarian/releases/tag/v0.1.0