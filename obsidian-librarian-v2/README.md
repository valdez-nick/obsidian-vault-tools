# Obsidian Librarian v2

**High-Performance Python-Rust Hybrid Intelligent Content Management System for Obsidian Vaults**

## Overview

Obsidian Librarian v2 is a complete rewrite that combines the performance of Rust with the rich AI/ML ecosystem of Python. This hybrid architecture enables:

- **10-100x performance improvement** for file operations
- **Support for 100,000+ notes** without performance degradation
- **Intelligent research assistant** that fetches and organizes external content
- **Real-time vault monitoring** with microsecond-level responsiveness
- **AI-powered content analysis** and organization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Python Layer                          │
│         (AI/ML, Research Assistant, API, CLI)              │
├─────────────────────────────────────────────────────────────┤
│                    Python Bindings                         │
│                      (PyO3)                                │
├─────────────────────────────────────────────────────────────┤
│                      Rust Core                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    File    │  │    Vector    │  │   Concurrent     │   │
│  │Operations  │  │    Search    │  │   Web Fetcher    │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Core Vault Management
- **High-Performance File Operations**: Rust-powered concurrent file processing
- **Real-Time Monitoring**: Sub-second change detection with intelligent debouncing  
- **Smart Caching**: Memory-efficient note caching with LRU eviction
- **Comprehensive Parsing**: Full Obsidian markdown support including frontmatter, wiki links, tasks

### Research Assistant
- **Natural Language Queries**: Ask for research on any topic
- **Multi-Source Fetching**: Concurrent scraping from academic papers, GitHub, blogs
- **Intelligent Organization**: Automatic categorization and note creation
- **Progress Streaming**: Real-time updates as research is gathered

### AI-Powered Analysis
- **Semantic Duplicate Detection**: Vector similarity for finding related content
- **Quality Scoring**: Automated assessment of note completeness and formatting
- **Template Matching**: Intelligent application of templates to existing notes
- **Content Refinement**: AI-suggested improvements and standardization

### Scalability
- **Vector Database**: High-performance similarity search with Qdrant
- **Analytics Database**: Complex queries with DuckDB
- **Concurrent Processing**: Tokio-based async operations
- **Memory Efficiency**: Zero-copy operations where possible

## Quick Start

### Installation

```bash
# Install from PyPI (when published)
pip install obsidian-librarian

# Or build from source
git clone https://github.com/obsidian-librarian/obsidian-librarian-v2
cd obsidian-librarian-v2
pip install maturin
maturin develop --release
```

### Basic Usage

```python
import asyncio
from obsidian_librarian import Vault, ResearchService

async def main():
    # Initialize vault
    vault = await Vault.create("/path/to/obsidian/vault")
    
    # Research assistant query
    research = ResearchService(vault)
    results = await research.query(
        "Find me modern implementations of self-supervised learning"
    )
    
    # Results are automatically organized in your vault
    print(f"Found {len(results)} research items")

asyncio.run(main())
```

### CLI Usage

```bash
# Initialize for your vault
obsidian-librarian init /path/to/vault

# Research query
obsidian-librarian research "self-supervised learning implementations"

# Analyze vault for improvements
obsidian-librarian analyze --full

# Start real-time monitoring
obsidian-librarian monitor
```

## Development

### Project Structure

```
obsidian-librarian-v2/
├── rust-core/                 # Rust workspace
│   ├── librarian-core/        # File operations, parsing
│   ├── librarian-search/      # Vector search engine  
│   ├── librarian-web/         # Web scraping engine
│   └── python-bindings/       # PyO3 Python bindings
├── python/                    # Python package
│   └── obsidian_librarian/
│       ├── ai/               # LLM integration
│       ├── api/              # FastAPI server
│       ├── services/         # Business logic
│       └── cli/              # Command interface
├── tests/                     # Integrated test suite
└── benchmarks/               # Performance benchmarks
```

### Building

```bash
# Build Rust components
cd rust-core
cargo build --release

# Build Python package
cd ../python
maturin develop --release

# Run tests
cargo test  # Rust tests
pytest     # Python tests
```

### Performance Benchmarks

Run benchmarks to validate performance targets:

```bash
# Rust benchmarks
cd rust-core/librarian-core
cargo bench

# Python integration benchmarks  
cd ../../python
pytest benchmarks/ -v
```

## Research Assistant Usage

### Natural Language Queries

```python
research = ResearchService(vault)

# Academic research
await research.query(
    "Latest papers on transformer architectures from 2024"
)

# Implementation examples
await research.query(
    "Production-ready GraphQL implementations in Rust"
)

# Documentation search
await research.query(
    "FastAPI async database patterns and best practices"
)
```

### Automated Organization

Research results are automatically organized in your vault:

```
Research Library/
├── By Topic/
│   └── Machine Learning/
│       ├── Transformers/
│       └── Self Supervised Learning/
├── By Date/
│   └── 2024-01-15/
└── By Source/
    ├── ArXiv/
    ├── GitHub/
    └── Documentation/
```

## Performance Targets

- **File Processing**: 1,000+ notes/second
- **Search**: <100ms for 1M+ embeddings  
- **Memory Usage**: <500MB for 10,000 notes
- **Startup**: <2 seconds for large vaults
- **Research Queries**: 100+ concurrent web requests

## Contributing

1. **Rust Components**: High-performance operations, parsing, search
2. **Python Services**: AI/ML integration, research logic, API endpoints
3. **Testing**: Both unit tests and integration tests required
4. **Documentation**: Comprehensive docs for all public APIs

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built for Obsidian users who need both performance and intelligence in their knowledge management workflow.