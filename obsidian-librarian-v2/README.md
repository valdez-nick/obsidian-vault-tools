# Obsidian Librarian v2 🚀

**High-Performance Python-Rust Hybrid Intelligent Content Management System for Obsidian Vaults**

[![Version](https://img.shields.io/badge/version-v0.2.0-blue.svg)](https://github.com/obsidian-librarian/obsidian-librarian-v2/releases/tag/v0.2.0)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://rustlang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

Obsidian Librarian v2 is a complete rewrite that combines the performance of Rust with the rich AI/ML ecosystem of Python. This hybrid architecture enables:

- **🚀 10-100x performance improvement** for file operations
- **📈 Support for 100,000+ notes** without performance degradation
- **🔍 Intelligent research assistant** that fetches and organizes external content
- **⚡ Real-time vault monitoring** with microsecond-level responsiveness
- **🤖 AI-powered content analysis** and organization
- **🏷️ Advanced tag management** with semantic analysis and hierarchy detection
- **📁 Intelligent directory organization** with ML-powered content classification

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

## ✨ Key Features

### 🏷️ Advanced Tag Management (NEW in v0.2.0)
- **Intelligent Tag Analysis**: Fuzzy matching and semantic similarity detection
- **Tag Hierarchy Detection**: Automatic discovery of tag relationships and structures
- **Duplicate Tag Cleanup**: Smart identification and merging of redundant tags
- **AI-Powered Tag Suggestions**: Content-based tag recommendations
- **Bulk Tag Operations**: Efficient batch processing with conflict resolution
- **Tag Quality Scoring**: Assessment of tag consistency and usefulness

### 📁 Smart Directory Organization (NEW in v0.2.0)
- **AI-Powered Content Classification**: Automatic categorization of notes by content
- **Pattern-Based Routing Rules**: Flexible file organization with custom patterns
- **Machine Learning Optimization**: Learns from user preferences over time
- **Real-Time Auto-Organization**: Monitors file creation and moves notes automatically
- **Safe File Operations**: Validation, backup, and rollback capabilities
- **Organization Planning**: Preview and approve changes before execution

### 🔍 Core Vault Management
- **High-Performance File Operations**: Rust-powered concurrent file processing
- **Real-Time Monitoring**: Sub-second change detection with intelligent debouncing  
- **Smart Caching**: Memory-efficient note caching with LRU eviction
- **Comprehensive Parsing**: Full Obsidian markdown support including frontmatter, wiki links, tasks

### 🔬 Research Assistant
- **Natural Language Queries**: Ask for research on any topic
- **Multi-Source Fetching**: Concurrent scraping from academic papers, GitHub, blogs
- **Intelligent Organization**: Automatic categorization and note creation
- **Progress Streaming**: Real-time updates as research is gathered

### 🤖 AI-Powered Analysis
- **Semantic Duplicate Detection**: Vector similarity for finding related content
- **Quality Scoring**: Automated assessment of note completeness and formatting
- **Template Matching**: Intelligent application of templates to existing notes
- **Content Refinement**: AI-suggested improvements and standardization

### ⚡ Scalability
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

# NEW: Tag management commands
obsidian-librarian tags analyze /path/to/vault
obsidian-librarian tags suggest /path/to/vault --ai
obsidian-librarian tags cleanup /path/to/vault --merge-similar

# NEW: Directory organization commands
obsidian-librarian organize plan /path/to/vault
obsidian-librarian organize execute /path/to/vault --plan-id abc123
obsidian-librarian organize auto /path/to/vault --enable

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

## 🏷️ Tag Management Usage (NEW in v0.2.0)

### Tag Analysis and Cleanup

```python
from obsidian_librarian.services import TagManagerService

tag_manager = TagManagerService(vault)

# Analyze tag usage and quality
analysis = await tag_manager.analyze_tags()
print(f"Found {analysis.total_tags} tags, {len(analysis.duplicate_candidates)} duplicates")

# Find similar tags
similar_tags = await tag_manager.find_similar_tags(threshold=0.8)
for cluster in similar_tags:
    print(f"Similar tags: {cluster.tags}")

# Cleanup redundant tags
await tag_manager.merge_tags("machine-learning", "ml", "machinelearning")
```

### Intelligent Tag Suggestions

```python
# Get AI-powered tag suggestions for a note
suggestions = await tag_manager.suggest_tags_for_note(note_id)
for suggestion in suggestions:
    print(f"Suggested tag: {suggestion.tag} (confidence: {suggestion.confidence})")

# Auto-tag all notes in vault
await tag_manager.auto_tag_vault(min_confidence=0.7)
```

## 📁 Directory Organization Usage (NEW in v0.2.0)

### Smart Content Classification

```python
from obsidian_librarian.services import AutoOrganizer

organizer = AutoOrganizer(vault)

# Classify content and suggest organization
plan = await organizer.create_organization_plan(
    notes=await vault.get_all_notes(),
    rules=[
        DirectoryRule(
            name="Daily Notes",
            pattern="Daily Notes/*.md",
            destination="Archive/Daily/"
        )
    ]
)

# Execute organization plan
await organizer.execute_plan(plan)
```

### Real-time Auto-Organization

```python
# Enable automatic organization for new files
await organizer.enable_auto_organization(
    confidence_threshold=0.8,
    exclude_patterns=["Daily Notes/*", "Templates/*"]
)

# Files will be automatically organized as they're created
```

## 🔬 Research Assistant Usage

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

## 📋 Changelog

### v0.2.0 (2025-12-06) - Advanced Tag Management & Directory Organization

#### 🚀 Major New Features
- **Tag Management System**: Complete tag analysis, cleanup, and AI-powered suggestions
- **Directory Organization**: Intelligent auto-organization with ML-powered classification
- **Enhanced CLI**: Rich terminal interface with comprehensive command coverage
- **Multi-Agent Development**: Parallel development workflow using git worktrees

#### 🏷️ Tag Management
- Fuzzy matching and semantic similarity for tag deduplication
- Automatic tag hierarchy detection and suggestions
- Bulk tag operations with conflict resolution
- AI-powered tag suggestions based on content analysis
- Tag quality scoring and optimization recommendations

#### 📁 Directory Organization
- AI-powered content classification for automatic file placement
- Pattern-based directory routing with custom rules
- Machine learning optimization from user behavior
- Real-time file monitoring and auto-organization
- Safe file operations with validation and rollback

#### 🔧 Technical Improvements
- Enhanced service layer with graceful degradation
- Comprehensive data models for advanced features
- Improved error handling and conflict resolution
- Complete test coverage for new functionality

### v0.1.0 (Previous) - Core Foundation
- Hybrid Python-Rust architecture
- Basic vault management and file operations
- Research assistant with web scraping
- AI-powered content analysis

## 🤝 Contributing

We welcome contributions in several areas:

1. **🦀 Rust Components**: High-performance operations, parsing, search engines
2. **🐍 Python Services**: AI/ML integration, research logic, API endpoints  
3. **🧪 Testing**: Both unit tests and integration tests for all components
4. **📚 Documentation**: Comprehensive docs for all public APIs
5. **🏷️ Tag Management**: Enhanced algorithms for tag analysis and suggestions
6. **📁 Organization**: Improved classification models and organization rules

### Development Workflow
- See `DEVELOPMENT_PLAN.md` for roadmap and priorities
- Check `CLAUDE.md` for development commands and patterns
- Use `make dev` for complete development environment setup

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built for Obsidian users who need both performance and intelligence in their knowledge management workflow. Special thanks to the multi-agent development approach that enabled parallel feature development.

---

**⭐ Star this repo if you find it useful!** | **🐛 Report issues** | **💬 Join discussions**