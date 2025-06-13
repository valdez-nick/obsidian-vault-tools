# Obsidian Librarian

[![PyPI version](https://badge.fury.io/py/obsidian-librarian.svg)](https://badge.fury.io/py/obsidian-librarian)
[![Python Support](https://img.shields.io/pypi/pyversions/obsidian-librarian.svg)](https://pypi.org/project/obsidian-librarian/)
[![Documentation Status](https://readthedocs.org/projects/obsidian-librarian/badge/?version=latest)](https://obsidian-librarian.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/obsidian-librarian/obsidian-librarian/actions/workflows/ci.yml/badge.svg)](https://github.com/obsidian-librarian/obsidian-librarian/actions/workflows/ci.yml)

An intelligent content management system for Obsidian vaults that combines high-performance Rust operations with Python's rich AI ecosystem.

## 🚀 Features

- **🧹 Intelligent Organization** - Automatically organize notes by content, tags, or custom rules
- **🔍 Duplicate Detection** - Find and merge similar content with configurable thresholds
- **🔬 Research Assistant** - Research topics and automatically organize findings in your vault
- **📝 Template Management** - Apply Templater templates intelligently to existing notes
- **🔄 Git Integration** - Automatic backups with intelligent commit messages
- **🤖 AI-Powered** - Smart summarization, content analysis, and query understanding
- **⚡ High Performance** - Rust core for blazing-fast operations on large vaults

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install obsidian-librarian
```

### From Source

```bash
git clone https://github.com/obsidian-librarian/obsidian-librarian.git
cd obsidian-librarian
pip install -e .
```

## 🎯 Quick Start

### 1. Initialize Your Vault

```bash
cd /path/to/your/obsidian/vault
obsidian-librarian init
```

### 2. Basic Commands

```bash
# View vault statistics
obsidian-librarian stats

# Find duplicate content
obsidian-librarian duplicates

# Organize your vault
obsidian-librarian organize --dry-run

# Research a topic
obsidian-librarian research "machine learning transformers"

# Curate vault content intelligently
obsidian-librarian curate --dry-run
```

### 3. Interactive Mode

```bash
obsidian-librarian interactive
```

## 📚 User Guide

### Getting Started

#### First-Time Setup
1. **Install Obsidian Librarian**:
   ```bash
   pip install obsidian-librarian
   ```

2. **Initialize your vault**:
   ```bash
   cd /path/to/your/obsidian/vault
   obsidian-librarian init
   ```

3. **Configure your preferences** (optional):
   ```bash
   obsidian-librarian config --ai-provider openai --api-key YOUR_API_KEY
   ```

#### Basic Workflow
```bash
# Analyze your vault health
obsidian-librarian stats

# Find and review duplicates
obsidian-librarian duplicates --threshold 0.85

# Organize content intelligently
obsidian-librarian organize --strategy content --dry-run
obsidian-librarian organize --strategy content  # Apply changes

# Research and add new content
obsidian-librarian research "quantum computing applications" --max-results 10

# Comprehensive content curation
obsidian-librarian curate --dry-run  # Preview improvements
obsidian-librarian curate --interactive  # Apply with review
```

### Core Features

#### 🔍 Duplicate Detection
Find similar notes with configurable similarity thresholds:

```bash
# Find exact duplicates (95%+ similarity)
obsidian-librarian duplicates --threshold 0.95

# Find near-duplicates for review
obsidian-librarian duplicates --threshold 0.75 --interactive

# Auto-merge high-confidence duplicates
obsidian-librarian duplicates --threshold 0.95 --auto-merge --backup
```

**What it detects:**
- Identical content with different titles
- Notes with overlapping information
- Accidentally copied content
- Similar meeting notes or daily logs

#### 🔬 Research Assistant
Automatically research topics and organize findings:

```bash
# Basic research
obsidian-librarian research "transformer neural networks"

# Advanced research with specific sources
obsidian-librarian research "quantum computing" \
  --sources arxiv.org,github.com \
  --max-results 15 \
  --organize-by topic

# Research with custom output location
obsidian-librarian research "machine learning papers" \
  --output "Research Library/ML Papers" \
  --format structured
```

**Research sources:**
- ArXiv (academic papers)
- GitHub (code repositories)
- Web articles and blog posts
- Wikipedia and reference sites

#### 🧹 Intelligent Organization
Organize your vault using AI-powered content analysis:

```bash
# Organize by content similarity
obsidian-librarian organize --strategy content

# Organize by tags and metadata
obsidian-librarian organize --strategy tags

# Organize by creation date
obsidian-librarian organize --strategy date

# Custom organization rules
obsidian-librarian organize --strategy custom --rules-file my-rules.yaml
```

**Organization strategies:**
- **Content**: Groups similar topics together
- **Tags**: Organizes by existing tag structure
- **Date**: Chronological organization
- **Links**: Groups by connection patterns
- **Custom**: User-defined rules

#### 🎯 Intelligent Content Curation
Comprehensive vault improvement with advanced analysis:

```bash
# Full curation with dry-run preview
obsidian-librarian curate --dry-run

# Focus on specific improvements
obsidian-librarian curate --duplicates --quality --interactive

# Templates and structure only
obsidian-librarian curate --templates --structure

# Automated curation with backup
obsidian-librarian curate --backup --batch-size 100
```

**Curation features:**
- **Duplicate Detection**: Find and merge similar content
- **Quality Analysis**: Identify improvement opportunities
- **Structure Enhancement**: Apply templates and improve organization  
- **Interactive Mode**: Step-by-step review and approval
- **Safe Operations**: Automatic backups and dry-run preview

#### 📝 Template Management
Apply templates intelligently to existing notes:

```bash
# Auto-detect and apply appropriate templates
obsidian-librarian templates --auto-apply

# Apply specific template to notes
obsidian-librarian templates --template "meeting-notes" --pattern "Meeting*"

# Suggest templates for unstructured notes
obsidian-librarian templates --suggest --min-words 50
```

#### 📊 Vault Analysis
Get insights into your vault's health and structure:

```bash
# Basic statistics
obsidian-librarian stats

# Detailed analysis
obsidian-librarian analyze --detailed

# Export analysis report
obsidian-librarian analyze --export analysis-report.html

# Performance metrics
obsidian-librarian analyze --performance
```

**Analysis includes:**
- Note count and size distribution
- Orphaned notes (no links)
- Broken links and references
- Tag usage patterns
- Content quality scores
- Growth trends over time

### Advanced Features

#### 🔄 Git Integration
Automatic backup and version control:

```bash
# Enable auto-backup
obsidian-librarian git --enable --threshold 10

# Manual backup with smart commit message
obsidian-librarian backup --message "Added research on quantum algorithms"

# Restore from backup
obsidian-librarian restore --commit abc123f
```

#### 🤖 AI Configuration
Configure AI providers and models:

```bash
# Set up OpenAI
obsidian-librarian config --ai-provider openai --api-key YOUR_KEY

# Set up Anthropic Claude
obsidian-librarian config --ai-provider anthropic --api-key YOUR_KEY

# Use local models
obsidian-librarian config --ai-provider local --model-path /path/to/model

# Configure embedding models
obsidian-librarian config --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

#### ⚙️ Performance Tuning
Optimize for large vaults:

```bash
# Enable high-performance mode (requires Rust bindings)
obsidian-librarian config --performance-mode high

# Configure batch processing
obsidian-librarian config --batch-size 1000 --max-workers 4

# Enable caching
obsidian-librarian config --cache-enabled --cache-size 1GB
```

### Interactive Mode

Launch the interactive interface for guided operations:

```bash
obsidian-librarian interactive
```

**Interactive features:**
- Step-by-step vault analysis
- Guided duplicate resolution
- Template selection assistant
- Research topic suggestions
- Real-time vault monitoring

### Configuration Reference

Create `.obsidian-librarian/config.yaml` in your vault:

```yaml
# General settings
vault:
  path: "."
  backup_enabled: true
  auto_organize: false

# AI and ML settings
ai:
  provider: "openai"  # openai, anthropic, local
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  embedding_model: "text-embedding-3-small"
  max_tokens: 4000
  temperature: 0.3

# Organization settings
organization:
  strategy: "content"  # content, tags, date, links, custom
  preserve_structure: true
  create_folders: true
  folder_prefix: ""
  
# Duplicate detection
duplicates:
  threshold: 0.85  # 0.0-1.0, higher = more strict
  auto_merge: false
  backup_before_merge: true
  ignore_patterns:
    - "Daily Notes/*"
    - "Templates/*"

# Research settings
research:
  default_sources:
    - "arxiv.org"
    - "github.com"
  max_results: 20
  output_folder: "Research Library"
  auto_organize: true
  quality_threshold: 0.7

# Template settings
templates:
  auto_apply: false
  template_folder: "Templates"
  rules_file: "template-rules.yaml"
  
# Performance settings
performance:
  mode: "balanced"  # low, balanced, high
  batch_size: 100
  max_workers: 4
  cache_enabled: true
  cache_size: "512MB"

# Git integration
git:
  auto_backup: true
  backup_threshold: 10  # changes before auto-backup
  commit_message_template: "Obsidian Librarian: {operation} - {timestamp}"
  
# Web scraping
scraping:
  rate_limit: 2  # requests per second
  timeout: 30
  user_agent: "Obsidian-Librarian/1.0"
  respect_robots_txt: true
```

### Troubleshooting

#### Common Issues

**"Command not found: obsidian-librarian"**
```bash
# Ensure installation was successful
pip install --upgrade obsidian-librarian

# Check if it's in your PATH
python -m obsidian_librarian --help
```

**"No Rust bindings available"**
```bash
# For better performance, install with Rust support
pip install obsidian-librarian[rust]

# Or compile from source
git clone https://github.com/obsidian-librarian/obsidian-librarian
cd obsidian-librarian
make dev
```

**Performance issues with large vaults**
```bash
# Enable high-performance mode
obsidian-librarian config --performance-mode high

# Increase batch processing
obsidian-librarian config --batch-size 1000

# Enable caching
obsidian-librarian config --cache-enabled
```

**API rate limits**
```bash
# Reduce request frequency
obsidian-librarian config --rate-limit 1

# Use local models instead
obsidian-librarian config --ai-provider local
```

#### Getting Help

```bash
# General help
obsidian-librarian --help

# Command-specific help
obsidian-librarian research --help

# Check configuration
obsidian-librarian config --show

# Run diagnostics
obsidian-librarian diagnose
```

## 📖 Additional Documentation

- [API Reference](https://obsidian-librarian.readthedocs.io/en/latest/api-reference/)
- [Advanced Configuration](https://obsidian-librarian.readthedocs.io/en/latest/configuration/)
- [Developer Guide](https://obsidian-librarian.readthedocs.io/en/latest/developer-guide/)
- [Tutorials](https://obsidian-librarian.readthedocs.io/en/latest/tutorials/)

## 🐍 Python API

The Python API provides programmatic access to all features:

### Basic Usage

```python
import asyncio
from pathlib import Path
from obsidian_librarian.librarian import ObsidianLibrarian
from obsidian_librarian.models import LibrarianConfig

async def main():
    # Create configuration
    config = LibrarianConfig()
    
    # Initialize librarian
    async with ObsidianLibrarian(config) as librarian:
        # Create session for your vault
        vault_path = Path("/path/to/your/vault")
        session_id = await librarian.create_session(vault_path)
        
        # Analyze vault
        async for result in librarian.analyze_vault(session_id):
            if result['type'] == 'complete':
                print(f"Analysis complete: {result['data']}")
        
        # Research a topic
        async for result in librarian.research(session_id, "quantum computing"):
            if result['type'] == 'result':
                print(f"Found: {result['data']['title']}")
        
        # Get session status
        status = await librarian.get_session_status(session_id)
        print(f"Vault has {status['vault_stats']['note_count']} notes")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced API Usage

```python
import asyncio
from pathlib import Path
from obsidian_librarian.vault import Vault, VaultConfig
from obsidian_librarian.services.analysis import AnalysisService
from obsidian_librarian.services.research import ResearchService

async def advanced_example():
    vault_path = Path("/path/to/vault")
    vault_config = VaultConfig()
    
    # Direct service usage
    async with Vault(vault_path, vault_config) as vault:
        # Analysis service
        analysis_service = AnalysisService(vault)
        
        # Find duplicates
        duplicates = await analysis_service.find_duplicates()
        for cluster in duplicates:
            print(f"Found {len(cluster.note_ids)} similar notes")
        
        # Research service
        research_service = ResearchService(vault)
        
        # Research with streaming results
        async for result in research_service.research("machine learning"):
            print(f"Found: {result.title} - Quality: {result.quality_score}")

asyncio.run(advanced_example())
```

## 🔧 Requirements

- Python 3.8 or higher
- An Obsidian vault
- Git (optional, for auto-backup features)
- OpenAI/Anthropic API key (optional, for AI features)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyO3](https://pyo3.rs/) for Python-Rust interop
- Inspired by the amazing Obsidian community
- Uses [LangChain](https://langchain.com/) for AI features

## 📞 Support

- **Documentation**: [Read the Docs](https://obsidian-librarian.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/obsidian-librarian/obsidian-librarian/issues)
- **Discussions**: [GitHub Discussions](https://github.com/obsidian-librarian/obsidian-librarian/discussions)
- **Discord**: [Join our Discord](https://discord.gg/obsidian-librarian)

---

Made with ❤️ for the Obsidian community