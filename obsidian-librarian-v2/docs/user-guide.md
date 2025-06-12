# Obsidian Librarian User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Features](#core-features)
5. [Command Line Interface](#command-line-interface)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## Introduction

Obsidian Librarian is an intelligent content management system for Obsidian vaults that combines high-performance Rust operations with Python's AI capabilities. It acts as your personal librarian and research assistant, helping you:

- Organize and curate your notes automatically
- Detect and manage duplicate content
- Research topics and organize findings
- Apply templates intelligently
- Maintain vault health and integrity

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust 1.70 or higher (for building from source)
- Git (for automatic backups)
- An Obsidian vault

### Install from PyPI (Recommended)

```bash
pip install obsidian-librarian
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/obsidian-librarian.git
cd obsidian-librarian

# Install in development mode
make dev
```

## Quick Start

### 1. Initialize Your Vault

```bash
obsidian-librarian init /path/to/your/vault
```

This creates a configuration file and sets up the necessary directories.

### 2. Basic Commands

```bash
# Organize your vault
obsidian-librarian organize

# Detect duplicate notes
obsidian-librarian duplicates

# Research a topic
obsidian-librarian research "machine learning transformers"

# Apply templates to unstructured notes
obsidian-librarian template apply
```

### 3. Start the Interactive Session

```bash
obsidian-librarian interactive
```

## Core Features

### Vault Organization

The librarian can automatically organize your notes based on:

- **Content similarity**: Groups related notes together
- **Tags and metadata**: Organizes by existing tags
- **Date patterns**: Creates time-based structures
- **Link relationships**: Maintains note hierarchies

Example:
```bash
obsidian-librarian organize --strategy content --dry-run
```

### Duplicate Detection

Identifies similar or duplicate content across your vault:

```bash
# Find all duplicates
obsidian-librarian duplicates

# Set similarity threshold (0.0-1.0)
obsidian-librarian duplicates --threshold 0.85

# Merge duplicates interactively
obsidian-librarian duplicates --merge
```

### Research Assistant

Research topics and automatically organize findings:

```bash
# Basic research
obsidian-librarian research "quantum computing"

# Research with specific sources
obsidian-librarian research "deep learning" --sources arxiv.org,github.com

# Limit results
obsidian-librarian research "rust programming" --max-results 20
```

Research results are automatically:
- Summarized and formatted
- Saved to your Research Library
- Tagged and categorized
- Linked to existing notes

### Template Application

Apply Templater templates to existing notes:

```bash
# List available templates
obsidian-librarian template list

# Apply template to specific notes
obsidian-librarian template apply --template daily --notes "2024-01-*"

# Auto-detect and apply templates
obsidian-librarian template auto-apply
```

### Git Integration

Automatic backups during major changes:

```bash
# Enable auto-backup
obsidian-librarian config set git.auto_backup true

# Set backup threshold
obsidian-librarian config set git.change_threshold 10

# Manual backup
obsidian-librarian backup "Manual backup before reorganization"
```

## Command Line Interface

### Global Options

```bash
obsidian-librarian [OPTIONS] COMMAND [ARGS]

Options:
  --vault PATH     Path to Obsidian vault (or set OBSIDIAN_VAULT env var)
  --config PATH    Path to config file
  --verbose        Enable verbose output
  --quiet          Suppress non-error output
  --help           Show help message
```

### Commands Reference

#### `init`
Initialize a vault for use with Obsidian Librarian.

```bash
obsidian-librarian init [VAULT_PATH] [OPTIONS]

Options:
  --force          Overwrite existing configuration
  --minimal        Create minimal configuration
```

#### `organize`
Organize vault contents.

```bash
obsidian-librarian organize [OPTIONS]

Options:
  --strategy       Organization strategy: content, tags, date, links
  --dry-run        Preview changes without applying
  --interactive    Confirm each change
```

#### `duplicates`
Find and manage duplicate content.

```bash
obsidian-librarian duplicates [OPTIONS]

Options:
  --threshold      Similarity threshold (0.0-1.0, default: 0.85)
  --merge          Merge duplicates interactively
  --export PATH    Export duplicate report
```

#### `research`
Research topics and organize findings.

```bash
obsidian-librarian research QUERY [OPTIONS]

Options:
  --sources        Comma-separated list of domains
  --max-results    Maximum number of results
  --organize       Auto-organize results (default: true)
  --summarize      Generate summaries (default: true)
```

#### `template`
Manage and apply templates.

```bash
obsidian-librarian template SUBCOMMAND [OPTIONS]

Subcommands:
  list             List available templates
  apply            Apply template to notes
  auto-apply       Automatically apply appropriate templates
```

#### `stats`
Show vault statistics.

```bash
obsidian-librarian stats [OPTIONS]

Options:
  --detailed       Show detailed statistics
  --export PATH    Export stats to file
```

#### `validate`
Validate vault integrity.

```bash
obsidian-librarian validate [OPTIONS]

Options:
  --fix            Attempt to fix issues
  --report PATH    Save validation report
```

## Configuration

### Configuration File

The configuration file is located at `.obsidian-librarian/config.yaml` in your vault.

```yaml
# Vault settings
vault:
  path: /path/to/vault
  exclude_dirs:
    - .obsidian
    - .trash
    - Archive
  
# Organization settings
organization:
  auto_organize: true
  strategy: content
  preserve_structure: true
  
# Duplicate detection
duplicates:
  threshold: 0.85
  auto_merge: false
  ignore_patterns:
    - "daily/*"
    - "templates/*"
  
# Research settings
research:
  library_path: "Research Library"
  max_concurrent: 5
  rate_limit: 10
  sources:
    - arxiv.org
    - github.com
    - papers.nips.cc
  
# Template settings
templates:
  auto_apply: true
  template_dir: "Templates"
  rules:
    - pattern: "daily/*"
      template: "daily"
    - pattern: "projects/*"
      template: "project"
  
# Git integration
git:
  auto_backup: true
  change_threshold: 10
  commit_message_template: "Obsidian Librarian: {action}"
  
# AI settings
ai:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
```

### Environment Variables

```bash
# Set default vault path
export OBSIDIAN_VAULT=/path/to/vault

# Set API keys for AI features
export OPENAI_API_KEY=your-key-here

# Configure log level
export LIBRARIAN_LOG_LEVEL=INFO
```

## Advanced Usage

### Python API

Use Obsidian Librarian programmatically:

```python
import asyncio
from obsidian_librarian import Vault, ResearchService, Librarian
from obsidian_librarian.models import LibrarianConfig

async def main():
    # Initialize librarian
    config = LibrarianConfig(vault_path="/path/to/vault")
    librarian = Librarian(config)
    await librarian.initialize()
    
    # Research a topic
    results = await librarian.research_and_organize({
        "query": "machine learning optimization",
        "sources": ["arxiv.org"],
        "max_results": 10
    })
    
    # Find duplicates
    duplicates = await librarian.find_duplicates(threshold=0.9)
    
    # Organize vault
    await librarian.organize_vault(strategy="content")
    
    await librarian.shutdown()

asyncio.run(main())
```

### Custom Templates

Create custom template rules:

```yaml
# In .obsidian-librarian/templates.yaml
rules:
  - name: "Meeting Notes"
    pattern: "meetings/*.md"
    template: "meeting"
    variables:
      attendees: "extract:attendees"
      date: "extract:date"
      
  - name: "Book Notes"
    pattern: "books/*.md"
    detect:
      - "author:"
      - "isbn:"
    template: "book-note"
```

### Research Sources

Configure custom research sources:

```yaml
# In .obsidian-librarian/sources.yaml
sources:
  - name: "ArXiv"
    domain: "arxiv.org"
    type: "academic"
    rate_limit: 5
    
  - name: "Custom Wiki"
    domain: "wiki.company.com"
    type: "documentation"
    auth:
      type: "bearer"
      token_env: "WIKI_TOKEN"
```

### Automation

Set up automated tasks:

```bash
# Crontab example
# Daily organization at 2 AM
0 2 * * * obsidian-librarian organize --quiet

# Weekly duplicate check
0 10 * * 0 obsidian-librarian duplicates --export /tmp/duplicates.json

# Research digest every morning
0 9 * * * obsidian-librarian research "daily tech news" --max-results 10
```

## Troubleshooting

### Common Issues

#### 1. Rust Bindings Not Available

**Error**: `Rust core components not available`

**Solution**:
```bash
# Rebuild with maturin
cd python
maturin develop --release
```

#### 2. High Memory Usage

**Issue**: Memory usage grows with large vaults

**Solutions**:
- Adjust cache size in configuration
- Use `--no-cache` flag for one-time operations
- Process vault in batches

#### 3. Slow Performance

**Issue**: Operations take too long

**Solutions**:
- Enable multi-threading: `--threads 4`
- Reduce search depth
- Exclude large directories

#### 4. Git Conflicts

**Issue**: Automatic commits cause conflicts

**Solutions**:
- Disable auto-backup during manual edits
- Use `--no-git` flag
- Configure merge strategy

### Debug Mode

Enable detailed logging:

```bash
# Set log level
export LIBRARIAN_LOG_LEVEL=DEBUG

# Or use verbose flag
obsidian-librarian --verbose organize

# Save logs to file
obsidian-librarian --verbose organize 2> debug.log
```

### Performance Tuning

For large vaults (>10,000 notes):

```yaml
# In config.yaml
performance:
  cache_size: 10000
  chunk_size: 100
  max_concurrent: 8
  use_mmap: true
  
  # Rust-specific
  rust:
    thread_pool_size: 8
    channel_buffer: 1000
```

### Getting Help

1. Check the built-in help:
   ```bash
   obsidian-librarian --help
   obsidian-librarian COMMAND --help
   ```

2. View the documentation:
   ```bash
   obsidian-librarian docs
   ```

3. Report issues:
   - GitHub: https://github.com/yourusername/obsidian-librarian/issues
   - Email: support@obsidian-librarian.dev

## Best Practices

1. **Regular Backups**: Always backup your vault before major operations
2. **Test First**: Use `--dry-run` to preview changes
3. **Start Small**: Test on a subset of notes first
4. **Monitor Performance**: Check stats regularly
5. **Update Regularly**: Keep the tool updated for best performance

## Conclusion

Obsidian Librarian is designed to be your intelligent assistant for managing knowledge in Obsidian. With its powerful features and flexible configuration, it can adapt to any workflow and vault size.

For more information and updates:
- Documentation: https://obsidian-librarian.dev/docs
- GitHub: https://github.com/yourusername/obsidian-librarian
- Community: https://discord.gg/obsidian-librarian