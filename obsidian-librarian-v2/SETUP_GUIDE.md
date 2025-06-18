# Obsidian Librarian v2 Setup Guide

## Prerequisites

### 1. Python 3.11+ (Required)
You currently have Python 3.10.13, which is **too old**. You need Python 3.11 or later.

```bash
# Check your Python version
python --version

# Install Python 3.11+ using pyenv
pyenv install 3.11.7
pyenv local 3.11.7  # Set for this project
```

### 2. Rust (Required for building)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 3. Build Tools
```bash
# macOS (you already have these)
xcode-select --install

# Install maturin for building Python-Rust bindings
pip install maturin
```

## Installation Steps

### Step 1: Build the Rust Components
```bash
cd /Users/nvaldez/Documents/repos/Obsidian/obsidian-librarian-v2/rust-core

# Build all Rust crates
cargo build --release

# Run tests to ensure everything works
cargo test
```

### Step 2: Build Python Package with Rust Bindings
```bash
cd /Users/nvaldez/Documents/repos/Obsidian/obsidian-librarian-v2/python

# Build and install the package
maturin develop --release

# Install with development dependencies
pip install -e ".[dev]"
```

### Step 3: Verify Installation
```bash
# Test the CLI
obsidian-librarian --help

# Run the test suite
pytest
```

## Using Obsidian Librarian v2

### Basic Commands

#### 1. Analyze Your Vault
```bash
# Basic analysis
obsidian-librarian analyze /Users/nvaldez/Documents/repos/Obsidian

# Detailed analysis with specific checks
obsidian-librarian analyze /Users/nvaldez/Documents/repos/Obsidian \
  --duplicates \
  --quality \
  --structure
```

#### 2. Tag Management (What You Need!)

```bash
# Analyze tag usage and find issues
obsidian-librarian tags analyze /Users/nvaldez/Documents/repos/Obsidian

# Find duplicate/similar tags
obsidian-librarian tags duplicates /Users/nvaldez/Documents/repos/Obsidian \
  --threshold 0.8

# Clean up tags (dry run first!)
obsidian-librarian tags cleanup /Users/nvaldez/Documents/repos/Obsidian \
  --fix-case \
  --remove-special \
  --remove-unused \
  --dry-run

# Apply the cleanup
obsidian-librarian tags cleanup /Users/nvaldez/Documents/repos/Obsidian \
  --fix-case \
  --remove-special \
  --remove-unused

# Auto-tag untagged notes using AI
obsidian-librarian tags auto-tag /Users/nvaldez/Documents/repos/Obsidian \
  --confidence 0.7 \
  --max-tags 5

# Merge similar tags
obsidian-librarian tags merge /Users/nvaldez/Documents/repos/Obsidian \
  --source "daily-notes" \
  --target "daily-note"

# Fix tag hierarchies
obsidian-librarian tags hierarchy /Users/nvaldez/Documents/repos/Obsidian \
  --create-missing \
  --max-depth 4
```

#### 3. Research Assistant
```bash
# Perform intelligent research
obsidian-librarian research /Users/nvaldez/Documents/repos/Obsidian \
  "fraud detection machine learning approaches"

# Research with specific sources
obsidian-librarian research /Users/nvaldez/Documents/repos/Obsidian \
  "payment protection strategies" \
  --sources github,arxiv,documentation
```

#### 4. Content Curation
```bash
# Intelligent content curation
obsidian-librarian curate /Users/nvaldez/Documents/repos/Obsidian \
  --quality \
  --structure \
  --relationships
```

#### 5. Template Application
```bash
# Apply templates based on content analysis
obsidian-librarian templates /Users/nvaldez/Documents/repos/Obsidian \
  --auto \
  --template-dir /path/to/templates
```

### Configuration

Create a config file at `~/.obsidian-librarian/config.yaml`:

```yaml
# Tag management settings
tag_management:
  fuzzy_similarity_threshold: 0.8
  semantic_similarity_threshold: 0.7
  case_insensitive: true
  normalize_special_chars: true
  min_tag_length: 2
  max_tag_length: 50
  max_hierarchy_depth: 4
  min_usage_threshold: 2
  auto_tag_confidence: 0.7
  max_auto_tags_per_note: 5

# AI settings
ai:
  provider: openai  # or anthropic
  model: gpt-4
  api_key: ${OPENAI_API_KEY}  # From environment variable

# Research settings
research:
  max_concurrent_searches: 10
  cache_results: true
  output_format: markdown

# Vault settings
vault:
  exclude_patterns:
    - "*.tmp"
    - ".obsidian/workspace*"
    - "Archive/*"
```

## Recommended Workflow for Tag Cleanup

### 1. Backup First
```bash
python backup_vault.py /Users/nvaldez/Documents/repos/Obsidian --type pre-tags
```

### 2. Analyze Current State
```bash
obsidian-librarian tags analyze /Users/nvaldez/Documents/repos/Obsidian > tag_analysis_detailed.txt
```

### 3. Find Duplicates
```bash
obsidian-librarian tags duplicates /Users/nvaldez/Documents/repos/Obsidian --threshold 0.7
```

### 4. Preview Cleanup
```bash
obsidian-librarian tags cleanup /Users/nvaldez/Documents/repos/Obsidian \
  --fix-case \
  --remove-special \
  --remove-unused \
  --dry-run > cleanup_preview.txt
```

### 5. Apply Cleanup
```bash
obsidian-librarian tags cleanup /Users/nvaldez/Documents/repos/Obsidian \
  --fix-case \
  --remove-special \
  --remove-unused
```

### 6. Auto-tag Untagged Notes
```bash
# First, see what would be tagged
obsidian-librarian tags suggest /Users/nvaldez/Documents/repos/Obsidian \
  --untagged-only \
  --confidence 0.7

# Then apply auto-tagging
obsidian-librarian tags auto-tag /Users/nvaldez/Documents/repos/Obsidian \
  --confidence 0.7 \
  --max-tags 5
```

### 7. Fix Hierarchies
```bash
obsidian-librarian tags hierarchy /Users/nvaldez/Documents/repos/Obsidian \
  --create-missing \
  --optimize
```

## Performance Tips

1. **Use Rust-powered features**: The Rust core makes operations 10-100x faster
2. **Batch operations**: Process multiple files at once
3. **Use caching**: Enable cache in config for repeated operations
4. **Exclude patterns**: Configure exclusions to skip unnecessary files

## Troubleshooting

### Common Issues

1. **Python version error**: Make sure you're using Python 3.11+
2. **Rust build errors**: Ensure Rust is properly installed
3. **Import errors**: Run `maturin develop --release` again
4. **Permission errors**: Check file permissions in your vault

### Getting Help

```bash
# View detailed help
obsidian-librarian --help
obsidian-librarian tags --help

# Check version and dependencies
obsidian-librarian --version --verbose
```

## Alternative: Quick Tag Fix

If you don't want to set up the full v2 tool yet, use the simpler script:

```bash
# Backup first
python backup_vault.py /Users/nvaldez/Documents/repos/Obsidian --type pre-tags

# Run the simple tag fixer
python fix_vault_tags.py /Users/nvaldez/Documents/repos/Obsidian --apply
```

This will handle the basic tag formatting issues while you set up the full v2 tool for more advanced features.