# Getting Started with Obsidian Librarian

This tutorial will walk you through setting up and using Obsidian Librarian for the first time.

## Prerequisites

Before we begin, make sure you have:

1. **Python 3.8+** installed
2. **An Obsidian vault** (even an empty one works)
3. **Basic command line knowledge**

## Step 1: Installation

### Option A: Install from PyPI (Recommended)

```bash
pip install obsidian-librarian
```

### Option B: Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/obsidian-librarian.git
cd obsidian-librarian

# Install in development mode
make dev
```

Verify the installation:

```bash
obsidian-librarian --version
```

## Step 2: Initialize Your Vault

Navigate to your Obsidian vault directory and initialize it:

```bash
cd /path/to/your/obsidian/vault
obsidian-librarian init
```

This creates:
- `.obsidian-librarian/` directory for configuration
- `Research Library/` directory for research results
- Default configuration file

## Step 3: Your First Organization

Let's organize your existing notes:

```bash
# Preview what will happen (dry run)
obsidian-librarian organize --dry-run

# If it looks good, run it for real
obsidian-librarian organize
```

The librarian will:
- Analyze your note content
- Suggest organizational improvements
- Create a more logical structure

## Step 4: Finding Duplicates

Check for duplicate content in your vault:

```bash
# Find duplicates
obsidian-librarian duplicates

# Find with lower similarity threshold (more results)
obsidian-librarian duplicates --threshold 0.75

# Export results for review
obsidian-librarian duplicates --export duplicates-report.md
```

## Step 5: Research Assistant

Let's research a topic and automatically organize the findings:

```bash
# Basic research
obsidian-librarian research "getting started with obsidian plugins"

# Research from specific sources
obsidian-librarian research "machine learning basics" --sources arxiv.org,github.com

# Limit results
obsidian-librarian research "python async programming" --max-results 10
```

The librarian will:
1. Search for relevant content
2. Summarize findings
3. Create organized notes in your Research Library
4. Link to existing related notes

## Step 6: Template Application

If you use Templater, the librarian can apply templates to existing notes:

```bash
# List available templates
obsidian-librarian template list

# Apply a template to specific notes
obsidian-librarian template apply --template daily --pattern "2024-01-*"

# Auto-detect and apply appropriate templates
obsidian-librarian template auto-apply
```

## Step 7: Interactive Mode

For a more conversational experience:

```bash
obsidian-librarian interactive
```

In interactive mode, you can:
- Ask questions about your vault
- Request specific operations
- Get suggestions for improvement

Example commands:
```
> find all notes about python
> organize my meeting notes
> research quantum computing and create summary
> show me duplicate notes from this week
```

## Step 8: Basic Configuration

Edit `.obsidian-librarian/config.yaml` to customize behavior:

```yaml
# Basic configuration
vault:
  exclude_dirs:
    - .obsidian
    - .trash
    - Archive

organization:
  strategy: content  # or: tags, date, links
  preserve_structure: true

research:
  max_concurrent: 5
  default_sources:
    - arxiv.org
    - github.com

git:
  auto_backup: true
  change_threshold: 10
```

## Step 9: Automation

Set up automated tasks using cron (Linux/Mac) or Task Scheduler (Windows):

### Daily Organization (2 AM)
```bash
# Add to crontab with: crontab -e
0 2 * * * cd /path/to/vault && obsidian-librarian organize --quiet
```

### Weekly Duplicate Check (Sundays at 10 AM)
```bash
0 10 * * 0 cd /path/to/vault && obsidian-librarian duplicates --export /tmp/weekly-duplicates.md
```

## Step 10: Best Practices

### 1. Start Small
Test operations on a subset of notes first:
```bash
obsidian-librarian organize --pattern "test/*" --dry-run
```

### 2. Regular Backups
Before major operations:
```bash
obsidian-librarian backup "Before major reorganization"
```

### 3. Monitor Performance
Check vault statistics regularly:
```bash
obsidian-librarian stats --detailed
```

### 4. Use Dry Run
Always preview changes:
```bash
obsidian-librarian [command] --dry-run
```

## Common Workflows

### Research Workflow
```bash
# 1. Research a topic
obsidian-librarian research "rust programming best practices"

# 2. Review the generated notes in Research Library/

# 3. Find related existing notes
obsidian-librarian search "rust programming"

# 4. Merge or link related content
obsidian-librarian duplicates --merge
```

### Maintenance Workflow
```bash
# 1. Check vault health
obsidian-librarian validate

# 2. Find and resolve duplicates
obsidian-librarian duplicates --merge

# 3. Organize by content similarity
obsidian-librarian organize --strategy content

# 4. Apply templates to unstructured notes
obsidian-librarian template auto-apply

# 5. Backup changes
obsidian-librarian backup "Weekly maintenance"
```

### Content Creation Workflow
```bash
# 1. Research topic
obsidian-librarian research "topic name"

# 2. Create note from template
obsidian-librarian template create --template article

# 3. Find related notes to link
obsidian-librarian search "related keywords"

# 4. Check for duplicates
obsidian-librarian duplicates --pattern "new-note*"
```

## Troubleshooting

### Issue: "Command not found"
```bash
# Check if installed
pip list | grep obsidian-librarian

# Reinstall if needed
pip install --upgrade obsidian-librarian
```

### Issue: "Vault not initialized"
```bash
# Initialize in your vault directory
cd /path/to/vault
obsidian-librarian init
```

### Issue: "Permission denied"
```bash
# Check file permissions
ls -la .obsidian-librarian/

# Fix permissions if needed
chmod -R u+rw .obsidian-librarian/
```

## Next Steps

Now that you have the basics down:

1. Read the [User Guide](../user-guide.md) for detailed features
2. Explore the [API Reference](../api-reference.md) for programmatic usage
3. Check out [Advanced Tutorials](./advanced-usage.md) for power user features
4. Join our community for tips and support

## Getting Help

- **Documentation**: Run `obsidian-librarian docs`
- **Help**: Run `obsidian-librarian --help`
- **GitHub Issues**: https://github.com/yourusername/obsidian-librarian/issues
- **Discord**: https://discord.gg/obsidian-librarian

Happy organizing! 📚