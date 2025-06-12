# Obsidian Librarian Example Vault

This is an example Obsidian vault demonstrating the features of Obsidian Librarian.

## What's Included

### 📁 Structure

```
example-vault/
├── .obsidian/              # Obsidian configuration
├── .obsidian-librarian/    # Librarian configuration
├── Daily Notes/            # Example daily notes
├── Projects/               # Sample project notes
├── Knowledge Base/         # Reference notes
├── Research Library/       # Research results (auto-generated)
├── Templates/              # Note templates
├── Archive/                # Archived/duplicate content
└── Welcome.md             # Getting started guide
```

### 🎯 Features Demonstrated

1. **Duplicate Detection**
   - `Python Best Practices.md` vs `Python Development Guidelines.md`
   - Run: `obsidian-librarian duplicates`

2. **Organization**
   - Mixed organizational structure ready for optimization
   - Run: `obsidian-librarian organize --dry-run`

3. **Templates**
   - Daily note template
   - Project template
   - Research template
   - Run: `obsidian-librarian template list`

4. **Research Integration**
   - Example research result in Research Library
   - Run: `obsidian-librarian research "your topic"`

5. **Links and References**
   - Wiki-style links between notes
   - Backlinks and forward links
   - Run: `obsidian-librarian validate`

## Quick Start

1. **Copy this vault** to your desired location:
   ```bash
   cp -r example-vault /path/to/my-test-vault
   cd /path/to/my-test-vault
   ```

2. **Initialize Obsidian Librarian**:
   ```bash
   obsidian-librarian init
   ```

3. **Explore features**:
   ```bash
   # View statistics
   obsidian-librarian stats
   
   # Find duplicates
   obsidian-librarian duplicates
   
   # Organize notes
   obsidian-librarian organize --dry-run
   
   # Research a topic
   obsidian-librarian research "machine learning"
   ```

## Configuration

The vault includes a pre-configured `.obsidian-librarian/config.yaml` with sensible defaults. Customize it to match your workflow.

## Sample Content

### Daily Notes
- Demonstrates task management
- Shows meeting notes structure
- Includes links and tags

### Projects
- Project Alpha: ML pipeline optimization example
- Shows project tracking structure
- Includes status, tasks, and team info

### Knowledge Base
- Python Best Practices: Technical reference
- Demonstrates knowledge organization
- Includes code examples

### Templates
- **daily.md**: Daily note template with tasks and reflection
- **project.md**: Comprehensive project tracking template
- **research.md**: Research note template with citations

## Testing Scenarios

### Scenario 1: Duplicate Management
```bash
# Find the duplicate Python guides
obsidian-librarian duplicates --threshold 0.8

# Review and merge
obsidian-librarian duplicates --merge
```

### Scenario 2: Research Workflow
```bash
# Research a topic
obsidian-librarian research "distributed systems"

# Check the Research Library for results
ls "Research Library/"
```

### Scenario 3: Template Application
```bash
# Create a new daily note without template
echo "# Today's Notes" > "Daily Notes/2024-01-16.md"

# Apply template
obsidian-librarian template apply --template daily --pattern "Daily Notes/2024-01-16.md"
```

## Tips

1. **Experiment freely** - This is a demo vault meant for testing
2. **Check logs** - Look in `.obsidian-librarian/librarian.log` for details
3. **Try automation** - Test the auto-organization features
4. **Customize** - Modify the config to see different behaviors

## Support

- Documentation: Run `obsidian-librarian docs`
- Help: Run `obsidian-librarian --help`
- GitHub: https://github.com/obsidian-librarian/obsidian-librarian

Happy exploring! 🚀