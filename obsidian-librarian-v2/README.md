# Obsidian Librarian v2 🚀

**Transform your Obsidian vault into an AI-powered knowledge powerhouse**

[![Version](https://img.shields.io/badge/version-v0.2.0-blue.svg)](https://github.com/obsidian-librarian/obsidian-librarian-v2/releases/tag/v0.2.0)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://rustlang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🤔 What is Obsidian Librarian?

Obsidian Librarian is your **AI-powered assistant** for managing and organizing your Obsidian knowledge vault. It automatically:

- 🏷️ **Cleans up your messy tags** - finds duplicates, suggests mergers, and builds tag hierarchies
- 📁 **Organizes your files intelligently** - uses AI to categorize and file your notes automatically  
- 🔍 **Researches topics for you** - fetches content from the web and creates organized notes
- 🤖 **Analyzes your content** - finds duplicate notes, suggests improvements, and scores quality
- ⚡ **Works at lightning speed** - handles vaults with 100,000+ notes without breaking a sweat

Think of it as having a **personal librarian** who understands your content and keeps everything perfectly organized while you focus on writing.

## 🎯 Why Use Obsidian Librarian?

### The Problem
- 😫 **Tag chaos**: `#machinelearning`, `#machine-learning`, `#ml` all meaning the same thing
- 🗂️ **Messy folders**: Notes scattered everywhere with no consistent organization
- 🔍 **Lost content**: Can't find that note you wrote 6 months ago
- 📚 **Research overload**: Manually copying content from web sources
- 🔄 **Duplicate notes**: Same ideas written multiple times in different places

### The Solution
Obsidian Librarian uses **advanced AI and machine learning** to:
- Automatically detect and merge similar tags
- Intelligently file notes based on their content
- Find and merge duplicate content
- Research topics and create well-organized notes
- Monitor your vault in real-time and keep it organized

## 🚀 Quick Start

### Install in 30 seconds
```bash
pip install obsidian-librarian
obsidian-librarian init /path/to/your/vault
```

### Clean up your tags instantly
```bash
# Find all your messy duplicate tags
obsidian-librarian tags analyze

# Auto-merge similar tags (with preview)
obsidian-librarian tags cleanup --merge-similar

# Get AI suggestions for better tags
obsidian-librarian tags suggest --ai
```

### Auto-organize your vault
```bash
# See how AI would organize your notes
obsidian-librarian organize plan

# Enable real-time auto-organization
obsidian-librarian organize auto --enable
```

### Research anything
```bash
# Research a topic and auto-create organized notes
obsidian-librarian research "quantum computing basics"
```

## ✨ Core Features

### 🏷️ Smart Tag Management
Transform your chaotic tags into a well-organized taxonomy:

```
Before: #ml, #machinelearning, #machine-learning, #ML, #ai/ml
After:  #machine-learning (with all notes properly tagged)
```

- **Find duplicate tags** automatically (even with typos!)
- **Merge similar tags** with one command
- **Build tag hierarchies** like `#programming/languages/python`
- **Get AI suggestions** for missing tags on any note
- **Bulk operations** to clean up years of tag mess

### 📁 Intelligent File Organization  
Let AI organize your vault like a professional librarian:

```
Before: 500 notes in the root folder 😱
After:  Everything neatly organized by topic, type, and date
```

- **AI categorization** - understands note content and files it correctly
- **Custom rules** - "Put all daily notes in Archive/Daily/"
- **Auto-organization** - new notes get filed automatically
- **Safe operations** - preview changes before applying
- **Learn your style** - improves organization over time

### 🔍 Research Assistant
Turn web research into organized notes automatically:

```bash
obsidian-librarian research "transformers in NLP"
# Creates notes from ArXiv papers, GitHub repos, blogs, and more
```

- **Multi-source search** - ArXiv, GitHub, documentation, blogs
- **Auto-summarization** - key points extracted automatically
- **Smart organization** - research filed by topic and date
- **Citation tracking** - keeps source links for everything

### 🤖 Content Analysis
Find and fix issues in your vault:

- **Duplicate detection** - finds notes with similar content
- **Quality scoring** - identifies incomplete or poorly formatted notes
- **Link analysis** - finds broken links and orphaned notes
- **Template matching** - suggests templates for existing notes

### ⚡ Lightning Fast Performance
Built with Rust for incredible speed:

- Process **1,000+ notes per second**
- Handle vaults with **100,000+ notes**
- **Real-time monitoring** with instant updates
- **Minimal memory usage** even with huge vaults

## 💡 Real-World Examples

### Example 1: Clean up years of messy tags
```bash
$ obsidian-librarian tags analyze

📊 Tag Analysis Report:
- Total tags: 1,847
- Duplicate clusters found: 127
- Suggested merges: 
  • #ml, #ML, #machinelearning → #machine-learning (47 notes)
  • #todo, #TODO, #to-do → #todo (183 notes)
  • #book, #books, #reading → #books (91 notes)

$ obsidian-librarian tags cleanup --merge-similar --preview
```

### Example 2: Auto-organize a chaotic vault
```bash
$ obsidian-librarian organize plan

🗂️ Organization Plan:
- Notes to organize: 523
- Suggested structure:
  📁 Projects/
    📁 Active/ (73 notes)
    📁 Archive/ (112 notes)
  📁 Areas/
    📁 Work/ (89 notes)
    📁 Personal/ (124 notes)
  📁 Resources/
    📁 Articles/ (67 notes)
    📁 Books/ (58 notes)

$ obsidian-librarian organize execute --auto-approve
```

### Example 3: Research and create notes automatically
```bash
$ obsidian-librarian research "RAG systems implementation"

🔍 Researching: RAG systems implementation
✓ Found 23 ArXiv papers
✓ Found 15 GitHub repositories  
✓ Found 31 blog posts
✓ Creating organized notes...

📝 Created notes:
- Research/AI/RAG Systems Overview.md
- Research/AI/RAG Implementation Guide.md
- Research/AI/Vector Database Comparison.md
- Research/AI/RAG Best Practices.md
```

## 📦 Installation

### Option 1: Quick Install (Recommended)
```bash
pip install obsidian-librarian
```

### Option 2: Install from source
```bash
git clone https://github.com/valdez-nick/obsidian-librarian-v2
cd obsidian-librarian-v2
pip install -e .
```

### Requirements
- Python 3.9+
- Obsidian vault (obviously! 😄)
- Optional: Rust 1.70+ (for building from source)

## 🚀 Getting Started

### 1. Initialize your vault
```bash
obsidian-librarian init /path/to/vault
```

### 2. See what needs fixing
```bash
obsidian-librarian analyze
```

### 3. Clean up tags
```bash
obsidian-librarian tags cleanup --preview
```

### 4. Organize your files
```bash
obsidian-librarian organize auto --enable
```

That's it! Your vault is now self-organizing. 🎉

## 🐍 Python API

For developers who want to integrate Obsidian Librarian into their workflows:

```python
import asyncio
from obsidian_librarian import Vault, TagManagerService, AutoOrganizer

async def organize_my_vault():
    # Connect to vault
    vault = await Vault.create("/path/to/vault")
    
    # Clean up tags
    tag_manager = TagManagerService(vault)
    await tag_manager.merge_similar_tags(threshold=0.8)
    
    # Auto-organize files
    organizer = AutoOrganizer(vault)
    await organizer.enable_auto_organization()

asyncio.run(organize_my_vault())
```

## 🛠️ Advanced Configuration

### Custom Organization Rules
```yaml
# .obsidian-librarian/config.yaml
organization:
  rules:
    - name: "Daily Notes"
      pattern: "**/Daily Note*.md"
      destination: "Journal/Daily/{year}/{month}/"
    
    - name: "Meeting Notes"  
      pattern: "**/meeting*.md"
      destination: "Work/Meetings/{year}/"
    
    - name: "Book Notes"
      content_match: "#book OR #reading"
      destination: "Resources/Books/"
```

### AI Configuration
```yaml
ai:
  provider: "openai"  # or "anthropic", "local"
  model: "gpt-4"
  tag_suggestions: true
  auto_summarize: true
```

## 🎯 Why Obsidian Librarian?

### Without Obsidian Librarian 😩
- Hours spent organizing files manually
- Duplicate notes you can't find
- Inconsistent tagging makes search useless
- Research scattered across random notes
- Constant worry about vault organization

### With Obsidian Librarian 🚀
- Vault organizes itself automatically
- AI understands and categorizes your content
- Tags stay clean and consistent
- Research gets filed perfectly
- Focus on writing, not organizing

## 📊 Performance

- **Handles 100,000+ notes** without breaking a sweat
- **Processes 1,000+ notes/second** during analysis
- **Real-time monitoring** with <100ms response time
- **Low memory usage** - typically under 500MB
- **Works with your existing vault** - no migration needed

## 🤝 Contributing

We'd love your help! Check out our [Contributing Guide](CONTRIBUTING.md) for ways to get involved.

## 📄 License

MIT License - because knowledge tools should be free.

## 🙏 Support

- 🐛 [Report bugs](https://github.com/valdez-nick/obsidian-librarian-v2/issues)
- 💬 [Join discussions](https://github.com/valdez-nick/obsidian-librarian-v2/discussions)
- ⭐ Star this repo to show support!
- 📧 Contact: obsidian.librarian@example.com

---

<p align="center">
  <b>Transform your Obsidian vault from chaotic to organized in minutes.</b><br>
  <a href="https://github.com/valdez-nick/obsidian-librarian-v2">Get Started</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-core-features">Features</a> •
  <a href="https://github.com/valdez-nick/obsidian-librarian-v2/wiki">Documentation</a>
</p>