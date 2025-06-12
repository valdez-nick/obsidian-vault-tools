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
```

### 3. Interactive Mode

```bash
obsidian-librarian interactive
```

## 📖 Documentation

- [User Guide](https://obsidian-librarian.readthedocs.io/en/latest/user-guide/)
- [API Reference](https://obsidian-librarian.readthedocs.io/en/latest/api-reference/)
- [Configuration](https://obsidian-librarian.readthedocs.io/en/latest/configuration/)
- [Tutorials](https://obsidian-librarian.readthedocs.io/en/latest/tutorials/)

## 🛠️ Configuration

Create `.obsidian-librarian/config.yaml` in your vault:

```yaml
organization:
  strategy: content  # Options: content, tags, date, links
  preserve_structure: true

duplicates:
  threshold: 0.85
  auto_merge: false

research:
  default_sources:
    - arxiv.org
    - github.com
  max_results: 20

git:
  auto_backup: true
  change_threshold: 10
```

## 🐍 Python API

```python
import asyncio
from obsidian_librarian import Librarian, LibrarianConfig

async def main():
    config = LibrarianConfig(vault_path="/path/to/vault")
    
    async with Librarian(config) as librarian:
        # Find duplicates
        duplicates = await librarian.find_duplicates(threshold=0.9)
        
        # Research a topic
        results = await librarian.research_and_organize({
            "query": "quantum computing",
            "max_results": 10
        })
        
        # Organize vault
        await librarian.organize_vault(strategy="content")

asyncio.run(main())
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