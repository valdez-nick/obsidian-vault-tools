#!/usr/bin/env python3
"""
Setup script for obsidian-vault-tools
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="obsidian-vault-tools",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive toolkit for managing Obsidian vaults",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/obsidian-vault-tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business",
        "Topic :: Text Processing :: Markup :: Markdown",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0",
        "rich>=10.0",
        "pyyaml>=5.4",
        "pillow>=8.0",  # For ASCII art
        "pygame>=2.0",  # For audio
        "aiohttp>=3.8",  # For async operations
        "python-dotenv>=0.19",
        "pathlib>=1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "ai": [
            "openai>=0.27",
            "transformers>=4.20",
            "sentence-transformers>=2.0",
            "numpy>=1.21",
            "torch>=1.9",
        ],
        "mcp": [
            "mcp>=0.9.0",
            "cryptography>=3.4.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "obsidian-tools=obsidian_vault_tools.cli:main",
            "ovt=obsidian_vault_tools.cli:main",  # Short alias
        ],
    },
    package_data={
        "obsidian_vault_tools": [
            "audio/sounds/**/*.wav",
            "config/*.yaml",
        ],
    },
    include_package_data=True,
)