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
    version="2.3.0",
    author="Nick Valdez",
    author_email="nvaldez@siftscience.com",
    description="Comprehensive toolkit for managing Obsidian vaults with AI-powered features and PM automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valdez-nick/obsidian-vault-tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business",
        "Topic :: Text Processing :: Markup :: Markdown",
        "Topic :: Software Development :: Libraries :: Python Modules",
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
        "pandas>=1.5.0",  # Core data processing
        "pydantic>=1.10.0",  # Data validation
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-asyncio>=0.21.0",
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
            "chromadb>=0.4.0",
        ],
        "mcp": [
            "mcp>=0.9.0",
            "cryptography>=3.4.8",
        ],
        "pm-automation": [
            # Core PM Automation dependencies
            "scikit-learn>=1.2.0",
            "plotly>=5.0.0",
            "tenacity>=8.0.0",
            "python-pptx>=0.6.21",
            "keyring>=23.0.0",
            "cryptography>=3.4.8",
            # Optional integrations
            "snowflake-connector-python>=3.0.0;python_version>='3.8'",
            "google-api-python-client>=2.70.0",
            "google-auth>=2.0.0",
            "google-auth-oauthlib>=0.5.0",
            "atlassian-python-api>=3.40.0",
            "jira>=3.4.0",
            # ML and Analytics (optional)
            "tensorflow>=2.10.0;python_version>='3.8'",
            "prometheus-client>=0.14.0",
        ],
        "all": [
            # Include all features
            "openai>=0.27",
            "transformers>=4.20",
            "sentence-transformers>=2.0",
            "numpy>=1.21",
            "torch>=1.9",
            "chromadb>=0.4.0",
            "mcp>=0.9.0",
            "cryptography>=3.4.8",
            "scikit-learn>=1.2.0",
            "plotly>=5.0.0",
            "tenacity>=8.0.0",
            "python-pptx>=0.6.21",
            "keyring>=23.0.0",
            "snowflake-connector-python>=3.0.0;python_version>='3.8'",
            "google-api-python-client>=2.70.0",
            "google-auth>=2.0.0",
            "google-auth-oauthlib>=0.5.0",
            "atlassian-python-api>=3.40.0",
            "jira>=3.4.0",
            "tensorflow>=2.10.0;python_version>='3.8'",
            "prometheus-client>=0.14.0",
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
        "pm_automation_suite": [
            "templates/*.json",
            "templates/*.yaml",
            "examples/*.py",
            "docs/*.md",
            ".env.example",
            "requirements.txt",
        ],
    },
    include_package_data=True,
)