"""
Setup configuration for PM Automation Suite

This file defines the package metadata and dependencies for installation.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core dependencies (minimal set)
CORE_DEPS = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "dataclasses-json>=0.6.0",
    "aiohttp>=3.8.0",
    "click>=8.1.0",
    "structlog>=23.0.0",
    "rich>=13.0.0",
]

# Optional dependency groups
EXTRAS = {
    # AI/ML features
    "ai": [
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "langchain>=0.1.0",
        "tiktoken>=0.5.0",
        "transformers>=4.35.0",
        "scikit-learn>=1.3.0",
    ],
    # API integrations
    "integrations": [
        "atlassian-python-api>=3.41.0",
        "google-api-python-client>=2.100.0",
        "google-auth>=2.23.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth-oauthlib>=1.0.0",
        "snowflake-connector-python>=3.5.0",
        "slack-sdk>=3.23.0",
    ],
    # Web API server
    "api": [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic-settings>=2.0.0",
    ],
    # Task scheduling
    "scheduler": [
        "apscheduler>=3.10.0",
        "celery>=5.3.0",
        "redis>=5.0.0",
    ],
    # Development tools
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.12.0",
        "faker>=20.0.0",
        "black>=23.0.0",
        "flake8>=6.1.0",
        "mypy>=1.7.0",
        "isort>=5.12.0",
        "pre-commit>=3.5.0",
    ],
    # Documentation
    "docs": [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.4.0",
        "mkdocstrings>=0.24.0",
    ],
}

# All extras combined
EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))

setup(
    name="pm-automation-suite",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive automation suite for Product Management workflows",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/valdez-nick/obsidian-vault-tools",
    project_urls={
        "Bug Tracker": "https://github.com/valdez-nick/obsidian-vault-tools/issues",
        "Documentation": "https://github.com/valdez-nick/obsidian-vault-tools/wiki",
        "Source Code": "https://github.com/valdez-nick/obsidian-vault-tools",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Topic :: Office/Business :: Scheduling",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=CORE_DEPS,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "pm-suite=pm_automation_suite.cli:main",
            "pm-api=pm_automation_suite.api.main:run",
            "pm-scheduler=pm_automation_suite.orchestration.scheduler:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pm_automation_suite": [
            "templates/*.json",
            "templates/*.yaml",
            "templates/*.md",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "product management",
        "automation",
        "workflow",
        "jira",
        "confluence",
        "google sheets",
        "ai",
        "obsidian",
        "productivity",
    ],
    # Additional metadata
    license="MIT",
    platforms=["any"],
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    # Options
    options={
        "bdist_wheel": {
            "universal": False  # This is a Python 3 only package
        },
    },
)