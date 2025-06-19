#!/bin/bash
# Set up the Python package structure

echo "Creating package structure..."

# Create main package directory
mkdir -p obsidian_vault_tools/{manager,analysis,organization,creative,audio,ai,backup,utils}

# Create __init__.py files
touch obsidian_vault_tools/__init__.py
touch obsidian_vault_tools/manager/__init__.py
touch obsidian_vault_tools/analysis/__init__.py
touch obsidian_vault_tools/organization/__init__.py
touch obsidian_vault_tools/creative/__init__.py
touch obsidian_vault_tools/audio/__init__.py
touch obsidian_vault_tools/ai/__init__.py
touch obsidian_vault_tools/backup/__init__.py
touch obsidian_vault_tools/utils/__init__.py

# Move files to appropriate locations
echo "Organizing files into package structure..."

# Manager module
mv vault_manager.py obsidian_vault_tools/manager/
mv vault_manager_enhanced.py obsidian_vault_tools/manager/
mv menu_navigator.py obsidian_vault_tools/manager/

# Analysis module
mv analyze_tags_simple.py obsidian_vault_tools/analysis/tag_analyzer.py
cp vault_query_system*.py obsidian_vault_tools/analysis/

# Organization module
mv fix_vault_tags.py obsidian_vault_tools/organization/tag_fixer.py

# Creative module
mv ascii_*.py obsidian_vault_tools/creative/
mv better_ascii_converter.py obsidian_vault_tools/creative/

# Audio module (move entire directory)
mv audio/* obsidian_vault_tools/audio/
rmdir audio

# AI module
mv llm_model_manager.py obsidian_vault_tools/ai/model_manager.py
mv vault_query_system_llm.py obsidian_vault_tools/ai/llm_query_system.py
mv natural_language_query.py obsidian_vault_tools/ai/
mv query_router.py obsidian_vault_tools/ai/
mv models obsidian_vault_tools/ai/adapters

# Backup module
mv backup_vault.py obsidian_vault_tools/backup/

# Utils module
mv file_versioning.py obsidian_vault_tools/utils/
mv feedback_collector.py obsidian_vault_tools/utils/
mv cli_config.py obsidian_vault_tools/utils/

# Create directories for non-code files
mkdir -p {tests,docs,examples}

# Move test files
mv comprehensive_e2e_test.py tests/
mv *_test_*.py tests/
mv test_*.py tests/

# Move demo files to examples
mv demo_*.py examples/

# Move documentation
mv *.md docs/

# Keep shell scripts in scripts directory
mkdir -p scripts
mv *.sh scripts/

echo "Package structure created!"