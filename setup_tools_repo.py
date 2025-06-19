#!/usr/bin/env python3
"""
Setup script to organize the obsidian-vault-tools repository
Run this from the obsidian-vault-tools directory
"""
import os
import shutil
from pathlib import Path

def setup_package_structure():
    """Set up the Python package structure"""
    
    # Create package directories
    package_dirs = [
        "obsidian_vault_tools",
        "obsidian_vault_tools/manager",
        "obsidian_vault_tools/analysis", 
        "obsidian_vault_tools/organization",
        "obsidian_vault_tools/creative",
        "obsidian_vault_tools/audio",
        "obsidian_vault_tools/ai",
        "obsidian_vault_tools/ai/adapters",
        "obsidian_vault_tools/backup",
        "obsidian_vault_tools/utils",
        "tests",
        "docs", 
        "examples",
        "scripts",
        "homebrew"
    ]
    
    for dir_path in package_dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    # Create __init__.py files
    init_dirs = [d for d in package_dirs if d.startswith("obsidian_vault_tools")]
    for dir_path in init_dirs:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, 'a').close()
    
    # File movements mapping
    file_moves = {
        # Manager module
        "vault_manager.py": "obsidian_vault_tools/manager/vault_manager.py",
        "vault_manager_enhanced.py": "obsidian_vault_tools/manager/enhanced_manager.py",
        "menu_navigator.py": "obsidian_vault_tools/manager/menu_navigator.py",
        
        # Analysis module  
        "analyze_tags_simple.py": "obsidian_vault_tools/analysis/tag_analyzer.py",
        "vault_query_system.py": "obsidian_vault_tools/analysis/vault_query_system.py",
        
        # Organization module
        "fix_vault_tags.py": "obsidian_vault_tools/organization/tag_fixer.py",
        
        # Creative module
        "ascii_art_converter.py": "obsidian_vault_tools/creative/ascii_art_converter.py",
        "ascii_magic_converter.py": "obsidian_vault_tools/creative/ascii_magic_converter.py",
        "ascii_flowchart_generator.py": "obsidian_vault_tools/creative/flowchart_generator.py",
        "better_ascii_converter.py": "obsidian_vault_tools/creative/advanced_ascii_converter.py",
        
        # AI module
        "llm_model_manager.py": "obsidian_vault_tools/ai/model_manager.py",
        "vault_query_system_llm.py": "obsidian_vault_tools/ai/llm_query_system.py",
        "natural_language_query.py": "obsidian_vault_tools/ai/natural_language_query.py",
        "query_router.py": "obsidian_vault_tools/ai/query_router.py",
        
        # Backup module
        "backup_vault.py": "obsidian_vault_tools/backup/backup_manager.py",
        
        # Utils module
        "file_versioning.py": "obsidian_vault_tools/utils/file_versioning.py",
        "feedback_collector.py": "obsidian_vault_tools/utils/feedback_collector.py",
        "cli_config.py": "obsidian_vault_tools/utils/config.py",
        
        # Test files
        "comprehensive_e2e_test.py": "tests/test_e2e.py",
        "audio_test_improved.py": "tests/test_audio_improved.py",
        "audio_test_isolated.py": "tests/test_audio_isolated.py",
        "audio_test_alternative.py": "tests/test_audio_alternative.py",
        "test_obvious_sounds.py": "tests/test_obvious_sounds.py",
        
        # Demo files
        "demo_vault_manager.py": "examples/demo_vault_manager.py",
        "demo_enhanced_manager.py": "examples/demo_enhanced_manager.py",
        "audio_demo.py": "examples/audio_demo.py",
        
        # Scripts
        "quick_backup.sh": "scripts/quick_backup.sh",
        "quick_incremental_backup.sh": "scripts/quick_incremental_backup.sh", 
        "run_tests.sh": "scripts/run_tests.sh",
        "setup_kopia_backup.sh": "scripts/setup_kopia_backup.sh",
    }
    
    # Move files
    for src, dst in file_moves.items():
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")
    
    # Move entire audio directory contents
    if os.path.exists("audio"):
        for item in os.listdir("audio"):
            src = os.path.join("audio", item)
            dst = os.path.join("obsidian_vault_tools/audio", item)
            if os.path.exists(src):
                shutil.move(src, dst)
        os.rmdir("audio")
        
    # Move models to AI adapters
    if os.path.exists("models"):
        shutil.move("models", "obsidian_vault_tools/ai/adapters/")
    
    # Move obsidian-librarian-v2 to a temporary location for integration
    if os.path.exists("obsidian-librarian-v2"):
        shutil.move("obsidian-librarian-v2", "librarian_to_integrate")
        
    # Move documentation
    md_files = [f for f in os.listdir('.') if f.endswith('.md')]
    for md_file in md_files:
        shutil.move(md_file, f"docs/{md_file}")
        
    # Move remaining audio diagnostic/fix files
    audio_files = [f for f in os.listdir('.') if 'audio_' in f and f.endswith('.py')]
    for audio_file in audio_files:
        shutil.move(audio_file, f"obsidian_vault_tools/audio/{audio_file}")
        
    print("\nPackage structure created successfully!")
    print("\nNext steps:")
    print("1. Create setup.py and pyproject.toml")
    print("2. Integrate librarian_to_integrate features")
    print("3. Update imports in all Python files")
    print("4. Create CLI entry point")
    print("5. Write comprehensive documentation")

if __name__ == "__main__":
    setup_package_structure()