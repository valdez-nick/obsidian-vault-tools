#!/usr/bin/env python3
"""Quick diagnostic to check all imports"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Quick Import Diagnostic")
print("=" * 50)

modules_to_test = [
    ("Intelligence System", [
        "from obsidian_vault_tools.intelligence import IntentDetector",
        "from obsidian_vault_tools.intelligence import ActionExecutor", 
        "from obsidian_vault_tools.intelligence import ContextManager",
        "from obsidian_vault_tools.intelligence import IntelligenceOrchestrator"
    ]),
    ("Research Assistant", [
        "from obsidian_vault_tools.research_assistant import ResearchAssistant"
    ]),
    ("LLM System", [
        "from vault_query_system_llm import VaultQuerySystemLLM"
    ]),
    ("Audio System", [
        "from audio.audio_manager import AudioManager"
    ]),
    ("File Versioning", [
        "from file_versioning import FileVersioning"
    ]),
    ("Menu Navigation", [
        "from menu_navigator import MenuNavigator"
    ]),
    ("MCP Tools", [
        "from obsidian_vault_tools.mcp_tools import MCPClientManager",
        "from obsidian_vault_tools.mcp_tools import MCPToolDiscovery",
        "from obsidian_vault_tools.mcp_tools import MCPToolExecutor",
        "from obsidian_vault_tools.mcp_tools import DynamicMenuBuilder"
    ])
]

for module_name, imports in modules_to_test:
    print(f"\n{module_name}:")
    for import_stmt in imports:
        try:
            # SECURITY: Using safe import instead of exec() to prevent code injection
            # Parse the import statement safely
            if import_stmt.startswith("from ") and " import " in import_stmt:
                parts = import_stmt.replace("from ", "").split(" import ")
                module_path = parts[0].strip()
                import_name = parts[1].strip()
                
                # Use importlib for safe importing
                import importlib
                module = importlib.import_module(module_path)
                # Verify the attribute exists
                if hasattr(module, import_name):
                    print(f"  ✓ {import_stmt}")
                else:
                    print(f"  ✗ {import_stmt}")
                    print(f"    Error: {import_name} not found in {module_path}")
            else:
                print(f"  ✗ {import_stmt}")
                print(f"    Error: Invalid import format")
        except Exception as e:
            print(f"  ✗ {import_stmt}")
            print(f"    Error: {e}")

print("\nDiagnostic complete.")