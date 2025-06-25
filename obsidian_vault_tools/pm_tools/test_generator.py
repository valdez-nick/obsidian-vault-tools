#!/usr/bin/env python3
"""Test script for PM Daily Template Generator."""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from obsidian_vault_tools.pm_tools.daily_template_generator import PMDailyTemplateGenerator


def test_generator():
    """Test the daily template generator functionality."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", "/Users/nvaldez/Documents/Obsidian Vault")
    
    print(f"Testing PM Daily Template Generator")
    print(f"Vault Path: {vault_path}")
    print("-" * 50)
    
    generator = PMDailyTemplateGenerator(vault_path)
    
    # Test 1: Read WSJF priorities
    print("\n1. Reading WSJF Priorities:")
    tasks = generator.read_wsjf_priorities()
    if tasks:
        for i, task in enumerate(tasks[:3]):
            print(f"   {i+1}. [WSJF: {task['score']}] {task['task']}")
    else:
        print("   No WSJF tasks found")
    
    # Test 2: Check product focus
    print("\n2. Product Focus Schedule:")
    today = datetime.now()
    focus = generator.get_product_focus(today)
    print(f"   Today ({today.strftime('%A')}): {focus}")
    
    # Test 3: Calculate completion rates
    print("\n3. Completion Rates:")
    yesterday_rate, week_average = generator.calculate_completion_rates()
    print(f"   Yesterday: {yesterday_rate:.0f}%")
    print(f"   7-Day Average: {week_average:.0f}%")
    
    # Test 4: Generate template content (preview)
    print("\n4. Template Generation Preview:")
    content = generator.generate_daily_note()
    
    # Show just the new sections
    start_marker = "## 🎯 Today's Top 3 WSJF Priorities"
    end_marker = "## ✅ Tasks"
    
    if start_marker in content and end_marker in content:
        start_idx = content.index(start_marker)
        end_idx = content.index(end_marker)
        preview = content[start_idx:end_idx].strip()
        print(preview)
    
    print("\n" + "-" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_generator()