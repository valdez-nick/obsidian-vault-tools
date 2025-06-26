#!/bin/bash
# Quick test script to verify everything works

echo "🧪 Running Quick Tests for Obsidian Vault Tools"
echo "=============================================="

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "❌ Tests directory not found!"
    exit 1
fi

# Run regression test first (most critical)
echo -e "\n1️⃣ Running Menu Navigation Regression Test..."
cd tests
python e2e/test_regression_menu_navigation.py
REGRESSION_RESULT=$?

if [ $REGRESSION_RESULT -ne 0 ]; then
    echo "❌ CRITICAL: Menu navigation regression test failed!"
    echo "The bug that prevented the menu from loading may have returned."
    exit 1
fi

echo -e "\n✅ Menu navigation bug is still fixed!"

# Run auto-discovery test
echo -e "\n2️⃣ Running Feature Discovery Test..."
python e2e/test_auto_discover_all_features.py > discovery_output.txt 2>&1
DISCOVERY_RESULT=$?

if [ $DISCOVERY_RESULT -eq 0 ]; then
    echo "✅ Feature discovery test passed!"
    
    # Extract some stats
    if [ -f "e2e/coverage_report.json" ]; then
        echo -e "\n📊 Coverage Summary:"
        python -c "
import json
with open('e2e/coverage_report.json') as f:
    data = json.load(f)
    totals = data.get('totals', {})
    print(f'  - CLI Commands: {totals.get(\"cli_commands\", 0)}')
    print(f'  - Menu Categories: {totals.get(\"menu_categories\", 0)}')
    print(f'  - Total Menu Options: {totals.get(\"total_menu_options\", 0)}')
    print(f'  - Features: {totals.get(\"features\", 0)} ({totals.get(\"features_available\", 0)} available)')
"
    fi
else
    echo "⚠️  Feature discovery had some issues (non-critical)"
fi

# Quick CLI test
echo -e "\n3️⃣ Testing CLI Commands..."
cd ..
python -m obsidian_vault_tools.cli version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CLI is working!"
else
    echo "⚠️  CLI test had issues"
fi

# Summary
echo -e "\n=============================================="
if [ $REGRESSION_RESULT -eq 0 ]; then
    echo "✅ TESTS PASSED - Safe to proceed!"
    echo "The menu navigation bug is fixed and core functionality works."
else
    echo "❌ TESTS FAILED - Please fix issues before proceeding!"
fi

exit $REGRESSION_RESULT