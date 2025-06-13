#!/bin/bash
# Script to run comprehensive test coverage for Obsidian Librarian

echo "==================================="
echo "Obsidian Librarian Test Coverage"
echo "==================================="
echo ""

# Change to python directory
cd "$(dirname "$0")" || exit 1

# Create coverage directory if it doesn't exist
mkdir -p coverage

# Install coverage tools if not already installed
echo "Checking dependencies..."
pip install -q pytest pytest-cov pytest-asyncio pytest-xdist coverage

# Clear previous coverage data
echo "Clearing previous coverage data..."
coverage erase

# Run unit tests with coverage
echo ""
echo "Running unit tests..."
echo "-------------------"
pytest tests/unit/ \
    --cov=obsidian_librarian \
    --cov-report=term-missing \
    --cov-report=html:coverage/unit \
    --cov-report=json:coverage/unit.json \
    -v \
    --tb=short

# Save unit test coverage
cp .coverage .coverage.unit

# Run integration tests with coverage
echo ""
echo "Running integration tests..."
echo "--------------------------"
pytest tests/integration/ \
    --cov=obsidian_librarian \
    --cov-report=term-missing \
    --cov-report=html:coverage/integration \
    --cov-report=json:coverage/integration.json \
    -v \
    --tb=short

# Save integration test coverage
cp .coverage .coverage.integration

# Run e2e tests with coverage (if example vault exists)
if [ -d "example-vault" ]; then
    echo ""
    echo "Running end-to-end tests..."
    echo "-------------------------"
    pytest tests/e2e/ \
        --cov=obsidian_librarian \
        --cov-report=term-missing \
        --cov-report=html:coverage/e2e \
        --cov-report=json:coverage/e2e.json \
        -v \
        --tb=short \
        -m "not slow"
    
    # Save e2e test coverage
    cp .coverage .coverage.e2e
fi

# Combine coverage data
echo ""
echo "Combining coverage data..."
echo "------------------------"
coverage combine .coverage.unit .coverage.integration .coverage.e2e 2>/dev/null || \
coverage combine .coverage.unit .coverage.integration 2>/dev/null || \
coverage combine

# Generate final reports
echo ""
echo "Generating final coverage report..."
echo "---------------------------------"
coverage report --precision=2
coverage html -d coverage/combined
coverage json -o coverage/combined.json

# Extract coverage percentage
COVERAGE_PERCENT=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')

echo ""
echo "==================================="
echo "Test Coverage Summary"
echo "==================================="
echo ""

# Show coverage by module
echo "Coverage by module:"
echo "------------------"
coverage report --skip-covered --precision=2 | grep -E "obsidian_librarian/(services|cli|ai|database)" || true

echo ""
echo "Overall Coverage: ${COVERAGE_PERCENT}%"
echo ""

# Check if we meet the 80% threshold
if (( $(echo "$COVERAGE_PERCENT >= 80" | bc -l) )); then
    echo "✅ Coverage goal of 80% achieved!"
else
    echo "❌ Coverage is below 80% target"
    echo ""
    echo "Areas needing more tests:"
    echo "-----------------------"
    coverage report --skip-covered --sort=cover | tail -10
fi

echo ""
echo "Coverage reports generated in:"
echo "- HTML: coverage/combined/index.html"
echo "- JSON: coverage/combined.json"
echo ""

# Run specific feature coverage analysis
echo "Feature-specific coverage:"
echo "------------------------"

# Tag management coverage
echo -n "Tag Management: "
coverage report --include="*/services/tag_manager.py" 2>/dev/null | grep TOTAL | awk '{print $4}' || echo "N/A"

# Auto-organization coverage  
echo -n "Auto Organization: "
coverage report --include="*/services/auto_organizer.py" 2>/dev/null | grep TOTAL | awk '{print $4}' || echo "N/A"

# CLI coverage
echo -n "CLI Commands: "
coverage report --include="*/cli/commands/*.py" 2>/dev/null | grep TOTAL | awk '{print $4}' || echo "N/A"

echo ""

# Generate coverage badge (if coverage-badge is installed)
if command -v coverage-badge &> /dev/null; then
    echo "Generating coverage badge..."
    coverage-badge -o coverage/badge.svg -f
fi

# Cleanup temporary files
rm -f .coverage.unit .coverage.integration .coverage.e2e 2>/dev/null

echo "Coverage analysis complete!"