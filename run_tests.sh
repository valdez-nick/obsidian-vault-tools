#!/bin/bash
#
# Obsidian Vault Manager Test Runner
# Run comprehensive end-to-end tests for all vault manager features
#

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Show usage
show_usage() {
    echo "Obsidian Vault Manager Test Runner"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  all       Run all tests (default)"
    echo "  quick     Run quick tests only (skip backups)"
    echo "  backup    Run only backup tests"
    echo "  clean     Clean up test outputs"
    echo "  help      Show this help message"
}

# Clean up test outputs
clean_test_outputs() {
    print_status "Cleaning up test outputs..."
    rm -f test_report_*.txt
    rm -rf __pycache__
    rm -f *.pyc
    print_success "Test outputs cleaned"
}

# Run tests based on option
case "${1:-all}" in
    all)
        print_status "Running all tests..."
        python3 comprehensive_e2e_test.py
        ;;
    quick)
        print_status "Running quick tests (skipping backups)..."
        python3 comprehensive_e2e_test.py --quick
        ;;
    backup)
        print_status "Running backup tests only..."
        python3 comprehensive_e2e_test.py --backup-only
        ;;
    clean)
        clean_test_outputs
        ;;
    help|--help|-h)
        show_usage
        exit 0
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Tests completed successfully!"
    
    # Show latest test report
    LATEST_REPORT=$(ls -t test_report_*.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        print_status "Test report saved to: $LATEST_REPORT"
        print_status "View report with: cat $LATEST_REPORT"
    fi
else
    print_error "Tests failed with exit code: $EXIT_CODE"
    
    # Show latest test report
    LATEST_REPORT=$(ls -t test_report_*.txt 2>/dev/null | head -1)
    if [ -n "$LATEST_REPORT" ]; then
        print_warning "Check the test report for details: $LATEST_REPORT"
    fi
fi

exit $EXIT_CODE