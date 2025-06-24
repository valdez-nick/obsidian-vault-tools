#!/usr/bin/env python3
"""
Test tag safety - ensure tag processing doesn't corrupt hierarchical tags
"""

import re
import sys

def test_tag_patterns():
    """Test that our tag patterns don't corrupt hierarchical tags"""
    
    # Test cases with hierarchical tags that should NOT be changed
    test_cases = [
        ("#initiative/dfp-revamp is important", "#initiative/dfp-revamp is important"),
        ("Working on #todo/urgent task", "Working on #todo/urgent task"),
        ("#identity/device-fingerprinting feature", "#identity/device-fingerprinting feature"),
        ("Multiple tags: #todo/api-platform #initiative/payments", "Multiple tags: #todo/api-platform #initiative/payments"),
    ]
    
    # Test cases with tags that SHOULD be changed
    fix_cases = [
        ("#todo/ is incomplete", "#todo is incomplete"),  # Trailing slash
        ("#initiative/ needs work", "#initiative needs work"),  # Trailing slash
        ("#productresearch findings", "#product-research findings"),  # Missing dash
        ("#todo-urgent item", "#todo/urgent item"),  # Should be hierarchical
    ]
    
    # Patterns from our fixed tag fixer
    tag_patterns = [
        # Fix trailing slashes only (not hierarchical tags)
        (r'#todo/(?=\s|$)', '#todo'),
        (r'#initiative/(?=\s|$)', '#initiative'),
        (r'#identity/(?=\s|$)', '#identity'),
        
        # Standardize various tags
        (r'#productresearch\b', '#product-research'),
        (r'#todo-urgent\b', '#todo/urgent'),
    ]
    
    print("Testing tag patterns for safety...")
    print("=" * 60)
    
    # Test that hierarchical tags are preserved
    print("\nTesting preservation of hierarchical tags:")
    all_good = True
    for test_input, expected in test_cases:
        result = test_input
        for pattern, replacement in tag_patterns:
            result = re.sub(pattern, replacement, result)
        
        if result == expected:
            print(f"✅ PASS: '{test_input}'")
        else:
            print(f"❌ FAIL: '{test_input}' → '{result}' (expected: '{expected}')")
            all_good = False
    
    # Test that problematic tags are fixed
    print("\nTesting fixes for problematic tags:")
    for test_input, expected in fix_cases:
        result = test_input
        for pattern, replacement in tag_patterns:
            result = re.sub(pattern, replacement, result)
        
        if result == expected:
            print(f"✅ PASS: '{test_input}' → '{result}'")
        else:
            print(f"❌ FAIL: '{test_input}' → '{result}' (expected: '{expected}')")
            all_good = False
    
    # Test edge cases
    print("\nTesting edge cases:")
    edge_cases = [
        ("#todo/", "#todo"),  # Just trailing slash
        ("#todo/ #initiative/dfp-revamp", "#todo #initiative/dfp-revamp"),  # Mixed
        ("tag includes #initiative/dfp-revamp", "tag includes #initiative/dfp-revamp"),  # In query
    ]
    
    for test_input, expected in edge_cases:
        result = test_input
        for pattern, replacement in tag_patterns:
            result = re.sub(pattern, replacement, result)
        
        if result == expected:
            print(f"✅ PASS: '{test_input}' → '{result}'")
        else:
            print(f"❌ FAIL: '{test_input}' → '{result}' (expected: '{expected}')")
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All tests passed! Tag patterns are safe.")
    else:
        print("❌ Some tests failed. Tag patterns need adjustment.")
        sys.exit(1)

if __name__ == '__main__':
    test_tag_patterns()