#!/usr/bin/env python3
"""
Test structured negation patterns without loading GLiNER model
"""

import re

def detect_negation_simple(text, entity_start, entity_end):
    """Simplified version of detect_negation to test structured patterns"""
    
    # Special patterns for structured medical reports (entity: status)
    negated_status_patterns = [
        r':\s*none\b',
        r':\s*absent\b', 
        r':\s*negative\b',
        r':\s*normal\b',
        r':\s*unremarkable\b',
        r':\s*no\b',
        r':\s*not\s+seen\b',
        r':\s*not\s+present\b',
        r':\s*not\s+identified\b'
    ]
    
    # Positive status patterns that cancel negation
    positive_status_patterns = [
        r':\s*present\b',
        r':\s*positive\b',
        r':\s*seen\b',
        r':\s*identified\b',
        r':\s*abnormal\b',
        r':\s*enlarged\b',
        r':\s*increased\b',
        r':\s*decreased\b'
    ]
    
    # Check for structured patterns first (entity: status)
    # Look for patterns after the entity
    text_after_entity = text[entity_end:entity_end + 50].lower()
    
    # Check for negated status patterns right after the entity
    for pattern in negated_status_patterns:
        if re.search(pattern, text_after_entity):
            return True
    
    # Check for positive status patterns that override negation
    for pattern in positive_status_patterns:
        if re.search(pattern, text_after_entity):
            return False
    
    return False

def main():
    # Test cases for structured patterns
    test_cases = [
        ("Hydronephrosis: None", "Hydronephrosis", True),
        ("Hydronephrosis: Present", "Hydronephrosis", False),
        ("Size: Normal", "Size", True),
        ("Size: Enlarged", "Size", False),
        ("Tumor: Negative", "Tumor", True),
        ("Tumor: Positive", "Tumor", False),
        ("Mass: Absent", "Mass", True),
        ("Mass: Seen", "Mass", False),
        ("Inflammation: Unremarkable", "Inflammation", True),
        ("Inflammation: Abnormal", "Inflammation", False),
        ("Stone: Not seen", "Stone", True),
        ("Stone: Identified", "Stone", False),
    ]
    
    print("Testing structured negation patterns:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for text, entity_name, expected in test_cases:
        entity_start = text.find(entity_name)
        entity_end = entity_start + len(entity_name)
        
        actual = detect_negation_simple(text, entity_start, entity_end)
        
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status}: '{text}' -> {actual} (expected {expected})")
        
        if actual == expected:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All structured negation patterns work correctly!")
    else:
        print(f"❌ {failed} patterns need fixing")

if __name__ == "__main__":
    main()
