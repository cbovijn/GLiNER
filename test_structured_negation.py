#!/usr/bin/env python3
"""
Test structured negation patterns in medical reports.
This tests patterns like:
- "Hydronephrosis: None" (should be negated)
- "Hydronephrosis: Present" (should NOT be negated)
- "Size: Normal" (should be negated)
- "Size: Enlarged" (should NOT be negated)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from relationsSpacy import detect_negation, expand_entity_with_negation

def test_structured_negation():
    """Test structured medical report patterns"""
    
    test_cases = [
        # Basic structured patterns that should be negated
        {
            "text": "Hydronephrosis: None. The patient shows no other abnormalities.",
            "entity_start": 0,
            "entity_end": 14,  # "Hydronephrosis"
            "expected_negated": True,
            "description": "Hydronephrosis: None"
        },
        {
            "text": "Kidney stones: Absent. Bladder appears normal.",
            "entity_start": 0,
            "entity_end": 13,  # "Kidney stones"
            "expected_negated": True,
            "description": "Kidney stones: Absent"
        },
        {
            "text": "Tumor: Negative. No masses identified.",
            "entity_start": 0,
            "entity_end": 5,  # "Tumor"
            "expected_negated": True,
            "description": "Tumor: Negative"
        },
        {
            "text": "Size: Normal. Shape is regular.",
            "entity_start": 0,
            "entity_end": 4,  # "Size"
            "expected_negated": True,
            "description": "Size: Normal"
        },
        {
            "text": "Inflammation: Unremarkable. Patient feels well.",
            "entity_start": 0,
            "entity_end": 12,  # "Inflammation"
            "expected_negated": True,
            "description": "Inflammation: Unremarkable"
        },
        {
            "text": "Masses: Not seen. Liver appears healthy.",
            "entity_start": 0,
            "entity_end": 6,  # "Masses"
            "expected_negated": True,
            "description": "Masses: Not seen"
        },
        
        # Positive patterns that should NOT be negated
        {
            "text": "Hydronephrosis: Present. Requires immediate attention.",
            "entity_start": 0,
            "entity_end": 14,  # "Hydronephrosis"
            "expected_negated": False,
            "description": "Hydronephrosis: Present"
        },
        {
            "text": "Tumor: Positive. Biopsy recommended.",
            "entity_start": 0,
            "entity_end": 5,  # "Tumor"
            "expected_negated": False,
            "description": "Tumor: Positive"
        },
        {
            "text": "Mass: Seen in the upper pole. Further evaluation needed.",
            "entity_start": 0,
            "entity_end": 4,  # "Mass"
            "expected_negated": False,
            "description": "Mass: Seen"
        },
        {
            "text": "Size: Enlarged. Patient requires treatment.",
            "entity_start": 0,
            "entity_end": 4,  # "Size"
            "expected_negated": False,
            "description": "Size: Enlarged"
        },
        {
            "text": "Inflammation: Abnormal. Shows signs of infection.",
            "entity_start": 0,
            "entity_end": 12,  # "Inflammation"
            "expected_negated": False,
            "description": "Inflammation: Abnormal"
        },
        
        # Mixed cases within same text
        {
            "text": "Right kidney: Hydronephrosis: None. Left kidney: Mass: Present.",
            "entity_start": 14,
            "entity_end": 28,  # "Hydronephrosis"
            "expected_negated": True,
            "description": "Mixed text - Hydronephrosis: None"
        },
        {
            "text": "Right kidney: Hydronephrosis: None. Left kidney: Mass: Present.",
            "entity_start": 53,
            "entity_end": 57,  # "Mass"
            "expected_negated": False,
            "description": "Mixed text - Mass: Present"
        }
    ]
    
    print("Testing structured negation patterns...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        entity_start = test_case["entity_start"]
        entity_end = test_case["entity_end"]
        expected = test_case["expected_negated"]
        description = test_case["description"]
        
        # Test detect_negation function
        actual = detect_negation(text, entity_start, entity_end)
        
        print(f"Test {i}: {description}")
        print(f"  Text: '{text}'")
        print(f"  Entity: '{text[entity_start:entity_end]}'")
        print(f"  Expected negated: {expected}")
        print(f"  Actual negated: {actual}")
        
        if actual == expected:
            print("  ✓ PASSED")
            passed += 1
        else:
            print("  ✗ FAILED")
            failed += 1
        
        # Also test expand_entity_with_negation to see the expanded text
        entity = {
            "text": text[entity_start:entity_end],
            "start": entity_start,
            "end": entity_end
        }
        expanded = expand_entity_with_negation(text, entity)
        print(f"  Expanded text: '{expanded['text']}'")
        print(f"  Is negated: {expanded.get('is_negated', False)}")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed")
        return False

if __name__ == "__main__":
    test_structured_negation()
