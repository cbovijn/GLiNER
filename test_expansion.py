#!/usr/bin/env python3
"""
Test entity expansion with structured patterns
"""

import re

def expand_entity_with_negation_simple(text, entity_start, entity_end, entity_text):
    """Simplified version to test entity expansion"""
    
    # Check for structured patterns after the entity
    text_after_entity = text[entity_end:entity_end + 50]
    
    # Pattern for structured medical reports like "Hydronephrosis: None"
    structured_pattern = r':\s*(none|absent|negative|normal|unremarkable|no|not\s+seen|not\s+present|not\s+identified)\b'
    match = re.search(structured_pattern, text_after_entity, re.IGNORECASE)
    
    if match:
        # For structured patterns, include the entity and its status
        expanded_end = entity_end + match.end()
        new_text = text[entity_start:expanded_end].strip()
        return {
            "text": new_text,
            "start": entity_start,
            "end": expanded_end,
            "original_text": entity_text,
            "is_negated": True
        }
    
    # Check for positive patterns
    positive_pattern = r':\s*(present|positive|seen|identified|abnormal|enlarged|increased|decreased)\b'
    match = re.search(positive_pattern, text_after_entity, re.IGNORECASE)
    
    if match:
        # For positive patterns, include the entity and its status but mark as not negated
        expanded_end = entity_end + match.end()
        new_text = text[entity_start:expanded_end].strip()
        return {
            "text": new_text,
            "start": entity_start,
            "end": expanded_end,
            "original_text": entity_text,
            "is_negated": False
        }
    
    # No structured pattern found
    return {
        "text": entity_text,
        "start": entity_start,
        "end": entity_end,
        "original_text": entity_text,
        "is_negated": False
    }

def main():
    # Test cases for entity expansion
    test_cases = [
        ("Hydronephrosis: None detected", "Hydronephrosis", 0, 14),
        ("Hydronephrosis: Present in left kidney", "Hydronephrosis", 0, 14),
        ("Size: Normal for patient age", "Size", 0, 4),
        ("Size: Enlarged significantly", "Size", 0, 4),
    ]
    
    print("Testing entity expansion with structured patterns:")
    print("=" * 70)
    
    for text, entity_name, start, end in test_cases:
        result = expand_entity_with_negation_simple(text, start, end, entity_name)
        
        print(f"Text: '{text}'")
        print(f"Original entity: '{entity_name}'")
        print(f"Expanded entity: '{result['text']}'")
        print(f"Is negated: {result['is_negated']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
