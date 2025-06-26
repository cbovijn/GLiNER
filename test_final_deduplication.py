#!/usr/bin/env python3
"""
Test script to verify deduplication works correctly with repeated mentions
by creating a specific test case.
"""

from relationsSpacy import extract_relations_from_paragraph

def test_specific_deduplication():
    """Test with text that has repeated mentions of the same entities"""
    
    # Create test text with repeated mentions
    test_text = """
    The patient has chest pain in the chest area. Later, the chest pain returned 
    and the chest became more tender. Additional chest pain was noted in the chest region.
    """
    
    print("Testing deduplication with repeated mentions:")
    print(f"Text: {test_text.strip()}")
    print("\n" + "="*80)
    
    entities, relations = extract_relations_from_paragraph(test_text)
    
    print(f"\nFound {len(entities)} entities:")
    chest_entities = []
    pain_entities = []
    
    for i, entity in enumerate(entities):
        print(f"{i+1}. '{entity['text']}' ({entity['label']}) at positions {entity['start']}-{entity['end']}")
        if "chest" in entity['text'].lower():
            chest_entities.append(entity)
        if "pain" in entity['text'].lower():
            pain_entities.append(entity)
    
    print(f"\n{len(chest_entities)} chest-related entities:")
    for entity in chest_entities:
        print(f"  - '{entity['text']}' at {entity['start']}-{entity['end']}")
        
    print(f"\n{len(pain_entities)} pain-related entities:")
    for entity in pain_entities:
        print(f"  - '{entity['text']}' at {entity['start']}-{entity['end']}")
    
    print(f"\nFound {len(relations)} relations:")
    for i, relation in enumerate(relations):
        print(f"{i+1}. {relation['head']} --{relation['relation']}--> {relation['tail']} (distance: {relation['distance']})")
    
    # Count unique relation patterns
    relation_patterns = {}
    for relation in relations:
        pattern = f"{relation['head']} -> {relation['tail']}"
        if pattern in relation_patterns:
            relation_patterns[pattern] += 1
        else:
            relation_patterns[pattern] = 1
    
    print(f"\nRelation pattern summary:")
    for pattern, count in relation_patterns.items():
        print(f"  - {pattern}: {count} times")
        
    print(f"\nExpected: Multiple relations for repeated mentions at different positions")
    print(f"Actual: {len(relations)} total relations found")

if __name__ == "__main__":
    test_specific_deduplication()
