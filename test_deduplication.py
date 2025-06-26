#!/usr/bin/env python3
"""
Test script to verify that the updated deduplication logic correctly handles repeated mentions
of the same entity text at different positions.
"""

from relationsSpacy import extract_relations_from_paragraph

def test_repeated_mentions():
    """Test case with repeated mentions of same entity at different positions"""
    
    # Create a test paragraph with repeated mentions
    test_text = """
    Patient presents with chest pain. The chest shows evidence of inflammation. 
    Later examination reveals additional chest findings with ongoing chest pain.
    The chest X-ray confirms abnormalities in the chest cavity.
    """
    
    print("Testing paragraph with repeated mentions:")
    print(f"Text: {test_text}")
    print("\n" + "="*60)
    
    # Extract entities and relations
    entities, relations = extract_relations_from_paragraph(test_text)
    
    print(f"\nFound {len(entities)} entities:")
    for i, entity in enumerate(entities):
        print(f"{i+1}. {entity['text']} ({entity['label']}) at positions {entity['start']}-{entity['end']}")
    
    print(f"\nFound {len(relations)} relations:")
    for i, relation in enumerate(relations):
        print(f"{i+1}. {relation['head']} --{relation['relation']}--> {relation['tail']} (distance: {relation['distance']})")
    
    # Count how many times "chest" appears in different positions
    chest_entities = [e for e in entities if "chest" in e['text'].lower()]
    print(f"\nEntities containing 'chest': {len(chest_entities)}")
    for entity in chest_entities:
        print(f"  - '{entity['text']}' at {entity['start']}-{entity['end']}")
    
    # Count relations involving "chest"
    chest_relations = [r for r in relations if "chest" in r['head'].lower() or "chest" in r['tail'].lower()]
    print(f"\nRelations involving 'chest': {len(chest_relations)}")
    for relation in chest_relations:
        print(f"  - {relation['head']} --{relation['relation']}--> {relation['tail']}")

def test_exact_duplicates():
    """Test that exact duplicate relations (same text, same positions) are still filtered out"""
    
    # This should not happen in practice, but let's test the edge case
    test_text = "Patient has chest pain affecting the chest."
    
    print("\n\nTesting paragraph with potential exact duplicates:")
    print(f"Text: {test_text}")
    print("\n" + "="*60)
    
    entities, relations = extract_relations_from_paragraph(test_text)
    
    print(f"\nFound {len(entities)} entities:")
    for i, entity in enumerate(entities):
        print(f"{i+1}. {entity['text']} ({entity['label']}) at positions {entity['start']}-{entity['end']}")
    
    print(f"\nFound {len(relations)} relations:")
    for i, relation in enumerate(relations):
        print(f"{i+1}. {relation['head']} --{relation['relation']}--> {relation['tail']} (distance: {relation['distance']})")

if __name__ == "__main__":
    test_repeated_mentions()
    test_exact_duplicates()
