#!/usr/bin/env python3
"""
Test script to verify the deduplication logic without loading models.
This tests the key-generation logic for relation deduplication.
"""

def test_relation_key_logic():
    """Test the relation key generation logic"""
    
    # Simulate entities with same text but different positions
    ent1_first = {"text": "chest pain", "label": "ClinicalFinding", "start": 10, "end": 20}
    ent2_anatomy = {"text": "chest", "label": "BodyStructure", "start": 25, "end": 30}
    ent1_second = {"text": "chest pain", "label": "ClinicalFinding", "start": 50, "end": 60}
    
    # Test the key generation logic
    def create_relation_key(head_text, relation_type, tail_text, head_pos, tail_pos):
        return (head_text, relation_type, tail_text, head_pos, tail_pos)
    
    # First relation: chest pain (10-20) -> chest (25-30)
    key1 = create_relation_key("chest pain", "has_anatomy", "chest", (10, 20), (25, 30))
    
    # Second relation: chest pain (50-60) -> chest (25-30) 
    key2 = create_relation_key("chest pain", "has_anatomy", "chest", (50, 60), (25, 30))
    
    # Third relation: same as first (exact duplicate)
    key3 = create_relation_key("chest pain", "has_anatomy", "chest", (10, 20), (25, 30))
    
    print("Testing relation key generation:")
    print(f"Key 1 (first mention): {key1}")
    print(f"Key 2 (second mention): {key2}")
    print(f"Key 3 (duplicate): {key3}")
    
    print(f"\nKey 1 == Key 2: {key1 == key2}")  # Should be False
    print(f"Key 1 == Key 3: {key1 == key3}")    # Should be True
    
    # Test with a set to simulate deduplication
    processed_pairs = set()
    
    print(f"\nTesting deduplication with set:")
    
    # Add first relation
    if key1 not in processed_pairs:
        processed_pairs.add(key1)
        print(f"Added relation 1: {key1}")
    else:
        print(f"Skipped duplicate relation 1")
    
    # Add second relation (different positions)
    if key2 not in processed_pairs:
        processed_pairs.add(key2)
        print(f"Added relation 2: {key2}")
    else:
        print(f"Skipped duplicate relation 2")
    
    # Try to add third relation (exact duplicate)
    if key3 not in processed_pairs:
        processed_pairs.add(key3)
        print(f"Added relation 3: {key3}")
    else:
        print(f"Skipped duplicate relation 3")
    
    print(f"\nFinal set size: {len(processed_pairs)}")
    print("Expected: 2 (first and second mentions should be different, third should be filtered as duplicate)")

if __name__ == "__main__":
    test_relation_key_logic()
