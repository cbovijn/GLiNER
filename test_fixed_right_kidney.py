#!/usr/bin/env python3
"""
Test the fixed RIGHT KIDNEY case
"""

from relationsSpacy import extract_relations_from_paragraph

def test_fixed_right_kidney():
    """Test the fixed RIGHT KIDNEY case"""
    
    text = "RIGHT KIDNEY: Normal size and echogenicity. 10.3 x 4.6 x 4.4 cm. Mild right pelvocaliectasis without frank hydronephrosis. No stones. No gross masses."
    
    print("Testing fixed RIGHT KIDNEY case:")
    print(f"Text: {text}")
    print(f"Text length: {len(text)} chars")
    print("\n" + "="*80)
    
    entities, relations = extract_relations_from_paragraph(text)
    
    print("Entities found:")
    for i, ent in enumerate(entities):
        negation_flag = " [NEGATED]" if ent.get('is_negated', False) else ""
        original_text = f" (original: '{ent.get('original_text', ent['text'])}')" if ent.get('is_negated', False) else ""
        print(f"{i+1}. {ent['label']}: '{ent['text']}'{negation_flag} at {ent['start']}-{ent['end']} (score: {ent['score']:.3f}){original_text}")
    
    print(f"\nRelations found: {len(relations)}")
    for i, rel in enumerate(relations):
        print(f"{i+1}. {rel['head']} -> {rel['tail']} (distance: {rel['distance']}, max allowed: {rel['max_distance_used']})")
    
    # Check if "No gross masses" now has a relation
    gross_masses_relation = any(
        'gross masses' in rel['head'].lower() or 'gross masses' in rel['tail'].lower() 
        for rel in relations
    )
    
    print(f"\nResult: 'No gross masses' relation found: {'✅ YES' if gross_masses_relation else '❌ NO'}")
    
    # Show distance calculation details
    print(f"\nDistance calculation details:")
    kidney_entity = next((ent for ent in entities if ent['label'] == 'BodyStructure'), None)
    gross_masses_entity = next((ent for ent in entities if 'gross masses' in ent['text'].lower()), None)
    
    if kidney_entity and gross_masses_entity:
        distance = abs(gross_masses_entity["start"] - kidney_entity["end"])
        paragraph_length = len(text)
        base_max_distance = max(50, int(paragraph_length * 0.85))
        special_max_distance = max(base_max_distance, int(paragraph_length * 0.9))
        
        print(f"  - RIGHT KIDNEY ends at: {kidney_entity['end']}")
        print(f"  - No gross masses starts at: {gross_masses_entity['start']}")
        print(f"  - Distance: {distance}")
        print(f"  - Base max distance (85% of {paragraph_length}): {base_max_distance}")
        print(f"  - Special max distance (90% for anatomy at start): {special_max_distance}")
        print(f"  - Used max distance: {special_max_distance} (kidney at start: {kidney_entity['start'] < 30})")

if __name__ == "__main__":
    test_fixed_right_kidney()
