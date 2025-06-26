#!/usr/bin/env python3
"""
Debug the specific RIGHT KIDNEY case to understand why "No gross masses" doesn't create a relation
"""

from relationsSpacy import extract_relations_from_paragraph

def debug_right_kidney():
    """Debug the RIGHT KIDNEY case"""
    
    text = "RIGHT KIDNEY: Normal size and echogenicity. 10.3 x 4.6 x 4.4 cm. Mild right pelvocaliectasis without frank hydronephrosis. No stones. No gross masses."
    
    print("Debugging RIGHT KIDNEY case:")
    print(f"Text: {text}")
    print(f"Text length: {len(text)}")
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
    
    # Let's manually check distances for all entities
    print(f"\nManual distance analysis:")
    kidney_entity = None
    for ent in entities:
        if ent['label'] == 'BodyStructure' and 'KIDNEY' in ent['text'].upper():
            kidney_entity = ent
            break
    
    if kidney_entity:
        print(f"Reference entity: '{kidney_entity['text']}' ends at position {kidney_entity['end']}")
        
        for ent in entities:
            if ent != kidney_entity:
                distance = abs(ent["start"] - kidney_entity["end"])
                print(f"  - '{ent['text']}' starts at {ent['start']}, distance = {distance}")
    
    # Check paragraph length and thresholds
    paragraph_length = len(text)
    max_distance_ratio = 0.8
    min_distance = 50
    max_distance = max(min_distance, int(paragraph_length * max_distance_ratio))
    
    print(f"\nDistance thresholds:")
    print(f"  - Paragraph length: {paragraph_length}")
    print(f"  - Max distance ratio: {max_distance_ratio}")
    print(f"  - Min distance: {min_distance}")
    print(f"  - Calculated max distance: {max_distance}")
    print(f"  - Max distance for relation (80% of paragraph): {max(max_distance, paragraph_length * 0.8)}")

if __name__ == "__main__":
    debug_right_kidney()
