#!/usr/bin/env python3
"""
Test negation detection with the kidney example
"""

from relationsSpacy import extract_relations_from_paragraph

def test_kidney_negation():
    """Test with the kidney example that was problematic"""
    
    test_text = """Left kidney: There are no renal or ureteral calculi. No hydronephrosis or hydroureter. No enhancing masses. No cortical cysts. No retroperitoneal mass."""
    
    print("Testing negation detection with kidney example:")
    print(f"Text: {test_text}")
    print("\n" + "="*80)
    
    entities, relations = extract_relations_from_paragraph(test_text)
    
    print(f"\nFound {len(entities)} entities:")
    for i, entity in enumerate(entities):
        negation_flag = " [NEGATED]" if entity.get('is_negated', False) else ""
        original_text = f" (original: '{entity.get('original_text', entity['text'])}')" if entity.get('is_negated', False) else ""
        print(f"{i+1}. '{entity['text']}' ({entity['label']}){negation_flag} at positions {entity['start']}-{entity['end']}{original_text}")
    
    print(f"\nFound {len(relations)} relations:")
    for i, relation in enumerate(relations):
        print(f"{i+1}. {relation['head']} --{relation['relation']}--> {relation['tail']} (distance: {relation['distance']})")
    
    # Check specific expectations
    print(f"\nExpected behavior:")
    print(f"- Each 'No ...' should be a separate negated entity")
    print(f"- Should NOT combine across sentences")
    print(f"- 'no renal or ureteral calculi' should be one entity")
    print(f"- 'No hydronephrosis' should be separate from 'No enhancing masses'")

if __name__ == "__main__":
    test_kidney_negation()
    
    # Test case from user example
    test_text = """Left kidney: There are no renal or ureteral calculi. No hydronephrosis or hydroureter. No enhancing masses. No cortical cysts. No retroperitoneal mass."""
    
    print("Testing negation detection:")
    print(f"Text: {test_text}")
    print("\n" + "="*80)
    
    entities, relations = extract_relations_from_paragraph(test_text)
    
    print(f"\nFound {len(entities)} entities:")
    for i, entity in enumerate(entities):
        negation_flag = " [NEGATED]" if entity.get('is_negated', False) else ""
        original_text = f" (original: '{entity.get('original_text', entity['text'])}')" if entity.get('is_negated', False) else ""
        print(f"{i+1}. {entity['label']}: '{entity['text']}'{negation_flag} (score: {entity['score']:.3f}, pos: {entity['start']}-{entity['end']}){original_text}")
    
    print(f"\nFound {len(relations)} relations:")
    for i, relation in enumerate(relations):
        print(f"{i+1}. {relation['head']} --{relation['relation']}--> {relation['tail']}")
    
    # Count negated vs non-negated entities
    negated_count = sum(1 for e in entities if e.get('is_negated', False))
    print(f"\nSummary:")
    print(f"  - Total entities: {len(entities)}")
    print(f"  - Negated entities: {negated_count}")
    print(f"  - Non-negated entities: {len(entities) - negated_count}")

def test_additional_negation_cases():
    """Test additional negation patterns"""
    
    test_cases = [
        "Patient denies chest pain and shortness of breath.",
        "No evidence of pneumonia or pleural effusion.",
        "Heart examination was unremarkable with no murmurs.",
        "The patient is free of any neurological symptoms.",
        "Negative for appendicitis but positive for gastritis.",
        "Normal kidney function without any abnormalities."
    ]
    
    print("\n\nTesting additional negation patterns:")
    print("="*80)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_text}")
        entities, _ = extract_relations_from_paragraph(test_text)
        
        if entities:
            for entity in entities:
                negation_flag = " [NEGATED]" if entity.get('is_negated', False) else ""
                print(f"  - {entity['label']}: '{entity['text']}'{negation_flag}")
        else:
            print("  - No entities found")

if __name__ == "__main__":
    test_negation_detection()
    test_additional_negation_cases()
