#!/usr/bin/env python3
"""
Test the complete system with kidney example
"""

from relationsSpacy import extract_relations_from_paragraph

def test_kidney_example():
    """Test with the kidney example from the user"""
    
    text = "Left kidney: There are no renal or ureteral calculi. No hydronephrosis or hydroureter. No enhancing masses. No cortical cysts. No retroperitoneal mass."
    
    print("Testing complete system with kidney example:")
    print(f"Text: {text}")
    print("\n" + "="*80)
    
    entities, relations = extract_relations_from_paragraph(text)
    
    print(f"Entities found: {len(entities)}")
    for i, ent in enumerate(entities):
        negation_flag = " [NEGATED]" if ent.get('is_negated', False) else ""
        original_text = f" (original: '{ent.get('original_text', ent['text'])}')" if ent.get('is_negated', False) else ""
        print(f"{i+1}. {ent['label']}: '{ent['text']}'{negation_flag}{original_text}")
    
    print(f"\nRelations found: {len(relations)}")
    for i, rel in enumerate(relations):
        print(f"{i+1}. {rel['head']} --{rel['relation']}--> {rel['tail']}")

if __name__ == "__main__":
    test_kidney_example()
