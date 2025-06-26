from gliner import GLiNER

# Test different confidence thresholds
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

test_text = "Patient has difficulty moving the arm and experiences pain in the shoulder."

print("Test text:", test_text)
print("\n" + "="*60)

# Test with different thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]

for threshold in thresholds:
    print(f"\nThreshold: {threshold}")
    print("-" * 30)
    
    entities = model.predict_entities(
        test_text, 
        labels=["ClinicalFinding", "BodyStructure", "Symptom", "FunctionalImpairment"], 
        flat_ner=False,
        threshold=threshold
    )
    
    print(f"Entities found: {len(entities)}")
    for ent in entities:
        print(f"  - {ent['label']}: '{ent['text']}' (score: {ent['score']:.3f})")
    
    if not entities:
        print("  No entities found at this threshold")
