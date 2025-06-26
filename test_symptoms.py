from gliner import GLiNER

# Quick test to see what GLiNER detects as functional impairments
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

test_text = "Patient experiences difficulty moving the arm and has trouble walking. There is limited range of motion in the shoulder. The patient reports inability to lift objects and reduced mobility in the knee joint. Weakness in the left hand affects daily activities."

entities = model.predict_entities(test_text, labels=["ClinicalFinding", "BodyStructure", "Symptom", "FunctionalImpairment"], flat_ner=False)

print("Test text:", test_text)
print("\nFound entities:")
for ent in entities:
    print(f"  - {ent['label']}: '{ent['text']}' (score: {ent['score']:.3f})")

print("\nFunctional impairments specifically:")
functional_impairments = [ent for ent in entities if ent['label'] == 'FunctionalImpairment']
if functional_impairments:
    for ent in functional_impairments:
        print(f"  - '{ent['text']}' (score: {ent['score']:.3f})")
else:
    print("  - No functional impairments detected")
