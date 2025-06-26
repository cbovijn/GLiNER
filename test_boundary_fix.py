import re

def preprocess_medical_text(text):
    """Preprocess medical text to improve entity boundary detection"""
    # Fix medical report structure by adding proper punctuation
    # Add period after field values that end a line (before next field or end)
    text = re.sub(r'(\b(?:Normal|None|Present|Adequate|Increased|Decreased|Stable|Improved)\b)(?=\s*\n\s*[A-Z][a-z]*\s*:|$)', r'\1.', text)
    
    # Add period after measurements
    text = re.sub(r'(\d+\.?\d*\s*cm)(?!\s*[.,:;])', r'\1.', text)
    text = re.sub(r'(\d+\.?\d*\s*mm)(?!\s*[.,:;])', r'\1.', text)
    
    # Add period after grades/classifications
    text = re.sub(r'(grade\s+\d+/\d+)(?!\s*[.,:;])', r'\1.', text)
    
    # Add period after long descriptive phrases that end a line
    text = re.sub(r'(\b\w+(?:\s+\w+){2,}?)(?=\s*\n\s*[A-Z][a-z]*\s*:|$)', r'\1.', text)
    
    return text

# Test with your problematic text
test_text = """RIGHT kidney:
Echogenicity: Increased with preserved corticomedullary differentiation
Position: Normal
Hydronephrosis: Present: SFU grade 2/3. The renal pelvis measures 1.8 cm. Stable ureteral dilation measuring up to 1.0 cm.
Size: Normal, 7.7 cm"""

print("Original text:")
print(test_text)
print("\nProcessed text:")
processed = preprocess_medical_text(test_text)
print(processed)

# Test with GLiNER
from gliner import GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

print("\nEntities from original text:")
entities_orig = model.predict_entities(test_text, labels=["ClinicalFinding", "BodyStructure", "Symptom"], flat_ner=False, threshold=0.1)
for ent in entities_orig:
    print(f"  - {ent['label']}: '{ent['text']}' (pos: {ent['start']}-{ent['end']})")

print("\nEntities from processed text:")
entities_proc = model.predict_entities(processed, labels=["ClinicalFinding", "BodyStructure", "Symptom"], flat_ner=False, threshold=0.1)
for ent in entities_proc:
    print(f"  - {ent['label']}: '{ent['text']}' (pos: {ent['start']}-{ent['end']})")
