
from gliner import GLiNER
import spacy
import os
import re

# Configuration
CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence score for entity recognition (0.0 to 1.0)
MAX_DISTANCE_RATIO = 0.8    # Maximum distance between entities as ratio of paragraph length (increased for medical reports)
MIN_DISTANCE = 50           # Minimum distance threshold in characters (increased for medical reports)

# Laad GLiNER en spaCy
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
nlp = spacy.load("en_core_web_sm")

def read_files_from_directory(directory_path):
    """Read all text files from a directory"""
    files_content = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    files_content.append((filename, content))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return files_content

def preprocess_medical_text(text):
    """Preprocess medical text to improve entity boundary detection"""
    import re
    
    # Fix medical report structure by adding proper punctuation
    # Add period after field values that end a line (before next field or end)
    text = re.sub(r'(\b(?:Normal|None|Present|Adequate|Increased|Decreased|Stable|Improved)\b)(?=\s*\n\s*[A-Z][a-z]*\s*:|$)', r'\1.', text)
    
    # Add period after measurements
    text = re.sub(r'(\d+\.?\d*\s*cm)(?!\s*[.,:;])', r'\1.', text)
    text = re.sub(r'(\d+\.?\d*\s*mm)(?!\s*[.,:;])', r'\1.', text)
    text = re.sub(r'(\d+\.?\d*\s*x\s*\d+\.?\d*\s*x\s*\d+\.?\d*\s*cm)(?!\s*[.,:;])', r'\1.', text)
    
    # Add period after grades/classifications
    text = re.sub(r'(grade\s+\d+/\d+)(?!\s*[.,:;])', r'\1.', text)
    
    # Add period after long descriptive phrases that end a line
    text = re.sub(r'(\b\w+\s+\w+\s+\w+(?:\s+\w+)*?)(?=\s*\n\s*[A-Z][a-z]*\s*:|$)', r'\1.', text)
    
    return text
    
    # Add period after grades/classifications
    text = re.sub(r'(grade\s+\d+/\d+)(?!\s*[.,:;])', r'\1.', text)
    
    return text

def split_into_paragraphs(text):
    """Split text into paragraphs based on double newlines or single newlines"""
    # Preprocess the text first
    text = preprocess_medical_text(text)
    
    # Split by double newlines first, then by single newlines if no double newlines found
    paragraphs = re.split(r'\n\s*\n', text.strip())
    if len(paragraphs) == 1:
        paragraphs = text.split('\n')
    
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def improve_text_segmentation(text):
    """Use spaCy to improve text segmentation for better entity detection"""
    # Process with spaCy to get better sentence boundaries
    doc = nlp(text)
    
    # Reconstruct text with proper sentence boundaries
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            # Ensure sentence ends with punctuation
            if not sent_text.endswith(('.', '!', '?', ':')):
                sent_text += '.'
            sentences.append(sent_text)
    
    return ' '.join(sentences)

def extract_relations_from_paragraph(paragraph, max_distance_ratio=MAX_DISTANCE_RATIO, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Extract entities and relations from a single paragraph"""
    paragraph_length = len(paragraph)
    max_distance = max(MIN_DISTANCE, int(paragraph_length * max_distance_ratio))  # At least MIN_DISTANCE chars or ratio of paragraph
    
    # Improve text segmentation before entity extraction
    processed_paragraph = improve_text_segmentation(paragraph)
    
    # Extract entities with confidence threshold
    entities = model.predict_entities(
        processed_paragraph, 
        labels=["ClinicalFinding", "BodyStructure", "Symptom", "FunctionalImpairment"], 
        flat_ner=False,
        threshold=confidence_threshold
    )
    
    # Find relations
    relations = []
    processed_pairs = set()  # Track processed entity pairs to avoid duplicates
    
    for ent1 in entities:
        for ent2 in entities:
            # Skip if same entity
            if ent1 == ent2:
                continue
                
            # Calculate distance between entities
            distance = abs(ent2["start"] - ent1["end"])
            
            # Special case: If BodyStructure is at the beginning of paragraph (position < 20)
            # and is followed by clinical findings/symptoms, use more lenient distance
            if ((ent1["label"] == "BodyStructure" and ent1["start"] < 20) or 
                (ent2["label"] == "BodyStructure" and ent2["start"] < 20)):
                # For medical reports where anatomy is mentioned first, allow larger distances
                max_distance_for_relation = max(max_distance, paragraph_length * 0.8)
            else:
                max_distance_for_relation = max_distance
            
            # ClinicalFinding -> BodyStructure relations (bidirectional)
            if ((ent1["label"] == "ClinicalFinding" and ent2["label"] == "BodyStructure") or
                (ent1["label"] == "BodyStructure" and ent2["label"] == "ClinicalFinding")):
                if distance < max_distance_for_relation:
                    # Determine head and tail based on which is the clinical finding
                    if ent1["label"] == "ClinicalFinding":
                        head, tail = ent1["text"], ent2["text"]
                        head_pos, tail_pos = (ent1["start"], ent1["end"]), (ent2["start"], ent2["end"])
                    else:
                        head, tail = ent2["text"], ent1["text"]
                        head_pos, tail_pos = (ent2["start"], ent2["end"]), (ent1["start"], ent1["end"])
                    
                    # Create unique identifier including positions to avoid duplicates while allowing repeated mentions
                    relation_key = (head, "has_anatomy", tail, head_pos, tail_pos)
                    if relation_key not in processed_pairs:
                        relations.append({
                            "head": head,
                            "relation": "has_anatomy", 
                            "tail": tail,
                            "distance": distance,
                            "paragraph_length": paragraph_length,
                            "max_distance_used": max_distance_for_relation
                        })
                        processed_pairs.add(relation_key)
            
            # Symptom -> BodyStructure relations (bidirectional)
            elif ((ent1["label"] == "Symptom" and ent2["label"] == "BodyStructure") or
                  (ent1["label"] == "BodyStructure" and ent2["label"] == "Symptom")):
                if distance < max_distance_for_relation:
                    # Determine head and tail based on which is the symptom
                    if ent1["label"] == "Symptom":
                        head, tail = ent1["text"], ent2["text"]
                        head_pos, tail_pos = (ent1["start"], ent1["end"]), (ent2["start"], ent2["end"])
                    else:
                        head, tail = ent2["text"], ent1["text"]
                        head_pos, tail_pos = (ent2["start"], ent2["end"]), (ent1["start"], ent1["end"])
                    
                    # Create unique identifier including positions to avoid duplicates while allowing repeated mentions
                    relation_key = (head, "affects_anatomy", tail, head_pos, tail_pos)
                    if relation_key not in processed_pairs:
                        relations.append({
                            "head": head,
                            "relation": "affects_anatomy", 
                            "tail": tail,
                            "distance": distance,
                            "paragraph_length": paragraph_length,
                            "max_distance_used": max_distance_for_relation
                        })
                        processed_pairs.add(relation_key)
            
            # FunctionalImpairment -> BodyStructure relations (bidirectional)
            elif ((ent1["label"] == "FunctionalImpairment" and ent2["label"] == "BodyStructure") or
                  (ent1["label"] == "BodyStructure" and ent2["label"] == "FunctionalImpairment")):
                if distance < max_distance_for_relation:
                    # Determine head and tail based on which is the functional impairment
                    if ent1["label"] == "FunctionalImpairment":
                        head, tail = ent1["text"], ent2["text"]
                        head_pos, tail_pos = (ent1["start"], ent1["end"]), (ent2["start"], ent2["end"])
                    else:
                        head, tail = ent2["text"], ent1["text"]
                        head_pos, tail_pos = (ent2["start"], ent2["end"]), (ent1["start"], ent1["end"])
                    
                    # Create unique identifier including positions to avoid duplicates while allowing repeated mentions
                    relation_key = (head, "impairs_function_of", tail, head_pos, tail_pos)
                    if relation_key not in processed_pairs:
                        relations.append({
                            "head": head,
                            "relation": "impairs_function_of", 
                            "tail": tail,
                            "distance": distance,
                            "paragraph_length": paragraph_length,
                            "max_distance_used": max_distance_for_relation
                        })
                        processed_pairs.add(relation_key)
    
    return entities, relations

def process_directory(directory_path):
    """Process all files in directory"""
    files_content = read_files_from_directory(directory_path)
    all_results = []
    
    for filename, content in files_content:
        print(f"\n=== Processing {filename} ===")
        paragraphs = split_into_paragraphs(content)
        
        file_results = {
            "filename": filename,
            "total_paragraphs": len(paragraphs),
            "paragraphs": []
        }
        
        for i, paragraph in enumerate(paragraphs):
            print(f"\nParagraph {i+1} (length: {len(paragraph)} chars):")
            print(f"Original: {paragraph[:100]}...")
            
            # Show processed text for debugging
            processed_text = improve_text_segmentation(paragraph)
            if processed_text != paragraph:
                print(f"Processed: {processed_text[:100]}...")
            
            entities, relations = extract_relations_from_paragraph(paragraph)
            
            paragraph_result = {
                "paragraph_number": i+1,
                "text": paragraph,
                "length": len(paragraph),
                "entities": entities,
                "relations": relations
            }
            
            file_results["paragraphs"].append(paragraph_result)
            
            print(f"Entities found: {len(entities)}")
            if entities:
                for ent in entities:
                    print(f"  - {ent['label']}: '{ent['text']}' (score: {ent['score']:.3f}, pos: {ent['start']}-{ent['end']})")
            
            print(f"Relations found: {len(relations)}")
            if relations:
                for rel in relations:
                    print(f"  - {rel['head']} -> {rel['tail']} (distance: {rel['distance']}, max allowed: {rel['max_distance_used']})")
        
        all_results.append(file_results)
    
    return all_results

# Example usage
if __name__ == "__main__":
    # Change this to your directory path
    directory_path = input("Enter the directory path containing text files: ").strip()
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        # Use current directory as fallback for demo
        directory_path = "."
        print(f"Using current directory: {directory_path}")
    
    results = process_directory(directory_path)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    total_files = len(results)
    total_paragraphs = sum(r["total_paragraphs"] for r in results)
    total_relations = sum(len(rel) for r in results for p in r["paragraphs"] for rel in p["relations"])
    
    print(f"Files processed: {total_files}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total relations found: {total_relations}")
    
    # Show all relations
    if total_relations > 0:
        print("\nAll relations found:")
        for result in results:
            for paragraph in result["paragraphs"]:
                for relation in paragraph["relations"]:
                    print(f"  {result['filename']} - P{paragraph['paragraph_number']}: {relation['head']} -> {relation['tail']}")
