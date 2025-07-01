from gliner import GLiNER
import spacy
import os
import re

# Configuration
CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence score for entity recognition (0.0 to 1.0)
MAX_DISTANCE_RATIO = 0.85   # Maximum distance between entities as ratio of paragraph length (increased for medical reports)
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
    
    # Process entities to handle negation
    processed_entities = []
    for entity in entities:
        expanded_entity = expand_entity_with_negation(processed_paragraph, entity)
        processed_entities.append(expanded_entity)
    
    # Filter overlapping entities (prefer negated/longer entities)
    entities = filter_overlapping_entities(processed_entities)
    
    # Filter out spurious field value entities (like standalone "None", "Present", etc.)
    entities = filter_field_value_entities(entities, processed_paragraph)
    
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
            
            # Special case: If BodyStructure is at the beginning of paragraph (position < 30)
            # and is followed by clinical findings/symptoms, use more lenient distance
            if ((ent1["label"] == "BodyStructure" and ent1["start"] < 30) or 
                (ent2["label"] == "BodyStructure" and ent2["start"] < 30)):
                # For medical reports where anatomy is mentioned first, allow larger distances
                max_distance_for_relation = max(max_distance, int(paragraph_length * 0.9))
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

def detect_negation(text, entity_start, entity_end):
    """Detect if an entity is negated by looking for negation words before it within the same sentence"""
    # Common negation patterns in medical text
    negation_patterns = [
        r'\bno\b',
        r'\bnot\b', 
        r'\babsent\b',
        r'\bwithout\b',
        r'\bdenies\b',
        r'\bnegative\s+for\b',
        r'\brule\s+out\b',
        r'\bunremarkable\b',
        r'\bnormal\b',
        r'\bno\s+evidence\s+of\b',
        r'\bno\s+signs\s+of\b',
        r'\bfree\s+of\b',
        r'\bnone\b'  # Added for structured field patterns like "Hydronephrosis: None"
    ]
    
    # Special handling for structured field patterns (Field: Value)
    # Look for patterns like "Hydronephrosis: None" or "Pain: Present"
    entity_text = text[entity_start:entity_end]
    
    # Check if this entity is followed by ": None" pattern
    after_entity = text[entity_end:entity_end + 20].strip().lower()
    if after_entity.startswith(': none') or after_entity.startswith(':none'):
        return True
    
    # Check if this entity is preceded by "None:" pattern (value before field)
    before_entity = text[max(0, entity_start - 20):entity_start].strip().lower()
    if before_entity.endswith('none :') or before_entity.endswith('none:'):
        return True
    
    # Check for affirmative patterns that should NOT be negated
    affirmative_patterns_after = [': present', ':present', ': seen', ':seen', ': positive', ':positive']
    for pattern in affirmative_patterns_after:
        if after_entity.startswith(pattern):
            return False
    
    affirmative_patterns_before = ['present :', 'present:', 'seen :', 'seen:', 'positive :', 'positive:']
    for pattern in affirmative_patterns_before:
        if before_entity.endswith(pattern):
            return False
    
    # Find the start of the current sentence (look backwards for sentence boundaries)
    # In medical reports, don't treat colons as sentence boundaries
    sentence_start = entity_start
    for i in range(entity_start - 1, max(0, entity_start - 200), -1):
        if text[i] in '.!?':
            # Found sentence boundary, start after it
            sentence_start = i + 1
            break
        elif i == 0:
            # Reached beginning of text
            sentence_start = 0
            break
    
    # Get text from sentence start to entity start
    search_text = text[sentence_start:entity_start].strip().lower()
    
    # If search text is empty or very short, expand the search window
    if len(search_text) < 10:
        # Look further back but still respect sentence boundaries
        extended_start = max(0, entity_start - 100)
        extended_text = text[extended_start:entity_start].lower()
        
        # Find the last sentence boundary in the extended text
        last_boundary = -1
        for i in range(len(extended_text) - 1, -1, -1):
            if extended_text[i] in '.!?':
                last_boundary = i
                break
        
        if last_boundary >= 0:
            search_text = extended_text[last_boundary + 1:].strip()
        else:
            search_text = extended_text.strip()
    
    # If search text is too long, limit it
    if len(search_text) > 150:
        search_text = search_text[-150:]
    
    # Check for negation patterns
    for pattern in negation_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            # Use the last match (closest to entity)
            last_match = matches[-1]
            # Make sure there's no intervening positive words that would cancel negation
            intervening_text = search_text[last_match.end():]
            positive_patterns = [r'\bbut\b', r'\bhowever\b', r'\bexcept\b', r'\balthough\b', r'\bpresent\b', r'\bseen\b']
            
            # If no positive words intervene, it's negated
            if not any(re.search(pos_pattern, intervening_text) for pos_pattern in positive_patterns):
                return True
    
    return False

def expand_entity_with_negation(text, entity):
    """Expand entity text to include negation if present within the same sentence"""
    entity_start = entity["start"]
    entity_end = entity["end"]
    entity_text = entity["text"]
    
    # Check if entity is negated
    if detect_negation(text, entity_start, entity_end):
        # For structured field patterns (e.g., "Hydronephrosis: None"), 
        # expand text to include the field value and mark as negated
        after_entity = text[entity_end:entity_end + 20].strip().lower()
        if (after_entity.startswith(': none') or after_entity.startswith(':none')):
            # Find the end of the field value
            colon_pos = text.find(':', entity_end)
            if colon_pos != -1:
                # Look for the end of the value (space, newline, or punctuation)
                value_start = colon_pos + 1
                value_end = value_start
                while value_end < len(text) and text[value_end] not in '\n.;,':
                    value_end += 1
                
                # Create expanded text that includes "Field: Value"
                expanded_text = text[entity_start:value_end].strip()
                return {
                    **entity,
                    "text": expanded_text,
                    "end": value_end,
                    "original_text": entity_text,
                    "is_negated": True
                }
        elif (after_entity.startswith(': present') or after_entity.startswith(':present') or
              after_entity.startswith(': positive') or after_entity.startswith(':positive')):
            # Affirmative structured field pattern - expand but mark as not negated
            colon_pos = text.find(':', entity_end)
            if colon_pos != -1:
                value_start = colon_pos + 1
                value_end = value_start
                while value_end < len(text) and text[value_end] not in '\n.;,':
                    value_end += 1
                
                expanded_text = text[entity_start:value_end].strip()
                return {
                    **entity,
                    "text": expanded_text,
                    "end": value_end,
                    "original_text": entity_text,
                    "is_negated": False
                }
        # Find the sentence start (don't cross sentence boundaries marked by ., !, ?)
        sentence_start = 0
        for i in range(entity_start - 1, max(0, entity_start - 200), -1):
            if text[i] in '.!?':
                sentence_start = i + 1
                break
            elif i == 0:
                sentence_start = 0
                break
        
        # Get text from sentence start to entity start
        search_text = text[sentence_start:entity_start]
        
        # Find the negation word and include it
        negation_words = ['no', 'not', 'absent', 'without', 'denies', 'unremarkable', 'normal']
        negation_phrases = ['negative for', 'rule out', 'no evidence of', 'no signs of', 'free of']
        
        # Check for phrases first
        for phrase in negation_phrases:
            phrase_pattern = phrase.replace(' ', r'\s+')
            match = re.search(rf'\b{phrase_pattern}\b', search_text.lower())
            if match:
                negation_start = sentence_start + match.start()
                new_text = text[negation_start:entity_end].strip()
                return {
                    **entity,
                    "text": new_text,
                    "start": negation_start,
                    "original_text": entity_text,
                    "is_negated": True
                }
        
        # Check for single words (find the closest one to the entity)
        best_match = None
        best_distance = float('inf')
        
        for word in negation_words:
            word_pattern = rf'\b{word}\b'
            matches = list(re.finditer(word_pattern, search_text.lower()))
            for match in matches:
                # Calculate distance from negation word to entity
                distance = entity_start - (sentence_start + match.end())
                if distance < best_distance and distance >= 0:
                    best_distance = distance
                    best_match = match
        
        if best_match:
            negation_start = sentence_start + best_match.start()
            new_text = text[negation_start:entity_end].strip()
            return {
                **entity,
                "text": new_text,
                "start": negation_start,
                "original_text": entity_text,
                "is_negated": True
            }
    
    # Not negated, return original entity with flag
    return {
        **entity,
        "original_text": entity_text,
        "is_negated": False
    }

def filter_overlapping_entities(entities):
    """Filter overlapping entities while preserving nested entities"""
    if not entities:
        return entities
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    filtered = []
    
    for entity in sorted_entities:
        # Check if this entity overlaps with any already filtered entity
        should_add = True
        entities_to_remove = []
        
        for i, existing in enumerate(filtered):
            # Check for overlap
            if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                # Determine the type of overlap
                entity_contains_existing = (entity['start'] <= existing['start'] and entity['end'] >= existing['end'])
                existing_contains_entity = (existing['start'] <= entity['start'] and existing['end'] >= entity['end'])
                
                # Case 1: Perfect duplicate (same boundaries) - keep the better one
                if (entity['start'] == existing['start'] and entity['end'] == existing['end']):
                    keep_new = False
                    # Prefer negated entities for same boundaries
                    if entity.get('is_negated', False) and not existing.get('is_negated', False):
                        keep_new = True
                    elif not entity.get('is_negated', False) and existing.get('is_negated', False):
                        keep_new = False
                    # If both or neither are negated, prefer higher confidence
                    elif entity['score'] > existing['score']:
                        keep_new = True
                    
                    if keep_new:
                        entities_to_remove.append(i)
                    else:
                        should_add = False
                        break
                
                # Case 2: Nested entities - keep both (this preserves nested NER)
                elif entity_contains_existing or existing_contains_entity:
                    # Keep both entities as they represent different levels of annotation
                    # e.g., "right kidney" (BodyStructure) nested within "right kidney dysfunction" (ClinicalFinding)
                    continue
                
                # Case 3: Partial overlap - resolve conflict by keeping the longer/better entity
                else:
                    keep_new = False
                    
                    # Prefer negated entities
                    if entity.get('is_negated', False) and not existing.get('is_negated', False):
                        keep_new = True
                    elif not entity.get('is_negated', False) and existing.get('is_negated', False):
                        keep_new = False
                    # If both or neither are negated, prefer longer entity
                    elif len(entity['text']) > len(existing['text']):
                        keep_new = True
                    elif len(entity['text']) < len(existing['text']):
                        keep_new = False
                    # If same length, prefer higher confidence
                    elif entity['score'] > existing['score']:
                        keep_new = True
                    
                    if keep_new:
                        entities_to_remove.append(i)
                    else:
                        should_add = False
                        break
        
        # Remove entities marked for removal (in reverse order to maintain indices)
        for i in reversed(entities_to_remove):
            filtered.pop(i)
        
        if should_add:
            filtered.append(entity)
    
    return filtered

def filter_field_value_entities(entities, text):
    """Filter out spurious field value entities that are part of structured patterns"""
    if not entities:
        return entities
    
    # Common field values that should not be standalone entities
    field_values = {
        'none', 'present', 'positive', 'negative', 'normal', 'abnormal',
        'seen', 'not seen', 'stable', 'improved', 'decreased', 'increased',
        'adequate', 'unremarkable', 'clear', 'enlarged', 'small'
    }
    
    filtered_entities = []
    
    for entity in entities:
        entity_text = entity['text'].lower().strip()
        
        # If this entity is a common field value, check if it's part of a structured pattern
        if entity_text in field_values:
            entity_start = entity['start']
            entity_end = entity['end']
            
            # Check text immediately before this entity (look for "Field: " pattern)
            # Look back up to 50 characters to find the pattern
            search_start = max(0, entity_start - 50)
            before_text = text[search_start:entity_start]
            
            # Look for colon followed by optional whitespace/newlines immediately before entity
            # This catches patterns like "Hydronephrosis: None" or "Size:\nNormal"
            if re.search(r':\s*$', before_text):
                # This appears to be a field value (e.g., "Hydronephrosis: None")
                # Skip this entity
                continue
            
            # Additional check: look for the pattern where the entity is at the start of a line
            # and preceded by a line ending with colon
            lines_before = before_text.split('\n')
            if len(lines_before) > 1 and lines_before[-2].strip().endswith(':'):
                # Entity is on a new line after a line ending with colon
                continue
        
        # Keep all other entities (including field names that are followed by colons)
        filtered_entities.append(entity)
    
    return filtered_entities

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
                    negation_flag = " [NEGATED or NORMAL]" if ent.get('is_negated', False) else ""
                    original_text = f" (original: '{ent.get('original_text', ent['text'])}')" if ent.get('is_negated', False) else ""
                    print(f"  - {ent['label']}: '{ent['text']}'{negation_flag} (score: {ent['score']:.3f}, pos: {ent['start']}-{ent['end']}){original_text}")
            
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
