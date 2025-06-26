#!/usr/bin/env python3
"""
Test the corrected negation expansion functionality
"""

import re

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
        r'\bfree\s+of\b'
    ]
    
    # Find the start of the current sentence (look backwards for sentence boundaries)
    sentence_start = entity_start
    for i in range(entity_start - 1, max(0, entity_start - 200), -1):
        if text[i] in '.!?':
            sentence_start = i + 1
            break
        elif i == 0:
            sentence_start = 0
            break
    
    # Get text from sentence start to entity start
    search_text = text[sentence_start:entity_start].strip().lower()
    
    # If search text is empty or very short, expand the search window
    if len(search_text) < 10:
        extended_start = max(0, entity_start - 100)
        extended_text = text[extended_start:entity_start].lower()
        
        last_boundary = -1
        for i in range(len(extended_text) - 1, -1, -1):
            if extended_text[i] in '.!?':
                last_boundary = i
                break
        
        if last_boundary >= 0:
            search_text = extended_text[last_boundary + 1:].strip()
        else:
            search_text = extended_text.strip()
    
    if len(search_text) > 150:
        search_text = search_text[-150:]
    
    # Check for negation patterns
    for pattern in negation_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            last_match = matches[-1]
            intervening_text = search_text[last_match.end():]
            positive_patterns = [r'\bbut\b', r'\bhowever\b', r'\bexcept\b', r'\balthough\b', r'\bpresent\b', r'\bseen\b']
            
            if not any(re.search(pos_pattern, intervening_text) for pos_pattern in positive_patterns):
                return True
    
    return False

def expand_entity_with_negation(text, entity):
    """Expand entity text to include negation if present within the same sentence"""
    entity_start = entity["start"]
    entity_end = entity["end"]
    entity_text = entity["text"]
    
    if detect_negation(text, entity_start, entity_end):
        # Find the sentence start
        sentence_start = 0
        for i in range(entity_start - 1, max(0, entity_start - 200), -1):
            if text[i] in '.!?':
                sentence_start = i + 1
                break
            elif i == 0:
                sentence_start = 0
                break
        
        search_text = text[sentence_start:entity_start]
        
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
    
    return {
        **entity,
        "original_text": entity_text,
        "is_negated": False
    }

def test_corrected_negation():
    """Test the corrected negation detection and expansion"""
    
    test_text = "Left kidney: There are no renal or ureteral calculi. No hydronephrosis or hydroureter. No enhancing masses. No cortical cysts. No retroperitoneal mass."
    
    test_entities = [
        {"text": "renal or ureteral calculi", "start": 26, "end": 51, "score": 0.921, "label": "ClinicalFinding"},
        {"text": "hydronephrosis", "start": 56, "end": 70, "score": 0.730, "label": "Symptom"},
        {"text": "enhancing masses", "start": 90, "end": 106, "score": 0.856, "label": "ClinicalFinding"},
        {"text": "cortical cysts", "start": 111, "end": 125, "score": 0.843, "label": "ClinicalFinding"},
        {"text": "retroperitoneal mass", "start": 130, "end": 150, "score": 0.852, "label": "ClinicalFinding"}
    ]
    
    print("Testing corrected negation expansion:")
    print(f"Text: {test_text}")
    print("\n" + "="*80)
    
    print("BEFORE expansion:")
    for i, entity in enumerate(test_entities):
        print(f"{i+1}. '{entity['text']}' at {entity['start']}-{entity['end']}")
    
    print("\nAFTER expansion:")
    for i, entity in enumerate(test_entities):
        expanded = expand_entity_with_negation(test_text, entity)
        negation_flag = " [NEGATED]" if expanded.get('is_negated', False) else ""
        original_note = f" (was: '{expanded.get('original_text')}')" if expanded.get('is_negated', False) else ""
        
        print(f"{i+1}. '{expanded['text']}'{negation_flag} at {expanded['start']}-{expanded['end']}{original_note}")

if __name__ == "__main__":
    test_corrected_negation()
