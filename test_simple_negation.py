#!/usr/bin/env python3
"""
Simple test for negation detection logic without loading models
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

def test_kidney_example():
    """Test with the kidney example"""
    text = "Left kidney: There are no renal or ureteral calculi. No hydronephrosis or hydroureter. No enhancing masses. No cortical cysts. No retroperitoneal mass."
    
    # Simulate entities found by GLiNER
    entities = [
        {"text": "renal or ureteral calculi", "start": 26, "end": 51},
        {"text": "hydronephrosis", "start": 56, "end": 70},
        {"text": "enhancing masses", "start": 90, "end": 106},
        {"text": "cortical cysts", "start": 111, "end": 125},
        {"text": "retroperitoneal mass", "start": 130, "end": 150},
    ]
    
    print("Testing negation detection logic:")
    print(f"Text: {text}")
    print("\nEntity negation analysis:")
    
    for entity in entities:
        is_negated = detect_negation(text, entity["start"], entity["end"])
        print(f"- '{entity['text']}' at {entity['start']}-{entity['end']}: {'NEGATED' if is_negated else 'NOT NEGATED'}")
        
        # Show the sentence context
        sentence_start = entity["start"]
        for i in range(entity["start"] - 1, max(0, entity["start"] - 200), -1):
            if text[i] in '.!?':
                sentence_start = i + 1
                break
            elif i == 0:
                sentence_start = 0
                break
        
        sentence = text[sentence_start:entity["end"]+20].strip()
        if len(sentence) > 80:
            sentence = sentence[:80] + "..."
        print(f"  Context: '{sentence}'")
        print()

if __name__ == "__main__":
    test_kidney_example()
