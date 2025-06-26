#!/usr/bin/env python3
"""
Debug the position issue with the first entity
"""

def debug_positions():
    text = "Left kidney: There are no renal or ureteral calculi. No hydronephrosis or hydroureter. No enhancing masses. No cortical cysts. No retroperitoneal mass."
    
    print(f"Full text: {text}")
    print(f"Text length: {len(text)}")
    print()
    
    # Find "no renal or ureteral calculi"
    target = "renal or ureteral calculi"
    start_pos = text.find(target)
    end_pos = start_pos + len(target)
    
    print(f"Target: '{target}'")
    print(f"Position: {start_pos}-{end_pos}")
    print(f"Text at position: '{text[start_pos:end_pos]}'")
    print()
    
    # Check what's before it
    print("Text before entity:")
    before_text = text[:start_pos]
    print(f"'{before_text}'")
    print()
    
    # Find sentence boundaries
    sentence_start = start_pos
    for i in range(start_pos - 1, max(0, start_pos - 200), -1):
        if text[i] in '.!?':
            sentence_start = i + 1
            break
        elif i == 0:
            sentence_start = 0
            break
    
    print(f"Sentence starts at: {sentence_start}")
    print(f"Sentence text: '{text[sentence_start:start_pos]}'")
    
    # Check if "no" is in there
    search_text = text[sentence_start:start_pos].lower()
    print(f"Search text (lowercase): '{search_text}'")
    
    import re
    no_match = re.search(r'\bno\b', search_text)
    if no_match:
        print(f"Found 'no' at relative position: {no_match.start()}")
        print(f"Absolute position: {sentence_start + no_match.start()}")
    else:
        print("'no' not found in search text")

if __name__ == "__main__":
    debug_positions()
