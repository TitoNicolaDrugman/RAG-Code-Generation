import re
from typing import List

def robust_code_tokenizer_for_s5(text_input) -> List[str]:
    if not isinstance(text_input, str):
        return []
    text = text_input.lower()
    raw_tokens = re.split(r'[^a-z0-9_]+', text)
    return [t for t in raw_tokens if t and len(t) > 1 and not t.isdigit()]
