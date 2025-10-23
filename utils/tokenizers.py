import re

def robust_code_tokenizer_for_s5(text_input):
    """
    Tokenizer semplice per codice:
    - converte in lowercase
    - separa su caratteri non alfanumerici/underscore
    - rimuove stringhe vuote, singoli caratteri, numeri puri
    """
    if not isinstance(text_input, str):
        return []
    text = text_input.lower()
    raw_tokens = re.split(r'[^a-z0-9_]+', text)
    return [t for t in raw_tokens if t and len(t) > 1 and not t.isdigit()]
