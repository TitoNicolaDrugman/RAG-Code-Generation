# prompts/v4.py
def build_baseline_prompt_v4(instruction: str) -> str:
    """
    Restituisce solo la risposta grezza (nessun boilerplate).
    """
    return instruction.strip()

def build_rag_prompt_v4(instruction: str, retrieved: str) -> str:
    """
    RAG: guida + retrieved, ma richiede solo la risposta grezza come output.
    """
    return f"""\
You are a senior Python engineer.  Use the retrieved examples to guide your implementation.

**Retrieved Examples**:
{retrieved.strip()}

**Task**:
{instruction.strip()}

**Requirements**:
- Python 3 with type hints
- Clean function or class design with a docstring
- Adhere to PEP8 conventions
- Include at least one unit test
- Do **NOT** copy or repeat the retrieved examples; write a NEW solution
- Return exactly and only the raw answer to the user request, no extra text
**Implementation**:
```python
"""
