# prompts/v1.py
def build_baseline_prompt_v1(instruction: str) -> str:
    """
    Baseline: solo il task e il marcatore '### Code:'.
    """
    return f"""\
### Task:
{instruction.strip()}

### Code:
"""

def build_rag_prompt_v1(instruction: str, retrieved: str) -> str:
    """
    RAG: prima gli esempi recuperati, poi il task, poi '### Code:'.
    """
    return f"""\
### Retrieved Examples:
{retrieved.strip()}

### Task:
{instruction.strip()}

### Code:
"""
