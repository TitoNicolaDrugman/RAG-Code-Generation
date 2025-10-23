# prompts/v3.py
def build_baseline_prompt_v3(instruction: str) -> str:
    """
    Senior Python engineer – function/class + docstring + test.
    """
    return f"""\
You are a senior Python engineer.  Fulfill the following task by writing production-ready code.

**Task**:
{instruction.strip()}

**Requirements**:
- Python 3, include type hints
- One well-formed function or class with a descriptive name
- A docstring (inputs, outputs, edge cases)
- PEP8 style (4-space indent, snake_case)
- At least one unit test using `assert` or `unittest`

**Implementation**:
```python
"""

def build_rag_prompt_v3(instruction: str, retrieved: str) -> str:
    """
    RAG: come v3 baseline ma con retrieved examples in testa.
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


**Implementation**:
```python
"""
