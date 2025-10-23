# prompts/v5.py
from .utils import truncate_to_n_tokens

def build_baseline_prompt_v5(instruction: str) -> str:
    return f"""\
Write a complete Python 3 implementation for the following task.  Include type hints, a docstring, and at least one unit test.

Task:
{instruction.strip()}

```python
"""

def build_rag_prompt_v5(instruction: str, retrieved: str) -> str:
    retrieved = truncate_to_n_tokens(retrieved, 1800)
    return f"""\
You are a senior Python engineer.  Use the retrieved examples to inspire your implementation.

**Retrieved Examples**:
{retrieved}

**Task**:
{instruction.strip()}

**Requirements**:
- Python 3 with type hints
- Clean function or class design with a docstring
- Adhere to PEP8
- Include at least one unit test
- Do **NOT** copy or repeat the examples; write a NEW solution
**Implementation**:
```python
"""
