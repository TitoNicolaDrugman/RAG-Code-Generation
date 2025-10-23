# prompts/v8.py
from .utils import truncate_to_n_tokens

def build_baseline_prompt_v8(instruction: str) -> str:
    """
    v8 baseline – versione con checklist e regole di output.
    """
    return f"""\
# 📚 Library-Centric Python Code Generation (v8)

You are ChatGPT, a world-class Python engineer.

**Task**  
Write a **single, self-contained Python 3 file** that completely solves the problem below.  
The solution must make *substantial, idiomatic* use of the one Python library implicitly
required by the task.

---

## Problem

{instruction.strip()}

---

## Implementation Checklist ✅
1. Plan first (internally) — do **not** output the plan.  
2. Code quality  
   - all necessary imports (standard + target library)  
   - type hints everywhere  
   - concise docstrings explaining each public item’s intent and library usage  
   - PEP 8 compliance  
3. Demonstration – `if __name__ == "__main__":` **or** minimal pytest-style test  
4. Robustness – handle edge cases, raise informative errors  
5. Clarity > cleverness

---

## Output Rules

- **Output ONLY the code** for the single file.  
- No extra text or markdown.  
- Begin code immediately after ` ```python` and end with nothing else.

```python
"""

def build_rag_prompt_v8(instruction: str, retrieved: str) -> str:
    """
    v8 RAG – versione con esempi di libreria troncati.
    """
    retrieved_snippet = truncate_to_n_tokens(retrieved, 1800)

    return f"""\
# 📚 Library-Centric Python Code Generation with Context (v8)

You are ChatGPT, a world-class Python engineer.

You receive:  
1. **Problem description** (requires a specific Python library).  
2. **Retrieved code examples** from that library.

Craft a **single, self-contained Python 3 file** solving the problem, leveraging and adapting patterns from the examples.

---

## Problem

{instruction.strip()}

---

## Retrieved Library Examples

```python
{retrieved_snippet.strip()}
"""
