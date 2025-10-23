# prompts/v9.py
from .utils import truncate_to_n_tokens

def build_baseline_prompt_v9(instruction: str) -> str:
    """
    v9 baseline – focus su API-recall e self-review interna.
    """
    return f"""# 🔍 Library-Centric Python Code Generation (v9 – Baseline)

You are **a senior Python engineer**.  
Your mission is to deliver a **single, self-contained Python 3 file** that *fully* solves the problem below, *making substantial and correct use of the one Python library implicitly required*.

---

## Problem

{instruction.strip()}

---

## Workflow (do not output steps 1–3)

1. **Analyse** the task and *identify the target library* with high confidence.  
2. **Plan** the solution in your head (modules, functions, data flow).  
3. **Self-review checklist**  
   - Every function/class is type-annotated and has a concise docstring.  
   - Each external call **exactly matches an existing API** of the library  
     (verify names, parameter order, return types).  
   - No extraneous imports or unused variables.  
   - PEP 8 compliance.  
   - Provide a tiny runnable demo or `pytest`-style test in a  
     `if __name__ == "__main__":` guard.  
   - Script runs without internet access or extra files.

4. **Write the code** (start now).

---

## Output rules

- **Output ONLY the Python code.**  
- No explanations, markdown, or comments outside the source file.  
- Begin immediately after the next line:

```python
"""

def build_rag_prompt_v9(instruction: str, retrieved: str) -> str:
    """
    v9 RAG – aggiunge estratti verificati della libreria come contesto.
    """
    retrieved_snippet = truncate_to_n_tokens(retrieved, 1800)
    return f"""# 🔍 Library-Centric Python Code Generation with Context (v9 – RAG)

You are **a senior Python engineer**.  
Alongside the problem, you receive **verified code excerpts** from the relevant library.  
Leverage them to craft a **single, runnable Python 3 file** that solves the task, re-using APIs and patterns where appropriate.

---

## Problem

{instruction.strip()}

---

## Retrieved Library Excerpts (grounding context)

```python
{retrieved_snippet.strip()}
"""
