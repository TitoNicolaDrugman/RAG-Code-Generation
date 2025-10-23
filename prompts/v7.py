# prompts/v7.py
from .utils import truncate_to_n_tokens

def build_baseline_prompt_v7(instruction: str) -> str:
    """
    v7 baseline – script standalone, libreria implicita, qualità alta.
    """
    return f"""\
# Python Code Generation Task (Library-Focused)

You are an expert programmer and Python engineer tasked with writing a **complete, standalone Python 3 script** that solves the problem described below.  
This problem implicitly requires using a particular **Python library**, which must be inferred from the task.  
Focus on demonstrating strong understanding and appropriate use of that library’s API.

---

## Task Description

{instruction.strip()}

---

## Code Requirements

- Produce a **self-contained** and **runnable** Python 3 script. It must be a high-quality and clean code.
- Make **extensive and idiomatic use** of the target library.
- Include all necessary `import` statements.
- Use **type hints** for all function signatures and relevant variables.
- Add **clear and concise docstrings** to main classes/functions.
- Follow **PEP 8** style and organize code into logical sections.
- If the task involves a reusable component (e.g., function/class), demonstrate its usage with a clear **main block** or **unit test**.
- Your solution should be **robust**, modular, and easy to understand.
- If you follow all the rules we've given you, you'll make us very proud—and you might even become famous
---

## Output Format

- Output **only** the Python code.
- Do **not** include any explanation or markdown.
- Start writing your code exaclty now

"""

def build_rag_prompt_v7(instruction: str, retrieved: str) -> str:
    """
    v7 RAG – usa retrieved snippet come contesto.
    """
    retrieved_snippet = truncate_to_n_tokens(retrieved, 1800)

    return f"""\
# Python Code Generation Task (Library-Focused, with Context)

You are an expert Python developer. Your task is to write a **complete, single-file Python 3 script** that solves the problem described below.  
This problem implicitly involves using a specific **Python library**, and to support your implementation, you are also given **retrieved code examples** from that library which it might help you.

---

## Task Description

{instruction.strip()}

---

## Retrieved Code Examples

Use the examples below as context. You may adapt patterns, API usage, and techniques shown here if relevant:

```python
{retrieved_snippet.strip()}

---

## Code Requirements

- The script must be **fully self-contained** and **runnable**. It must be a high-quality and clean code.
- Use the relevant library’s APIs **extensively** and **idiomatically**.
- Include all **necessary imports**.
- Add **type annotations** and well-written **docstrings**.
- Ensure clean code structure, following **PEP 8** style.
- If applicable, include a simple **main execution block** or **test case**.
- Emphasize clarity, reusability, and correctness.

---

## Output Format

- Provide **only** the Python code.
- No extra explanation, markdown, or commentary.
- Start writing your code exaclty now

"""
