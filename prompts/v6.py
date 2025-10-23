# prompts/v6.py
import textwrap
from .utils import truncate_to_n_tokens, _sanitize_triple_backticks, SNIPPET_TOKEN_BUDGET

def build_baseline_prompt_v6(instruction: str) -> str:
    """
    Long Code Arena · baseline (no retrieval snippets).
    """
    instruction = _sanitize_triple_backticks(instruction.strip())

    return textwrap.dedent(f"""\
        ### Long Code Arena · Library-Based Code Generation

        You are an elite Python 3 engineer.

        **Goal** Generate a **single, runnable `.py` file** that fulfils the
        **Task Description** below.  
        *Everything you need* – target library name, expected behaviour, I/O
        format – is stated inside the task itself.

        ── Task Description ────────────────────────────────────────────────
        {instruction}

        ── Implementation Rules (strict) ───────────────────────────────────
        1. Use **only** public APIs of the target library + Python ≥ 3.9 stdlib.  
        2. Write production-ready code **without extra comments** (docstrings
           are optional but allowed).  This maximises evaluation similarity.  
        3. All imports at the very top; no third-party deps beyond the library.  
        4. If you define reusable functions/classes, add a minimal
           ``if __name__ == "__main__":`` demo that illustrates usage.  
        5. Keep runtime ≤ 30 s and RAM ≤ 2 GiB.  
        6. **Do not** reveal chain-of-thought; keep reasoning internal.

        ── Output Protocol ─────────────────────────────────────────────────
        • Reply with **only** the Python code, nothing else.  
        • Start your code right after the marker below and end with a matching
          fence.  The first executable line must be an import or module docstring.

        ```python
        """)

def build_rag_prompt_v6(instruction: str, retrieved: str) -> str:
    """
    Long Code Arena · RAG (with snippets).
    """
    instruction = _sanitize_triple_backticks(instruction.strip())
    snippet = truncate_to_n_tokens(retrieved, SNIPPET_TOKEN_BUDGET, keep="head")

    return textwrap.dedent(f"""\
        ### Long Code Arena · Library-Based Code Generation (with Snippets)

        You are an elite Python 3 engineer.

        **Goal** Generate a **single, runnable `.py` file** that fulfils the
        **Task Description** below.  
        *Everything you need* – target library name, expected behaviour, I/O
        format – is stated inside the task itself.

        ── Retrieved Library Snippets ──────────────────────────────────────
        The fenced block contains **real fragments** from the target library.
        • Treat them as *read-only* context.  
        • Copy verbatim **only** trivial boiler-plate (e.g. imports, constant
          tables).  For everything else, write original code inspired by them.  
        • Ignore snippets that do not help the task.

        ```python
        {snippet}
        ```

        ── Task Description ────────────────────────────────────────────────
        {instruction}

        ── Implementation Rules (strict) ───────────────────────────────────
        1. Use **only** public APIs of the target library + Python ≥ 3.9 stdlib.  
        2. Write production-ready code **without extra comments** (docstrings
           are optional but allowed).  This maximises evaluation similarity.  
        3. Replicate import aliases that you see in the snippets where useful
           (helps API-Recall).  
        4. No third-party deps beyond the target library.  
        5. If you define reusable functions/classes, add a minimal
           ``if __name__ == "__main__":`` demo that illustrates usage.  
        6. Keep runtime ≤ 30 s and RAM ≤ 2 GiB.  
        7. **Do not** reveal chain-of-thought; keep reasoning internal.

        ── Output Protocol ─────────────────────────────────────────────────
        • Reply with **only** the Python code, nothing else.  
        • Start your code right after the marker below and end with a matching
          fence.  The first executable line must be an import or module docstring.

        ```python
        """)
