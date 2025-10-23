# prompts/templates.py

from typing import List, Dict, Any, Optional

# Tipo “snippet” normalizzato (compatibile con i tuoi JSON normalizzati)
# {"doc_id": str, "score": float, "path": Optional[str], "text": str, "metadata": dict}
Snippet = Dict[str, Any]

def _format_snippets(snippets: List[Snippet], max_chars: int = 4000) -> str:
    """
    Concatena gli snippet in un unico blocco marcato.
    Taglia in modo “morbido” se supera max_chars (senza spezzare parole lunghissime).
    """
    parts = []
    for i, s in enumerate(snippets, 1):
        path = s.get("path") or s.get("metadata", {}).get("path") or s.get("doc_id")
        head = f"[{i}] {path}"
        text = s.get("text") or ""
        parts.append(head + "\n" + text)
    blob = "\n\n---\n\n".join(parts).strip()
    if len(blob) <= max_chars:
        return blob
    # soft truncate
    return blob[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n... [truncated]"

def _base_header(instruction: str, repo: str, extra_ctx: Optional[Dict[str, Any]] = None) -> str:
    repo_str = f"Repository target: {repo}" if repo else "Repository target: (unknown)"
    meta = []
    if extra_ctx:
        for k, v in extra_ctx.items():
            meta.append(f"{k}: {v}")
    meta_str = ("\n".join(meta) + "\n") if meta else ""
    return (
        f"{repo_str}\n"
        f"{meta_str}"
        "Task: Generate Python code that satisfies the instruction below.\n"
        "Return only code unless explicitly asked otherwise.\n\n"
        f"Instruction:\n{instruction.strip()}\n"
    )

# -------------- 9 VARIANTI DI PROMPT (v1..v9) -----------------
# Ogni funzione accetta instruction, repo e (opz.) snippets già filtrati

def v1_baseline(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nConstraints:\n"
        "- Use only public APIs from the library when possible.\n"
        "- Include minimal scaffolding needed to run.\n"
        "- Add brief inline comments for non-trivial steps.\n"
    )

def v2_structured_steps(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nPlan first:\n"
        "1) Outline key steps.\n"
        "2) Write the code in a single block.\n"
        "3) Keep imports tidy and grouped.\n"
        "Now produce the final code block only.\n"
    )

def v3_minimal_api(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nRules:\n"
        "- Prefer minimal API surface.\n"
        "- Avoid unused variables and imports.\n"
        "- If an exact class/function name is required by the library, use it.\n"
    )

def v4_safe_defaults(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nDefaults:\n"
        "- Choose safe defaults and sensible parameters.\n"
        "- Validate inputs when feasible.\n"
        "- Print final outputs/results at the end.\n"
    )

def v5_docstring_first(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nFormat:\n"
        "- Start code with a brief module-level docstring summarizing the task.\n"
        "- Then provide the complete implementation.\n"
    )

def v6_tests_smoke(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nTesting:\n"
        "- Include a minimal smoke test in a __main__ guard that demonstrates usage.\n"
    )

def v7_functional(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nStyle:\n"
        "- Favor small pure functions.\n"
        "- Separate data preparation from execution.\n"
    )

def v8_comments_sparse(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nComments:\n"
        "- Keep comments sparse: only where truly helpful.\n"
    )

def v9_explanatory(instruction: str, repo: str, **kw) -> str:
    return _base_header(instruction, repo) + (
        "\nExplain choices as short comments near the relevant lines (not a prose block).\n"
    )

# -------------- VERSIONI “CON SNIPPETS” -----------------
# Wrapper: ogni versione ha la controparte con contesto KB

def _with_snippets(template_fn, instruction: str, repo: str, snippets: List[Snippet], **kw) -> str:
    base = template_fn(instruction, repo, **kw)
    if snippets:
        kb_block = _format_snippets(snippets, max_chars=kw.get("snippets_max_chars", 4000))
        return (
            base
            + "\nContext snippets (from the library KB, use them for API names and patterns):\n"
            + kb_block
            + "\n\nNow write the final code block."
        )
    return base + "\n(No snippets available)\n"

# Esporta una mappa nome->funzione per le 9 versioni (baseline / con-snippets)
TEMPLATES = {
    "v1": dict(
        baseline=v1_baseline,
        with_snippets=lambda **a: _with_snippets(v1_baseline, **a),
    ),
    "v2": dict(
        baseline=v2_structured_steps,
        with_snippets=lambda **a: _with_snippets(v2_structured_steps, **a),
    ),
    "v3": dict(
        baseline=v3_minimal_api,
        with_snippets=lambda **a: _with_snippets(v3_minimal_api, **a),
    ),
    "v4": dict(
        baseline=v4_safe_defaults,
        with_snippets=lambda **a: _with_snippets(v4_safe_defaults, **a),
    ),
    "v5": dict(
        baseline=v5_docstring_first,
        with_snippets=lambda **a: _with_snippets(v5_docstring_first, **a),
    ),
    "v6": dict(
        baseline=v6_tests_smoke,
        with_snippets=lambda **a: _with_snippets(v6_tests_smoke, **a),
    ),
    "v7": dict(
        baseline=v7_functional,
        with_snippets=lambda **a: _with_snippets(v7_functional, **a),
    ),
    "v8": dict(
        baseline=v8_comments_sparse,
        with_snippets=lambda **a: _with_snippets(v8_comments_sparse, **a),
    ),
    "v9": dict(
        baseline=v9_explanatory,
        with_snippets=lambda **a: _with_snippets(v9_explanatory, **a),
    ),
}
