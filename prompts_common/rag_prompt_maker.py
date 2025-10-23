# prompts_common/rag_prompt_maker.py
from __future__ import annotations
import json, textwrap
from pathlib import Path
from typing import Dict, Any, List, Callable, Iterable

def format_snippets_block(snippets: List[Dict[str, Any]], repo_name: str, method: str, k: int) -> str:
    parts = [
        f"### CONTEXT SNIPPETS ({method.upper()}, repo={repo_name}, top-{k})",
        "Usa solo informazioni rilevanti, senza inventare dettagli non presenti.",
    ]
    for i, s in enumerate(snippets, 1):
        doc = s.get("doc_id") or s.get("id") or s.get("path") or f"doc_{i}"
        score = s.get("score")
        path  = s.get("path")
        txt   = s.get("text") or s.get("content") or s.get("snippet") or ""
        header = f"[{i}] doc_id={doc}"
        if isinstance(score, (int, float)):
            header += f" | score={score:.4f}"
        if path:
            header += f" | path={path}"
        parts.append(header)
        parts.append(textwrap.indent(txt.strip(), "    "))
    return "\n".join(parts)

def make_rag_prompt(
    base_builder: Callable[[str], str],
    instruction: str,
    snippets: List[Dict[str, Any]],
    repo_name: str,
    method: str,
    k: int,
) -> str:
    context = format_snippets_block(snippets, repo_name, method, k)
    task    = base_builder(instruction)
    return f"{context}\n\n### TASK\n{task}".strip()

def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
