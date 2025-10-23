from pathlib import Path
from typing import Dict, Any, List, Union
from .config import KB_NAME_MAP, KB_BASE_DIR

def resolve_kb_file(repo_key: str) -> Path:
    repo_key = (repo_key or "").strip().lower()
    kb_fname = KB_NAME_MAP.get(repo_key)
    if not kb_fname:
        raise FileNotFoundError(f"Nessuna KB mappata per repo '{repo_key}'")
    flat = KB_BASE_DIR / kb_fname
    if flat.exists():
        return flat
    nested = KB_BASE_DIR / repo_key / kb_fname
    if nested.exists():
        return nested
    if repo_key == "pyscf":
        alt_nested = KB_BASE_DIR / "pyscf__pyscf" / kb_fname
        if alt_nested.exists():
            return alt_nested
    raise FileNotFoundError(f"KB file non trovato per repo '{repo_key}'")

def norm_repo_name(item: Dict[str, Any]) -> str:
    for k in ("repo_name", "repo_full_name"):
        if k in item and item[k]:
            return str(item[k]).strip().lower()
    return ""

def norm_instruction(item: Dict[str, Any]) -> str:
    for k in ("instruction", "query", "prompt"):
        if k in item and item[k]:
            return str(item[k])
    return ""

def norm_query_id(item: Dict[str, Any], idx: int) -> str:
    for k in ("id", "query_id", "qid"):
        if k in item and item[k]:
            return str(item[k])
    return f"{norm_repo_name(item) or 'repo'}__{idx:06d}"

def norm_hit(hit: Union[Dict[str, Any], str], i: int) -> Dict[str, Any]:
    if isinstance(hit, str):
        return {"doc_id": f"doc_{i}", "score": 0.0, "path": None, "text": hit, "metadata": {}}
    if isinstance(hit, dict):
        doc_id = hit.get("doc_id") or hit.get("id") or hit.get("path") or f"doc_{i}"
        text   = hit.get("text") or hit.get("content") or hit.get("snippet") or ""
        score  = hit.get("score") or hit.get("similarity") or hit.get("bm25_score") or 0.0
        path   = hit.get("path")
        md     = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        try:
            score = float(score)
        except Exception:
            score = 0.0
        return {"doc_id": doc_id, "score": score, "path": path, "text": text, "metadata": md}
    return {"doc_id": f"doc_{i}", "score": 0.0, "path": None, "text": str(hit), "metadata": {}}
