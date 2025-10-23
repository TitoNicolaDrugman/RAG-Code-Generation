# rag_prompts/bm25/io_utils.py
from pathlib import Path
from typing import Dict, Any, List
import json

# --- Normalizzazioni comuni ---
def norm_hit(hit: Any, i: int) -> Dict[str, Any]:
    if isinstance(hit, str):
        return {"doc_id": f"doc_{i}", "score": 0.0, "path": None, "text": hit}
    if isinstance(hit, dict):
        doc_id = hit.get("doc_id") or hit.get("id") or hit.get("path") or f"doc_{i}"
        text   = hit.get("text") or hit.get("content") or hit.get("snippet") or ""
        score  = hit.get("score") or hit.get("similarity") or hit.get("bm25_score") or 0.0
        path   = hit.get("path")
        try: score = float(score)
        except Exception: score = 0.0
        return {"doc_id": doc_id, "score": score, "path": path, "text": text}
    return {"doc_id": f"doc_{i}", "score": 0.0, "path": None, "text": str(hit)}

def take_top_k(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not isinstance(results, list):
        return []
    sorted_res = sorted(
        results,
        key=lambda x: (x.get("score") if isinstance(x.get("score"), (int, float)) else -1.0),
        reverse=True
    )
    return sorted_res[:k]

def norm_repo_name(item: Dict[str, Any]) -> str:
    for k in ("repo_name", "repo_full_name"):
        v = item.get(k)
        if v: return str(v).strip().lower()
    return ""

def norm_instruction(item: Dict[str, Any]) -> str:
    for k in ("instruction", "query", "prompt"):
        v = item.get(k)
        if v: return str(v)
    return ""

def norm_query_id(item: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("id", "query_id", "qid"):
        v = item.get(k)
        if v: return str(v)
    repo = norm_repo_name(item) or "repo"
    return f"{repo}__{fallback_idx:06d}"

# --- I/O helpers ---
def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
