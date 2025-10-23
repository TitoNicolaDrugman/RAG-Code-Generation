# BM25/export.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, List
import json

def read_raw_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def iter_query_entries(raw: Dict[str, Any] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    for key in ("queries", "items", "data", "rows"):
        val = raw.get(key)
        if isinstance(val, list):
            return val
    if isinstance(raw, dict) and isinstance(raw.get("results_by_query"), list):
        return raw["results_by_query"]
    return []

def extract_results_list(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("results", "top_k", "retrieved", "hits"):
        val = entry.get(key)
        if isinstance(val, list):
            return val
    payload = entry.get("payload")
    if isinstance(payload, dict):
        for key in ("results", "top_k", "retrieved", "hits"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    return []

def norm_repo_name(entry: Dict[str, Any]) -> str:
    for key in ("repo_name", "repo_full_name"):
        v = entry.get(key)
        if v:
            return str(v).strip().lower()
    return ""

def norm_instruction(entry: Dict[str, Any]) -> str:
    for key in ("instruction", "query", "prompt"):
        v = entry.get(key)
        if v:
            return str(v)
    return ""

def norm_query_id(entry: Dict[str, Any], idx: int) -> str:
    for key in ("id", "query_id", "qid"):
        v = entry.get(key)
        if v:
            return str(v)
    repo = norm_repo_name(entry) or "repo"
    return f"{repo}__{idx:06d}"

def norm_hit(hit: Dict[str, Any], i: int) -> Dict[str, Any]:
    doc_id = hit.get("doc_id") or hit.get("id") or hit.get("path") or f"doc_{i}"
    text   = hit.get("text") or hit.get("content") or hit.get("snippet") or ""
    score  = hit.get("score") or hit.get("bm25_score") or hit.get("similarity") or 0.0
    path   = hit.get("path")
    md     = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    try:
        score = float(score)
    except Exception:
        score = 0.0
    return {"doc_id": doc_id, "score": score, "path": path, "text": text, "metadata": md}

def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_normalized_rows(
    entries: List[Dict[str, Any]],
    target_repos: set[str],
    top_k: int,
    retrieval_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    filtered = [e for e in entries if norm_repo_name(e) in target_repos]
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(filtered):
        repo = norm_repo_name(ex)
        instr = norm_instruction(ex)
        qid = norm_query_id(ex, idx)
        hits = extract_results_list(ex)
        hits_norm = [norm_hit(h, i) for i, h in enumerate(hits)]
        rows.append({
            "query_id": qid,
            "repo_name": repo,
            "instruction": instr,
            "retrieval_method": "bm25",
            "k": top_k,
            "retrieval_params": retrieval_params,
            "results": hits_norm
        })
    return rows
