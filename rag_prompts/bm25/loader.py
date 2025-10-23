# rag_prompts/bm25/loader.py
from pathlib import Path
from typing import Dict, Any, List, Set
import json

from .io_utils import norm_hit, norm_repo_name, norm_instruction, norm_query_id

def _load_normalized_jsonl(p: Path, target_repos: Set[str]) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            obj = json.loads(line)
            repo = norm_repo_name(obj)
            if repo not in target_repos:
                continue
            res = [norm_hit(h, j) for j, h in enumerate(obj.get("results") or [])]
            obj["results"] = res
            rows.append(obj)
    return rows

def _load_raw_json(p: Path, target_repos: Set[str]) -> List[Dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(data):
        repo = norm_repo_name(item)
        if repo not in target_repos:
            continue
        instr = norm_instruction(item)
        hits  = item.get("retrieved_snippets") or []
        res   = [norm_hit(h, j) for j, h in enumerate(hits)]
        rows.append({
            "query_id": norm_query_id(item, i),
            "repo_name": repo,
            "instruction": instr,
            "results": res,
        })
    return rows

def load_bm25_rows(norm_jsonl_path: Path, raw_json_path: Path, target_repos: Set[str]) -> List[Dict[str, Any]]:
    if norm_jsonl_path and norm_jsonl_path.exists():
        print(f"[BM25] Carico JSONL normalizzato: {norm_jsonl_path}")
        return _load_normalized_jsonl(norm_jsonl_path, target_repos)
    if raw_json_path and raw_json_path.exists():
        print(f"[BM25] JSONL non trovato. Carico RAW JSON: {raw_json_path}")
        return _load_raw_json(raw_json_path, target_repos)
    raise FileNotFoundError("Nessuno dei due file BM25 esiste nei path forniti.")
