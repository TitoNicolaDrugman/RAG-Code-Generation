# promptgen/normalize.py
from typing import Dict, Any

def norm_repo_name(item: Dict[str, Any]) -> str:
    for k in ("repo_name", "repo_full_name"):
        if k in item and item[k]:
            return str(item[k]).strip()
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
    repo = (norm_repo_name(item) or "repo").replace("/", "__")
    return f"{repo}__{idx:06d}"
