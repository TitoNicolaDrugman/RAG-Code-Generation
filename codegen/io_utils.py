# codegen/io_utils.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, List, Tuple

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # salta righe corrotte
                continue
    return rows

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def scan_prompt_files(dirs: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for d in dirs:
        if not d.exists(): 
            continue
        for p in sorted(d.glob("*.jsonl")):
            out.append(p)
    return out

def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")

def result_paths(root: Path, model_name: str, prompt_file: Path) -> Tuple[Path, Path]:
    """
    Restituisce: (per_prompt_output, aggregated_output)
    per_prompt_output: outputs/codegen/<model>/by_prompt/<promptfile_stem>.jsonl
    aggregated_output: outputs/codegen/<model>/all_generations.jsonl
    """
    m = sanitize_model_name(model_name)
    per_prompt = root / m / "by_prompt" / f"{prompt_file.stem}.jsonl"
    aggregated = root / m / "all_generations.jsonl"
    return per_prompt, aggregated
