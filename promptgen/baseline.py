# promptgen/baseline.py
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

from .normalize import norm_repo_name, norm_instruction, norm_query_id
from .prompts_discovery import (
    discover_prompt_modules,
    find_baseline_builder,
    call_baseline_builder_safely,
)

def _ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _iter_dataset(dataset_like: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(dataset_like)

def generate_baseline_prompts(
    dataset_like: Iterable[Dict[str, Any]],
    prompts_dir: str | Path = "prompts",
    out_dir: str | Path = "outputs/prompts/baseline",
    aggregate_filename: str = "_all_baseline.jsonl",
    fail_fast: bool = False,
) -> Tuple[List[Path], Path, int]:
    """
    Genera prompt baseline per tutte le query del dataset per tutti i template.
    Non legge file di retrieval e forza i template in modalità 'baseline'.
    """
    prompts_dir = Path(prompts_dir)
    out_dir = Path(out_dir)
    _ensure_out_dir(out_dir)

    # Flag d'ambiente per i template "chiacchieroni"
    os.environ["PROMPT_MODE"] = "baseline"

    root = Path(".").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    templates = discover_prompt_modules(prompts_dir)
    if not templates:
        raise RuntimeError(f"Nessun template trovato in {prompts_dir.resolve()}")

    data = _iter_dataset(dataset_like)

    per_template_paths: List[Path] = []
    all_rows: List[Dict[str, Any]] = []
    total_written = 0

    for tmpl in templates:
        try:
            builder = find_baseline_builder(tmpl, tmpl)
        except Exception as e:
            msg = f"[SKIP] prompts.{tmpl}: {e}"
            if fail_fast:
                raise
            print(msg)
            continue

        out_path = out_dir / f"{tmpl}_baseline.jsonl"
        written = 0
        with out_path.open("w", encoding="utf-8") as f_out:
            for i, ex in enumerate(data):
                qid  = norm_query_id(ex, i)
                repo = norm_repo_name(ex)
                instr = norm_instruction(ex)
                if not instr:
                    continue
                try:
                    prompt_text = call_baseline_builder_safely(builder, instr)
                except Exception as e:
                    warn = f"[WARN] {tmpl} qid={qid}: {e}"
                    if fail_fast:
                        raise
                    print(warn)
                    continue

                row = {
                    "query_id": qid,
                    "repo_name": repo,
                    "instruction": instr,
                    "template": tmpl,
                    "variant": "baseline",
                    "prompt": prompt_text,
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_rows.append(row)
                written += 1

        per_template_paths.append(out_path)
        total_written += written
        print(f"[OK] {tmpl}: salvati {written} prompt -> {out_path.resolve()}")

    # File aggregato
    agg_path = out_dir / aggregate_filename
    with agg_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nFile aggregato: {agg_path.resolve()}")
    print(f"Totale prompt baseline creati: {total_written}")

    return per_template_paths, agg_path, total_written
