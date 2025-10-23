# codegen/runner.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Set
from tqdm import tqdm

from .io_utils import (
    read_jsonl, append_jsonl, ensure_dir,
    scan_prompt_files, result_paths
)
from .infer import generate_once

# chiavi standard attese nei file dei prompt
_PROMPT_KEYS = ("query_id", "repo_name", "instruction", "template", "variant", "prompt")

def _load_done_keys(per_prompt_out: Path) -> Set[str]:
    """
    Crea un set di chiavi già generate (query_id) dal file di output, per resume.
    """
    if not per_prompt_out.exists():
        return set()
    done: Set[str] = set()
    for row in read_jsonl(per_prompt_out):
        qid = row.get("query_id")
        if qid:
            done.add(qid)
    return done

def run_codegen_over_prompt_files(
    tokenizer,
    model,
    prompt_files: Iterable[Path],
    out_root: Path = Path("outputs/codegen"),
    model_name: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False,
) -> List[Path]:
    """
    Esegue la generazione su una lista di file JSONL di prompt.
    Salva:
      - per-prompt file: outputs/codegen/<model>/by_prompt/<promptfile_stem>.jsonl (append con resume)
      - aggregato:       outputs/codegen/<model>/all_generations.jsonl (append)
    Ritorna la lista dei file per-prompt scritti/aggiornati.
    """
    if model_name is None:
        model_name = getattr(getattr(model, "config", None), "name_or_path", "local_model")

    written_files: List[Path] = []

    for pf in prompt_files:
        per_prompt_out, aggregated_out = result_paths(out_root, model_name, pf)
        ensure_dir(per_prompt_out.parent)
        ensure_dir(aggregated_out.parent)

        # resume: salta query_id già presenti nel file per-prompt
        done_keys = _load_done_keys(per_prompt_out)

        rows = read_jsonl(pf)
        to_write_per_prompt: List[Dict[str, Any]] = []
        to_write_agg: List[Dict[str, Any]] = []

        print(f"\n[CODEGEN] file: {pf} | items: {len(rows)} | resume hits: {len(done_keys)}")
        for row in tqdm(rows, desc=f"Generating -> {pf.stem}"):
            missing = [k for k in _PROMPT_KEYS if k not in row]
            if missing:
                # riga non conforme
                continue
            qid = row["query_id"]
            if qid in done_keys:
                continue

            prompt_text = row["prompt"]
            g = generate_once(
                tokenizer=tokenizer, model=model, prompt_text=prompt_text,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, do_sample=do_sample,
            )

            out_row = {
                "query_id": qid,
                "repo_name": row["repo_name"],
                "instruction": row["instruction"],
                "template": row["template"],
                "variant": row.get("variant", "baseline"),
                "prompt": prompt_text,
                "generation": g["generation"],
                "metrics": {
                    "prompt_tokens": g["prompt_tokens"],
                    "gen_tokens": g["gen_tokens"],
                    "elapsed_sec": g["elapsed_sec"],
                },
                "gen_params": g["gen_kwargs"],
                "model_name": model_name,
                "source_file": str(pf),
            }
            to_write_per_prompt.append(out_row)
            to_write_agg.append(out_row)

        if to_write_per_prompt:
            append_jsonl(per_prompt_out, to_write_per_prompt)
            append_jsonl(aggregated_out, to_write_agg)

        written_files.append(per_prompt_out)
        print(f"[DONE] -> {per_prompt_out} (+{len(to_write_per_prompt)})")

    return written_files
