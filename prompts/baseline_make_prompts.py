# prompts/baseline_make_prompts.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import importlib
import inspect
from tqdm.auto import tqdm


# ------------- paths & io -------------
def retrieval_json_path_for_k(top_k: int) -> Path:
    return Path("BM25") / f"retrieved_k{top_k}_samples.json"

def output_dir_for_k(top_k: int) -> Path:
    # cartella: prompts/prompts_baseline_{K}snip
    return Path(__file__).parent / f"prompts_baseline_{top_k}snip"

def output_json_path_for_builder(top_k: int, builder_name: str) -> Path:
    return output_dir_for_k(top_k) / f"{builder_name}.json"


# ------------- discover builders -------------
def _import_prompt_modules():
    candidates = [
        "prompts",
        "prompts.utils",
        "prompts.v1", "prompts.v2", "prompts.v3", "prompts.v4",
        "prompts.v5", "prompts.v6", "prompts.v6_2", "prompts.v6_3",
        "prompts.v7", "prompts.v8", "prompts.v9",
    ]
    loaded = []
    for name in candidates:
        try:
            loaded.append(importlib.import_module(name))
        except Exception:
            pass
    return loaded

def discover_baseline_builders() -> Dict[str, Callable[[str], str]]:
    """
    Trova tutte le funzioni 'build_baseline_prompt_*' nel package prompts.
    Ritorna: {nome_funzione: funzione}
    """
    modules = _import_prompt_modules()
    out: Dict[str, Callable] = {}
    for m in modules:
        for fname, obj in inspect.getmembers(m, inspect.isfunction):
            if fname.startswith("build_baseline_prompt_"):
                out[fname] = obj
    return out


# ------------- core -------------
def generate_baseline_prompts_from_queries(
    top_k: int = 3,
    preferred_first: Optional[str] = "build_baseline_prompt_v6_3",
    force: bool = False,
    show_progress: bool = True,
) -> Dict[str, Path]:
    """
    Per ogni builder 'build_baseline_prompt_*' crea i prompt baseline per tutte le query
    dal file BM25/retrieved_k{K}_samples.json e salva in
    'prompts/prompts_baseline_{K}snip/{builder}.json'.

    Se il file per un builder esiste e force=False → SKIP.

    Ritorna: {builder_name: path_output_json}
    """
    # 1) carica retrieval
    rpath = retrieval_json_path_for_k(top_k)
    if not rpath.is_file():
        raise FileNotFoundError(
            f"Retrieval file non trovato: {rpath}.\n"
            f"Genera prima BM25/retrieved_k{top_k}_samples.json (batch Sez. 5)."
        )
    payload = json.loads(rpath.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    num_queries = len(results)
    print(f"Loaded retrieval for K={top_k}: {num_queries} queries from '{rpath}'.")

    # 2) builders
    builders = discover_baseline_builders()
    if not builders:
        raise RuntimeError("Nessuna funzione 'build_baseline_prompt_*' trovata nel package 'prompts'.")

    names = sorted(builders.keys())
    if preferred_first and preferred_first in builders:
        names.remove(preferred_first)
        names.insert(0, preferred_first)

    # 3) output dir
    out_dir = output_dir_for_k(top_k)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")

    outputs: Dict[str, Path] = {}

    # 4) per builder
    for bname in names:
        out_path = output_json_path_for_builder(top_k, bname)
        if out_path.is_file() and not force:
            print(f"  Skip '{bname}': file già presente → {out_path.name} (usa force=True per rigenerare)")
            outputs[bname] = out_path
            continue

        print(f"Building baseline prompts with '{bname}' → {out_path.name}")
        builder = builders[bname]

        items: List[Dict[str, Any]] = []
        iterator = results if not show_progress else tqdm(results, desc=f"Baseline ({bname})", unit="query")

        for rec in iterator:
            idx = rec.get("idx")
            repo = rec.get("repo_full_name")
            instruction = rec.get("instruction") or ""

            try:
                prompt_str = builder(instruction)
            except TypeError:
                raise TypeError(f"Il builder '{bname}' deve avere firma (instruction).")

            items.append({
                "idx": idx,
                "repo_full_name": repo,
                "instruction": instruction,
                "prompt": prompt_str,
                "top_k": top_k,
            })

        # 5) salva un JSON per builder
        out_payload = {
            "meta": {
                "builder": bname,
                "top_k": top_k,
                "num_queries": num_queries,
                "timestamp": time.time(),
                "source_retrieval_file": str(rpath),
            },
            "prompts": items,
        }
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved {len(items)} baseline prompts → {out_path}")
        outputs[bname] = out_path

    return outputs


# ------------- cli -------------
if __name__ == "__main__":
    # Esecuzione semplice da riga di comando:
    #   python -m prompts.baseline_make_prompts
    generate_baseline_prompts_from_queries(top_k=3, force=False, show_progress=True)
