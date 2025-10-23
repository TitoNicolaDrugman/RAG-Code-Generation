from pathlib import Path
from typing import List, Tuple, Optional
from datasets import config as hf_config
from data import load_lca_dataset, filter_by_repos

def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_dataset_jsonl(ds, path: Path) -> None:
    """
    Salva un datasets.Dataset in JSONL.
    Prima tenta ds.to_json(path), in fallback usa pandas.
    """
    try:
        ds.to_json(str(path))
        return
    except Exception:
        pass

    try:
        import pandas as pd
        df = ds.to_pandas()
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Impossibile salvare {path} in JSONL: {e}")

def load_and_optionally_filter(
    dataset_name: str,
    split: str,
    target_repos: Optional[List[str]]
):
    """
    Carica lo split da HF e applica eventualmente il filtro su repo_name.
    Restituisce (ds_full, ds_filtered).
    """
    ds_full = load_lca_dataset(dataset_name, split)
    ds_filt = filter_by_repos(ds_full, target_repos) if target_repos else ds_full
    return ds_full, ds_filt

def print_summary(
    ds_full,
    ds_filt,
    split: str,
    target_repos: Optional[List[str]],
    out_all: Path,
    out_filt: Path
) -> None:
    n_all = len(ds_full)
    n_filt = len(ds_filt)
    print("HuggingFace cache dir:", hf_config.HF_DATASETS_CACHE)
    try:
        print("cache_files (full split):", getattr(ds_full, "cache_files", None))
    except Exception:
        pass
    print()
    print("===== SUMMARY =====")
    print(f"Split: {split}")
    print(f"Total in split: {n_all}")
    if target_repos:
        print(f"Filter repos: {target_repos}")
        print(f"Total after filter: {n_filt}")
    else:
        print("No filter applied")
        print(f"Filtered equals full split: {n_filt}")
    print()
    print("Saved files:")
    print(" - Full split   :", out_all.resolve())
    print(" - Filtered     :", out_filt.resolve())
