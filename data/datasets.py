# data/datasets.py
from __future__ import annotations
from typing import Iterable, Optional, List
from datasets import load_dataset

DEFAULT_DATASET_NAME = "JetBrains-Research/lca-library-based-code-generation"

def load_lca_dataset(dataset_name: str = DEFAULT_DATASET_NAME, split: str = "test"):
    print(f"\n  Loading dataset '{dataset_name}' (split='{split}')…")
    ds = load_dataset(dataset_name, split=split)
    print(f"Dataset loaded with {len(ds)} examples across all libraries.")
    return ds

def filter_by_repos(ds, repo_names: Optional[Iterable[str]] = None):
    if not repo_names:
        return ds
    names: List[str] = list(repo_names)
    print(f"Filtering to repos: {names}")
    fds = ds.filter(lambda ex: ex.get("repo_name") in names or ex.get("repo_full_name") in names)
    print(f"Filtered to {len(fds)} examples.")
    return fds
