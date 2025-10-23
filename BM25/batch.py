# BM25/batch.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi

# Reuse internal helpers from BM25/runner.py
from .runner import _kb_keys_from_github, _build_corpus


# -----------------------
# Index building (global)
# -----------------------
def build_global_bm25_index(
    tokenizer,
    library_filter: Optional[List[str]] = None,
    overwrite_kb_download: bool = False,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> Dict[str, Any]:
    """
    Costruisce UNA VOLTA il corpus globale e l'indice BM25.

    Returns:
        dict: {
            'bm25': BM25Okapi,
            'mapping': List[(library_key, original_doc_idx)],
            'docs': List[str],
            'selected_keys': List[str]
        }
    """
    # Scope KB (ALL o subset)
    available_keys = _kb_keys_from_github()
    selected_keys = library_filter or available_keys
    print(f"  Using {len(selected_keys)} KBs ({'ALL' if library_filter is None else 'subset'})")

    # Build corpus (tokenized docs + mapping)
    print("  Building global tokenized corpus…")
    tok_corpus, mapping, docs = _build_corpus(
        selected_keys,
        tokenizer,
        overwrite=overwrite_kb_download,
    )
    if not tok_corpus:
        raise RuntimeError("Empty tokenized corpus — no BM25 index built.")

    print(f"  Corpus size: {len(tok_corpus)} documents")
    bm25 = BM25Okapi(tok_corpus, k1=bm25_k1, b=bm25_b)
    print("  BM25 index built.\n")

    return {
        "bm25": bm25,
        "mapping": mapping,
        "docs": docs,
        "selected_keys": selected_keys,
    }


# -----------------------
# Single-instruction topK
# -----------------------
def topk_for_instruction(
    bm25: BM25Okapi,
    mapping: List[Tuple[str, int]],
    docs: List[str],
    tokenizer,
    instruction: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Calcola i Top-K per una singola istruzione.

    Returns:
        List[dict]: [{rank, library_key, snippet, snippet_len}, ...]
    """
    q_tokens = tokenizer(instruction) if isinstance(instruction, str) else []
    if not q_tokens or top_k <= 0:
        return []

    n = min(top_k, len(docs))
    idxs = bm25.get_top_n(q_tokens, list(range(len(docs))), n=n)
    results: List[Dict[str, Any]] = []
    for r, i in enumerate(idxs, start=1):
        lib, _ = mapping[i]
        snip = docs[i]
        results.append({
            "rank": r,
            "library_key": lib,
            "snippet": snip,
            "snippet_len": len(snip),
        })
    return results


# -----------------------
# Batch over dataset
# -----------------------
def retrieve_topk_for_dataset(
    dataset,                      # lca_dataset_split (list-like di esempi)
    tokenizer,
    bm25_bundle: Dict[str, Any],  # output di build_global_bm25_index
    top_k: int = 3,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Itera su TUTTE le query del dataset e calcola i Top-K per ciascuna.

    Returns:
        List[dict]: elemento per ogni sample:
        {
          "idx": i,
          "repo_full_name": <str or None>,
          "instruction": <str>,
          "topk": [ {rank, library_key, snippet, snippet_len}, ... ]
        }
    """
    bm25 = bm25_bundle["bm25"]
    mapping = bm25_bundle["mapping"]
    docs = bm25_bundle["docs"]

    N = len(dataset)
    if max_samples is not None:
        N = min(N, max_samples)

    iterator = range(N)
    if show_progress:
        iterator = tqdm(iterator, desc="BM25 retrieval over dataset", unit="sample")

    out: List[Dict[str, Any]] = []
    for i in iterator:
        ex = dataset[i]
        repo = ex.get("repo_full_name") or ex.get("repo_name")
        instr = ex.get("instruction") or ""
        res = topk_for_instruction(bm25, mapping, docs, tokenizer, instr, top_k=top_k)
        out.append({
            "idx": i,
            "repo_full_name": repo,
            "instruction": instr,
            "topk": res,
        })
    return out


# -----------------------
# Caching utilities (per K)
# -----------------------
def _cache_dir() -> Path:
    """
    Directory dove salvare i JSON di retrieval per K.
    (Richiesto: BM25/retrieved_k{K}_samples.json)
    """
    return Path("BM25")


def cache_path_for_k(top_k: int) -> Path:
    """
    Ritorna il path del JSON per questo top_k:
        BM25/retrieved_k{K}_samples.json
    """
    return _cache_dir() / f"retrieved_k{top_k}_samples.json"


def load_cached_results_if_any(top_k: int) -> Optional[Dict[str, Any]]:
    """
    Se esiste il file cache per K, lo legge e ritorna il contenuto.
    """
    path = cache_path_for_k(top_k)
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Cache hit: found existing results for K={top_k} at '{path}'.")
            return data
        except Exception as e:
            print(f"  WARNING: failed to read cache '{path}': {e}")
    return None


def save_results_for_k(top_k: int, data: Dict[str, Any]) -> Path:
    """
    Salva il dizionario 'data' nel file per K e ritorna il Path.
    """
    path = cache_path_for_k(top_k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved results to: {path}")
    return path


# -----------------------
# One-call convenience
# -----------------------
def get_or_build_retrieval_for_all_queries(
    dataset,
    tokenizer,
    top_k: int = 3,
    library_filter: Optional[List[str]] = None,
    overwrite_kb_download: bool = False,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Flusso completo con caching:
      - se BM25/retrieved_k{K}_samples.json esiste e force_rebuild=False: carica e ritorna
      - altrimenti costruisce indice, fa retrieval per tutto il dataset, salva su JSON e ritorna

    Returns:
        dict:
        {
          "meta": {
            "top_k": int, "bm25_k1": float, "bm25_b": float,
            "num_queries": int, "num_kbs": int, "library_filter": List[str] | None,
            "timestamp": float
          },
          "results": [ ... ]  # vedi retrieve_topk_for_dataset
        }
    """
    if not force_rebuild:
        cached = load_cached_results_if_any(top_k)
        if cached is not None:
            return cached

    print(f"[1/2] Build ONE global BM25 index (K={top_k})…")
    bundle = build_global_bm25_index(
        tokenizer=tokenizer,
        library_filter=library_filter,
        overwrite_kb_download=overwrite_kb_download,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )

    print("[2/2] Run retrieval over ALL dataset queries…")
    results = retrieve_topk_for_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        bm25_bundle=bundle,
        top_k=top_k,
        show_progress=show_progress,
        max_samples=max_samples,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "top_k": top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "num_queries": len(results),
            "num_kbs": len(bundle["selected_keys"]),
            "library_filter": bundle["selected_keys"] if library_filter else None,
            "timestamp": time.time(),
        },
        "results": results,
    }

    save_results_for_k(top_k, payload)
    return payload


__all__ = [
    "build_global_bm25_index",
    "topk_for_instruction",
    "retrieve_topk_for_dataset",
    "cache_path_for_k",
    "load_cached_results_if_any",
    "save_results_for_k",
    "get_or_build_retrieval_for_all_queries",
]
