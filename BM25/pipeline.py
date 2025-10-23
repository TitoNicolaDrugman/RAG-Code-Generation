# BM25/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

from BM25 import robust_code_tokenizer_for_s5  # già esistente nel tuo progetto
from BM25.batch import get_or_build_retrieval_for_all_queries  # già esistente

from .export import (
    read_raw_json, iter_query_entries, build_normalized_rows, save_jsonl
)

def run_bm25_and_export(
    dataset,                          # es. lca_dataset_split (HF Dataset o struttura compatibile attesa dalla tua batch API)
    target_repos: List[str],          # es. ["seed-emulator", "pyscf__pyscf"]
    top_k: int = 3,
    out_dir: Path = Path("outputs/retrieval"),
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    overwrite_kb_download: bool = False,
    max_samples: Optional[int] = None,
    force_rebuild: bool = False,
    tokenizer = robust_code_tokenizer_for_s5,
) -> Dict[str, Any]:
    """
    Esegue BM25 limitandosi alle repo target (library_filter=target_repos),
    poi normalizza in JSONL con una riga per query (top-k risultati).
    Ritorna un dict con percorsi e conteggi.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bm25_raw_json = Path("BM25") / f"retrieved_k{top_k}_samples.json"
    out_jsonl = out_dir / f"bm25_topk_k{top_k}.jsonl"

    payload = get_or_build_retrieval_for_all_queries(
        dataset=dataset,
        tokenizer=tokenizer,
        top_k=top_k,
        library_filter=target_repos,             # filtro chiave richiesto
        overwrite_kb_download=overwrite_kb_download,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        show_progress=True,
        max_samples=max_samples,
        force_rebuild=force_rebuild,
    )

    # Normalizzazione
    raw = read_raw_json(bm25_raw_json)
    entries = iter_query_entries(raw)

    rows = build_normalized_rows(
        entries=entries,
        target_repos={r.lower() for r in target_repos},
        top_k=top_k,
        retrieval_params={"bm25_k1": bm25_k1, "bm25_b": bm25_b, "tokenizer": getattr(tokenizer, "__name__", "robust_code_tokenizer_for_s5")},
    )

    save_jsonl(rows, out_jsonl)

    # Ritorna summary
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return {
        "num_queries_payload": meta.get("num_queries"),
        "num_kbs_payload": meta.get("num_kbs"),
        "raw_json_path": str(bm25_raw_json.resolve()),
        "normalized_jsonl_path": str(out_jsonl.resolve()),
        "exported_queries": len(rows),
        "k": top_k,
    }
