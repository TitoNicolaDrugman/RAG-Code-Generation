import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from snippet_retrievers.factory import get_retriever

from .config import OUT_DIR
from .utils import resolve_kb_file, norm_repo_name, norm_instruction, norm_query_id, norm_hit

def run_hybrid_retrieval(
    dataset,                              # Iterable[Dict[str, Any]] (HF Dataset o lista di dict)
    target_repos: List[str],
    top_k: int = 5,
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 32,
    rrf_k: int = 60,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    max_samples: Optional[int] = None,
    out_dir: Path = OUT_DIR,
) -> Dict[str, Any]:
    """
    Esegue retrieval 'hybrid' per tutte le query del dataset appartenenti a target_repos.
    Salva:
      - outputs/retrieval/hybrid_raw_k{top_k}.json
      - outputs/retrieval/hybrid_topk_k{top_k}.jsonl
      - outputs/retrieval/hybrid_topk_k{top_k}.json
    Ritorna i path e stats.
    """
    METRIC = "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json   = out_dir / f"{METRIC}_raw_k{top_k}.json"
    jsonl_path = out_dir / f"{METRIC}_topk_k{top_k}.jsonl"
    json_path  = out_dir / f"{METRIC}_topk_k{top_k}.json"

    # Materializza dataset e filtra per repo
    data_iter = list(dataset.select(range(max_samples))) if max_samples is not None else list(dataset)
    target_set = {r.lower() for r in target_repos}
    data_iter = [ex for ex in data_iter if norm_repo_name(ex) in target_set]

    print(f"[HYBRID] queries selezionate: {len(data_iter)} | k={top_k}")

    raw_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []

    for i, item in enumerate(tqdm(data_iter, desc="HYBRID")):
        repo_key = norm_repo_name(item)
        query = norm_instruction(item)

        # Retrieval (con gestione errori)
        try:
            kb_file_path = resolve_kb_file(repo_key)
            ret = get_retriever(
                metric=METRIC,
                kb_file=str(kb_file_path),
                kb_root=None,
                k1=bm25_k1,
                b=bm25_b,
                model_name=model_name,
                batch_size=batch_size,
                rrf_k=rrf_k,
            )
            hits = ret.retrieve(query, k=top_k)
            raw_rows.append({
                "query_id": i,
                "repo_full_name": repo_key,
                "instruction": query,
                "retrieved_snippets": hits
            })
        except Exception as e:
            hits = []
            raw_rows.append({
                "query_id": i,
                "repo_full_name": repo_key,
                "instruction": query,
                "retrieved_snippets": [],
                "error": str(e)
            })

        # Normalizzazione robusta
        hits_norm = [norm_hit(h, j) for j, h in enumerate(hits)]
        norm_rows.append({
            "query_id": norm_query_id(item, i),
            "repo_name": repo_key,
            "instruction": query,
            "retrieval_method": METRIC,
            "k": top_k,
            "retrieval_params": {
                "model_name": model_name,
                "batch_size": batch_size,
                "rrf_k": rrf_k,
                "bm25_k1": bm25_k1,
                "bm25_b": bm25_b
            },
            "results": hits_norm
        })

    # Salvataggi
    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2, ensure_ascii=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in norm_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(norm_rows, f, indent=2, ensure_ascii=False)

    print("[HYBRID] Saved files:")
    print(" - RAW          :", raw_json.resolve())
    print(" - JSONL        :", jsonl_path.resolve())
    print(" - JSON (array) :", json_path.resolve())
    print("[HYBRID] Exported queries:", len(norm_rows), "| k =", top_k, "| metric =", METRIC)

    return {
        "num_queries": len(norm_rows),
        "raw_json": str(raw_json.resolve()),
        "jsonl": str(jsonl_path.resolve()),
        "json": str(json_path.resolve()),
    }
