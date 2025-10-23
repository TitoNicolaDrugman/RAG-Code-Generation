# rag_prompts/bm25/build_prompts.py
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from .loader import load_bm25_rows
from .io_utils import take_top_k, save_jsonl

# Riutilizziamo i builder e il maker già definiti nel tuo progetto
from prompts_common.templates import load_all_prompt_builders
from prompts_common.rag_prompt_maker import make_rag_prompt

DEFAULT_TARGET_REPOS = {"seed-emulator", "seed-labs__seed-emulator", "pyscf", "pyscf__pyscf"}

def build_bm25_prompts(
    norm_jsonl_path: Path,
    raw_json_path: Path,
    out_dir: Path,
    top_k_snippets: int = 3,
    method_tag: str = "rag_bm25_top3",
    target_repos: Set[str] = DEFAULT_TARGET_REPOS,
) -> Tuple[List[Path], Path, int]:
    """
    Genera i prompt RAG (BM25) per tutti i template.
    Ritorna: (lista_file_per_template, file_aggregato, num_righe_aggregate)
    """
    rows = load_bm25_rows(norm_jsonl_path, raw_json_path, target_repos)
    print(f"[BM25] rows filtrate: {len(rows)}")
    builders = load_all_prompt_builders()
    print("[BM25] Template caricati:", ", ".join(sorted(builders.keys())))

    out_dir.mkdir(parents=True, exist_ok=True)
    agg_jsonl = out_dir / f"RAG_{method_tag}.jsonl"

    per_template_buffers: Dict[str, List[Dict[str, Any]]] = {name: [] for name in builders.keys()}
    agg_rows: List[Dict[str, Any]] = []

    for r in rows:
        qid   = str(r.get("query_id"))
        repo  = (r.get("repo_name") or "").lower()
        instr = r.get("instruction") or ""
        topk  = take_top_k(r.get("results") or [], top_k_snippets)

        for templ_name, builder in builders.items():
            prompt_text = make_rag_prompt(
                base_builder=builder,
                instruction=instr,
                snippets=topk,
                repo_name=repo,
                method="bm25",
                k=top_k_snippets,
            )
            row_out = {
                "query_id": qid,
                "repo_name": repo,
                "instruction": instr,
                "template": templ_name,
                "variant": "rag_bm25",
                "k_snippets": top_k_snippets,
                "retrieval_method": "bm25",
                "snippets": topk,
                "prompt": prompt_text,
            }
            per_template_buffers[templ_name].append(row_out)
            agg_rows.append(row_out)

    # Salvataggi per-template
    written: List[Path] = []
    for templ_name, rows_out in per_template_buffers.items():
        path = out_dir / f"{templ_name}_{method_tag}.jsonl"
        save_jsonl(path, rows_out)
        written.append(path)

    # Salvataggio aggregato
    save_jsonl(agg_jsonl, agg_rows)

    print("\n[BM25] Scritti i file RAG (BM25):")
    for p in written: 
        print(" -", p.resolve())
    print("Aggregato:")
    print(" -", agg_jsonl.resolve())

    return written, agg_jsonl, len(agg_rows)
