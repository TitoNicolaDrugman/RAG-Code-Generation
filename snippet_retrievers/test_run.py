"""
Minimal test runner for the snippet retrievers.

Examples (from repo root, PowerShell):

  # Single KB, BM25
  python -m snippet_retrievers.test_run --q "run bokeh tests with nosetests" --k 3 --metric bm25 `
    --kb-file "temp_downloaded_kbs\\bokeh__bokeh\\kb_bokeh__bokeh.json"

  # Single KB, Cosine (will build & cache vectors)
  python -m snippet_retrievers.test_run --q "websocket worker and bokeh server" --k 5 --metric cosine `
    --kb-file "temp_downloaded_kbs\\bokeh__bokeh\\kb_bokeh__bokeh.json"

  # All KBs under root
  python -m snippet_retrievers.test_run --q "websocket worker and bokeh server" --k 5 --metric cosine
"""

import argparse
from . import retrieve_snippets
from .utils import REPORT_PATH, PKG_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="query string")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--metric", default="bm25", choices=["bm25","cosine","hybrid"])
    ap.add_argument("--kb-root", default="temp_downloaded_kbs",
                    help="root dir containing per-repo folders with kb_*.json")
    ap.add_argument("--kb-file", default=None,
                    help="optional path to a single kb_*.json; overrides --kb-root")
    # BM25 params
    ap.add_argument("--k1", type=float, default=1.5, help="BM25 k1")
    ap.add_argument("--b", type=float, default=0.75, help="BM25 b")
    # Cosine params
    ap.add_argument("--model", default="microsoft/codebert-base",
                    help="HF model name for code embeddings (768-d)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rrf-k", type=int, default=60,
                help="RRF constant used when computing the hybrid score in reports")
    args = ap.parse_args()

    snippets = retrieve_snippets(
        q=args.q, k=args.k, metric=args.metric,
        kb_root=args.kb_root, kb_file=args.kb_file,
        # pass-through kwargs (retrievers will use what they need)
        k1=args.k1, b=args.b,
        model_name=args.model, batch_size=args.batch_size,
        rrf_k=args.rrf_k
    )

    print("\n=== Returned snippets (strings only) ===")
    for i, s in enumerate(snippets, 1):
        print(f"\n[{i}] --------------------------------")
        preview = s if len(s) < 800 else (s[:800] + "... [truncated]")
        print(preview)

    print("\nReport written to:", REPORT_PATH)
    print("Snippets-only copy:", PKG_DIR / "snippets_only.txt")

if __name__ == "__main__":
    main()
