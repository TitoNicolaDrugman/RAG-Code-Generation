"""
Inspect cached vectors or find the closest vector to a sentence.

Examples:

  # (1) Inspect by index
  python -m snippet_retrievers.inspect_vector --kb-file "temp_downloaded_kbs\\bokeh__bokeh\\kb_bokeh__bokeh.json" --i 0

  # (2) Given a sentence, find the closest cached vector (cosine)
  # (ensure you've run cosine/hybrid once so the cache exists)
  python -m snippet_retrievers.inspect_vector --kb-file "temp_downloaded_kbs\\bokeh__bokeh\\kb_bokeh__bokeh.json" ^
    --sentence "websocket worker and bokeh server"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .embedder import _VEC_DIR, _cache_id_for_items, CodeEmbedder
from .utils import build_corpus

def _cache_paths(cache_id: str):
    mat_path = _VEC_DIR / f"{cache_id}.npz"
    meta_path = _VEC_DIR / f"{cache_id}.meta.json"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing cache vectors: {mat_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cache meta: {meta_path}")
    return mat_path, meta_path

def _describe_vector(label: str, vec: np.ndarray):
    # length can mean both dimension and L2 norm; show both.
    l2 = float(np.linalg.norm(vec))
    print(f"\n[{label}]")
    print(f"Vector shape : {vec.shape}")
    print(f"dtype        : {vec.dtype}")
    print(f"L2 norm      : {l2:.6f} (should be ~1.0)")
    print(f"Max value    : {float(np.max(vec)):.6f}")
    print(f"Min value    : {float(np.min(vec)):.6f}")
    # print the entire vector (768 floats)
    print("Vector values:")
    print("se vuoi vedere il vettore dei scommentare la prossima riga nel codice")
    #print(np.array2string(vec, precision=6, floatmode='fixed', max_line_width=10_000))

def _load_cache_from_args(args):
    """
    Resolve cache id, load matrix + meta, and (if available) the aligned items list.
    Returns: (cache_id, mat, meta, items)
    """
    if args.cache_id:
        cid = args.cache_id
        items = []
    else:
        items = build_corpus(kb_root=args.kb_root, kb_file=args.kb_file)
        # Derive cache id exactly as the retrievers do
        cid = _cache_id_for_items(items, CodeEmbedder().model_name)
    mat_path, meta_path = _cache_paths(cid)
    mat = np.load(mat_path)["M"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return cid, mat, meta, items

def _closest_vector_for_sentence(sentence: str, mat: np.ndarray, meta_model_name: str) -> tuple[int, float, np.ndarray]:
    """
    Embed sentence with the same model used for the cache (from meta),
    then return (index, cosine_score, vector).
    """
    embedder = CodeEmbedder(model_name=meta_model_name)
    qv = embedder.embed_query(sentence)  # (768,) L2-normalized
    # cosine = dot product because both are L2-normalized
    sims = mat @ qv
    idx = int(np.argmax(sims))
    return idx, float(sims[idx]), mat[idx]

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--kb-file", default=None, help="Path to a kb_*.json file.")
    g.add_argument("--kb-root", default=None, help="Root folder containing subfolders with kb_*.json.")
    g.add_argument("--cache-id", default=None, help="Directly supply a cache id.")

    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--i", type=int, help="Row index to inspect (0-based).")
    m.add_argument("--sentence", type=str, help="Sentence to embed and match to the closest cached vector.")

    args = ap.parse_args()

    cid, mat, meta, items = _load_cache_from_args(args)
    print(f"Cache ID     : {cid}")
    print(f"Matrix shape : {mat.shape}  dtype={mat.dtype}")
    print(f"Model (meta) : {meta.get('model', 'unknown')}")

    if args.i is not None:
        i = args.i
        if i < 0 or i >= mat.shape[0]:
            raise IndexError(f"Index {i} out of range 0..{mat.shape[0]-1}")
        row = mat[i]
        _describe_vector(f"Row #{i}", row)

        # Optional: show which snippet this row belongs to (if we loaded items)
        if items and i < len(items):
            text = items[i].text
            preview = text if len(text) <= 300 else (text[:300] + "... [truncated]")
            print("\nSnippet preview:")
            print(preview)
        return

    # --sentence flow
    sent = args.sentence.strip()
    if not sent:
        raise ValueError("--sentence provided but empty after stripping.")

    idx, score, vec = _closest_vector_for_sentence(sent, mat, meta.get("model", CodeEmbedder().model_name))
    print(f"\nClosest to: \"{sent}\"")
    print(f"Best index  : {idx}")
    print(f"Cosine score: {score:.6f}")
    _describe_vector("Closest vector", vec)

    # Optional: show the underlying snippet text if available
    if items and idx < len(items):
        text = items[idx].text
        preview = text if len(text) <= 600 else (text[:600] + "... [truncated]")
        print("\nMatched snippet preview:")
        print(preview)

if __name__ == "__main__":
    main()
