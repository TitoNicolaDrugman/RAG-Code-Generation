# Small utilities to run and trace multihop experiments cleanly from notebooks.
# Printing text and comments are in English US, as requested.

from __future__ import annotations

import csv
import json
import os
import random
import time
import glob
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Normalization controls
NORMALIZE_SCORES: bool = False
"""
If True, functions that display or persist scores (_hits_compact, print_hits,
compute_final_metrics, save_topk_impact_csv) will scale them to [0,1] using
max-per-list normalization. This keeps strategies comparable in tables.
"""
NORMALIZE_MAX: float = 100.0        # 1.0 => 0–1, 100.0 => 0–100

def _norm_digits() -> int:
    # nicer rounding: 2 decimals when using 0–100, else 4
    return 2 if (NORMALIZE_SCORES and NORMALIZE_MAX > 1.0) else 4

def _normalize_list_scores(scores):
    mx = max(scores) if scores else 0.0
    if mx <= 0:
        return [0.0 for _ in scores]
    scale = float(NORMALIZE_MAX) / mx
    return [s * scale for s in scores]

def _metrics_empty() -> dict:
    return {
        "final_k": 0,
        "unique_repos": 0,
        "diversity": 0.0,
        "mean_score": 0.0,
        "max_score": 0.0,
        "repo_focus_rate": 0.0,
    }


# ---------------------------------------------------------------------------
# Hit handling (robust to dict or Doc)
# ---------------------------------------------------------------------------

def sort_key(hit: Any) -> float:
    """Return a numeric score for sorting, supporting both dict and Doc hits."""
    if isinstance(hit, dict):
        return float(hit.get("score", 0.0))
    return float(getattr(hit, "score", 0.0))


def hit_tuple(hit: Any) -> Tuple[Optional[str], Optional[int], float, Optional[str]]:
    """Normalize a hit to (repo, doc_id, score, text) for printing or CSV."""
    if isinstance(hit, dict):
        return (hit.get("repo"), hit.get("doc_id"), float(hit.get("score", 0.0)), hit.get("text"))
    return (
        getattr(hit, "repo", None),
        getattr(hit, "doc_id", None),
        float(getattr(hit, "score", 0.0)),
        getattr(hit, "text", None),
    )


def extract_hits(pack: Dict[str, Any], strategy_key: str) -> List[Any]:
    """Extract hits from the returned pack, tolerant to different pack schemas."""
    if not isinstance(pack, dict):
        return []
    if "hits" in pack and isinstance(pack["hits"], list):
        return pack["hits"]
    section = pack.get(strategy_key, {})
    if isinstance(section, dict) and isinstance(section.get("hits"), list):
        return section["hits"]
    return []


def extract_subqueries(pack: Dict[str, Any]) -> List[str]:
    """Extract subqueries if present, else return empty list."""
    if not isinstance(pack, dict):
        return []
    subs = pack.get("subqueries", [])
    return subs if isinstance(subs, list) else []


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_header(title: str, question: str, cfg: Any, filters: Optional[Dict[str, str]]) -> None:
    print(f"\n=== {title} ===")
    print(f"Question           : {question}")
    if filters and isinstance(filters, dict):
        print(f"Filters            : {filters}")
    print(f"Strategy           : {cfg.strategy}")
    print(f"k_sub / k_final    : {cfg.k_sub} / {cfg.k_final}")
    print(f"max_hops           : {cfg.max_hops}")
    print(f"Backend / Model    : {cfg.llm_backend} / {cfg.llm_model}")


def print_subqueries(pack: Dict[str, Any]) -> None:
    subs = extract_subqueries(pack)
    if subs:
        print("Subqueries generated:")
        for i, s in enumerate(subs):
            print(f"  {i:>2}: {s}")


def print_hits(hits: List[Any], show_top: int = 5) -> None:
    if not hits:
        print("No hits.")
        return
    ordered = sorted(hits, key=sort_key, reverse=True)[:show_top]

    if NORMALIZE_SCORES:
        all_scores = [hit_tuple(h)[2] for h in hits]
        norm_all = _normalize_list_scores(all_scores)
        idx_map = {id(h): i for i, h in enumerate(hits)}
        digits = 2 if NORMALIZE_MAX > 1.0 else 3
        print(f"Top-{len(ordered)} hits (normalized 0..{int(NORMALIZE_MAX) if NORMALIZE_MAX.is_integer() else NORMALIZE_MAX}):")
        for i, h in enumerate(ordered, 1):
            repo, doc_id, _raw, _ = hit_tuple(h)
            nsc = norm_all[idx_map[id(h)]]
            print(f"  {i:>2}. [{repo}] doc_id={doc_id}  score={nsc:.{digits}f}")
        return

    # raw (default)
    print(f"Top-{len(ordered)} hits (sorted by score desc):")
    for i, h in enumerate(ordered, 1):
        repo, doc_id, score, _text = hit_tuple(h)
        print(f"  {i:>2}. [{repo}] doc_id={doc_id}  score={score:.3f}")



# ---------------------------------------------------------------------------
# Filter arg normalization (back-compat bridge)
# ---------------------------------------------------------------------------

def _pick_filter_arg(
    filter_libs: Optional[Union[str, List[str]]],
    repo_filter: Optional[Union[str, List[str]]],
) -> Optional[Union[str, List[str]]]:
    """Prefer `filter_libs`; if None, fall back to `repo_filter`; else None (open-world)."""
    return filter_libs if filter_libs is not None else repo_filter


# ---------------------------------------------------------------------------
# Run one mode end-to-end
# ---------------------------------------------------------------------------

def run_mode(
    *,
    provider: Any,
    question: str,
    strategy_key: str,
    top_k: int,
    filter_libs: Optional[Union[str, List[str]]] = None,   # preferred modern arg
    repo_filter: Optional[Union[str, List[str]]] = None,   # legacy alias for backward-compat
    backend: str,
    model: str,
    cache_dir: str,
    planner_ctx_docs: int = 2,
    planner_ctx_chars: int = 300,
    decomposer_max_tokens: Optional[int] = None,
    planner_max_tokens: Optional[int] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    # NEW (optional): if None -> True for IR, False for DF; you can pass True/False explicitly if you want
    force_last_subquery_final: Optional[bool] = None,
) -> Tuple[Dict[str, Any], List[Any], Any, float]:
    """
    Build cfg for the given mode, invoke run_hops, and return (pack, hits, cfg, latency_ms).
    - decomposition_first: remote decomposer, stub planner, max_hops=1
    - iterative_refine   : stub decomposer, remote planner, max_hops=3
    Additionally, for iterative_refine we recompute final hits from the **last refined subquery**
    so that 'final top-k' reflects the final step, not the initial query.
    """
    from importlib import reload
    import S6.multihop as mh
    reload(mh)

    if strategy_key not in ("decomposition_first", "iterative_refine"):
        raise ValueError(f"Unsupported strategy_key: {strategy_key}")

    if strategy_key == "decomposition_first":
        k_sub = max(1, min(3, (top_k // 2) or 1)); max_hops = 1
        decomposer_mode, planner_mode = "remote", "stub"
    else:
        k_sub = max(1, min(2, (top_k // 3) or 1)); max_hops = 3
        decomposer_mode, planner_mode = "stub", "remote"

    # Normalize runtime library selection: None | str | list[str]
    libs_sel = filter_libs if filter_libs is not None else repo_filter
    target_repo = libs_sel if isinstance(libs_sel, str) else None

    cfg = mh.MultiHopConfig(
        strategy=strategy_key,
        k_sub=k_sub,
        k_final=top_k,
        max_hops=max_hops,
        target_repo=target_repo,
        llm_backend=backend,
        llm_model=model,
        decomposer_mode=decomposer_mode,
        planner_mode=planner_mode,
        cache_dir=cache_dir,
        planner_ctx_docs=planner_ctx_docs,
        planner_ctx_chars=planner_ctx_chars,
        decomposer_max_tokens=decomposer_max_tokens,
        planner_max_tokens=planner_max_tokens,
        **(cfg_overrides or {}),
    )

    # Build provider-compatible filters dict
    from S6.mh_helpers import filters_for
    filters = filters_for(libs_sel)  # None | {"repo": ...}

    # Default policy: override to last subquery only for iterative_refine
    if force_last_subquery_final is None:
        force_last_subquery_final = (strategy_key == "iterative_refine")

    t0 = time.time()
    pack = mh.run_hops(
        question,
        k=top_k,
        strategy=cfg.strategy,
        provider=provider,
        cfg=cfg,
        filters=filters,
    )

    # If requested, recompute final hits from the LAST refined subquery
    # so that "final top-k (after the strategy)" is truly final.
    if force_last_subquery_final:
        subs = extract_subqueries(pack)
        if subs:
            last_q = subs[-1]
            final_hits = provider.search(last_q, k=top_k, filters=filters)
            # annotate and override in-pack hits for downstream code/CSV
            pack.setdefault("meta", {})
            pack["meta"]["final_query_source"] = "last_subquery"
            pack["meta"]["final_query_text"] = last_q
            pack["meta"]["original_hits_present"] = bool(pack.get("hits"))
            pack["hits"] = final_hits  # <- override for consistency with expectation

    latency_ms = (time.time() - t0) * 1000.0
    hits = extract_hits(pack, strategy_key)
    return pack, hits, cfg, latency_ms


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp_iso", "strategy", "backend", "model", "retriever",
    "k_sub", "k_final", "max_hops", "repo_filter",
    "latency_ms", "total_hits",
    "question", "subqueries", "top_hits", "kb_dir", "libs", "cache_dir",
]

def ensure_dir(d: str) -> None:
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _hits_compact(hits: List[Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Compact representation for CSV (repo, doc_id, score). Respects NORMALIZE_*."""
    ordered = sorted(hits, key=sort_key, reverse=True)[:limit]
    scores_all = [hit_tuple(h)[2] for h in hits]
    digits = _norm_digits()
    if NORMALIZE_SCORES:
        norm_all = _normalize_list_scores(scores_all)
        idx_map = {id(h): i for i, h in enumerate(hits)}
    out: List[Dict[str, Any]] = []
    for h in ordered:
        repo, doc_id, score, _ = hit_tuple(h)
        if NORMALIZE_SCORES:
            score = norm_all[idx_map[id(h)]]
        out.append({"repo": repo, "doc_id": doc_id, "score": round(float(score), digits)})
    return out


def build_csv_row(
    *,
    strategy: str,
    backend: str,
    model: str,
    retriever: str,
    cfg: Any,
    repo_filter: Optional[str],
    latency_ms: float,
    hits: List[Any],
    question: str,
    kb_dir: str,
    libs: List[str],
) -> Dict[str, Any]:
    """Build a single CSV row with reproducibility metadata."""
    import datetime as _dt
    row = {
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "backend": backend,
        "model": model,
        "retriever": retriever,
        "k_sub": getattr(cfg, "k_sub", None),
        "k_final": getattr(cfg, "k_final", None),
        "max_hops": getattr(cfg, "max_hops", None),
        "repo_filter": repo_filter or "",
        "latency_ms": round(float(latency_ms), 0),
        "total_hits": len(hits),
        "question": question,
        "subqueries": "",  # caller fills with actual subqueries if desired
        "top_hits": json.dumps(_hits_compact(hits, limit=10), ensure_ascii=False),
        "kb_dir": kb_dir,
        "libs": json.dumps(libs, ensure_ascii=False),
        "cache_dir": getattr(cfg, "cache_dir", ""),
    }
    return row


def save_run_csv(csv_dir: str, row: Dict[str, Any], filename: str = "multihop_runs.csv") -> str:
    """Append one row to CSV; create file with header if missing. Returns file path."""
    ensure_dir(csv_dir)
    path = os.path.join(csv_dir, filename)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        # filter to known columns only, in case extra keys were provided
        safe_row = {k: row.get(k, "") for k in CSV_COLUMNS}
        w.writerow(safe_row)
    return path


# === Extra helpers for model/strategy comparison and top_k impact ============

def hit_key(hit: Any) -> Tuple[Optional[str], Optional[int]]:
    """Stable key for de-dup and set ops: (repo, doc_id)."""
    if isinstance(hit, dict):
        return (hit.get("repo"), hit.get("doc_id"))
    return (getattr(hit, "repo", None), getattr(hit, "doc_id", None))


def jaccard_hits(a: List[Any], b: List[Any]) -> float:
    """Jaccard similarity between two final result sets based on (repo, doc_id)."""
    A = {hit_key(x) for x in a}
    B = {hit_key(x) for x in b}
    if not A and not B:
        return 1.0
    U = A | B
    I = A & B
    return float(len(I)) / float(len(U)) if U else 0.0


def compute_final_metrics(hits: List[Any] | None, repo_filter: Optional[str] = None) -> Dict[str, Any]:
    hits = hits or []
    k = len(hits)
    if k == 0:
        return _metrics_empty()

    repos  = [hit_tuple(h)[0] for h in hits]
    scores = [hit_tuple(h)[2] for h in hits]
    if NORMALIZE_SCORES:
        scores = _normalize_list_scores(scores)

    uniq_repos = len({r for r in repos if r is not None})
    diversity = round(uniq_repos / max(1, k), 4)
    digits = _norm_digits()
    mean_score = round(sum(scores) / k, digits) if k else 0.0
    max_score  = round(max(scores), digits) if k else 0.0

    repo_focus_rate = 0.0
    if isinstance(repo_filter, str) and repo_filter:
        repo_focus_rate = round(sum(1 for r in repos if r == repo_filter) / k, 4) if k else 0.0

    return {
        "final_k": k,
        "unique_repos": uniq_repos,
        "diversity": diversity,
        "mean_score": mean_score,
        "max_score": max_score,
        "repo_focus_rate": repo_focus_rate,
    }


def build_compare_row(
    *,
    strategy: str, backend: str, model: str, retriever: str,
    cfg: Any, repo_filter: Optional[str], latency_ms: float,
    hits: List[Any], question: str, kb_dir: str, libs: List[str],
    jaccard_vs_baseline: Optional[float] = None,
) -> Dict[str, Any]:
    """Row for a compact compare CSV."""
    row = build_csv_row(
        strategy=strategy, backend=backend, model=model, retriever=retriever,
        cfg=cfg, repo_filter=repo_filter, latency_ms=latency_ms,
        hits=hits, question=question, kb_dir=kb_dir, libs=libs,
    )
    # Merge lightweight metrics (robust default)
    m = compute_final_metrics(hits, repo_filter=repo_filter) or _metrics_empty()
    row.update({
        "unique_repos": m["unique_repos"],
        "diversity": m["diversity"],
        "mean_score": m["mean_score"],
        "max_score": m["max_score"],
        "repo_focus_rate": m["repo_focus_rate"],
        "jaccard_vs_baseline": (None if jaccard_vs_baseline is None else round(float(jaccard_vs_baseline), 4)),
    })
    return row


def save_compare_csv(csv_dir: str, rows: List[Dict[str, Any]], filename: str = "multihop_compare.csv") -> str:
    """Append multiple rows to a comparison CSV with header."""
    ensure_dir(csv_dir)
    path = os.path.join(csv_dir, filename)
    new_file = not os.path.isfile(path)
    fieldnames = CSV_COLUMNS + [
        "unique_repos","diversity","mean_score","max_score","repo_focus_rate","jaccard_vs_baseline"
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            safe = {k: r.get(k, "") for k in fieldnames}
            w.writerow(safe)
    return path


def per_query_hits_dict(
    queries: List[str],
    provider: Any,
    top_k: int,
    filters: Optional[Dict[str, Union[str, List[str]]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {query: compact_top_hits[]} honoring the provided filters."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for q in queries or []:
        hits = provider.search(q, k=top_k, filters=filters)  # previously ignored filters
        out[q] = _hits_compact(hits, limit=top_k)
    return out


def save_topk_impact_csv(
    csv_dir: str,
    *,
    strategy: str, backend: str, model: str, retriever: str,
    top_k: int, question: str, repo_filter: Optional[str],
    subqueries: List[str], per_query_hits: Dict[str, List[Dict[str, Any]]],
    pack_hits: List[Any], latency_ms: float, kb_dir: str, libs: List[str],
    filename: str = "multihop_topk_impact.csv",
) -> str:
    """Save one row capturing the impact of top_k on queries and per-query hits."""
    ensure_dir(csv_dir)
    path = os.path.join(csv_dir, filename)
    new_file = not os.path.isfile(path)
    fieldnames = [
        "timestamp_iso","strategy","backend","model","retriever","top_k","question","repo_filter",
        "subqueries","per_query_hits","final_top_hits","latency_ms","kb_dir","libs"
    ]
    import datetime as _dt
    row = {
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "backend": backend,
        "model": model,
        "retriever": retriever,
        "top_k": top_k,
        "question": question,
        "repo_filter": repo_filter or "",
        "subqueries": " || ".join(subqueries or []),
        "per_query_hits": json.dumps(per_query_hits, ensure_ascii=False),
        "final_top_hits": json.dumps(_hits_compact(pack_hits, limit=top_k), ensure_ascii=False),
        "latency_ms": round(float(latency_ms), 0),
        "kb_dir": kb_dir,
        "libs": json.dumps(libs, ensure_ascii=False),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return path


def discover_libs(kb_dir: str) -> List[str]:
    """Return a sorted list of lib names by scanning kb_<lib>.json under kb_dir and subfolders."""
    pattern = os.path.join(kb_dir, "**", "kb_*.json")
    libs = set()
    for path in glob.glob(pattern, recursive=True):
        base = os.path.basename(path)
        m = re.match(r"kb_(.+)\.json$", base)
        if m:
            libs.add(m.group(1))
    return sorted(libs)


# --- Library filters helper ---------------------------------------------------

def filters_for(libs: Optional[Union[str, List[str]]] = None) -> Optional[Dict[str, Union[str, List[str]]]]:
    """
    Build a filters dict compatible with providers: None | {"repo": "lib"} | {"repo": ["libA","libB"]}.
    - libs=None         -> None (open-world: all indexed libs)
    - libs=str          -> {"repo": "that_lib"}
    - libs=list[str]    -> {"repo": [...]}  (filter to this subset)
    """
    if libs is None:
        return None
    if isinstance(libs, str):
        libs = libs.strip()
        return {"repo": libs} if libs else None
    libs = [str(x).strip() for x in libs if str(x).strip()]
    return {"repo": libs} if libs else None


# --- Instruction selection switch --------------------------------------------

Selection = Union[str, int, List[int]]  # "custom" | "all" | 5 | "idx:12" | [0,2,5]

def resolve_instructions(
    selection: Selection,
    *,
    dataset=None,                  # HuggingFace Dataset or list[dict] with "instruction"
    custom: List[str] | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Resolve user's single-parameter selection into a list of questions.
    Returns: {"mode": "custom"|"dataset", "items": [{"id": str, "instruction": str, "repo_full_name": str|None}]}
    Accepted 'selection':
      - "custom"      -> use the 'custom' list provided
      - "all"         -> use all dataset rows
      - int (e.g., 5) -> sample N random rows from dataset
      - "idx:12"      -> pick a single dataset row by index 12
      - [1,5,7]       -> pick those exact indices from dataset
    """
    def _mk_item(i: int, ex: dict):
        rid = f"{ex.get('repo_full_name','unk')}##{i}"
        return {"id": rid, "instruction": ex.get("instruction", ""), "repo_full_name": ex.get("repo_full_name")}

    if isinstance(selection, str) and selection.lower() == "custom":
        if not custom:
            raise ValueError("selection='custom' but no custom questions provided")
        return {"mode": "custom", "items": [{"id": f"custom##{i}", "instruction": q, "repo_full_name": None} for i, q in enumerate(custom)]}

    # everything below requires a dataset
    if dataset is None:
        raise ValueError("Dataset is required for non-'custom' selections")

    # normalize dataset to list of dicts
    rows = list(dataset)  # HF Dataset supports iteration

    if isinstance(selection, str) and selection.lower() == "all":
        return {"mode": "dataset", "items": [_mk_item(i, ex) for i, ex in enumerate(rows)]}

    if isinstance(selection, int) and selection > 0:
        rnd = random.Random(seed)
        if selection > len(rows):
            selection = len(rows)
        idxs = rnd.sample(range(len(rows)), selection)
        return {"mode": "dataset", "items": [_mk_item(i, rows[i]) for i in idxs]}

    if isinstance(selection, str) and selection.lower().startswith("idx:"):
        idx = int(selection.split(":", 1)[1])
        if not (0 <= idx < len(rows)):
            raise IndexError(f"idx out of range: {idx} / {len(rows)}")
        return {"mode": "dataset", "items": [_mk_item(idx, rows[idx])]}

    if isinstance(selection, list) and all(isinstance(i, int) for i in selection):
        items = []
        for i in selection:
            if 0 <= i < len(rows):
                items.append(_mk_item(i, rows[i]))
        return {"mode": "dataset", "items": items}

    raise ValueError(f"Unsupported selection: {selection!r}")

# --- Snippet extraction (final retrieval set -> List[str]) -------------------
def snippets_from_hits(
    hits: List[Any],
    provider: Any,
    *,
    dedupe: bool = True,
) -> List[str]:
    """
    Convert the final retrieval set (hits) into plain snippet texts ready for prompting.
    - Preserves the order of `hits`.
    - Dedupes by (repo, doc_id) by default (set dedupe=False to keep duplicates).
    - If a hit has no 'text', tries provider.get_doc(repo, doc_id).

    Returns: List[str] of snippet texts. Intended to be joined by your prompt builder.
    """
    out: List[str] = []
    if not hits:
        return out

    seen = set()
    for h in hits:
        if isinstance(h, dict):
            repo = h.get("repo")
            doc_id = h.get("doc_id")
            txt = h.get("text")
        else:
            repo = getattr(h, "repo", None)
            doc_id = getattr(h, "doc_id", None)
            txt = getattr(h, "text", None)

        key = (repo, doc_id)
        if dedupe and key in seen:
            continue
        if dedupe:
            seen.add(key)

        if not txt and hasattr(provider, "get_doc"):
            try:
                txt = provider.get_doc(repo, doc_id)
            except Exception:
                txt = None
        if not txt:
            continue

        out.append(txt)
    return out
