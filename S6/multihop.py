# S6/multihop.py
from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional

from models.retrieval_providers import RetrievalProvider, Doc
from models import llm_clients as llm


# =========================
# Config
# =========================

@dataclass
class MultiHopConfig:
    # Strategy
    strategy: str = "decomposition_first"      # or "iterative_refine"
    k_sub: int = 5
    k_final: int = 10
    max_hops: int = 3
    decomposer_max_tokens: int = 64
    planner_max_tokens: int = 48
    planner_ctx_docs: int = 2
    planner_ctx_chars: int = 300

    # Target repo name for coverage calculations
    target_repo: Optional[str] = None

    # LLM selection
    llm_backend: str = "openrouter"            # openrouter | gemini | local
    llm_model: Optional[str] = None            # model id or local folder
    # Back-compat: if only openrouter_model is set, use that
    openrouter_model: Optional[str] = None

    # Planner/decomposer modes
    decomposer_mode: str = "remote"            # remote | stub
    planner_mode: str = "remote"               # remote | stub

    # Logging & caching (optional)
    cache_dir: Optional[str] = None
    log_dir: str = "results/multihop_logs"
    seed: int = 42


# =========================
# Helpers
# =========================

def _ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _safe_cache_key(*parts) -> str:
    """
    Build a filesystem-safe cache key from arbitrary parts.
    Replaces slashes and non-filename chars with underscores.
    """
    s = "_".join(map(str, parts))
    s = s.replace(os.sep, "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def _cache_get(cache_dir: Optional[str], key: str) -> Optional[str]:
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, key + ".txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def _cache_put(cache_dir: Optional[str], key: str, val: str) -> None:
    if not cache_dir:
        return
    path = os.path.join(cache_dir, key + ".txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(val or "")

def _llm_for(cfg: MultiHopConfig) -> llm.BaseLLMClient:
    # Prefer new fields; fall back to openrouter_model for back-compat
    model = cfg.llm_model or cfg.openrouter_model
    backend = (cfg.llm_backend or ("openrouter" if cfg.openrouter_model else "openrouter")).lower()
    return llm.get_client(backend, model)

def _rrf_merge(results_per_subquery: List[List[Doc]], k_final: int, K: int = 60) -> List[Doc]:
    """
    Reciprocal Rank Fusion over multiple subquery result lists.
    Exposed as a top-level symbol so bench.runner can monkey-patch it with alternative fusions.
    """
    from collections import defaultdict
    fused = defaultdict(float)
    best_by_key: Dict[tuple, Doc] = {}
    for docs in results_per_subquery:
        ranked = sorted(docs, key=lambda d: d.score, reverse=True)
        for rank, d in enumerate(ranked, start=1):
            key = (d.repo, d.doc_id)
            fused[key] += 1.0 / (K + rank)
            if key not in best_by_key or d.score > best_by_key[key].score:
                best_by_key[key] = d
    ranked_keys = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k_final]
    return [best_by_key[key] for key, _ in ranked_keys]


# =========================
# Decomposition-first
# =========================

def _decompose(question: str, cfg: MultiHopConfig) -> List[str]:
    if cfg.decomposer_mode == "stub":
        # Simple deterministic decomposition
        return [
            question,
            "core function(s) implementing this behavior",
            "supporting utilities or constants",
            "unit or integration examples",
        ][:max(2, min(cfg.k_sub, 4))]

    client = _llm_for(cfg)

    # Stable cache key (md5 over the prompt text)
    cache_key = None
    if cfg.cache_dir:
        model_id = (cfg.llm_model or cfg.openrouter_model or "none")
        h = hashlib.md5(question.encode("utf-8")).hexdigest()
        cache_key = _safe_cache_key("decomp", cfg.llm_backend, model_id, h)
        cached = _cache_get(cfg.cache_dir, cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass  # ignore bad cache

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior code-search planner for multi-hop retrieval across BM25, "
                "KNN, and hybrid retrievers. Turn a software code-generation request into "
                "high-signal, non-redundant search subqueries. Prioritize exact library/package "
                "names and aliases, API artifacts (functions, classes, methods), config/CLI keys, "
                "error strings, file/dir hints (e.g., *.py, src/, tests/), and language hints "
                "(e.g., python). You may combine terms with AND/OR and use quoted phrases for "
                "multi-word identifiers. Keep each line concise (≤14 tokens). "
                "OUTPUT FORMAT: a plain bullet list using '- ' with one subquery per line. "
                "No explanations, numbering, JSON, code fences, or extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Return 3–5 distinct subqueries as a bullet list only."
            ),
        },
    ]
    chat_kwargs = {"temperature": 0.0}
    if cfg.decomposer_max_tokens is not None:
        chat_kwargs["max_tokens"] = cfg.decomposer_max_tokens
    out = client.chat(messages, **chat_kwargs)
    # Parse bullet lines
    subs = [s.strip("-• ").strip() for s in out.splitlines() if s.strip()]
    subs = [s for s in subs if len(s) > 2][:max(2, cfg.k_sub)]

    if cache_key:
        try:
            _cache_put(cfg.cache_dir, cache_key, json.dumps(subs))
        except Exception:
            pass

    return subs or [question]

def _run_decomposition_first(
    question: str,
    provider: RetrievalProvider,
    cfg: MultiHopConfig,
    filters: Optional[Dict] = None,
) -> Dict:
    t0 = time.time()
    subqueries = _decompose(question, cfg)

    results_per_sub: List[List[Doc]] = []
    hops_log = []
    for sq in subqueries:
        docs = provider.search(sq, k=cfg.k_final, filters=filters)
        hops_log.append({"type": "hop", "subquery": sq, "docs": [d.__dict__ for d in docs]})
        results_per_sub.append(docs)

    fused = _rrf_merge(results_per_sub, cfg.k_final)
    latency_ms = int((time.time() - t0) * 1000)
    cov = _coverage(fused, cfg.target_repo, cfg.k_final)

    return {
        "strategy": "decomposition_first",
        "subqueries": subqueries,
        "hits": [d.__dict__ for d in fused],
        "coverage": cov,
        "latency_ms": latency_ms,
        "hops": hops_log,
    }


# =========================
# Iterative refine
# =========================

def _refine_queries_iteratively(question: str, cfg: MultiHopConfig, prev_docs: List[Doc]) -> Optional[str]:
    if cfg.planner_mode == "stub":
        if not prev_docs:
            return None
        txt = prev_docs[0].text[:min(320, getattr(cfg, "planner_ctx_chars", 300))]
        return f"Refine: focus on implementation details mentioned here: {txt}"

    client = _llm_for(cfg)
    prompt = (
        "Using the original question and the current top document excerpts, write ONE "
        "refined follow-up search query suitable for BM25, KNN, and hybrid retrieval. "
        "Include at least one exact identifier/path/error string confirmed by the excerpts; "
        "add specific language/file hints if clear (e.g., python, *.py, tests/). "
        "Remove vague or generic words; prefer exact API/class/function names; avoid redundancy; "
        "keep it concise (≤16 tokens). If the excerpts add no concrete detail, compress the "
        "original question to its 3–8 strongest keywords. Output the query only—no quotes, "
        "no prose, no bullets, no code fences."
    )
    n_docs = max(1, int(getattr(cfg, "planner_ctx_docs", 2)))
    n_chars = max(80, int(getattr(cfg, "planner_ctx_chars", 300)))
    context_docs = [d.text[:n_chars] for d in prev_docs[:n_docs]]
    context = "\n\n".join(context_docs)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question: {question}\nContext:\n{context}\nOne refined query:"},
    ]
    chat_kwargs = {"temperature": 0.0}
    if cfg.planner_max_tokens is not None:
        chat_kwargs["max_tokens"] = cfg.planner_max_tokens
    out = client.chat(messages, **chat_kwargs)
    return out.strip().splitlines()[0] if out else None

def _run_iterative_refine(
    question: str,
    provider: RetrievalProvider,
    cfg: MultiHopConfig,
    filters: Optional[Dict] = None,
) -> Dict:
    t0 = time.time()
    subqs: List[str] = [question]
    all_docs: List[Doc] = []

    # hop 0
    docs0 = provider.search(question, k=cfg.k_final, filters=filters)
    all_docs.extend(docs0)
    hops_log = [{"type": "hop", "subquery": question, "docs": [d.__dict__ for d in docs0]}]

    for _ in range(1, cfg.max_hops):
        refined = _refine_queries_iteratively(question, cfg, docs0)
        if not refined:
            break
        subqs.append(refined)
        docsH = provider.search(refined, k=cfg.k_final, filters=filters)
        hops_log.append({"type": "hop", "subquery": refined, "docs": [d.__dict__ for d in docsH]})
        # union + score-based re-sort (de-dup later)
        all_docs.extend(docsH)
        # early stop if no new docs are coming
        if not docsH:
            break
        docs0 = docsH

    # de-dup by (repo, doc_id) and keep best score
    by_key: Dict[tuple, Doc] = {}
    for d in all_docs:
        key = (d.repo, d.doc_id)
        if key not in by_key or d.score > by_key[key].score:
            by_key[key] = d
    ranked = sorted(by_key.values(), key=lambda d: d.score, reverse=True)[:cfg.k_final]

    latency_ms = int((time.time() - t0) * 1000)
    cov = _coverage(ranked, cfg.target_repo, cfg.k_final)

    return {
        "strategy": "iterative_refine",
        "subqueries": subqs,
        "hits": [d.__dict__ for d in ranked],
        "coverage": cov,
        "latency_ms": latency_ms,
        "hops": hops_log,
    }


# =========================
# Shared
# =========================

def _coverage(hits: List[Doc], target_repo: Optional[str], k: int) -> Dict:
    if not target_repo:
        return {}
    k = min(k, len(hits))
    good = sum(1 for d in hits[:k] if d.repo == target_repo)
    pct = 100.0 * good / max(1, k)
    return {"target_repo": target_repo, "pct_from_target_repo": pct, "k": k}

def run_hops(
    question: str,
    k: int,
    strategy: str,
    provider: RetrievalProvider,
    cfg: MultiHopConfig,
    filters: Optional[Dict] = None,
) -> Dict:
    """
    Public entrypoint used by the notebook/runner.
    If k is provided and differs, it overrides cfg.k_final for this call.
    """
    if k and k != cfg.k_final:
        # local override (non-destructive)
        cfg = MultiHopConfig(**{**cfg.__dict__, "k_final": k})

    if strategy == "decomposition_first":
        # pass filters by name to avoid positional mistakes
        return _run_decomposition_first(question, provider, cfg, filters=filters)
    elif strategy == "iterative_refine":
        return _run_iterative_refine(question, provider, cfg, filters=filters)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
