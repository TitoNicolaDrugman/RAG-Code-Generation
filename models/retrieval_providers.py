from __future__ import annotations
"""
Retrieval providers for BM25 (and factory hooks for KNN/Hybrid).

Public API (unchanged):
- Doc                (dataclass)
- RetrievalProvider  (Protocol)
- BM25Provider       (class)
- make_provider      (factory)

Behavioral notes (unchanged):
- BM25Provider.search(q, k, filters) respects the caller's `k` and pre-pools
  max(k*5, 50) before filter/truncate.
- `filters` may include {"repo": "<name>"}; this version also safely accepts
  {"repo": ["libA","libB"]} without changing external behavior if you pass a str.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple
import glob
import json
import logging
import os
import re

try:
    from rank_bm25 import BM25Okapi
except Exception as e:  # pragma: no cover
    raise RuntimeError("rank_bm25 is required. Add it to requirements.txt") from e

# --- Logging -----------------------------------------------------------------
logger = logging.getLogger("retrieval_providers")
if not logger.handlers:  # avoid duplicate handlers in notebooks
    import sys
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# =========================
# Data types
# =========================

@dataclass(frozen=True)
class Doc:
    """A retrieved snippet from the KB."""
    repo: str
    doc_id: int
    text: str
    score: float


class RetrievalProvider(Protocol):
    def search(self, q: str, k: int, filters: Optional[Dict] = None) -> List[Doc]:  # pragma: no cover
        ...


# =========================
# Utilities (internal)
# =========================

def _default_tokenizer(text: str) -> List[str]:
    """Lowercase and split on non-alnum/_ characters."""
    text = (text or "").lower()
    return [t for t in re.split(r"[^a-z0-9_]+", text) if t]


def _normalize_repo_filter(filters: Optional[Dict]) -> Optional[Set[str]]:
    """
    Accepts:
      filters=None
      filters={"repo": "lib"}
      filters={"repo": ["libA", "libB"]}
    Returns a set of repo names or None.
    """
    if not filters:
        return None
    value = filters.get("repo")
    if not value:
        return None
    if isinstance(value, str):
        v = value.strip()
        return {v} if v else None
    try:
        vals = {str(x).strip() for x in value if str(x).strip()}
        return vals or None
    except Exception:
        return None


# =========================
# BM25 Provider
# =========================

class BM25Provider:
    """
    Lightweight BM25 wrapper over your KB jsons.

    Use either:
      - local_kb_dir="temp_download" (or ".../knowledge_bases_prod") to read local files, OR
      - omit local_kb_dir to fall back to utils.kb_manager (may fetch from GitHub if missing).
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        library_filter: Optional[List[str]] = None,
        overwrite_kb_download: bool = False,
        min_len: int = 1,
        local_kb_dir: Optional[str] = None,   # ✅ OPTIONAL local source
    ) -> None:
        # fallback path (still available)
        from utils.kb_manager import list_available_kbs, load_kb  # lazy import to avoid cycles
        self._list_available_kbs = list_available_kbs
        self._load_kb = load_kb

        self._bm25: Optional[BM25Okapi] = None
        self._docs: List[str] = []
        self._mapping: List[Tuple[str, int]] = []           # (repo, doc_id)
        self._key_to_text: Dict[Tuple[str, int], str] = {}  # for O(1) get_doc
        self._libs: List[str] = []                          # for introspection

        self._library_filter = library_filter
        self._overwrite = overwrite_kb_download
        self._min_len = min_len
        self._tokenizer = tokenizer or _default_tokenizer
        self._local_kb_dir = local_kb_dir

    # ---------------- Internal helpers -----------------

    def _iter_local_kb_files(self) -> Dict[str, str]:
        """
        Discover KB jsons recursively:
          <local_kb_dir>/**/kb_<lib>.json  OR  <local_kb_dir>/kb_<lib>.json
        Returns: {lib_name: file_path}
        """
        if not self._local_kb_dir:
            return {}
        patterns = [
            os.path.join(self._local_kb_dir, "kb_*.json"),
            os.path.join(self._local_kb_dir, "**", "kb_*.json"),
        ]
        files: List[str] = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        lib_to_path: Dict[str, str] = {}
        for path in files:
            base = os.path.basename(path)
            m = re.match(r"kb_(.+)\.json$", base)
            if m:
                lib_to_path[m.group(1)] = path
        return lib_to_path

    def _append_snippet(
        self, lib: str, idx: int, snippet: str, tokenized_corpus: List[List[str]]
    ) -> None:
        """Validate+add a single snippet to internal arrays and tokenized corpus."""
        if not isinstance(snippet, str):
            return
        s = snippet.strip()
        if len(s) < self._min_len:
            return
        toks = self._tokenizer(s)
        if not toks:
            return
        tokenized_corpus.append(toks)
        self._mapping.append((lib, idx))
        self._docs.append(s)
        self._key_to_text[(lib, idx)] = s

    def _build_corpus(self) -> List[List[str]]:
        tokenized_corpus: List[List[str]] = []

        # ---------- LOCAL: prefer local_kb_dir if provided ----------
        local = self._iter_local_kb_files()
        if self._local_kb_dir:
            if not local:
                raise RuntimeError(f"No local KBs found under: {self._local_kb_dir}")
            # optional subset filter
            if self._library_filter:
                keep = set(self._library_filter)
                local = {lib: p for lib, p in local.items() if lib in keep}
            if not local:
                raise RuntimeError(f"Library filter {self._library_filter} matched no local KBs.")
            self._libs = sorted(local.keys())

            for lib, path in sorted(local.items()):
                with open(path, "r", encoding="utf-8") as f:
                    kb_json = json.load(f)
                for idx, snippet in enumerate(kb_json):
                    self._append_snippet(lib, idx, snippet, tokenized_corpus)

            if not tokenized_corpus:
                raise RuntimeError("Empty tokenized corpus from local KBs.")
            logger.info("BM25 indexed docs: %d across %d libs (local)", len(self._docs), len(self._libs))
            return tokenized_corpus

        # ---------- FALLBACK: kb_manager path (may download) ----------
        kb_entries = [
            x for x in self._list_available_kbs()
            if isinstance(x, dict) and x.get("name", "").endswith(".json")
        ]
        if self._library_filter:
            keys = set(self._library_filter)
            kb_entries = [x for x in kb_entries if x["name"][3:-5] in keys]
        if not kb_entries:
            raise RuntimeError("No KBs available to index via kb_manager().")

        libs_seen = set()
        for entry in kb_entries:
            lib = entry["name"][3:-5]
            kb_json, _ = self._load_kb(lib, overwrite=self._overwrite)
            if not kb_json:
                logger.warning("KB '%s' is empty or missing; skipping.", lib)
                continue
            libs_seen.add(lib)
            for idx, snippet in enumerate(kb_json):
                self._append_snippet(lib, idx, snippet, tokenized_corpus)

        self._libs = sorted(libs_seen)
        if not tokenized_corpus:
            raise RuntimeError("Empty tokenized corpus; verify KB creation and tokenizer.")
        logger.info("BM25 indexed docs: %d across %d libs (kb_manager)", len(self._docs), len(self._libs))
        return tokenized_corpus

    # ---------------- Public API -----------------

    def index(self) -> "BM25Provider":
        tokenized_corpus = self._build_corpus()
        self._bm25 = BM25Okapi(tokenized_corpus)
        return self

    def search(self, q: str, k: int = 10, filters: Optional[Dict] = None) -> List[Doc]:
        """
        Search contract (unchanged):
          - pre-pool max(k*5, 50)
          - apply optional repo filter
          - return at most k results
        """
        assert self._bm25 is not None, "Call index() before search()."
        toks = self._tokenizer(q) or []
        scores = self._bm25.get_scores(toks)

        topn = max(k * 5, 50)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topn]

        repo_allow = _normalize_repo_filter(filters)

        results: List[Doc] = []
        for i in idxs:
            repo, doc_id = self._mapping[i]
            if repo_allow and repo not in repo_allow:
                continue
            results.append(Doc(repo=repo, doc_id=doc_id, text=self._docs[i], score=float(scores[i])))
            if len(results) >= k:
                break
        return results

    # Introspection helpers
    def libs_indexed(self) -> List[str]:
        return self._libs[:]

    def get_doc(self, repo: str, doc_id: int) -> Optional[str]:
        return self._key_to_text.get((repo, doc_id))


# --------------------------------------------------------------------------
# Factory to build the desired retriever on-the-fly
# --------------------------------------------------------------------------

class _FilterProxy:
    """
    Wrap any provider so that repo filtering works even if the underlying
    provider does not accept 'filters'. Assumes results are Doc objects.
    """
    def __init__(self, base: Any):
        self._base = base

    def search(self, q: str, k: int, filters: Optional[Dict] = None) -> List[Doc]:
        # Prefer provider's own filters signature if available
        try:
            docs = self._base.search(q, k, filters)  # type: ignore[attr-defined]
        except TypeError:
            docs = self._base.search(q, k)          # type: ignore[attr-defined]

        # Apply repo filter here if needed
        allow = _normalize_repo_filter(filters)
        if allow:
            docs = [d for d in docs if getattr(d, "repo", None) in allow][:k]
        return docs

    def __getattr__(self, name: str) -> Any:  # passthrough helpers/attrs
        return getattr(self._base, name)


def make_provider(kind: str, **kwargs: Any) -> RetrievalProvider:
    """
    kind: "bm25" | "knn" | "hybrid"
    Note: KNN/Hybrid are loaded lazily and wrapped so repo filters work uniformly.
    """
    kind = (kind or "bm25").lower()

    if kind == "bm25":
        p = BM25Provider(
            tokenizer=kwargs.get("tokenizer"),
            library_filter=kwargs.get("library_filter"),
            overwrite_kb_download=kwargs.get("overwrite_kb_download", False),
            min_len=kwargs.get("min_len", 1),
            local_kb_dir=kwargs.get("local_kb_dir"),
        ).index()
        return p

    if kind == "knn":
        # Expect: retrievers/KNN.py exposing class Provider with .search(q,k[,filters])
        try:
            from retrievers.KNN import Provider as KNNProvider  # type: ignore
        except Exception as e:
            raise NotImplementedError(
                "KNN retriever not available. Ensure retrievers/KNN.py with class Provider exists."
            ) from e

        knn = KNNProvider(
            index_dir=kwargs.get("index_dir"),
            index_type=kwargs.get("index_type", "flat"),
            ef_search=kwargs.get("ef_search", 128),
            nprobe=kwargs.get("nprobe", 16),
            embedder=kwargs.get("embedder", None),
        )
        return _FilterProxy(knn)

    if kind == "hybrid":
        # Expect: retrievers/HYBRID.py exposing class Provider with .search(q,k[,filters])
        try:
            from retrievers.HYBRID import Provider as HybridProvider  # type: ignore
        except Exception as e:
            raise NotImplementedError(
                "HYBRID retriever not available. Ensure retrievers/HYBRID.py with class Provider exists."
            ) from e

        # Build sub-providers if not supplied
        bm25 = kwargs.get("bm25")
        if bm25 is None:
            bm25 = make_provider("bm25", **kwargs.get("bm25_kwargs", {}))

        knn = kwargs.get("knn")
        if knn is None:
            try:
                from retrievers.KNN import Provider as KNNProvider  # type: ignore
            except Exception as e:
                raise NotImplementedError(
                    "HYBRID requires KNN. Ensure retrievers/KNN.py is present."
                ) from e
            kk = kwargs.get("knn_kwargs", {})
            knn = KNNProvider(
                index_dir=kk.get("index_dir"),
                index_type=kk.get("index_type", "flat"),
                ef_search=kk.get("ef_search", 128),
                nprobe=kk.get("nprobe", 16),
                embedder=kk.get("embedder", None),
            )

        hybrid = HybridProvider(
            bm25=bm25,
            knn=knn,
            fusion=kwargs.get("fusion", "rrf"),
            alpha=kwargs.get("alpha", 0.5),
            rrf_k=kwargs.get("rrf_k", 60),
        )
        return _FilterProxy(hybrid)

    raise ValueError(f"Unknown retriever kind: {kind}")
