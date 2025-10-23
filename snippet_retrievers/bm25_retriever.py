# snippet_retrievers/bm25_retriever.py

from __future__ import annotations
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

from .base import SnippetRetriever
from .utils import simple_tokenize, build_corpus, write_report_multi, CorpusItem, ranks_desc
from .embedder import CodeEmbedder, DEFAULT_MODEL


class BM25Retriever(SnippetRetriever):
    """
    BM25-based snippet retriever that logs bm25, cosine, and rrf for chosen snippets.
    """
    def __init__(self, *, kb_root: str | None, kb_file: str | None,
                 k1: float = 1.5, b: float = 0.75,
                 model_name: str = DEFAULT_MODEL, batch_size: int = 32,
                 rrf_k: int = 60):
        super().__init__(kb_root=kb_root, kb_file=kb_file)
        self.rrf_k = int(rrf_k)

        items: List[CorpusItem] = build_corpus(kb_root=kb_root, kb_file=kb_file)
        self.items = items
        self.docs_tok: List[List[str]] = [simple_tokenize(it.text.lower()) for it in items] # Drop empties consistently across items and tokens
        nz_items, nz_tokens = [], []
        for it, toks in zip(self.items, self.docs_tok):
            if toks:
                nz_items.append(it)
                nz_tokens.append(toks)
        self.items, self.docs_tok = nz_items, nz_tokens
        if not self.items:
            raise RuntimeError("Empty corpus: no tokenizable snippets were found.")

        self.bm25 = BM25Okapi(self.docs_tok, k1=k1, b=b)
        self._bm25_params = {"k1": k1, "b": b, "num_docs": len(self.items)}

        # Prepare embedder+cache to compute cosine scores for reporting
        self.embedder = CodeEmbedder(model_name=model_name, batch_size=batch_size)
        
        # --- FIX IS HERE ---
        # We need embeddings aligned to self.items
        # OLD, BUGGY LINE: self.mat, _ = self.embedder.get_or_build_embeddings(self.items)
        # We need embeddings aligned to self.items
        snippets = [it.text for it in self.items]
        self.mat, _ = self.embedder.get_or_build_embeddings(snippets, kb_file=self.kb_file)

        # --- END OF FIX ---

    def _rank(self, q: str, k: int) -> List[Tuple[int, float]]:
        if not q or k <= 0:
            return []
        q_tokens = simple_tokenize(q.lower())
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
        return ranked[: min(k, len(ranked))]

    def retrieve(self, q: str, k: int) -> List[str]:
        ranked = self._rank(q, k)

        # Compute full-corpus scores for RRF
        q_tokens = simple_tokenize(q.lower())
        bm25_all = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.items), dtype=np.float32)

        qv = self.embedder.embed_query(q) if ranked else np.zeros(768, dtype=np.float32)
        cos_all = (self.mat @ qv) if ranked else np.zeros(len(self.items), dtype=np.float32)

        # RRF
        bm25_rank = ranks_desc(bm25_all)
        cos_rank  = ranks_desc(cos_all)
        k0 = float(self.rrf_k)
        rrf_all = (1.0 / (k0 + bm25_rank)) + (1.0 / (k0 + cos_rank))

        results_for_report, snippets_only = [], []
        for rank_idx, (doc_i, bm25_score) in enumerate(ranked, start=1):
            item = self.items[doc_i]
            scores_dict = {
                "bm25":  float(bm25_score),
                "cosine": float(cos_all[doc_i]),
                "rrf":   float(rrf_all[doc_i]),
            }
            results_for_report.append((rank_idx, scores_dict, item))
            snippets_only.append(item.text)

        write_report_multi(
            metric_name="bm25",
            q=q, k=k,
            params=self._bm25_params | {"model": self.embedder.model_name, "rrf_k": self.rrf_k},
            results=results_for_report,
        )
        return snippets_only