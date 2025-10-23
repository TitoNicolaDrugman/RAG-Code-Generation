from __future__ import annotations
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from .base import SnippetRetriever
from .utils import CorpusItem, build_corpus, simple_tokenize, write_report_multi, ranks_desc
from .embedder import CodeEmbedder, DEFAULT_MODEL

"""
def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    
    #Convert a score vector into 1-based ranks for descending sort.
    #Higher score -> smaller rank number (1 is best).
    
    # argsort descending:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    # ranks[doc_index] = rank_position
    ranks[order] = np.arange(1, len(scores) + 1, dtype=order.dtype)
    return ranks
"""
class HybridRRFRetriever(SnippetRetriever):
    """
    Hybrid retriever with Reciprocal Rank Fusion (RRF) of BM25 and Cosine.
    RRF(d) = sum_s 1 / (rrf_k + rank_s(d))  ; typically rrf_k in [50,100], default 60.
    """

    def __init__(
        self,
        *,
        kb_root: str | None,
        kb_file: str | None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
    ):
        super().__init__(kb_root=kb_root, kb_file=kb_file)
        self.rrf_k = int(rrf_k)

        # Build corpus
        items: List[CorpusItem] = build_corpus(kb_root=kb_root, kb_file=kb_file)

        # BM25 tokens (lower-cased for term match)
        docs_tok: List[List[str]] = [simple_tokenize(it.text.lower()) for it in items]

        # Drop empties consistently across items/tokens
        nz_items, nz_tokens = [], []
        for it, toks in zip(items, docs_tok):
            if toks:
                nz_items.append(it)
                nz_tokens.append(toks)

        self.items: List[CorpusItem] = nz_items
        self.docs_tok: List[List[str]] = nz_tokens
        if not self.items:
            raise RuntimeError("Empty corpus after tokenization.")

        # BM25 index
        self.bm25 = BM25Okapi(self.docs_tok, k1=bm25_k1, b=bm25_b)
        self._bm25_params = {"k1": bm25_k1, "b": bm25_b, "num_docs": len(self.items)}

        # Embeddings for cosine
        self.embedder = CodeEmbedder(model_name=model_name, batch_size=batch_size)
        kb_paths = [it.kb_path for it in self.items]
        snippets = [it.text    for it in self.items]
        self.mat, _ = self.embedder.get_or_build_embeddings(snippets, kb_file=self.kb_file)
        #self.mat, _ = self.embedder.get_or_build_embeddings(self.items)  # (N,768) L2-normalized

    def _rank_rrf(self, q: str, k: int) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          top_idx: list of doc indices sorted by RRF (len <= k)
          bm25_scores: (N,)
          cos_scores:  (N,)
          rrf_scores:  (N,)
        """
        if not q or k <= 0 or len(self.items) == 0:
            return [], np.zeros(0), np.zeros(0), np.zeros(0)

        # BM25 scores
        q_tokens = simple_tokenize(q.lower())
        bm25_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.items), dtype=np.float32)

        # Cosine scores
        qv = self.embedder.embed_query(q)  # (768,)
        cos_scores = self.mat @ qv  # cosine = dot since vectors are L2-normalized

        # Convert to ranks (1 is best)
        bm25_rank = ranks_desc(bm25_scores)
        cos_rank  = ranks_desc(cos_scores)

        # RRF fusion
        k0 = float(self.rrf_k)
        rrf_scores = (1.0 / (k0 + bm25_rank)) + (1.0 / (k0 + cos_rank))

        # Top-k by RRF
        order = np.argsort(-rrf_scores, kind="mergesort")[: min(k, len(self.items))]
        return order.tolist(), bm25_scores, cos_scores, rrf_scores

    def retrieve(self, q: str, k: int) -> List[str]:
        top_idx, bm25_scores, cos_scores, rrf_scores = self._rank_rrf(q, k)

        results_for_report = []
        snippets_only: List[str] = []
        for rank_idx, doc_i in enumerate(top_idx, start=1):
            item = self.items[doc_i]
            scores_dict = {
                "bm25":  float(bm25_scores[doc_i]),
                "cosine": float(cos_scores[doc_i]),
                "rrf":   float(rrf_scores[doc_i]),
            }
            results_for_report.append((rank_idx, scores_dict, item))
            snippets_only.append(item.text)

        write_report_multi(
            metric_name="hybrid_rrf",
            q=q,
            k=k,
            params=self._bm25_params | {"model": self.embedder.model_name, "rrf_k": self.rrf_k},
            results=results_for_report,
        )
        return snippets_only
