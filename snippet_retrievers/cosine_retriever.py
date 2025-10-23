# snippet_retrievers/cosine_retriever.py
from __future__ import annotations
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi

from .base import SnippetRetriever
from .utils import build_corpus, write_report_multi, simple_tokenize, CorpusItem
from .embedder import CodeEmbedder, DEFAULT_MODEL

class CosineRetriever(SnippetRetriever):
    def __init__(
        self,
        *,
        kb_root: str | None,
        kb_file: str | None,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        pooling: str = "mean",
        remove_top_pcs: int = 3,
        max_length: int = 512,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        **_
    ):
        super().__init__(kb_root=kb_root, kb_file=kb_file)
        self.embedder = CodeEmbedder(
            model_name=model_name,
            max_length=max_length,
            pooling=pooling,
            remove_top_pcs=remove_top_pcs,
            batch_size=batch_size
        )
        self.rrf_k = int(rrf_k)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b  = float(bm25_b)

    def retrieve(self, q: str, k: int) -> List[str]:
        # Load corpus
        items: List[CorpusItem] = build_corpus(kb_root=self.kb_root, kb_file=self.kb_file)
        if not items or k <= 0:
            write_report_multi(metric_name="cosine", q=q, k=k,
                               params={"num_docs": 0}, results=[])
            return []

        snippets = [it.text for it in items]
        kb_paths = sorted({it.kb_path for it in items})

        # Embeddings
        #mat, meta = self.embedder.get_or_build_embeddings(kb_paths, snippets)  # (N,768)
        mat, meta = self.embedder.get_or_build_embeddings(snippets, kb_file=self.kb_file)  # (N,768)

        qv = self.embedder.embed_query(q)                                      # (768,)

        sims = mat @ qv  # cosine (dot) because rows & qv are L2-normalized
        lengths = np.array([max(1, len(it.text)) for it in items], dtype=np.float32)  # character count
        length_prior = 1.0 - np.exp(-lengths / 120.0)  # tune 120 if needed
        prior_w = 0.10  # 0..0.2 is typically safe; set to 0 to disable
        sims = (1.0 - prior_w) * sims + prior_w * (sims * length_prior)



        order = np.argsort(-sims, kind="mergesort")[: min(k, len(items))]

        # BM25 (for reporting only)
        docs_tok = [simple_tokenize(it.text.lower()) for it in items]
        bm25 = BM25Okapi(docs_tok, k1=self.bm25_k1, b=self.bm25_b)
        q_tokens = simple_tokenize(q.lower())
        bm25_scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(items), dtype=np.float32)

        # Prepare report rows
        results = []
        out_snippets: List[str] = []
        for rank, idx in enumerate(order, start=1):
            it = items[idx]
            cos = float(sims[idx])
            bm = float(bm25_scores[idx])
            rrf = 1.0 / (self.rrf_k + rank)  # simple per-list RRF value for display
            results.append((rank, {"bm25": bm, "cosine": cos, "rrf": rrf}, it))
            out_snippets.append(it.text)

        write_report_multi(
            metric_name="cosine",
            q=q,
            k=k,
            params={
                "model": meta.get("model", self.embedder.model_name),
                "pooling": meta.get("pooling", "mean"),
                "remove_top_pcs": meta.get("remove_top_pcs", 3),
                "max_length": meta.get("max_length", 512),
                "num_docs": len(items),
                "rrf_k": self.rrf_k,
                "bm25_k1": self.bm25_k1,
                "bm25_b": self.bm25_b,
            },
            results=results
        )
        return out_snippets
