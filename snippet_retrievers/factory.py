from __future__ import annotations
from .base import SnippetRetriever
from .bm25_retriever import BM25Retriever
from .cosine_retriever import CosineRetriever
from .hybrid_retriever import HybridRRFRetriever

def get_retriever(metric: str, **kwargs) -> SnippetRetriever:
    m = (metric or "bm25").strip().lower()
    if m == "bm25":
        return BM25Retriever(**kwargs)
    elif m == "cosine":
        if "k1" in kwargs or "b" in kwargs:
            kwargs["bm25_k1"] = kwargs.pop("k1", 1.5)
            kwargs["bm25_b"]  = kwargs.pop("b", 0.75)
        return CosineRetriever(**kwargs)
    elif m == "hybrid":
        # remap bm25 args for consistency
        if "k1" in kwargs or "b" in kwargs:
            kwargs["bm25_k1"] = kwargs.pop("k1", 1.5)
            kwargs["bm25_b"]  = kwargs.pop("b", 0.75)
        return HybridRRFRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown metric: {metric}")
