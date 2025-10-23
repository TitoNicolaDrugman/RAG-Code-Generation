from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass
class Config:
    # Selezione sample
    target_repo_full_name: Optional[str] = "pyscf__pyscf"
    sample_index_within_repo: int = 0

    # Parametri BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k_snippets: int = 1

    # Tokenizer
    tokenizer: Callable[[str], List[str]] = None

    # Opzioni display
    show_query_tokens: bool = True
    highlight_keywords: bool = True

    # Scope KB (None = tutte le KB, altrimenti subset di repo_full_name)
    library_filter: Optional[List[str]] = None

    # Overwrite KB cache
    overwrite_kb_download: bool = False
