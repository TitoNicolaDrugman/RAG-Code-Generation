# prompts/__init__.py
from __future__ import annotations
from typing import Callable, Dict

# Utils condivisi
from .utils import truncate_to_n_tokens

# Versioni baseline/RAG
from .v1 import build_baseline_prompt_v1, build_rag_prompt_v1
from .v2 import build_baseline_prompt_v2, build_rag_prompt_v2
from .v3 import build_baseline_prompt_v3, build_rag_prompt_v3
from .v4 import build_baseline_prompt_v4, build_rag_prompt_v4
from .v5 import build_baseline_prompt_v5, build_rag_prompt_v5
from .v6 import build_baseline_prompt_v6, build_rag_prompt_v6
from .v6_2 import build_baseline_prompt_v6_2, build_rag_prompt_v6_2
from .v6_3 import build_baseline_prompt_v6_3, build_rag_prompt_v6_3
from .v7 import build_baseline_prompt_v7, build_rag_prompt_v7
from .v8 import build_baseline_prompt_v8, build_rag_prompt_v8
from .v9 import build_baseline_prompt_v9, build_rag_prompt_v9

# Registry opzionale per accesso dinamico
PROMPT_REGISTRY: Dict[str, Callable] = {
    # v1
    "baseline_v1": build_baseline_prompt_v1,
    "rag_v1": build_rag_prompt_v1,
    # v2
    "baseline_v2": build_baseline_prompt_v2,
    "rag_v2": build_rag_prompt_v2,
    # v3
    "baseline_v3": build_baseline_prompt_v3,
    "rag_v3": build_rag_prompt_v3,
    # v4
    "baseline_v4": build_baseline_prompt_v4,
    "rag_v4": build_rag_prompt_v4,
    # v5
    "baseline_v5": build_baseline_prompt_v5,
    "rag_v5": build_rag_prompt_v5,
    # v6
    "baseline_v6": build_baseline_prompt_v6,
    "rag_v6": build_rag_prompt_v6,
    # v6_2
    "baseline_v6_2": build_baseline_prompt_v6_2,
    "rag_v6_2": build_rag_prompt_v6_2,
    # v6_3
    "baseline_v6_3": build_baseline_prompt_v6_3,
    "rag_v6_3": build_rag_prompt_v6_3,
    # v7
    "baseline_v7": build_baseline_prompt_v7,
    "rag_v7": build_rag_prompt_v7,
    # v8
    "baseline_v8": build_baseline_prompt_v8,
    "rag_v8": build_rag_prompt_v8,
    # v9
    "baseline_v9": build_baseline_prompt_v9,
    "rag_v9": build_rag_prompt_v9,
}

__all__ = [
    # utils
    "truncate_to_n_tokens",
    # v1
    "build_baseline_prompt_v1", "build_rag_prompt_v1",
    # v2
    "build_baseline_prompt_v2", "build_rag_prompt_v2",
    # v3
    "build_baseline_prompt_v3", "build_rag_prompt_v3",
    # v4
    "build_baseline_prompt_v4", "build_rag_prompt_v4",
    # v5
    "build_baseline_prompt_v5", "build_rag_prompt_v5",
    # v6
    "build_baseline_prompt_v6", "build_rag_prompt_v6",
    # v6_2
    "build_baseline_prompt_v6_2", "build_rag_prompt_v6_2",
    # v6_3
    "build_baseline_prompt_v6_3", "build_rag_prompt_v6_3",
    # v7
    "build_baseline_prompt_v7", "build_rag_prompt_v7",
    # v8
    "build_baseline_prompt_v8", "build_rag_prompt_v8",
    # v9
    "build_baseline_prompt_v9", "build_rag_prompt_v9",
    # registry
    "PROMPT_REGISTRY",
]