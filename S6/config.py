from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class S6Config:
    builder_func: Callable[[str, str], str]  # not used inside multihop
    snippet_separator: str = "\n\n# --- Snippet Separator ---\n\n"
    preview_width: int = 700
    verbose: bool = True

    # New (multi-hop)
    strategy: str = "decomposition_first"
    max_hops: int = 3
    k_first: int = 8
    k_follow: int = 5
    beam: int = 2
    target_repo: Optional[str] = None
    log_dir: str = "results/logs/multihop"
    seed: int = 42
