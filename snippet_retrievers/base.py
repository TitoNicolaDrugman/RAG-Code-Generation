from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class SnippetRetriever(ABC):
    """Abstract interface for snippet retrievers."""

    def __init__(self, *, kb_root: str | None = None, kb_file: str | None = None):
        self.kb_root = kb_root
        self.kb_file = kb_file

    @abstractmethod
    def retrieve(self, q: str, k: int) -> List[str]:
        """Return k snippets ONLY (no metadata). Must also write a detailed TXT report."""
        raise NotImplementedError
