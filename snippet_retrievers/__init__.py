from .factory import get_retriever

def retrieve_snippets(q: str, k: int, metric: str = "bm25",
                      kb_root: str = "temp_downloaded_kbs",
                      kb_file: str | None = None,
                      **kwargs) -> list[str]:
    retriever = get_retriever(metric, kb_root=kb_root, kb_file=kb_file, **kwargs)
    return retriever.retrieve(q, k)
