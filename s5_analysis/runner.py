import os, json, textwrap
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from IPython.display import HTML, display

from utils.kb_manager import list_available_kbs, load_kb, kb_base_url
from .highlight import highlight_tokens_html

def _valid_strings_only(seq):
    return [str(x) for x in seq if isinstance(x, str) and str(x).strip()]

def _dataset_filter_by_repo(ds, repo_full_name: str):
    # ds atteso tipo list-like con .filter(lambda ex: ...) oppure fallback
    if hasattr(ds, "filter"):
        out = ds.filter(lambda ex: ex.get("repo_full_name") == repo_full_name)
        # Assicuriamoci sia indicizzabile
        return list(out)
    # fallback
    return [ex for ex in ds if ex.get("repo_full_name") == repo_full_name]

def _dataset_get_by_index(ds, idx: int):
    return ds[idx]

def _kb_keys_from_github() -> List[str]:
    kbs = list_available_kbs()  # [{'name':'kb_repo.json', ...}, ...]
    names = {it["name"] for it in kbs if "name" in it}
    keys = sorted({n[len("kb_"):-len(".json")] for n in names if n.startswith("kb_") and n.endswith(".json")})
    return keys

def _build_global_kb_index(selected_keys: List[str], tokenizer) -> Tuple[List[List[str]], List[Tuple[str,int]], List[str]]:
    """
    Ritorna:
      - corpus_tokenizzato: List[List[str]]
      - mappa: List[(library_key, original_doc_idx)]
      - docs: List[str] (documenti originali in stringa)
    """
    all_docs = []
    mapping = []
    tok_corpus = []

    for lib in selected_keys:
        kb_json, local_path = load_kb(lib, overwrite=False)  # path dalla tua config
        if not kb_json:
            continue
        valid_docs = _valid_strings_only(kb_json)
        # tokenizza e filtra vuoti
        for i, doc in enumerate(valid_docs):
            toks = tokenizer(doc)
            if toks:
                tok_corpus.append(toks)
                mapping.append((lib, i))
                all_docs.append(doc)

    return tok_corpus, mapping, all_docs

def _print_wrapped(prefix: str, text: str):
    print(prefix + textwrap.fill(text, width=100, initial_indent="", subsequent_indent=" " * len(prefix)))

def run_bm25_analysis(cfg, lca_dataset_split):
    # 0) Sanity
    assert cfg.tokenizer is not None, "Config.tokenizer non impostata."

    print("--- Section 5 (Generalized): Executing BM25 Retrieval Analysis over multi-KB ---")

    # 1) Selezione campione (istruzione + repo)
    if cfg.target_repo_full_name:
        lib_key = cfg.target_repo_full_name
        lib_samples = _dataset_filter_by_repo(lca_dataset_split, lib_key)
        if not lib_samples:
            raise ValueError(f"Nessun sample trovato per la libreria '{lib_key}'.")
        if not (0 <= cfg.sample_index_within_repo < len(lib_samples)):
            raise IndexError(f"sample_index_within_repo={cfg.sample_index_within_repo} out of range (0..{len(lib_samples)-1}) per '{lib_key}'.")
        target = lib_samples[cfg.sample_index_within_repo]
        print(f"  Analyzing instruction #{cfg.sample_index_within_repo} from library: '{lib_key}'")
    else:
        if not (0 <= cfg.sample_index_within_repo < len(lca_dataset_split)):
            raise IndexError(f"Indice globale {cfg.sample_index_within_repo} fuori range per dataset.")
        target = _dataset_get_by_index(lca_dataset_split, cfg.sample_index_within_repo)
        lib_key = target.get("repo_full_name")
        print(f"  Analyzing global sample #{cfg.sample_index_within_repo} (derived library: '{lib_key}')")

    s5_instruction_to_s6 = target.get("instruction")
    if not s5_instruction_to_s6:
        raise ValueError("Il sample selezionato non ha il campo 'instruction'.")

    print(f"\n  KB base URL: {kb_base_url()}/kb_LIBRARY_KEY.json")
    _print_wrapped("  Instruction Text: ", s5_instruction_to_s6)

    query_tokens = cfg.tokenizer(s5_instruction_to_s6)
    if cfg.show_query_tokens:
        print(f"\n  Query Tokens (using '{cfg.tokenizer.__name__}'): {query_tokens}")

    # 2) Determinazione scope KB (all vs subset)
    available_keys = _kb_keys_from_github()  # tutte le KB
    if cfg.library_filter:
        selected_keys = [k for k in available_keys if k in set(cfg.library_filter)]
        not_found = sorted(set(cfg.library_filter) - set(selected_keys))
        if not_found:
            print(f"  WARNING: {len(not_found)} library keys richiesti non presenti su GitHub: {not_found[:10]}{'...' if len(not_found)>10 else ''}")
        print(f"  Using subset of KBs: {len(selected_keys)} libs.")
    else:
        selected_keys = available_keys
        print(f"  Using ALL KBs: {len(selected_keys)} libs.")

    if not selected_keys:
        print("  ERROR: Nessuna KB selezionata. Interrompo.")
        return s5_instruction_to_s6, []

    # 3) Costruzione indice globale BM25
    print(f"\n  Building global BM25 index (this may read many KB files)…")
    tok_corpus, mapping, docs = _build_global_kb_index(selected_keys, cfg.tokenizer)

    if not tok_corpus:
        print("  Tokenized KB is empty after filtering. BM25 index not built.")
        return s5_instruction_to_s6, []

    print(f"  BM25 corpus size: {len(tok_corpus)} documents across {len(selected_keys)} libraries.")
    bm25 = BM25Okapi(tok_corpus, k1=cfg.bm25_k1, b=cfg.bm25_b)
    print("  BM25 index built.")

    # 4) Retrieval
    s5_snippets_to_s6: List[str] = []
    if query_tokens and cfg.top_k_snippets > 0:
        n = min(cfg.top_k_snippets, len(tok_corpus))
        print(f"\n  --- Retrieving and Displaying Top {n} Snippets (GLOBAL) ---")
        # usa get_top_n per ottenere direttamente gli indici
        indices = bm25.get_top_n(query_tokens, list(range(len(tok_corpus))), n=n)
        for rank, idx in enumerate(indices, start=1):
            lib, orig_i = mapping[idx]
            snip = docs[idx]
            print(f"\n    [{rank}/{n}] From library: {lib} | Length: {len(snip)} chars")
            if cfg.highlight_keywords:
                highlight_tokens_html(snip, query_tokens)
            else:
                print(textwrap.indent(textwrap.fill(snip, width=100, subsequent_indent='      '), '      '))
            s5_snippets_to_s6.append(snip)
        print(f"\n    Stored {len(s5_snippets_to_s6)} snippets in 's5_snippets_to_s6' for Section 6.")
    elif not query_tokens:
        print("  Query tokens empty. BM25 retrieval skipped.")
    else:
        print(f"  top_k_snippets ({cfg.top_k_snippets}) is not positive. No snippets retrieved.")

    print(f"\n--- Section 5 (Generalized): Retrieval Analysis Execution Complete ---")
    return s5_instruction_to_s6, s5_snippets_to_s6
