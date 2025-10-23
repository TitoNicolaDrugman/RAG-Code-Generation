import os, textwrap, re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from utils.kb_manager import list_available_kbs, load_kb, kb_base_url
from IPython.display import display, HTML
from .highlight import highlight_tokens_html

def _valid_strings_only(seq):
    return [str(x) for x in seq if isinstance(x, str) and str(x).strip()]

def _dataset_filter_by_repo(ds, repo_full_name: str):
    if hasattr(ds, "filter"):
        return list(ds.filter(lambda ex: ex.get("repo_full_name") == repo_full_name))
    return [ex for ex in ds if ex.get("repo_full_name") == repo_full_name]

def _kb_keys_from_github() -> List[str]:
    kbs = list_available_kbs()
    names = {it["name"] for it in kbs if "name" in it}
    return sorted({n[len("kb_"):-len(".json")] for n in names if n.startswith("kb_") and n.endswith(".json")})

def _build_corpus(selected_keys: List[str], tokenizer, overwrite=False):
    all_docs, mapping, tok_corpus = [], [], []
    for lib in selected_keys:
        kb_json, _ = load_kb(lib, overwrite=overwrite)
        if not kb_json:
            continue
        valid_docs = _valid_strings_only(kb_json)
        for i, doc in enumerate(valid_docs):
            toks = tokenizer(doc)
            if toks:
                tok_corpus.append(toks)
                mapping.append((lib, i))
                all_docs.append(doc)
    return tok_corpus, mapping, all_docs

def run_bm25_analysis(cfg, lca_dataset_split):
    assert cfg.tokenizer is not None, "Config.tokenizer non impostata"

    print("--- BM25 Retrieval Analysis ---")

    # 1) Selezione sample
    if cfg.target_repo_full_name:
        lib_key = cfg.target_repo_full_name
        samples = _dataset_filter_by_repo(lca_dataset_split, lib_key)
        target = samples[cfg.sample_index_within_repo]
        print(f"  Analyzing instruction #{cfg.sample_index_within_repo} from library '{lib_key}'")
    else:
        target = lca_dataset_split[cfg.sample_index_within_repo]
        lib_key = target.get("repo_full_name")
        print(f"  Analyzing global sample #{cfg.sample_index_within_repo} (derived library '{lib_key}')")

    s5_instruction_to_s6 = target.get("instruction")
    print(f"\n  KB base URL: {kb_base_url()}/kb_LIBRARY_KEY.json")
    print("  Instruction Text:")
    print(textwrap.fill(s5_instruction_to_s6, width=100, initial_indent="    ", subsequent_indent="    "))

    query_tokens = cfg.tokenizer(s5_instruction_to_s6)
    if cfg.show_query_tokens:
        print(f"\n  Query Tokens (using '{cfg.tokenizer.__name__}'):\n    {query_tokens}")

    # 2) Scope KB
    available_keys = _kb_keys_from_github()
    selected_keys = cfg.library_filter or available_keys
    print(f"\n  Using {len(selected_keys)} KBs ({'ALL' if cfg.library_filter is None else 'subset'})")

    # 3) Costruzione indice
    tok_corpus, mapping, docs = _build_corpus(selected_keys, cfg.tokenizer, overwrite=cfg.overwrite_kb_download)
    if not tok_corpus:
        print("  No documents for BM25.")
        return s5_instruction_to_s6, []

    print(f"  Building BM25 index on {len(tok_corpus)} docs…")
    bm25 = BM25Okapi(tok_corpus, k1=cfg.bm25_k1, b=cfg.bm25_b)

    # 4) Retrieval
    s5_snippets_to_s6 = []
    if query_tokens and cfg.top_k_snippets > 0:
        n = min(cfg.top_k_snippets, len(tok_corpus))
        indices = bm25.get_top_n(query_tokens, list(range(len(tok_corpus))), n=n)
        for rank, idx in enumerate(indices, 1):
            lib, orig_idx = mapping[idx]
            snip = docs[idx]
            print(f"\n    [{rank}/{n}] From library '{lib}' | Length {len(snip)} chars")
            if cfg.highlight_keywords:
                highlight_tokens_html(snip, query_tokens)
            else:
                print(textwrap.indent(textwrap.fill(snip, width=100), '      '))
            s5_snippets_to_s6.append(snip)
        print(f"\n  Stored {len(s5_snippets_to_s6)} snippets in 's5_snippets_to_s6'")
    else:
        print("  Skipping retrieval (empty query or top_k<=0)")

    print("\n--- BM25 Retrieval Complete ---")
    return s5_instruction_to_s6, s5_snippets_to_s6
