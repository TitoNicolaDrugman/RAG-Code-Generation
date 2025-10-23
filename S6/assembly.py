import textwrap
from typing import List, Optional, Callable

def _builder_needs_tokenizer(builder: Callable) -> bool:
    co = getattr(builder, "__code__", None)
    return bool(co and "truncate_to_n_tokens" in getattr(co, "co_names", []))

def assemble_rag_prompt(
    instruction: Optional[str],
    snippets: Optional[List[str]],
    cfg,
    tokenizer=None,
) -> Optional[str]:
    if instruction is None or snippets is None:
        raise NameError("Variables 's5_instruction_to_s6' or 's5_snippets_to_s6' not found. Ensure Section 5 has been run successfully.")

    build_rag_prompt_to_use_in_s6 = cfg.builder_func
    if build_rag_prompt_to_use_in_s6 is None or not callable(build_rag_prompt_to_use_in_s6):
        raise NameError("The function assigned to 'builder_func' is not defined or not callable.")

    print(f"INFO: Using '{build_rag_prompt_to_use_in_s6.__name__}' for RAG prompt assembly in this section.")
    needs_tok = _builder_needs_tokenizer(build_rag_prompt_to_use_in_s6)
    if needs_tok:
        if tokenizer is None:
            print(f"  INFO: Selected prompt builder '{build_rag_prompt_to_use_in_s6.__name__}' may use the global 'tokenizer'. Ensure 'tokenizer' is correctly loaded.")
        else:
            print(f"  INFO: Selected prompt builder '{build_rag_prompt_to_use_in_s6.__name__}' may use the global 'tokenizer'. Ensure 'tokenizer' is correctly loaded.")
    else:
        print(f"  INFO: Selected prompt builder '{build_rag_prompt_to_use_in_s6.__name__}' does not appear to directly require the global 'tokenizer' for truncation via 'truncate_to_n_tokens'.")

    if instruction is None:
        print("  ERROR: No instruction ('s5_instruction_to_s6') available from Section 5. Cannot assemble prompt.")
        return None
    else:
        if cfg.verbose:
            print(f"  Using instruction: '{instruction[:100]}...'")
            print(f"  Using {len(snippets)} retrieved snippets from 's5_snippets_to_s6'.")

    snippets_block = cfg.snippet_separator.join(snippets) if snippets else ""
    if not snippets:
        print("  INFO: No snippets were provided from Section 5 ('s5_snippets_to_s6' is empty). The RAG prompt will be assembled with an empty retrieved context.")

    s6_final_rag_prompt_output = None
    try:
        s6_final_rag_prompt_output = build_rag_prompt_to_use_in_s6(instruction, snippets_block)
        if s6_final_rag_prompt_output:
            if cfg.verbose:
                print("\n  Final RAG Prompt Assembled (First "
                      f"{cfg.preview_width} Characters):")
                print(textwrap.shorten(s6_final_rag_prompt_output, width=cfg.preview_width, placeholder="... (prompt truncated) ..."))
        else:
            print("  ERROR: Prompt assembly using your 'build_rag_prompt' function resulted in an empty or None prompt.")
    except Exception as e:
        print(f"  ERROR during prompt assembly with '{build_rag_prompt_to_use_in_s6.__name__}': {e}")
        s6_final_rag_prompt_output = None

    if s6_final_rag_prompt_output:
        if cfg.verbose:
            print(f"\n--- Section 6: RAG Prompt Assembly Complete. Output in 's6_final_rag_prompt_output' (Length: {len(s6_final_rag_prompt_output)} chars) ---")
    else:
        print(f"\n--- Section 6: RAG Prompt Assembly Failed or Produced No Output ---")
    return s6_final_rag_prompt_output
