import importlib, inspect, textwrap
from typing import Dict, List, Optional, Callable
from .config import S6Config
from .assembly import _builder_needs_tokenizer

def _import_prompt_modules():
    candidates = [
        "prompts",
        "prompts.utils",
        "prompts.v1", "prompts.v2", "prompts.v3", "prompts.v4",
        "prompts.v5", "prompts.v6", "prompts.v6_2", "prompts.v6_3",
        "prompts.v7", "prompts.v8", "prompts.v9",
    ]
    loaded = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            loaded.append(mod)
        except Exception:
            pass
    return loaded

def _discover_builders() -> Dict[str, Callable]:
    modules = _import_prompt_modules()
    builders = {}
    for m in modules:
        for fname, obj in inspect.getmembers(m, inspect.isfunction):
            if fname.startswith("build_rag_prompt_"):
                builders[fname] = obj
    return builders

def assemble_all_rag_prompts(
    instruction: str,
    snippets: List[str],
    cfg: S6Config,
    tokenizer=None,
    prefer_builder_name: str = "build_rag_prompt_v6_3",
) -> Dict[str, Optional[str]]:
    if instruction is None or snippets is None:
        raise NameError("Variables 's5_instruction_to_s6' or 's5_snippets_to_s6' not found. Ensure Section 5 ran successfully.")

    builders = _discover_builders()
    if not builders:
        raise NameError("No 'build_rag_prompt_*' functions found in the 'prompts' package.")

    snippets_block = cfg.snippet_separator.join(snippets) if snippets else ""
    if not snippets:
        print("  INFO: No snippets from Section 5; assembling prompts with empty retrieved context.")

    if cfg.verbose:
        print(f"  Using instruction: '{instruction[:100]}...'")
        print(f"  Using {len(snippets)} retrieved snippets from 's5_snippets_to_s6'.")

    all_outputs: Dict[str, Optional[str]] = {}
    names_sorted = sorted(builders.keys())
    if prefer_builder_name in builders:
        names_sorted.remove(prefer_builder_name)
        names_sorted.insert(0, prefer_builder_name)

    for name in names_sorted:
        builder = builders[name]
        print(f"\nINFO: Using '{name}' for RAG prompt assembly in this section.")
        needs_tok = _builder_needs_tokenizer(builder)
        if needs_tok:
            if tokenizer is None:
                print(f"  INFO: Selected prompt builder '{name}' may use the global 'tokenizer'. Ensure it's loaded (or truncation will be skipped).")
            else:
                print(f"  INFO: Selected prompt builder '{name}' may use the global 'tokenizer'.")

        prompt_out = None
        try:
            prompt_out = builder(instruction, snippets_block)
            if prompt_out:
                if cfg.verbose:
                    print("\n  Final RAG Prompt Assembled (First "
                          f"{cfg.preview_width} Characters):")
                    print(textwrap.shorten(prompt_out, width=cfg.preview_width, placeholder="... (prompt truncated) ..."))
            else:
                print("  ERROR: Builder returned empty prompt.")
        except Exception as e:
            print(f"  ERROR during prompt assembly with '{name}': {e}")
            prompt_out = None

        all_outputs[name] = prompt_out

        if prompt_out:
            if cfg.verbose:
                print(f"\n--- Section 6: RAG Prompt Assembly Complete for '{name}'. Length: {len(prompt_out)} chars ---")
        else:
            print(f"\n--- Section 6: Prompt Assembly FAILED for '{name}' ---")
    return all_outputs
