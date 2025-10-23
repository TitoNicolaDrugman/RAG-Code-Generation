# promptgen/prompts_discovery.py
import inspect
import importlib
from pathlib import Path
from typing import Callable, List, Dict, Any

def discover_prompt_modules(prompts_dir: Path) -> List[str]:
    mods = []
    for p in prompts_dir.glob("*.py"):
        if p.name == "__init__.py":
            continue
        mods.append(p.stem)
    return sorted(mods)

def find_baseline_builder(module_name: str, template_key: str) -> Callable[..., str]:
    mod = importlib.import_module(f"prompts.{module_name}")
    candidates = [
        "build_baseline_prompt",
        f"build_baseline_prompt_{template_key}",
        "build_baseline",
        "baseline_prompt",
        "make_baseline_prompt",
    ]
    for fn in candidates:
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return getattr(mod, fn)

    # fallback: qualunque funzione che contenga 'baseline' nel nome e prenda >=1 argomento
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if "baseline" in name and obj.__code__.co_argcount >= 1:
            return obj

    raise ImportError(
        f"Nessuna funzione baseline trovata in prompts.{module_name}. "
        f"Attesi uno tra: {', '.join(candidates)}"
    )

def call_baseline_builder_safely(builder: Callable[..., str], instruction: str) -> str:
    """
    Invoca il builder in modo robusto:
      - accetta sempre 'instruction'
      - se presenti, forza: snippets=[], top_k/k=0, mode/variant='baseline'
      - non passa altri argomenti
    """
    sig = inspect.signature(builder)
    kwargs: Dict[str, Any] = {}
    params = sig.parameters

    # obbligatorio: instruction
    if "instruction" in params:
        kwargs["instruction"] = instruction
    else:
        # builder potrebbe usare un altro nome (es. 'query' o 'prompt')
        if "query" in params:
            kwargs["query"] = instruction
        elif "prompt" in params:
            kwargs["prompt"] = instruction
        else:
            # se il primo parametro è posizionale, passiamo instruction lì
            # (senza usare kwargs)
            pass

    # opzionali: modalità baseline e nessuno snippet
    if "snippets" in params:
        kwargs["snippets"] = []
    if "top_k" in params:
        kwargs["top_k"] = 0
    if "k" in params and "top_k" not in kwargs:
        kwargs["k"] = 0
    if "mode" in params:
        kwargs["mode"] = "baseline"
    if "variant" in params:
        kwargs["variant"] = "baseline"

    # Costruiamo la chiamata rispettando l'ordine dei posizionali
    bound = sig.bind_partial(**kwargs)
    bound.apply_defaults()
    return builder(*bound.args, **bound.kwargs)
