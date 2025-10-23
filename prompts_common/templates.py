# prompts_common/templates.py
from __future__ import annotations
import importlib, inspect, re
from pathlib import Path
from typing import Callable, Dict

PROMPTS_DIR = Path("prompts")

def _find_builder(mod) -> Callable[[str], str]:
    # cerca una funzione che costruisce il prompt baseline
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if re.match(r"build_.*baseline.*", name):
            return fn
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("build_"):
            return fn
    raise RuntimeError(f"Nessun builder trovato in {mod.__name__}")

def load_all_prompt_builders() -> Dict[str, Callable]:
    """
    Ritorna { 'v1': builder, ..., 'v9': builder } includendo eventuali v6_2, v6_3.
    """
    builders: Dict[str, Callable] = {}
    for stem in [f"v{i}" for i in range(1, 10)] + ["v6_2", "v6_3"]:
        py = PROMPTS_DIR / f"{stem}.py"
        if not py.exists():
            continue
        mod = importlib.import_module(f"prompts.{stem}")
        builders[stem] = _find_builder(mod)
    if not builders:
        raise RuntimeError("Nessun template trovato nella cartella 'prompts'.")
    return builders
