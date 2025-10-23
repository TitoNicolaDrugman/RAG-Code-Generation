# prompts/utils.py
from __future__ import annotations
from typing import Final
import warnings

# Token budgets (usati dalle versioni v6+)
TOKEN_BUDGET:         Final[int] = 16_000
SNIPPET_TOKEN_BUDGET: Final[int] = 2_000

def _sanitize_triple_backticks(txt: str) -> str:
    """
    Evita che backticks tripli interrompano i fence markdown nei prompt.
    """
    return txt.replace("```", "'''")

def truncate_to_n_tokens(
    text: str,
    n: int = SNIPPET_TOKEN_BUDGET,
    keep: str = "head",  # "head" | "tail" | "both"
) -> str:
    """
    Taglia *text* a ≤ *n* token usando il tokenizer globale (se presente).

    keep = "head"  → primi n token
    keep = "tail"  → ultimi n token
    keep = "both"  → n//2 all'inizio + n//2 alla fine
    """
    if "tokenizer" not in globals() or not hasattr(globals()["tokenizer"], "encode"):
        warnings.warn("tokenizer missing – no truncation performed")
        return text

    tokenizer = globals()["tokenizer"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= n:
        return text

    if keep == "tail":
        keep_ids = ids[-n:]
    elif keep == "both":
        split = n // 2
        keep_ids = ids[:split] + ids[-split:]
    else:
        keep_ids = ids[:n]

    result = tokenizer.decode(keep_ids)
    # Evita chiusure accidentali di ``` nei prompt
    return result.replace("```", "'''")
