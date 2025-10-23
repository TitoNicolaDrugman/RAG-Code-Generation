
from __future__ import annotations
from typing import Sequence, List, Union, Tuple
import pandas as pd
from .core import codebleu_scores
from .config import DEFAULT_LANG, DEFAULT_WEIGHTS

def _ensure_list_refs(x) -> List[str]:
    """Accetta str o lista; ritorna sempre List[str]."""
    if isinstance(x, list):
        return [s for s in x if isinstance(s, str)]
    if isinstance(x, str):
        return [x]
    return []

def compute_grouped_codebleu(
    df: pd.DataFrame,
    prediction_col: str,
    references_col: str,
    group_cols: Sequence[str] = ("variant","prompt_type"),
    lang: str = DEFAULT_LANG,
    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS,
    average: str = "macro"
) -> pd.DataFrame:
    """
    Calcola CodeBLEU normalizzato per gruppi (es. per variant × prompt_type).
    - df[references_col] può contenere stringhe o liste di stringhe (multi-ref).
    Restituisce un nuovo DataFrame con colonne group + codebleu (0..1).
    """
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        preds = g[prediction_col].tolist()
        refs  = [ _ensure_list_refs(r) for r in g[references_col].tolist() ]
        res = codebleu_scores(preds, refs, lang=lang, weights=weights, average=average)
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {col: key for col, key in zip(group_cols, key_tuple)}
        row.update({"codebleu": res["aggregate"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out
