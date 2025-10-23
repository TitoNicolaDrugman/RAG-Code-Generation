
from __future__ import annotations
from typing import List, Union, Dict, Any, Tuple
import numpy as np
import warnings
from .config import DEFAULT_LANG, DEFAULT_WEIGHTS

try:
    from codebleu import calc_codebleu
    _HAS_CODEBLEU = True
except Exception as e:
    warnings.warn(f"Impossibile importare 'codebleu': {type(e).__name__}: {e}")
    _HAS_CODEBLEU = False

def _to_multi_ref_format(
    references: List[Union[str, List[str]]]
) -> List[List[str]]:
    """Converte references nel formato List[List[str]] richiesto da CodeBLEU."""
    refs_out = []
    for r in references:
        if isinstance(r, str):
            refs_out.append([r])
        elif isinstance(r, list) and all(isinstance(x, str) for x in r):
            refs_out.append(r)
        else:
            refs_out.append([])
    return refs_out

def _detect_codebleu_key(res: Dict[str, Any]) -> str | None:
    """Individua la chiave del punteggio CodeBLEU nel dict di ritorno."""
    if not isinstance(res, dict):
        return None
    candidates = [k for k in res.keys() if "code" in k.lower() and "bleu" in k.lower()]
    if "codebleu" in candidates: return "codebleu"
    if "code_bleu" in candidates: return "code_bleu"
    return candidates[0] if candidates else None

def _normalize_01(x: Any) -> float:
    """
    Porta il valore x in [0,1]:
    - None/NaN -> 0.0
    - Se 1 < x <= 100 assume percentuale e divide per 100
    - Clipping finale a [0,1]
    """
    try:
        v = float(x)
    except Exception:
        return 0.0
    if np.isnan(v) or np.isinf(v):
        return 0.0
    if v > 1.0:
        v = v / 100.0
    if v < 0.0: v = 0.0
    if v > 1.0: v = 1.0
    return v

def codebleu_scores(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    *,
    lang: str = DEFAULT_LANG,
    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS,
    average: str = "macro"  # "macro" | "corpus"
) -> Dict[str, Any]:
    """
    Calcola CodeBLEU con output SEMPRE in [0,1].
    Restituisce un dict con 'aggregate', 'per_sample' e 'details'.
    """
    if not _HAS_CODEBLEU:
        warnings.warn("CodeBLEU non disponibile: restituisco 0.0.")
        return {"aggregate": 0.0, "per_sample": [], "details": {"key_used": None, "raw_first_result": None}}

    if not isinstance(predictions, list) or not isinstance(references, list):
        raise ValueError("predictions e references devono essere liste della stessa lunghezza.")
    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere la stessa lunghezza.")

    refs_multi = _to_multi_ref_format(references)

    # Filtra/gestisce sample vuoti
    P, R = [], []
    for p, rlist in zip(predictions, refs_multi):
        p_ok = isinstance(p, str) and p.strip() != ""
        r_ok = any(isinstance(r, str) and r.strip() != "" for r in rlist)
        if p_ok and r_ok:
            P.append(p)
            R.append(rlist)
        else:
            if average == "macro":
                P.append("")
                R.append([""])

    if len(P) == 0:
        return {"aggregate": 0.0, "per_sample": [], "details": {"key_used": None, "raw_first_result": None}}

    # Corpus
    if average.lower() == "corpus":
        try:
            res = calc_codebleu(references=R, predictions=P, lang=lang, weights=weights)
            key = _detect_codebleu_key(res)
            raw_val = res.get(key, None) if isinstance(res, dict) and key else None
            agg = _normalize_01(raw_val)
            return {"aggregate": agg, "per_sample": [], "details": {"key_used": key, "raw_first_result": res}}
        except Exception as e:
            warnings.warn(f"CodeBLEU (corpus) fallito: {type(e).__name__}: {e}")
            return {"aggregate": 0.0, "per_sample": [], "details": {"key_used": None, "raw_first_result": None}}

    # Macro
    per_sample = []
    first_raw = None
    first_key = None
    for p, rlist in zip(P, R):
        if not p.strip() or not any(rr.strip() for rr in rlist):
            per_sample.append(0.0)
            continue
        try:
            res = calc_codebleu(references=[rlist], predictions=[p], lang=lang, weights=weights)
            if first_raw is None:
                first_raw = res if isinstance(res, dict) else None
                first_key = _detect_codebleu_key(res) if isinstance(res, dict) else None
            key = _detect_codebleu_key(res)
            raw_val = res.get(key, None) if isinstance(res, dict) and key else None
            per_sample.append(_normalize_01(raw_val))
        except Exception as e:
            warnings.warn(f"CodeBLEU (sample) fallito: {type(e).__name__}: {e}")
            per_sample.append(0.0)

    aggregate = float(np.mean(per_sample)) if per_sample else 0.0
    return {"aggregate": aggregate, "per_sample": per_sample, "details": {"key_used": first_key, "raw_first_result": first_raw}}
