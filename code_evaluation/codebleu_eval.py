# -*- coding: utf-8 -*-
"""
Valutazione baseline con CodeBLEU (se disponibile) o fallback BLEU-4.
- join tra generations e dataset su (repo_name, instruction) normalizzati
- salvataggio opzionale su out_dir: per-istanza JSONL + summary JSON
- summary include mean/min/max/std per prompt_type
"""
import json
import math
import re
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ---------- Loader robusto ----------
def load_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # tolleranza: salta righe malformate
                continue
    return items

def load_any_jsonish(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    # tenta JSONL
    try:
        return load_jsonl(path)
    except Exception:
        pass
    # tenta array JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception:
        pass
    # fallback: parse riga per riga
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if rows:
        return rows
    raise RuntimeError(f"Could not parse {path} as JSONL or JSON.")

# ---------- Estrazione codice ----------
FENCE_RE = re.compile(r"```(?:[Pp]ython)?\s*(.*?)```", re.DOTALL)
def extract_code(generation_field: str) -> str:
    if not generation_field:
        return ""
    m = FENCE_RE.search(generation_field)
    code = m.group(1) if m else generation_field
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()

# ---------- Indicizzazione dataset ----------
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def build_dataset_index(dataset_records: List[dict]) -> Dict[Tuple[str, str], dict]:
    index = {}
    for r in dataset_records:
        repo = (r.get("repo_name") or r.get("repo") or "").strip()
        instr = normalize_whitespace(r.get("instruction") or r.get("task") or "")
        key = (repo, instr)
        index[key] = r
    return index

# ---------- CodeBLEU import robusto + fallback BLEU-4 ----------
CODEBLEU_OK = True
_GET_CODE_BLEU = None
try:
    # Variante A
    try:
        from codebleu import calc_code_bleu as _ccb_mod
        _GET_CODE_BLEU = _ccb_mod.get_code_bleu
    except Exception:
        # Variante B
        from codebleu.calc_code_bleu import get_code_bleu as _GET_CODE_BLEU
except Exception:
    CODEBLEU_OK = False
    _GET_CODE_BLEU = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None

def _fallback_bleu_only(ref_code: str, hyp_code: str):
    """Ritorna un dict stile CodeBLEU ma calcolando solo BLEU-4 (sacrebleu)."""
    bleu = None
    if sacrebleu and ref_code and hyp_code:
        try:
            bleu = sacrebleu.corpus_bleu([hyp_code], [[ref_code]], force=True).score
        except Exception:
            bleu = None
    return {
        "codebleu": bleu,   # usiamo BLEU come "codebleu" nel fallback
        "ng": bleu,
        "weighted_ng": None,
        "syntax_tree": None,
        "dataflow": None,
    }

def _compute_scores(ref_code: str, hyp_code: str, lang: str,
                    alpha: float, beta: float, gamma: float, theta: float):
    if CODEBLEU_OK and _GET_CODE_BLEU is not None:
        return _GET_CODE_BLEU([ref_code], [hyp_code], lang=lang,
                              alpha=alpha, beta=beta, gamma=gamma, theta=theta)
    else:
        return _fallback_bleu_only(ref_code, hyp_code)

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

@dataclass
class InstanceRow:
    query_id: Optional[str]
    repo_name: Optional[str]
    prompt_type: Optional[str]
    model_name: Optional[str]
    language: str
    codebleu_a025: Optional[float]
    bleu_a025: Optional[float]
    w_bleu_a025: Optional[float]
    ast_a025: Optional[float]
    df_a025: Optional[float]
    codebleu_a1040: Optional[float]
    bleu_a1040: Optional[float]
    w_bleu_a1040: Optional[float]
    ast_a1040: Optional[float]
    df_a1040: Optional[float]
    error_reason: Optional[str]

def summarize(values: List[Optional[float]]):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None, None
    mean = sum(vals) / len(vals)
    vmin = min(vals)
    vmax = max(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    return mean, vmin, vmax, std

def compute_codebleu_for_generations(
    generations_path: str,
    dataset_path: str,
    lang: str = "python",
    prompt_field: str = "template",
    variant_field: str = "variant",
    variant_value: str = "baseline",
    out_dir: Optional[str] = None,
):
    gen_path = Path(generations_path)
    ds_path = Path(dataset_path)

    generations = load_any_jsonish(gen_path)
    dataset = load_any_jsonish(ds_path)
    ds_index = build_dataset_index(dataset)

    per_instance: List[InstanceRow] = []

    for r in generations:
        if r.get(variant_field) != variant_value:
            continue
        query_id = r.get("query_id")
        repo_name = (r.get("repo_name") or "").strip()
        instr = normalize_whitespace(r.get("instruction") or "")
        prompt_type = r.get(prompt_field) or "unknown"
        model_name = r.get("model_name")
        generation_field = r.get("generation") or ""

        code = extract_code(generation_field)
        ref_rec = ds_index.get((repo_name, instr))
        error_reason = None
        if ref_rec is None:
            error_reason = "reference_not_found_by_(repo_name,instruction)"
            ref_code = ""
        else:
            ref_code = ref_rec.get("clean_reference") or ref_rec.get("reference") or ""

        codebleu_a025 = bleu_a025 = w_bleu_a025 = ast_a025 = df_a025 = None
        codebleu_a1040 = bleu_a1040 = w_bleu_a1040 = ast_a1040 = df_a1040 = None

        if code and ref_code:
            try:
                scores = _compute_scores(ref_code, code, lang, 0.25, 0.25, 0.25, 0.25)
                codebleu_a025 = safe_float(scores.get("codebleu"))
                bleu_a025 = safe_float(scores.get("ng"))
                w_bleu_a025 = safe_float(scores.get("weighted_ng"))
                ast_a025 = safe_float(scores.get("syntax_tree"))
                df_a025 = safe_float(scores.get("dataflow"))

                scores2 = _compute_scores(ref_code, code, lang, 0.1, 0.1, 0.4, 0.4)
                codebleu_a1040 = safe_float(scores2.get("codebleu"))
                bleu_a1040 = safe_float(scores2.get("ng"))
                w_bleu_a1040 = safe_float(scores2.get("weighted_ng"))
                ast_a1040 = safe_float(scores2.get("syntax_tree"))
                df_a1040 = safe_float(scores2.get("dataflow"))
            except Exception as e:
                error_reason = f"codebleu_error: {type(e).__name__}: {e}"
        else:
            if not code and not error_reason:
                error_reason = "empty_generation_code"
            if not ref_code and not error_reason:
                error_reason = "empty_reference_code"

        per_instance.append(
            InstanceRow(
                query_id=query_id,
                repo_name=repo_name,
                prompt_type=prompt_type,
                model_name=model_name,
                language=lang,
                codebleu_a025=codebleu_a025,
                bleu_a025=bleu_a025,
                w_bleu_a025=w_bleu_a025,
                ast_a025=ast_a025,
                df_a025=df_a025,
                codebleu_a1040=codebleu_a1040,
                bleu_a1040=bleu_a1040,
                w_bleu_a1040=w_bleu_a1040,
                ast_a1040=ast_a1040,
                df_a1040=df_a1040,
                error_reason=error_reason,
            )
        )

    # Aggregazione per prompt_type
    by_prompt = defaultdict(lambda: {"count": 0, "codebleu_a025": [], "codebleu_a1040": []})
    for row in per_instance:
        p = row.prompt_type or "unknown"
        by_prompt[p]["count"] += 1
        by_prompt[p]["codebleu_a025"].append(row.codebleu_a025)
        by_prompt[p]["codebleu_a1040"].append(row.codebleu_a1040)

    summary = []
    for ptype, stats in sorted(by_prompt.items(), key=lambda kv: kv[0]):
        mean025, min025, max025, std025 = summarize(stats["codebleu_a025"])
        mean1040, min1040, max1040, std1040 = summarize(stats["codebleu_a1040"])
        summary.append({
            "prompt_type": ptype,
            "count": stats["count"],
            "mean_codebleu_025": mean025,
            "min_codebleu_025": min025,
            "max_codebleu_025": max025,
            "std_codebleu_025": std025,
            "mean_codebleu_1040": mean1040,
            "min_codebleu_1040": min1040,
            "max_codebleu_1040": max1040,
            "std_codebleu_1040": std1040,
        })

    # Salvataggi opzionali
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "codebleu_per_instance.jsonl").open("w", encoding="utf-8") as f:
            for row in per_instance:
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        with (out / "summary_by_prompt.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return [asdict(x) for x in per_instance], summary

# ---------- Utility: esporta summary in CSV ----------
def export_summary_csv(summary: List[dict], csv_path: str):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_type","count",
            "mean_codebleu_025","min_codebleu_025","max_codebleu_025","std_codebleu_025",
            "mean_codebleu_1040","min_codebleu_1040","max_codebleu_1040","std_codebleu_1040",
        ])
        for row in summary:
            writer.writerow([
                row.get("prompt_type"), row.get("count"),
                row.get("mean_codebleu_025"), row.get("min_codebleu_025"),
                row.get("max_codebleu_025"), row.get("std_codebleu_025"),
                row.get("mean_codebleu_1040"), row.get("min_codebleu_1040"),
                row.get("max_codebleu_1040"), row.get("std_codebleu_1040"),
            ])
