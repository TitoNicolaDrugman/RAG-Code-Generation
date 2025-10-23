from __future__ import annotations
import json, os, re, glob, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

PKG_DIR = Path(__file__).parent.resolve()
REPORT_PATH = PKG_DIR / "retrieval_report.txt"

# ---------- basic tokenization ----------
_word_re = re.compile(r"\w+")
def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    toks = _word_re.findall(text)
    # NOTE: keep original case to be case-sensitive for cosine encoder;
    # For BM25, rank_bm25 is fine with mixed case but we can lower if needed there.
    # Here we leave as-is, except we filter 1-char tokens sparsely:
    return [t for t in toks if len(t) > 1]

# ---------- KB loading ----------
def _load_kb_file(path: str | os.PathLike) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [str(x) for x in data if isinstance(x, str) and str(x).strip()]
    return docs

@dataclass
class CorpusItem:
    text: str
    folder_name: str     # e.g., "bokeh__bokeh"
    kb_path: str         # full path to kb_*.json

def build_corpus(kb_root: str | None = None, kb_file: str | None = None) -> List[CorpusItem]:
    corpus: List[CorpusItem] = []
    if kb_file:
        kb_path = Path(kb_file).resolve()
        folder = kb_path.parent.name
        for s in _load_kb_file(kb_path):
            corpus.append(CorpusItem(text=s, folder_name=folder, kb_path=str(kb_path)))
        return corpus

    if not kb_root:
        raise ValueError("Either kb_file must be provided or kb_root must be set.")

    root = Path(kb_root)
    if not root.exists():
        raise FileNotFoundError(f"KB root not found: {root}")

    kb_files = glob.glob(str(root / "*" / "kb_*.json"))
    for fp in kb_files:
        p = Path(fp)
        folder = p.parent.name
        try:
            for s in _load_kb_file(p):
                corpus.append(CorpusItem(text=s, folder_name=folder, kb_path=str(p)))
        except Exception as e:
            print(f"[WARN] Unable to read {p}: {e}")
            continue
    return corpus

# ---------- reporting ----------
def _fmt_scores(scores: Any) -> str:
    """
    Accept either a single float or a dict of floats (e.g., {'bm25':..., 'cosine':..., 'rrf':...}).
    """
    if isinstance(scores, dict):
        parts = []
        # print in a stable, friendly order:
        for k in ("bm25", "cosine", "rrf"):
            if k in scores:
                parts.append(f"{k}={scores[k]:.6f}")
        # include any others deterministically
        for k in sorted(scores.keys()):
            if k not in ("bm25", "cosine", "rrf"):
                parts.append(f"{k}={scores[k]:.6f}")
        return " | " + " ".join(parts) if parts else ""
    try:
        val = float(scores)
        return f" | score={val:.6f}"
    except Exception:
        return ""

def write_report_multi(*, metric_name: str, q: str, k: int,
                       params: dict,
                       results: List[Tuple[int, Dict[str, float], 'CorpusItem']]):
    """
    Overwrite a human-readable report with snippets + multi-metric scores + provenance.
    Each `results` item is (rank, scores_dict, CorpusItem).
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("="*80)
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Metric: {metric_name}")
    lines.append(f"k: {k}")
    if params:
        lines.append("Params: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    lines.append(f"Query: {q}")
    lines.append("-"*80)

    for rank, scores, item in results:
        score_str = _fmt_scores(scores)
        lines.append(f"[#{rank}]{score_str} | folder={item.folder_name} | kb_file={os.path.basename(item.kb_path)}")
        lines.append(item.text)
        lines.append("-"*80)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(PKG_DIR / "snippets_only.txt", "w", encoding="utf-8") as f:
        for _, _, item in results:
            f.write(item.text + "\n" + "-"*80 + "\n")

def ranks_desc(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks for scores sorted descending (1 = best)."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=order.dtype)
    return ranks