# generation/c
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList


# =========================
# Stopping criteria helpers
# =========================
class EosAndCodeEndStoppingCriteria(StoppingCriteria):
    """Stops generation on EOS token or a specific code-ending sequence."""
    def __init__(self, tokenizer, stop_on_eos_token: bool = True, code_end_sequence: Optional[str] = "\n```\n"):
        self.tokenizer = tokenizer
        self.stop_on_eos = stop_on_eos_token
        self.code_end_sequence_str = code_end_sequence
        self.code_end_sequence_ids: List[int] = []
        if self.code_end_sequence_str and self.tokenizer:
            try:
                self.code_end_sequence_ids = self.tokenizer.encode(
                    self.code_end_sequence_str, add_special_tokens=False
                )
            except Exception:
                self.code_end_sequence_ids = []

    def __call__(self, current_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # EOS stop
        if self.stop_on_eos and self.tokenizer and getattr(self.tokenizer, "eos_token_id", None) is not None:
            if current_ids[0, -1] == self.tokenizer.eos_token_id:
                return True
        # code fence stop
        if self.code_end_sequence_ids:
            seq_len = len(self.code_end_sequence_ids)
            if current_ids.shape[1] >= seq_len:
                tail = current_ids[0, -seq_len:]
                if torch.equal(tail, torch.tensor(self.code_end_sequence_ids, device=tail.device)):
                    return True
        return False


# =========================
# Generation configuration
# =========================
@dataclass
class GenParams:
    max_new_tokens: int = 384
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_on_eos: bool = True
    stop_on_code_end: bool = True
    code_end_sequence: Optional[str] = "\n```\n"
    # Per evitare overflow del contesto; None = nessun truncation (usa warning dei Transformers)
    max_input_tokens: Optional[int] = None
    # Salvataggio parziale dopo N esempi
    save_every: int = 10


# =========================
# Core generation function
# =========================
def _generate_llm_code_and_clean(
    prompt_text: str,
    llm_model,
    llm_tokenizer,
    params: GenParams,
    prompt_name: str = "Prompt",
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Esegue la generazione e ritorna (raw_output, cleaned_code, stats).
    Non solleva: intercetta eccezioni e ritorna (None, None, stats con 'error').
    """
    stats: Dict[str, Any] = {"prompt_name": prompt_name}
    try:
        # Tokenize (con eventuale truncation dell'input)
        tok_kwargs = {"return_tensors": "pt"}
        if params.max_input_tokens is not None:
            tok_kwargs.update({"truncation": True, "max_length": params.max_input_tokens})
        inputs = llm_tokenizer(prompt_text, **tok_kwargs).to(llm_model.device)

        prompt_len_tokens = int(inputs["input_ids"].shape[1])
        stats["prompt_len_tokens"] = prompt_len_tokens

        # Stopping criteria
        stop_list = None
        if params.stop_on_eos or params.stop_on_code_end:
            stop_list = StoppingCriteriaList(
                [EosAndCodeEndStoppingCriteria(
                    llm_tokenizer,
                    stop_on_eos_token=params.stop_on_eos,
                    code_end_sequence=(params.code_end_sequence if params.stop_on_code_end else None),
                )]
            )

        gen_args = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask", None),
            "max_new_tokens": params.max_new_tokens,
            "pad_token_id": getattr(llm_tokenizer, "eos_token_id", None),
            "repetition_penalty": params.repetition_penalty,
            "stopping_criteria": stop_list,
            "do_sample": params.do_sample,
        }
        if params.do_sample:
            gen_args.update({"temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k})

        t0 = time.time()
        with torch.no_grad():
            output_ids = llm_model.generate(**gen_args)
        stats["gen_time_sec"] = round(time.time() - t0, 3)

        # Estrarre solo la parte nuova
        generated_ids_part = output_ids[0, prompt_len_tokens:]
        raw_output = llm_tokenizer.decode(generated_ids_part, skip_special_tokens=True)

        # Pulizia "best effort": estrai blocco markdown ```python ... ```
        cleaned = None
        m = re.search(r"```python\s*\n(.*?)(?:\n```|\Z)", raw_output, re.DOTALL | re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()
        elif "\n```" in raw_output:
            cleaned = raw_output.split("\n```")[0].strip()
        else:
            cleaned = raw_output.strip()

        return raw_output, cleaned, stats

    except torch.cuda.OutOfMemoryError as e:
        stats["error"] = f"CUDA OOM: {e}"
        return None, None, stats
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        return None, None, stats


# =========================
# I/O helpers
# =========================
def _infer_mode_and_tag(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Dato il meta del file di prompt, riconosce:
    - mode: "rag" se builder inizia per build_rag_prompt_, altrimenti "baseline"
    - tag : per i RAG include top_k (es. 'rag_top3'), per baseline 'baseline' (o 'baseline_top3' per coerenza)
    """
    builder = meta.get("builder", "")
    top_k = meta.get("top_k")
    if builder.startswith("build_rag_prompt_"):
        tag = f"rag_top{top_k}" if isinstance(top_k, int) else "rag"
        return "rag", tag
    else:
        # includo comunque top_k nel tag per allineare i dataset (ricava dal file sorgente)
        tag = f"baseline_top{top_k}" if isinstance(top_k, int) else "baseline"
        return "baseline", tag


def _default_results_path(input_prompts_json: Path) -> Path:
    meta = json.loads(input_prompts_json.read_text(encoding="utf-8")).get("meta", {})
    mode, tag = _infer_mode_and_tag(meta)
    builder = meta.get("builder", "unknown_builder")
    out_dir = Path("results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{builder}_results.json"


def _load_existing_results(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# =========================
# Public API
# =========================
def run_generation_from_prompts_file(
    input_prompts_json: str | Path,
    model,
    tokenizer,
    out_json: Optional[str | Path] = None,
    gen_params: Optional[GenParams] = None,
    overwrite: bool = False,
    resume: bool = True,
    progress: bool = True,
) -> Path:
    """
    Esegue la generazione per TUTTI i prompt contenuti in un file JSON (baseline o RAG).

    Args:
        input_prompts_json: path al JSON creato da bm25_make_prompts.py o baseline_make_prompts.py
        model, tokenizer: istanze Transformers già caricate (Sezione 2)
        out_json: path di output. Se None, viene dedotto in results/<tag>/<builder>_results.json
        gen_params: parametri di generazione (default in GenParams)
        overwrite: se True, ignora eventuali risultati esistenti
        resume: se True e out_json esiste, salta gli item già presenti
        progress: mostra tqdm

    Returns:
        Path al file JSON di risultati.
    """
    in_path = Path(input_prompts_json)
    if not in_path.is_file():
        raise FileNotFoundError(f"Prompts file not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    items: List[Dict[str, Any]] = data.get("prompts", [])
    builder = meta.get("builder", "unknown_builder")

    mode, tag = _infer_mode_and_tag(meta)
    top_k = meta.get("top_k")
    print(f"Loaded prompts: builder='{builder}' | mode={mode} | queries={len(items)} | top_k={top_k}")

    # Output path
    out_path = Path(out_json) if out_json else _default_results_path(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume / overwrite handling
    existing = None
    processed_keys: set = set()
    if out_path.is_file() and not overwrite:
        existing = _load_existing_results(out_path)
        if existing and resume:
            # build key set (idx or (repo, idx_fallback))
            for r in existing.get("results", []):
                key = r.get("idx")
                if key is None:
                    key = (r.get("repo_full_name"), r.get("instruction", "")[:40])
                processed_keys.add(key)
            print(f"Resuming: found {len(processed_keys)} already processed items in '{out_path.name}'.")

    # Gen params
    gp = gen_params or GenParams()
    gp_dict = asdict(gp)

    # Meta output
    out_meta: Dict[str, Any] = {
        "builder": builder,
        "mode": mode,
        "tag": tag,
        "top_k": top_k,
        "num_queries": len(items),
        "source_prompts_file": str(in_path),
        "llm": {
            "name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None),
            "dtype": str(getattr(getattr(model, "config", None), "torch_dtype", None)),
            "device": str(next(model.parameters()).device) if hasattr(model, "parameters") else None,
        },
        "gen_params": gp_dict,
        "timestamp": time.time(),
    }

    # Results container (merge with existing if resuming)
    results_out: List[Dict[str, Any]] = []
    if existing and resume:
        # keep existing (in order)
        results_out = existing.get("results", [])
    processed_count = len(processed_keys)

    iterator = enumerate(items)
    if progress:
        iterator = tqdm(iterator, total=len(items), desc=f"LLM gen ({builder})", unit="query")

    for i, rec in iterator:
        # build unique key for resume checks
        rec_key = rec.get("idx")
        if rec_key is None:
            rec_key = (rec.get("repo_full_name"), (rec.get("instruction") or "")[:40])

        if rec_key in processed_keys:
            continue

        instruction = rec.get("instruction") or ""
        prompt_text = rec.get("prompt") or ""
        repo = rec.get("repo_full_name")

        raw_out, cleaned, stats = _generate_llm_code_and_clean(
            prompt_text=prompt_text,
            llm_model=model,
            llm_tokenizer=tokenizer,
            params=gp,
            prompt_name=f"{builder} | idx={rec.get('idx', i)}",
        )

        result_row = {
            "idx": rec.get("idx", i),
            "repo_full_name": repo,
            "instruction": instruction,
            "prompt_len_tokens": stats.get("prompt_len_tokens"),
            "gen_time_sec": stats.get("gen_time_sec"),
            "error": stats.get("error"),
            "generated": {
                "raw": raw_out,
                "cleaned": cleaned,
            },
        }
        # opzionali dal file di prompt
        if "top_k" in rec:
            result_row["top_k"] = rec["top_k"]
        if "num_snippets_used" in rec:
            result_row["num_snippets_used"] = rec["num_snippets_used"]

        results_out.append(result_row)
        processed_count += 1

        # salvataggio incrementale
        if processed_count % gp.save_every == 0:
            payload = {"meta": out_meta, "results": results_out}
            out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # salvataggio finale
    payload = {"meta": out_meta, "results": results_out}
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results_out)} generations to: {out_path}")
    return out_path
