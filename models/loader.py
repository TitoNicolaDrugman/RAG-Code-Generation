# app/loader.py
from __future__ import annotations
import os
import json
import traceback
from dataclasses import asdict
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .config import AppConfig

# ===== helpers =====

def resolve_compute_dtype(pref: str = "auto") -> torch.dtype:
    """
    auto => bf16 se CUDA + is_bf16_supported, altrimenti f16
    """
    if pref == "bfloat16":
        return torch.bfloat16
    if pref == "float16":
        return torch.float16
    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def qcfg_to_dict(qcfg: Optional[BitsAndBytesConfig]) -> Optional[Dict[str, Any]]:
    if qcfg is None:
        return None
    # estrai i campi principali per confronto e serializzazione
    return {
        "load_in_4bit": bool(getattr(qcfg, "load_in_4bit", False)),
        "bnb_4bit_quant_type": getattr(qcfg, "bnb_4bit_quant_type", None),
        "bnb_4bit_compute_dtype": str(getattr(qcfg, "bnb_4bit_compute_dtype", None)),
        "bnb_4bit_use_double_quant": bool(getattr(qcfg, "bnb_4bit_use_double_quant", False)),
    }

def build_quant_config(cfg: AppConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.quant.enable_4bit_if_cuda or not torch.cuda.is_available():
        return None
    compute_dtype = resolve_compute_dtype(cfg.quant.compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.quant.double_quant,
    )

def trust_remote_code_for(model_name: str, prefixes) -> bool:
    return any(model_name.startswith(p) for p in (prefixes or []))

def cache_paths(cfg: AppConfig) -> Tuple[str, str]:
    cache_dir = os.path.join(
        cfg.cache.root,
        cfg.model.model_name.replace("/", "_") + cfg.cache.suffix
    )
    meta_file = os.path.join(cache_dir, cfg.cache.meta_filename)
    return cache_dir, meta_file

# ===== core loader =====

def load_model_and_tokenizer(cfg: AppConfig, logger=None):
    """
    Carica tokenizer & modello, usando cache locale se metadati combaciano.
    Salva su cache se caricati da remoto.
    Ritorna: (tokenizer, model, used_cache: bool)
    """
    if logger is None:
        class _N: 
            def info(self,*a,**k): pass
            def warning(self,*a,**k): pass
            def error(self,*a,**k): pass
        logger = _N()

    cache_dir, meta_file = cache_paths(cfg)
    qcfg = build_quant_config(cfg)
    compute_dtype = resolve_compute_dtype(cfg.quant.compute_dtype)

    req_meta = {
        "model_name": cfg.model.model_name,
        "quant_cfg": qcfg_to_dict(qcfg),
    }

    # verifica cache
    use_cache = False
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            use_cache = (saved == req_meta)
            logger.info(f"Cache metadata match: {use_cache}")
        except Exception:
            logger.warning("Impossibile leggere metadata.json; ignoro la cache")

    trust_code = trust_remote_code_for(cfg.model.model_name, cfg.model.trust_prefixes)
    if trust_code:
        logger.warning("trust_remote_code=True: eseguirà codice Python dal repo Hugging Face del modello.")

    try:
        if use_cache:
            logger.info("Carico da cache locale…")
            tokenizer = AutoTokenizer.from_pretrained(
                cache_dir,
                local_files_only=True,
                trust_remote_code=trust_code
            )
            model = AutoModelForCausalLM.from_pretrained(
                cache_dir,
                device_map=cfg.model.device_map,
                low_cpu_mem_usage=cfg.model.low_cpu_mem_usage,
                trust_remote_code=trust_code,
                local_files_only=True,
            )
        else:
            do_4bit = (qcfg is not None)
            logger.info(f"Nessuna cache valida. CUDA disponibile? {torch.cuda.is_available()}")
            
            # --- START FIX ---
            # Proactively check if bitsandbytes is usable before attempting a 4-bit load.
            # This is more robust than relying on the downstream exception handler,
            # especially given bitsandbytes' tendency to fail on initialization due to CUDA issues.
            if do_4bit:
                try:
                    import bitsandbytes
                    logger.info("Verifica bitsandbytes: OK. Quantizzazione 4-bit abilitata.")
                except (ImportError, RuntimeError) as e:
                    logger.warning(
                        "Impossibile inizializzare bitsandbytes (necessario per il caricamento a 4-bit), "
                        "probabilmente a causa di un problema con la configurazione CUDA. "
                        "Il caricamento continuerà senza quantizzazione. Errore: %s",
                        type(e).__name__
                    )
                    do_4bit = False # Disable 4-bit quantization
            # --- END FIX ---

            logger.info("Quantizzazione 4-bit…" if do_4bit else f"Caricamento {str(compute_dtype)}… (no 4-bit)")

            # tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.model_name,
                trust_remote_code=trust_code
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

            # model
            if do_4bit:
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            cfg.model.model_name,
                            quantization_config=qcfg,
                            device_map=cfg.model.device_map,
                            trust_remote_code=trust_code,
                            low_cpu_mem_usage=cfg.model.low_cpu_mem_usage,
                        )
                    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as e:
                        logger.warning(
                            "4-bit load failed (%s). Falling back to bf16/fp16 without bitsandbytes.",
                            type(e).__name__,
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            cfg.model.model_name,
                            torch_dtype=compute_dtype,               # e.g., torch.bfloat16
                            device_map=cfg.model.device_map,
                            trust_remote_code=trust_code,
                            low_cpu_mem_usage=cfg.model.low_cpu_mem_usage,
                        )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.model_name,
                    torch_dtype=compute_dtype,
                    device_map=cfg.model.device_map,
                    trust_remote_code=trust_code,
                    low_cpu_mem_usage=cfg.model.low_cpu_mem_usage,
                )


            # salva cache
            logger.info("Scrivo la cache…")
            os.makedirs(cache_dir, exist_ok=True)
            tokenizer.save_pretrained(cache_dir)
            model.save_pretrained(cache_dir)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(req_meta, f)
            logger.info(f"Cache salvata in {cache_dir}")

        # allinea pad_token_id
        if getattr(model, "config", None) and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        logger.info("Modello e tokenizer pronti!")
        return tokenizer, model, use_cache

    except Exception:
        logger.error("Errore durante il caricamento di modello/tokenizer:")
        logger.error(traceback.format_exc())
        raise