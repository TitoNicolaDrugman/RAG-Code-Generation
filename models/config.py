# app/config.py
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict, Any
import yaml

# ==== sezioni di configurazione ====

@dataclass
class AppSection:
    name: str = "hf-causal-lm-loader"
    log_level: Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"] = "INFO"
    seed: int = 42

@dataclass
class ModelSection:
    model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    # se il nome modello inizia con uno di questi prefissi => trust_remote_code=True
    trust_prefixes: Optional[List[str]] = None  # impostato in __post_init__
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True

    def __post_init__(self):
        if self.trust_prefixes is None:
            self.trust_prefixes = ["microsoft/Phi-","Qwen/"]

@dataclass
class CacheSection:
    root: str = "./cache"
    suffix: str = "_4bit_nf4"  # verrà appeso al nome modello
    # file metadati per validare la cache
    meta_filename: str = "metadata.json"

@dataclass
class QuantSection:
    enable_4bit_if_cuda: bool = True
    quant_type: Literal["nf4","fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["auto","bfloat16","float16"] = "auto"  # "auto" => bf16 se supportato, altrimenti f16

@dataclass
class AppConfig:
    app: AppSection = field(default_factory=AppSection)
    model: ModelSection = field(default_factory=ModelSection)
    cache: CacheSection = field(default_factory=CacheSection)
    quant: QuantSection = field(default_factory=QuantSection)
# ==== util ====

def _env_override(section_obj, mapping: Dict[str, str]):
    for attr, env_key in mapping.items():
        v = os.getenv(env_key)
        if v is None:
            continue
        current = getattr(section_obj, attr)
        # prova a castare al tipo corrente quando possibile
        try:
            if isinstance(current, bool):
                v_cast = v.lower() in {"1","true","yes","y"}
            elif isinstance(current, int):
                v_cast = int(v)
            else:
                v_cast = type(current)(v)
        except Exception:
            v_cast = v
        setattr(section_obj, attr, v_cast)

def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Carica la configurazione da YAML e consente override tramite variabili d'ambiente.
    """
    cfg = AppConfig()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for section_name in ("app","model","cache","quant"):
            if section_name in raw and isinstance(raw[section_name], dict):
                sect_obj = getattr(cfg, section_name)
                for k, v in raw[section_name].items():
                    if hasattr(sect_obj, k):
                        setattr(sect_obj, k, v)

    # override comuni da env (opzionali)
    _env_override(cfg.model, {
        "model_name": "MODEL_NAME",
        "device_map": "DEVICE_MAP",
    })
    _env_override(cfg.cache, {
        "root": "CACHE_ROOT",
    })
    _env_override(cfg.app, {
        "log_level": "LOG_LEVEL",
        "seed": "SEED",
    })

    return cfg

def make_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger
