# models/llm_clients.py
from __future__ import annotations
import os, json, time, logging, shutil, pathlib
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# (top of models/llm_clients.py)
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


# --- add near top ---
import random, time
from collections import deque

class _RateLimiter:
    """Simple leaky bucket: at most rpm requests per minute (best-effort)."""
    def __init__(self, rpm: int = 30):
        self.rpm = max(1, int(rpm))
        self.window = deque()
        self.period = 60.0

    def wait(self):
        now = time.time()
        # discard old timestamps
        while self.window and (now - self.window[0]) > self.period:
            self.window.popleft()
        if len(self.window) >= self.rpm:
            sleep_s = self.period - (now - self.window[0]) + 0.01
            time.sleep(max(0.0, sleep_s))
        # record this call
        self.window.append(time.time())

def _backoff_sleep(attempt: int, base: float = 1.5, cap: float = 12.0):
    jitter = random.random()
    time.sleep(min(cap, (base ** attempt) + jitter))


_CLIENT_CACHE = {}

def get_client(backend: str, model: str) -> BaseLLMClient:
    key = (backend or "openrouter", model or "")
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    b = (backend or "openrouter").lower()
    if b == "openrouter":
        client = OpenRouterClient(model=model)
    elif b == "gemini":
        client = GeminiClient(model=model)
    elif b == "local":
        # Reuse one LocalHFClient on GPU to avoid re-loading/fragmentation
        client = LocalHFClient(model_or_path=model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    _CLIENT_CACHE[key] = client
    return client



# ========================
# Common interface
# ========================

class BaseLLMClient:
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 256) -> str:
        raise NotImplementedError

def _join_messages_for_prompt(messages: List[Dict[str, str]]) -> str:
    # simple deterministic join that works across providers
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)

# ========================
# OpenRouter
# ========================

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient(BaseLLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None, timeout: int = 120, rpm: Optional[int] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.limiter = _RateLimiter(int(os.getenv("OPENROUTER_RPM", rpm or 30)))

    def chat(self, messages, temperature=0.2, max_tokens=256) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        # retry loop
        for attempt in range(5):
            self.limiter.wait()
            r = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            if r.status_code in (429, 500, 502, 503, 504):
                _backoff_sleep(attempt)
                continue
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"{r.status_code} {r.reason} — {detail}")
        raise requests.HTTPError("LLM call failed after retries.")

# ========================
# Gemini (google-generativeai)
# ========================

class GeminiClient(BaseLLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None, rpm: Optional[int] = None):
        self.model = model
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini.")
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise RuntimeError("Install google-generativeai: pip install google-generativeai") from e
        genai.configure(api_key=key)
        self._genai = genai
        self._model = genai.GenerativeModel(model_name=model)
        self.limiter = _RateLimiter(int(os.getenv("GEMINI_RPM", rpm or 30)))

    def chat(self, messages, temperature=0.2, max_tokens=256) -> str:
        prompt = _join_messages_for_prompt(messages)
        for attempt in range(5):
            self.limiter.wait()
            try:
                resp = self._model.generate_content(
                    prompt,
                    generation_config=self._genai.GenerationConfig(
                        temperature=temperature, max_output_tokens=max_tokens
                    ),
                )
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                return resp.candidates[0].content.parts[0].text
            except Exception as e:
                # crude 429/5xx detection via message text
                msg = str(e).lower()
                if any(t in msg for t in ("rate limit", "429", "quota", "temporarily", "unavailable", "500", "503")):
                    _backoff_sleep(attempt)
                    continue
                raise
        raise RuntimeError("Gemini call failed after retries.")

# ========================
# Local (Transformers)
# ========================

# models/llm_clients.py  (REPLACE the LocalHFClient class with this one)

class LocalHFClient(BaseLLMClient):
    def __init__(
        self,
        model_or_path: str,
        device_map: Optional[dict | str] = None,     # NEW: allow explicit placement
        max_memory: Optional[dict] = None,            # OPTIONAL
        load_in_4bit: bool = True,                    # try 4-bit if available
    ):
        # Resolve local path
        if os.path.isdir(model_or_path):
            local_path = model_or_path
        else:
            local_path = os.path.join("cache", model_or_path)
            if not os.path.isdir(local_path):
                local_path = model_or_path
        if not os.path.isdir(local_path):
            raise RuntimeError(f"Local model path not found: {local_path}")

        # Prevent TF/Flax/JAX imports (keeps protobuf drama away)
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )  # type: ignore
        except Exception as e:
            raise RuntimeError("Install transformers (+ bitsandbytes for 4-bit).") from e

        # Select GPU 0 by default (avoid auto-shard)
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": 0}     # ← force entire model on GPU 0
            else:
                device_map = "cpu"

        # 4-bit config if bitsandbytes exists and requested
        quant_cfg = None
        if load_in_4bit:
            try:
                from bitsandbytes import __version__  # noqa: F401
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                quant_cfg = None  # fallback to full precision if bnb absent

        logger.info(f"[LocalHF] Loading model from {local_path} with device_map={device_map}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        # Try a strict GPU-only load first
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                device_map=device_map,
                torch_dtype=(torch.float16 if device_map != "cpu" else "auto"),
                trust_remote_code=True,
                quantization_config=quant_cfg,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
            )
        except ValueError as e:
            # Typical case: bnb complains some modules went to CPU/disk
            msg = str(e)
            if "dispatched on the CPU or the disk" in msg:
                raise RuntimeError(
                    "4-bit loader tried to offload to CPU/disk. "
                    "Fix: ensure GPU 0 has ~9-10GiB free, and use device_map={'':0}. "
                    "You can free memory with:\n"
                    "  import torch, gc; gc.collect(); torch.cuda.empty_cache()\n"
                    "Or reduce context (k_sub/k_final) to lower peak memory."
                ) from e
            raise
        except Exception as e:
            raise

        # Prefer chat template when available
        self._has_chat_template = hasattr(self.tokenizer, "apply_chat_template")
        self._device_is_gpu = (device_map != "cpu")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 256) -> str:
        import torch
        self.model.eval()

        if self._has_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = _join_messages_for_prompt(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self._device_is_gpu:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=max(0.01, temperature),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[-1].strip()
        return text.strip()


# ========================
# Router / factory
# ========================

def get_client(backend: str, model: str) -> BaseLLMClient:
    b = (backend or "openrouter").lower()
    if b == "openrouter":
        return OpenRouterClient(model=model)
    if b == "gemini":
        return GeminiClient(model=model)
    if b == "local":
        return LocalHFClient(model_or_path=model)
    raise ValueError(f"Unknown backend: {backend}")

def smart_client(model_hint: Optional[str] = None, backend: Optional[str] = None) -> BaseLLMClient:
    """
    Convenience: infer backend from model hint like:
      'openrouter:deepseek/deepseek-chat-v3.1:free'
      'gemini:gemini-2.5-flash'
      'local:Qwen_Qwen2.5-Coder-7B-Instruct_4bit_nf4'
    """
    if backend:
        return get_client(backend, model_hint or "")
    if not model_hint:
        raise ValueError("Provide model id or backend+model.")
    if ":" in model_hint and model_hint.split(":", 1)[0] in {"openrouter", "gemini", "local"}:
        b, m = model_hint.split(":", 1)
        return get_client(b, m)
    # default to openrouter
    return get_client("openrouter", model_hint)
