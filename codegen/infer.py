# codegen/infer.py
from __future__ import annotations
import time, torch
from typing import Dict, Any, Tuple

@torch.inference_mode()
def generate_once(
    tokenizer,
    model,
    prompt_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Dict[str, Any]:
    """
    Esegue una singola generazione.
    Restituisce: dict con 'generation', 'prompt_tokens', 'gen_tokens', 'elapsed_sec'
    """
    device = getattr(model, "device", "cpu")
    if pad_token_id is None and getattr(tokenizer, "pad_token_id", None) is None:
        # workaround tipico per modelli CodeLlama
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=eos_token_id if eos_token_id is not None else tokenizer.eos_token_id,
        pad_token_id=pad_token_id if pad_token_id is not None else tokenizer.pad_token_id,
    )

    t0 = time.perf_counter()
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        **gen_kwargs,
    )
    elapsed = time.perf_counter() - t0

    # prendi solo la porzione nuova
    new_tokens = output_ids[0, input_ids.size(1):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "generation": text,
        "prompt_tokens": int(input_ids.size(1)),
        "gen_tokens": int(new_tokens.size(0)),
        "elapsed_sec": elapsed,
        "gen_kwargs": gen_kwargs,
    }
