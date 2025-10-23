# snippet_retrievers/embedder.py
from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Public constants/paths expected by other modules
DEFAULT_MODEL = "microsoft/codebert-base"

PKG_DIR = Path(__file__).parent
CACHE_DIR = PKG_DIR / "vec_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Alias kept for inspect_vector.py compatibility
_VEC_DIR = CACHE_DIR





import os
import json
import time
import hashlib
from typing import List, Tuple

EMBED_CACHE_VERSION = "v2"  # bump to invalidate all old caches once

def _norm_path(p: str) -> str:
    # Normalize for Windows vs POSIX, remove ./.. differences
    return os.path.normcase(os.path.abspath(p))

def _kb_fingerprint(snippets: List[str]) -> str:
    """
    A stable fingerprint of the KB content (order-sensitive).
    If you prefer order-insensitive, sort the snippets first.
    """
    h = hashlib.md5()
    h.update(str(len(snippets)).encode("utf-8"))
    for s in snippets:
        # include length to reduce accidental collisions
        h.update(str(len(s)).encode("utf-8"))
        h.update(s.encode("utf-8", "ignore"))
    return h.hexdigest()

def _cache_id(kb_file: str, model_name: str, max_length: int, snippets: List[str]) -> str:
    """
    IMPORTANT: DO NOT include pooling, remove_top_pcs, rrf_k, etc. in the key.
    This guarantees a single cache used by cosine and hybrid alike.
    """
    key = "|".join([
        EMBED_CACHE_VERSION,
        _norm_path(kb_file),
        model_name.strip(),
        f"L={int(max_length)}",
        _kb_fingerprint(snippets),
    ])
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def _vec_cache_paths(cache_dir: str, cid: str) -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    return (
        os.path.join(cache_dir, f"{cid}.npz"),       # vectors
        os.path.join(cache_dir, f"{cid}.meta.json"), # meta
    )



def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        n = np.sqrt((x * x).sum()) + eps
        return x / n
    n = np.sqrt((x * x).sum(axis=1, keepdims=True)) + eps
    return x / n

def _hash_id(parts: List[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]

# Helper for inspect_vector.py (derive cache id the same way as the embedder)
def _cache_id_for_items(items, model_name: str,
                        pooling: str = "mean", max_length: int = 512, remove_top_pcs: int = 3) -> str:
    kb_paths = sorted({getattr(it, "kb_path", str(it)) for it in items})
    parts = [model_name, pooling, str(max_length), str(remove_top_pcs), *kb_paths]
    return _hash_id(parts)

@dataclass
class EmbeddingCacheMeta:
    model: str            # keep key name 'model' for inspect_vector.py
    model_name: str       # also store a duplicate
    pooling: str
    remove_top_pcs: int
    max_length: int
    num_snippets: int
    dim: int

class CodeEmbedder:
    """
    CodeBERT embedder with:
      - Masked mean pooling (better than CLS for RoBERTa/CodeBERT)
      - 'All-but-the-top' postprocessing to reduce anisotropy
    Cached at snippet_retrievers/vec_cache/.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 512,
        pooling: str = "mean",          # "mean" or "cls"
        remove_top_pcs: int = 3,        # 0 disables; 1-5 are common
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.remove_top_pcs = int(remove_top_pcs)
        self.batch_size = batch_size
        self.cache_dir = str(CACHE_DIR)  # ensure we have a cache dir attribute

        #self.device = torch.device("cpu")
        #self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.model = AutoModel.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, use_safetensors=True)

        self.model.eval()
        self.model.to(self.device)

        # postprocess params loaded from cache if present
        self._mu: Optional[np.ndarray] = None          # (768,)
        self._pc_top: Optional[np.ndarray] = None      # (768, k)

    # ---------- tokenization & pooling ----------

    def _ensure_model_loaded(self):
        # model & tokenizer are created in __init__, so nothing to do
        return


    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden_state: (B, T, H)
        attention_mask   : (B, T)
        """
        if self.pooling == "cls":
            return last_hidden_state[:, 0]  # (B, H)  (<s> position for RoBERTa/CodeBERT)
        # masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (last_hidden_state * mask).sum(dim=1)           # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)                 # (B, 1)
        return summed / counts                                   # (B, H)

    @torch.no_grad()
    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            tok = self._tokenize(batch).to(self.device)
            out = self.model(**tok)
            pooled = self._pool(out.last_hidden_state, tok["attention_mask"])  # (B, H)
            vecs.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(vecs, axis=0)  # (N, H)

    # ---------- post-processing (all-but-the-top) ----------
    def _fit_postprocess(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mu = X.mean(axis=0, keepdims=False)
        if self.remove_top_pcs <= 0:
            return mu.astype("float32"), None
        Xc = X - mu
        # economical SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc_top = Vt[: self.remove_top_pcs].T.astype("float32")  # (H, k)
        return mu.astype("float32"), pc_top

    def _apply_postprocess(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None:
            return X
        Xc = X - self._mu
        if self._pc_top is not None:
            Xc = Xc - Xc @ self._pc_top @ self._pc_top.T
        return Xc

    # ---------- cache id ----------
    def _cache_id(self, kb_paths: List[str]) -> str:
        parts = [self.model_name, self.pooling, str(self.max_length), str(self.remove_top_pcs)]
        parts.extend(sorted(kb_paths))
        return _hash_id(parts)

    # ---------- public API ----------
# --- paste this over the existing methods in embedder.py ---

    def get_or_build_embeddings(self, snippets: list[str], kb_file: str) -> tuple["np.ndarray", dict]:
        """
        Returns (M, meta) where M is (N,768) float32 L2-normalized.
        Caches one matrix per (KB, model, max_length, KB content).
        """
        import os, time, json
        import numpy as np
        import torch

        # cache id & paths
        cid = _cache_id(kb_file, self.model_name, self.max_length, snippets)
        npz_path, meta_path = _vec_cache_paths(self.cache_dir, cid)

        # ---- try load
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            M = np.load(npz_path)["M"]
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # if post-process params were saved, restore them for queries
            mu = np.load(npz_path).get("mu", None)
            pc = np.load(npz_path).get("pc_top", None)
            self._mu = mu.astype("float32") if mu is not None and mu.size > 0 else None
            self._pc_top = pc.astype("float32") if pc is not None and pc.size > 0 else None
            return M, meta

        # ---- build
        self._ensure_model_loaded()

        all_vecs = []
        B = self.batch_size
        for i in range(0, len(snippets), B):
            batch = snippets[i:i+B]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)           # [B,T,768]
                H = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                sum_vec = (H * mask).sum(dim=1)   # [B,768]
                lengths = mask.sum(dim=1).clamp(min=1)  # [B,1]
                vec = sum_vec / lengths           # [B,768]
                all_vecs.append(vec.cpu().numpy().astype("float32"))

        X = np.vstack(all_vecs).astype("float32")

        # fit anisotropy fix on the corpus and apply (keeps 768-dim)
        self._mu, self._pc_top = self._fit_postprocess(X)
        Xpp = self._apply_postprocess(X)
        M = _l2_normalize(Xpp).astype("float32")

        meta = {
            "id": cid,
            "version": EMBED_CACHE_VERSION,
            "model": self.model_name,
            "dim": int(M.shape[1]),
            "num_docs": int(M.shape[0]),
            "max_length": int(self.max_length),
            "kb_file": _norm_path(kb_file),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pooling": self.pooling,
            "remove_top_pcs": int(self.remove_top_pcs),
        }

        # save vectors AND post-process params so queries get the same transform
        np.savez_compressed(
            npz_path,
            M=M,
            mu=(self._mu if self._mu is not None else np.zeros((0,), dtype="float32")),
            pc_top=(self._pc_top if self._pc_top is not None else np.zeros((0,0), dtype="float32")),
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return M, meta


    def embed_query(self, text: str) -> np.ndarray:
        # raw pooled (no normalization)
        x = self._embed_raw([text])[0]                 # (768,)
        # apply SAME post-process learned on corpus (if available)
        if self._mu is not None:
            x = x - self._mu
            if self._pc_top is not None and self._pc_top.size > 0:
                x = x - x @ self._pc_top @ self._pc_top.T
        # L2 normalize
        return _l2_normalize(x).astype("float32")
