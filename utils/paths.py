# utils/paths.py
from __future__ import annotations
import os

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
