# utils/kb_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import requests

from .github_utils import download_github_raw_json

@dataclass
class KBConfig:
    username: str
    repo_name: str
    branch: str = "main"
    kb_folder: str = "knowledge_bases_prod"
    local_dir: str = "./temp_downloaded_kbs"

    @property
    def raw_base_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.username}/{self.repo_name}/{self.branch}/{self.kb_folder}"

    @property
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.username}/{self.repo_name}/contents/{self.kb_folder}?ref={self.branch}"

# --- stato globale semplice (config corrente) ---
_KB_CONFIG: Optional[KBConfig] = None

def set_kb_config(
    username: str,
    repo_name: str,
    branch: str = "main",
    kb_folder: str = "knowledge_bases_prod",
    local_dir: str = "./temp_downloaded_kbs",
) -> KBConfig:
    """
    Imposta la configurazione KB e crea la cartella locale se non esiste.
    """
    global _KB_CONFIG
    _KB_CONFIG = KBConfig(username=username, repo_name=repo_name, branch=branch, kb_folder=kb_folder, local_dir=local_dir)
    os.makedirs(_KB_CONFIG.local_dir, exist_ok=True)
    return _KB_CONFIG

def get_kb_config() -> KBConfig:
    if _KB_CONFIG is None:
        raise RuntimeError("KB config is not set. Call set_kb_config(...) first.")
    return _KB_CONFIG

def kb_base_url() -> str:
    return get_kb_config().raw_base_url

def list_available_kbs() -> List[Dict[str, Any]]:
    """
    Elenca i file KB disponibili nel repo GitHub (sola metadata list dal GitHub API).
    Ritorna una lista di dict con almeno: name, download_url (raw), size (se disponibile).
    """
    cfg = get_kb_config()
    print("Fetching KB files from GitHub…")
    resp = requests.get(cfg.api_url, timeout=30)
    resp.raise_for_status()
    items = resp.json()

    # tieni solo i .json
    kb_files = [it for it in items if isinstance(it, dict) and it.get("name", "").endswith(".json")]
    results = []
    for it in kb_files:
        name = it.get("name")
        raw_url = f"{cfg.raw_base_url}/{name}"
        size = it.get("size")  # può essere None
        results.append({"name": name, "raw_url": raw_url, "size": size})
    return results

def load_kb(library_key: str, overwrite: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """
    Scarica (se necessario) e carica la KB per la libreria indicata (repo_full_name).
    Esempio: library_key='pyscf__pyscf' → file 'kb_pyscf__pyscf.json'
    Ritorna: (kb_json, local_path) — kb_json può essere list o dict (a seconda del formato).
    """
    cfg = get_kb_config()
    filename = f"kb_{library_key}.json"
    raw_url  = f"{cfg.raw_base_url}/{filename}"
    local_subdir = os.path.join(cfg.local_dir, library_key)
    os.makedirs(local_subdir, exist_ok=True)

    kb_json = download_github_raw_json(
        raw_url=raw_url,
        local_save_dir=local_subdir,
        filename=filename,
        overwrite=overwrite,
    )
    local_path = os.path.join(local_subdir, filename) if kb_json is not None else None
    return kb_json, local_path

def print_kb_summary(kb: Any) -> None:
    """
    Stampa un riassunto del contenuto KB (list/dict).
    """
    if kb is None:
        print("KB is None (failed to load).")
        return
    if isinstance(kb, list):
        print(f"KB type: list — entries: {len(kb)}")
        sample = str(kb[0])[:300] + ("..." if len(str(kb[0])) > 300 else "") if kb else "(empty)"
        print("Sample entry:\n", sample)
    elif isinstance(kb, dict):
        keys = list(kb.keys())
        print(f"KB type: dict — top-level keys: {keys}")
    else:
        print(f"KB type: {type(kb).__name__} — unsupported preview.")
