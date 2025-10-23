# pipeline/kb_ops.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import os

# Dipendenze dal tuo progetto
from utils.kb_manager import (
    set_kb_config,
    kb_base_url,
    list_available_kbs,
)

# Alcuni ambienti non hanno "requests" preinstallato; lo usiamo solo in fallback.
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # gestiamo il fallback quando serve


# -----------------------------
# Configurazione
# -----------------------------
def configure_kb(
    username: str,
    repo_name: str,
    branch: str,
    kb_folder: str,
    local_dir: str | Path,
):
    """
    Wrapper sottile attorno a set_kb_config per centralizzare la configurazione KB.
    Ritorna l'oggetto cfg del tuo kb_manager (con .local_dir).
    """
    cfg = set_kb_config(
        username=username,
        repo_name=repo_name,
        branch=branch,
        kb_folder=kb_folder,
        local_dir=str(local_dir),
    )
    return cfg


# -----------------------------
# Mappatura target KB
# -----------------------------
def default_kb_name_map() -> Dict[str, str]:
    """
    Mappa repo_name -> nome file KB su GitHub (come presenti nella tua cartella KB).
    Adatta qui se necessario.
    """
    return {
        "seed-emulator": "kb_seed-labs__seed-emulator.json",
        # Il dataset può usare "pyscf" o "pyscf__pyscf" come repo_name; la KB è questa:
        "pyscf": "kb_pyscf__pyscf.json",
        "pyscf__pyscf": "kb_pyscf__pyscf.json",
    }


def resolve_kb_names_for_repos(target_repos: Iterable[str]) -> Dict[str, str]:
    """
    Dato un elenco di repo_name target, restituisce una mappa repo_name -> kb_filename.
    Solleva KeyError se una repo non ha una KB associata nella mappa.
    """
    kb_map = default_kb_name_map()
    out = {}
    for r in target_repos:
        if r not in kb_map:
            raise KeyError(f"Nessun KB noto per repo_name='{r}'. Aggiorna default_kb_name_map().")
        out[r] = kb_map[r]
    return out


# -----------------------------
# Download / Path resolution
# -----------------------------
def _local_kb_path(cfg, kb_filename: str) -> Path:
    """
    Percorso locale atteso per un KB, usando cfg.local_dir.
    """
    return Path(cfg.local_dir).resolve() / kb_filename


def _download_via_requests(url: str, dest_path: Path) -> None:
    if requests is None:
        raise RuntimeError(
            "Il modulo 'requests' non è disponibile. "
            "Installa 'requests' oppure fornisci un downloader in utils.kb_manager."
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest_path.write_bytes(r.content)


def ensure_kb_downloaded(cfg, kb_filename: str) -> Path:
    """
    Garantisce che il file KB richiesto sia presente in locale.
    1) Se già presente in cfg.local_dir, lo riusa.
    2) Altrimenti prova a scaricarlo da {kb_base_url()}/{kb_filename} via requests.
       Se nel tuo progetto esiste già una funzione dedicata al download,
       puoi sostituire questa parte per usarla.
    Ritorna il Path locale.
    """
    local_path = _local_kb_path(cfg, kb_filename)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    base = kb_base_url().rstrip("/")
    url = f"{base}/{kb_filename}"
    _download_via_requests(url, local_path)
    return local_path


def list_kbs_from_remote(max_print: int = 20) -> List[dict]:
    """
    Restituisce i metadati dei KB disponibili da GitHub tramite list_available_kbs().
    """
    kbs = list_available_kbs()
    # Troncamento eventuale è responsabilità della cella notebook (stampa)
    return kbs


# -----------------------------
# Orchestrazione per i target
# -----------------------------
def fetch_target_kbs(
    cfg,
    target_repos: Iterable[str],
) -> List[dict]:
    """
    Per ogni repo target:
      - risolve il nome del KB
      - assicura il download in locale
      - produce un record riassuntivo con path e size
    """
    repo_to_kb = resolve_kb_names_for_repos(target_repos)
    results = []
    for repo_name, kb_fname in repo_to_kb.items():
        local_path = ensure_kb_downloaded(cfg, kb_fname)
        size = local_path.stat().st_size if local_path.exists() else None
        results.append(
            {
                "repo_name": repo_name,
                "kb_filename": kb_fname,
                "local_path": str(local_path),
                "exists": local_path.exists(),
                "size_bytes": size,
            }
        )
    return results


def print_kb_summary(cfg, target_repos: Iterable[str], downloaded_rows: List[dict]) -> None:
    print("--- RAG Knowledge Base Configuration (GitHub) ---")
    print(f"  Base URL: {kb_base_url().rstrip('/')}/kb_LIBRARY_KEY.json")
    print(f"  Local directory: {Path(cfg.local_dir).resolve()}")
    print()
    print("Target KBs:")
    for row in downloaded_rows:
        print(
            f" - {row['repo_name']}: {row['kb_filename']} "
            f"-> {row['local_path']} (exists={row['exists']}, size={row['size_bytes']})"
        )
