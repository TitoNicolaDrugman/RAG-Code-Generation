from pathlib import Path

# Default output dir
OUT_DIR = Path("outputs/retrieval")

# Repo -> KB filename map
KB_NAME_MAP = {
    "seed-emulator": "kb_seed-labs__seed-emulator.json",
    "seed-labs__seed-emulator": "kb_seed-labs__seed-emulator.json",
    "pyscf": "kb_pyscf__pyscf.json",
    "pyscf__pyscf": "kb_pyscf__pyscf.json",
}

# Where KBs are stored locally (come nel resto del progetto)
KB_BASE_DIR = Path("temp_downloaded_kbs")
