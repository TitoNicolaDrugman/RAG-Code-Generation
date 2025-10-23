from pathlib import Path

# Output dir di default
OUT_DIR = Path("outputs/retrieval")

# Mappa repo -> filename della KB
KB_NAME_MAP = {
    "seed-emulator": "kb_seed-labs__seed-emulator.json",
    "seed-labs__seed-emulator": "kb_seed-labs__seed-emulator.json",
    "pyscf": "kb_pyscf__pyscf.json",
    "pyscf__pyscf": "kb_pyscf__pyscf.json",
}

# Root locale dove abbiamo scaricato le KB
KB_BASE_DIR = Path("temp_downloaded_kbs")
