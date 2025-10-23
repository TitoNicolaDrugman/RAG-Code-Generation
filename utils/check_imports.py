# utils/check_imports.py
import importlib.metadata as md

REQUIRED_LIBS = [
    "torch", "transformers", "datasets", "huggingface_hub",
    "tqdm", "rank-bm25", "numpy", "requests", "ipython"
]

def check_libraries(verbose: bool = True) -> dict:
    """
    Controlla se le librerie principali sono installate e stampa versioni.
    
    Args:
        verbose (bool): se True stampa le versioni.
    
    Returns:
        dict: {lib_name: version or None}
    """
    results = {}
    for lib in REQUIRED_LIBS:
        try:
            version = md.version(lib)
            results[lib] = version
            if verbose:
                print(f"{lib} ✓ (version {version})")
        except md.PackageNotFoundError:
            results[lib] = None
            if verbose:
                print(f"{lib} ✗ (not installed)")
    return results
