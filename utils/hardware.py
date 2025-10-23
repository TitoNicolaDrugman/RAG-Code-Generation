# utils/hardware.py
import torch

def check_gpu(verbose: bool = True) -> dict:
    """
    La seguente funzione controlla se una GPU CUDA è disponibile e ritorna info di base.
    
    Args:
        verbose (bool): se True, stampa le info a schermo.
    
    Returns:
        dict: contiene 'available', 'count', 'name', 'current_device'.
    """
    gpu_available = torch.cuda.is_available()
    info = {
        "available": gpu_available,
        "count": torch.cuda.device_count() if gpu_available else 0,
        "name": torch.cuda.get_device_name(0) if gpu_available else None,
        "current_device": torch.cuda.current_device() if gpu_available else None
    }

    if verbose:
        print("Is GPU available?:", gpu_available)
        if gpu_available:
            print("Number of GPUs:", info["count"])
            print("GPU Name:", info["name"])
            print("Current device:", info["current_device"])
        else:
            print("Using CPU only")

    return info
