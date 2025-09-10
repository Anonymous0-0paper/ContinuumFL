import torch
import psutil
import os

def get_rss_mb():
    try:
        # Cross-platform memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        # RSS (Resident Set Size) in MB
        return memory_info.rss / 1024 / 1024
    except Exception:
        # Fallback for systems where psutil doesn't work
        return 0.0

def log_mem(tag=""):
    rss = get_rss_mb()
    gpu = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"----------------------------------------\n"
          f"[{tag}] CPU={rss:.1f}MB, GPU={gpu:.1f}MB\n"
          f"----------------------------------------")