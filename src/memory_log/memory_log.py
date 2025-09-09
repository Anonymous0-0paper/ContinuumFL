import torch
import psutil, os

def get_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024
    return 0.0

def log_mem(tag=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1024**2
    gpu = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"----------------------------------------\n"
          f"[{tag}] CPU={rss:.1f}MB, GPU={gpu:.1f}MB\n"
          f"----------------------------------------")