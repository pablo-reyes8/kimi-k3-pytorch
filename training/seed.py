# ============================================================
# Seed
# ============================================================
import torch 
import random 
import numpy as np 

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Fix random seeds for Python, NumPy and PyTorch.

    Args:
        seed:
            Global random seed.
        deterministic:
            If True, enables deterministic CuDNN behavior.
            Useful for tests, but can slow down training.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Optional extra reproducibility.
        # Some operations may fail if no deterministic implementation exists.
        # Enable only if you truly need strict determinism.
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
