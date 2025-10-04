"""
乱数・再現性関連のユーティリティ関数
"""
import torch
import numpy as np
import random
import os


def replicability(seed=None):
    """
    再現性を確保するための乱数シード設定
    
    Args:
        seed (int): 乱数シード（Noneの場合は設定しない）
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # Sets the seed for generating random numbers. Returns a torch.Generator object.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It's safe to call this function if CUDA is not available; in that case, it is silently ignored.insufficient to get determinism
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
