"""
Utility functions for training and visualization
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


from .logger import TrainingVisualizer, visualize_training_results

__all__ = ['set_seed', 'TrainingVisualizer', 'visualize_training_results']
