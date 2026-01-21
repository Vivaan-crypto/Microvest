# dataset.py
from typing import Tuple

import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """
    Thin wrapper to hold:
        - features: [N, T, F]
        - targets:  [N, 1]
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        super().__init__()
        assert features.shape[0] == targets.shape[0], "X and y must have same N"
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.targets[idx]
        return x, y
