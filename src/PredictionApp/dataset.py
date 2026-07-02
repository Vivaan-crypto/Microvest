# dataset.py
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Lazy windowing: stores each ticker's [rows, F] feature matrix ONCE and
    slices the [WINDOW, F] view per sample in __getitem__.

    Why: materializing every window duplicates each row WINDOW times. For the
    S&P 500 universe over 20 years that's ~6 GB of float32; the lazy version
    is ~300 MB. Slicing a contiguous array is cheap, so training speed is the
    same.

    Index entries are (array_id, end_row) pairs: the sample is
    feats[array_id][end_row - WINDOW + 1 : end_row + 1].
    """

    def __init__(self, feats: List[np.ndarray], labels: List[np.ndarray],
                 index: List[Tuple[int, int]], window: int):
        super().__init__()
        self.feats = [np.ascontiguousarray(f, dtype=np.float32) for f in feats]
        self.labels = [np.asarray(l) for l in labels]
        self.index = index
        self.window = window

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr_id, end = self.index[idx]
        x = torch.from_numpy(self.feats[arr_id][end - self.window + 1: end + 1])
        # Shape [1] to match StockDataset's targets (LightningModule squeezes it).
        y = torch.tensor([int(self.labels[arr_id][end])], dtype=torch.long)
        return x, y


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
