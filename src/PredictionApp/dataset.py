import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Lazy windowing: stores each ticker's [rows, F] matrix once and slices the
    [WINDOW, F] view per sample, instead of materializing every window (~6 GB vs
    ~300 MB on sp500, same speed). Index entries are (array_id, end_row) pairs."""

    def __init__(
        self,
        feats: list[np.ndarray],
        labels: list[np.ndarray],
        index: list[tuple[int, int]],
        window: int,
    ):
        super().__init__()
        self.feats = [np.ascontiguousarray(f, dtype=np.float32) for f in feats]
        self.labels = [np.asarray(l) for l in labels]
        self.index = index
        self.window = window

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr_id, end = self.index[idx]
        x = torch.from_numpy(self.feats[arr_id][end - self.window + 1 : end + 1])
        # Shape [1] to match StockDataset (LightningModule squeezes it).
        y = torch.tensor([int(self.labels[arr_id][end])], dtype=torch.long)
        return x, y


class StockDataset(Dataset):
    """Thin wrapper holding features [N, T, F] and targets [N, 1]."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        super().__init__()
        assert features.shape[0] == targets.shape[0], "X and y must have same N"
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.targets[idx]
        return x, y
