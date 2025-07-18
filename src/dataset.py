import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """Unified dataset for stock sequences"""

    def __init__(self, features, targets, sequence_length=30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = self.features[idx:idx + self.seq_len]
        target = self.targets[idx + self.seq_len - 1]
        return sequence, target