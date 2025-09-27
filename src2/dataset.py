import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

#-----------Transforms-------------#

#-------------Dataset--------------#
class StockDataset(Dataset):
    """Unified dataset for stock sequences"""

    def __init__(self, features, price_targets):
        self.features = features
        self.price_targets = price_targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sequence = self.features[idx]
        target = self.price_targets[idx]
        return sequence, target


