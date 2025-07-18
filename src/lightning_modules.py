import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.utils.data import DataLoader
from dataset import StockDataset


class StockDataModule(L.LightningDataModule):
    """Simplified data module"""

    def __init__(self, train_features, train_targets, val_features, val_targets,
                 batch_size=64, sequence_length=30):
        super().__init__()
        self.train_features = train_features
        self.train_targets = train_targets
        self.val_features = val_features
        self.val_targets = val_targets
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def setup(self, stage=None):
        self.train_dataset = StockDataset(
            self.train_features, self.train_targets, self.sequence_length
        )
        self.val_dataset = StockDataset(
            self.val_features, self.val_targets, self.sequence_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class StockPredictor(L.LightningModule):
    """Simplified Lightning model"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2,
                 learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)