import torch
import torch.nn as nn
import pytorch_lightning as L


class AttentionLSTM(L.LightningModule):
    """Advanced LSTM with attention mechanism"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3,
                 attention_heads=4, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        lstm_output_size = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            lstm_output_size, attention_heads, dropout=dropout, batch_first=True
        )

        self.feature_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_output = attended_out[:, -1, :]
        normalized = self.batch_norm(last_output)
        return self.feature_layers(normalized)

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