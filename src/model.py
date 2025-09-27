"""
Attention-LSTM for stock prediction
-----------------------------------
1. LSTM captures **sequential** dependencies.
2. Attention lets the network *focus* on the most relevant timesteps.
3. The whole thing is wrapped as a LightningModule so we get
   training, validation, checkpointing for free.
"""

import torch, torch.nn as nn, pytorch_lightning as L
from torchmetrics import MeanSquaredError

class AttentionLSTM(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)          # keeps cfg in checkpoints
        C = cfg                                 # short alias

        # 1) LSTM: returns (batch, seq, hidden*2) because bidirectional=True
        self.lstm = nn.LSTM(
            input_size  = C.n_features,
            hidden_size = C.hidden,
            num_layers  = C.layers,
            dropout     = C.dropout if C.layers>1 else 0,
            batch_first = True,
            bidirectional = True
        )

        # 2) Attention: query = key = value = lstm_out
        self.attn = nn.MultiheadAttention(
            embed_dim = C.hidden*2,
            num_heads = C.heads,
            dropout   = C.dropout,
            batch_first = True
        )

        # 3) Head: maps hidden → 1 (regression of next-day return)
        self.head = nn.Sequential(
            nn.Linear(C.hidden*2, C.hidden//2),
            nn.ReLU(),
            nn.Dropout(C.dropout),
            nn.Linear(C.hidden//2, 1)
        )
        self.criterion = nn.MSELoss()

    # --------------------------------------------------
    # forward pass
    # --------------------------------------------------
    def forward(self, x):
        """
        x : (batch, seq_len, n_features)
        returns raw log-return prediction
        """
        lstm_out, _ = self.lstm(x)                  # (B,T,2H)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # same shape
        # we only need last timestep after attention
        return self.head(attn_out[:,-1,:]).squeeze(-1)

    # --------------------------------------------------
    # Lightning hooks
    # --------------------------------------------------
    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, _):
        loss, *_ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, *_ = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)