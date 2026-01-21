# model.py
import torch
import torch.nn as nn


class StockLSTMModel(nn.Module):
    """
    Two-branch model:
      - LSTM over sequential OHLCV:   price_seq  [B, T, 5]
      - MLP over last-step features:  indicators [B, K]   (K inferred automatically)

    Output:
      - scalar regression prediction [B]
    """

    def __init__(
        self,
        price_input_size: int = 5,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=price_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # LazyLinear infers indicator_input_size on first forward pass
        self.ff = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        combined_input_size = lstm_hidden_size + 32

        self.combined_ff = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1),
        )

    def forward(self, price_seq: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
        """
        price_seq:  [B, T, 5]
        indicators: [B, K]  (K inferred)
        """
        lstm_out, _ = self.lstm(price_seq)
        lstm_feat = lstm_out[:, -1]          # [B, hidden]

        ff_feat = self.ff(indicators)        # [B, 32]
        combined = torch.cat([lstm_feat, ff_feat], dim=1)

        output = self.combined_ff(combined)  # [B, 1]
        return output.squeeze(-1)            # [B]
