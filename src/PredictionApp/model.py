import torch
import torch.nn as nn


class StockLSTMModel(nn.Module):
    # input_size must match preprocess.FEATURE_COLS. Small on purpose: a tiny edge
    # means a big LSTM just memorizes, and CPU cost scales with size.
    def __init__(self, input_size=26, lstm_hidden_size=64, lstm_layers=2, dropout_prob=0.2, num_classes=3):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob if lstm_layers > 1 else 0.0,
        )
        # Raw logits, not softmax.
        self.classification_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_hidden_size, num_classes),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        """price_seq [B, T, F] -> logits [B, num_classes]."""
        lstm_out, _ = self.LSTM(price_seq)
        last_hidden = lstm_out[:, -1, :]  # final timestep
        return self.classification_head(last_hidden)


class StockTransformerModel(nn.Module):
    # input_size must match preprocess.FEATURE_COLS.
    def __init__(self, input_size=26, d_model=512, transformer_layers=3, dropout_prob=0.2, num_classes=3):
        super().__init__()
        nhead = max(1, min(8, d_model // 8))  # must divide d_model
        self.input_to_transformer_linear = nn.Linear(input_size, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_prob,
            batch_first=True,
            norm_first=True
        )
        self.Transformer = nn.TransformerEncoder(
            self.TransformerEncoderLayer,
            num_layers=transformer_layers,
            norm=self.LayerNorm,
            enable_nested_tensor=True
        )
        # Raw logits, not softmax.
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_prob),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        """price_seq [B, T, F] -> logits [B, num_classes]."""
        x = self.input_to_transformer_linear(price_seq)
        x = self.Transformer(x)
        x = x[:, -1, :]  # last token
        return self.classification_head(x)
