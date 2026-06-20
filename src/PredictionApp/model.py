# model.py
import torch
import torch.nn as nn


class StockLSTMModel(nn.Module):
    # input_size must match the feature count from preprocess.FEATURE_COLS.
    # Small on purpose: the target has a tiny edge, so a big LSTM just memorizes.
    def __init__(self, input_size=26, lstm_hidden_size=256, lstm_layers=3, dropout_prob=0.3, num_classes=3):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob if lstm_layers > 1 else 0.0,
        )
        # Classification head (outputs raw logits, NOT softmax)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_hidden_size, num_classes),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_seq: [B, T, F] - batch of sequences
        Returns:
            logits: [B, num_classes] - class logits for each sample
        """
        lstm_out, _ = self.LSTM(price_seq)   # [B, T, hidden_size]
        last_hidden = lstm_out[:, -1, :]     # [B, hidden_size] — take final timestep
        output = self.classification_head(last_hidden)  # [B, num_classes]
        return output


class StockTransformerModel(nn.Module):
    # input_size must match the feature count from preprocess.FEATURE_COLS.
    def __init__(self, input_size=26, d_model=512, transformer_layers=3, dropout_prob=0.2, num_classes=3):
        super().__init__()
        # nhead must divide d_model; pick the largest power-of-2 ≤ 8 that works.
        nhead = max(1, min(8, d_model // 8))
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
        # Classification head (outputs raw logits, NOT softmax)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_prob),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_seq: [B, T, F] - batch of sequences
        Returns:
            logits: [B, num_classes] - class logits for each sample
        """
        # Project input to d_model dimension
        x = self.input_to_transformer_linear(price_seq)  # [B, T, d_model]

        # Apply transformer encoder
        x = self.Transformer(x)  # [B, T, d_model]

        # Pool the sequence: use last token
        x = x[:, -1, :]  # [B, d_model]

        # Classification head
        output = self.classification_head(x)  # [B, num_classes]
        return output
