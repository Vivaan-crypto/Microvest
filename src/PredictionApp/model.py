# model.py
import torch
import torch.nn as nn


class StockLSTMModel(nn.Module):
    def __init__(self, input_size=17, lstm_hidden_size=128, lstm_layers=2, dropout_prob=0.2, num_classes=3):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
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
        lstm_out, _ = self.LSTM(price_seq)  # [B, T, hidden_size]
        #lstm_feat = lstm_out[:, -1, :]  # [B, hidden_size] - use last token

        output = self.classification_head(lstm_out)  # [B, num_classes]
        return output


class StockTransformerModel(nn.Module):
    def __init__(self, input_size=17, d_model=512, transformer_layers=3, dropout_prob=0.2, num_classes=3):
        super().__init__()
        self.input_to_transformer_linear = nn.Linear(input_size, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
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
