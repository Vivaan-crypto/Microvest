import torch
import torch.nn as nn


class StockLSTMModel(nn.Module):
    def __init__(self, price_input_size=5, indicator_input_size=6,
                 lstm_hidden_size=256, lstm_layers=3, dropout_prob=0.2):
        """
        Args:
            price_input_size: number of sequential features (OHLCV etc.)
            indicator_input_size: number of extra features (technical indicators, sentiment)
            lstm_hidden_size: hidden size of LSTM
            lstm_layers: number of LSTM layers
            dropout_prob: dropout rate
        """
        super().__init__()

        """
        OHLCV stock data goes through the LSTM
        Indicators go through 'ff'
        the output of the both heads goes through the combined head giving us the final output.
        """
        # LSTM branch for sequential price data
        self.lstm = nn.LSTM(
            input_size=price_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Feedforward branch for indicators
        self.ff = nn.Sequential(
            nn.Linear(indicator_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Combined head
        self.combined_ff = nn.Sequential(
            nn.Linear(lstm_hidden_size + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 1)
        )

    def forward(self, price_seq, indicators):
        """
        price_seq: [batch, seq_len, price_input_size]
        indicators: [batch, indicator_input_size]
        """
        # LSTM branch
        lstm_out, (hidden, cell) = self.lstm(price_seq)
        lstm_feat = lstm_out[:, -1, :]  # last time-step hidden

        # Feedforward branch
        ff_feat = self.ff(indicators)

            # Concatenate features
        combined = torch.cat([lstm_feat, ff_feat], dim=1)

        # Final regression head
        output = self.combined_ff(combined)
        return output.squeeze(-1)
