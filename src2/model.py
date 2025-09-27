import torch.nn as nn


class lstm_model(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)

        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # If x is 2D, add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layers
        output = self.fc_layers(last_hidden)
        return output.squeeze(-1)  # Remove last dimension for regression