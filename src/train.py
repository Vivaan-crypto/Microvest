import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """Load and preprocess stock data using pandas_ta"""

    def __init__(self, ticker: str, period: str = "2y"):
        self.ticker = ticker
        self.period = period
        self.scaler = StandardScaler()

    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(period=self.period)
            return data.dropna() if not data.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using pandas_ta"""
        if data.empty:
            return data

        # Calculate indicators
        indicators = ta.Strategy(
            name="Technical Indicators",
            ta=[
                {"kind": "sma", "length": 20},
                {"kind": "ema", "length": 12},
                {"kind": "rsi"},
                {"kind": "macd"},
                {"kind": "bbands"},
                {"kind": "stoch"},
                {"kind": "atr"},
                {"kind": "adx"},
            ],
        )

        # Run the strategy
        data.ta.strategy(indicators)

        # Add price features
        data["Price_Change"] = data["Close"].pct_change()
        data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
        return data.dropna()

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        if data.empty:
            return data

        # Select relevant features
        features = data.filter(like="_", axis=1)
        features = features.drop(
            columns=[col for col in features.columns if "BBU" in col or "BBL" in col]
        )
        return self.scaler.fit_transform(features)


class StockDataset(Dataset):
    """PyTorch Dataset for stock data"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.features[idx : idx + self.seq_length],
            self.targets[idx + self.seq_length],
        )


class LSTM_Model(nn.Module):
    """LSTM Model with Attention"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


class StockPredictor:
    """Stock Predictor with TensorBoard Logging"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.writer = SummaryWriter(log_dir="../runs/microvest_runs")
        logger.info(f"Using device: {self.device}")

    def prepare_data(self, ticker: str, seq_length: int = 30) -> DataLoader:
        """Prepare data loader"""
        loader = StockDataLoader(ticker)
        raw_data = loader.fetch_data()
        data_with_indicators = loader.add_indicators(raw_data)
        features = loader.prepare_features(data_with_indicators)

        # Create targets (next day's price change)
        targets = (
            data_with_indicators["Close"].pct_change().shift(-1).values[seq_length:]
        )

        dataset = StockDataset(features, targets, seq_length)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def train(self, train_loader: DataLoader, epochs: int = 50):
        """Train model with TensorBoard logging"""
        input_size = train_loader.dataset.features.shape[1]
        self.model = LSTM_Model(input_size).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.writer.add_scalar("Loss/train", avg_loss, epoch)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        self.writer.close()

    def save_model(self, path: str = "stock_model.pth"):
        """Save trained model"""
        if self.model:
            torch.save(
                {"model_state_dict": self.model.state_dict(), "scaler": self.scaler},
                path,
            )
            logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    predictor = StockPredictor()
    data_loader = predictor.prepare_data("AAPL")
    predictor.train(data_loader)
    predictor.save_model()
