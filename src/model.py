import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    "TICKERS": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
    "PERIOD": "2y",
    "INTERVAL": "1d",
    "EPOCHS": 100,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "MODEL_PATH": "stock_model.pth",
    "SCALER_PATH": "feature_scaler.pth",
    "SEQUENCE_LENGTH": 30,
    "DROPOUT_RATE": 0.3,
    "VALIDATION_SPLIT": 0.2,
    "TEST_SPLIT": 0.1,
}


class StockDataProcessor:
    """Handles data fetching and feature engineering"""

    def __init__(self, tickers, period="2y", interval="1d"):
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.scaler = StandardScaler()

    def fetch_data(self):
        """Fetch stock data for all tickers"""
        print(f"Fetching data for {len(self.tickers)} tickers...")
        data_dict = {}

        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker, period=self.period, interval=self.interval, progress=False
                )
                if not df.empty:
                    df = df.dropna()
                    data_dict[ticker] = df
                    print(f"✓ {ticker}: {len(df)} days")
                else:
                    print(f"✗ {ticker}: No data")
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")

        return data_dict

    def safe_divide(self, numerator, denominator, fill_value=0):
        """Safely divide two arrays/series, handling division by zero"""
        result = np.where(denominator != 0, numerator / denominator, fill_value)
        return result

    def calculate_sma(self, prices, window):
        """Calculate Simple Moving Average manually"""
        if len(prices) < window:
            return np.full(len(prices), np.nan)

        sma = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            sma[i] = np.mean(prices[i - window + 1 : i + 1])
        return sma

    def calculate_ema(self, prices, span):
        """Calculate Exponential Moving Average manually"""
        if len(prices) == 0:
            return np.array([])

        alpha = 2 / (span + 1)
        ema = np.full(len(prices), np.nan)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def calculate_rolling_std(self, prices, window):
        """Calculate rolling standard deviation manually"""
        if len(prices) < window:
            return np.full(len(prices), np.nan)

        rolling_std = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            rolling_std[i] = np.std(prices[i - window + 1 : i + 1])
        return rolling_std

    def calculate_rolling_min(self, prices, window):
        """Calculate rolling minimum manually"""
        if len(prices) < window:
            return np.full(len(prices), np.nan)

        rolling_min = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            rolling_min[i] = np.min(prices[i - window + 1 : i + 1])
        return rolling_min

    def calculate_rolling_max(self, prices, window):
        """Calculate rolling maximum manually"""
        if len(prices) < window:
            return np.full(len(prices), np.nan)

        rolling_max = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            rolling_max[i] = np.max(prices[i - window + 1 : i + 1])
        return rolling_max

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI manually with proper error handling"""
        if len(prices) < window + 1:
            return np.full(len(prices), np.nan)

        # Calculate price changes
        deltas = np.diff(prices)

        # Initialize arrays
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate initial averages
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])

        # Initialize RSI array
        rsi = np.full(len(prices), np.nan)

        # Calculate RSI for each point
        for i in range(window, len(prices)):
            if i == window:
                # Use simple average for first calculation
                current_avg_gain = avg_gain
                current_avg_loss = avg_loss
            else:
                # Use smoothed averages
                current_avg_gain = (avg_gain * (window - 1) + gains[i - 1]) / window
                current_avg_loss = (avg_loss * (window - 1) + losses[i - 1]) / window

            # Calculate RS and RSI
            if current_avg_loss == 0:
                rsi[i] = 100
            else:
                rs = current_avg_gain / current_avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

            # Update averages for next iteration
            avg_gain = current_avg_gain
            avg_loss = current_avg_loss

        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD manually"""
        if len(prices) < slow:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan)

        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        # Only calculate signal where MACD is not NaN
        valid_macd = macd_line[~np.isnan(macd_line)]
        if len(valid_macd) >= signal:
            signal_line = np.full(len(prices), np.nan)
            macd_signal = self.calculate_ema(valid_macd, signal)
            # Map back to original array
            valid_indices = np.where(~np.isnan(macd_line))[0]
            signal_line[valid_indices] = macd_signal
        else:
            signal_line = np.full(len(prices), np.nan)

        return macd_line, signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands manually"""
        if len(prices) < window:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan)

        # Calculate SMA and rolling standard deviation
        sma = self.calculate_sma(prices, window)
        rolling_std = self.calculate_rolling_std(prices, window)

        # Calculate bands
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)

        return upper_band, lower_band

    def calculate_stochastic(self, high, low, close, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator manually"""
        if len(close) < k_window:
            return np.full(len(close), np.nan), np.full(len(close), np.nan)

        # Calculate %K
        highest_high = self.calculate_rolling_max(high, k_window)
        lowest_low = self.calculate_rolling_min(low, k_window)

        # Calculate %K
        k_values = np.full(len(close), np.nan)
        for i in range(k_window - 1, len(close)):
            high_low_diff = highest_high[i] - lowest_low[i]
            if high_low_diff != 0:
                k_values[i] = ((close[i] - lowest_low[i]) / high_low_diff) * 100
            else:
                k_values[i] = 50  # Default to 50 when no range

        # Calculate %D (SMA of %K)
        d_values = self.calculate_sma(k_values, d_window)

        return k_values, d_values

    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators with robust manual calculations"""
        data = df.copy()

        # Convert to numpy arrays for calculations and ensure they're 1D
        close_prices = np.array(data["Close"]).flatten()
        high_prices = np.array(data["High"]).flatten()
        low_prices = np.array(data["Low"]).flatten()
        open_prices = np.array(data["Open"]).flatten()
        volume = np.array(data["Volume"]).flatten()

        n = len(close_prices)

        # Basic price features with consistent length
        returns = np.full(n, np.nan)
        log_returns = np.full(n, np.nan)
        gap = np.full(n, np.nan)

        if n > 1:
            price_changes = close_prices[1:] - close_prices[:-1]
            returns[1:] = price_changes / close_prices[:-1]
            log_returns[1:] = np.log(close_prices[1:] / close_prices[:-1])
            gap[1:] = open_prices[1:] - close_prices[:-1]

        data["returns"] = returns
        data["log_returns"] = log_returns
        data["price_change"] = close_prices - open_prices
        data["daily_range"] = high_prices - low_prices
        data["gap"] = gap
        data["high_low_ratio"] = self.safe_divide(high_prices, low_prices, 1)
        data["open_close_ratio"] = self.safe_divide(open_prices, close_prices, 1)

        # Moving averages
        for window in [5, 10, 20, 50]:
            sma = self.calculate_sma(close_prices, window)
            data[f"sma_{window}"] = sma
            data[f"price_sma_{window}_ratio"] = self.safe_divide(close_prices, sma, 1)
            data[f"sma_{window}_slope"] = np.concatenate([[np.nan], np.diff(sma)])

        # Exponential moving averages
        ema_12 = self.calculate_ema(close_prices, 12)
        ema_26 = self.calculate_ema(close_prices, 26)
        data["ema_12"] = ema_12
        data["ema_26"] = ema_26
        data["ema_cross"] = ema_12 - ema_26

        # Volatility indicators
        data["volatility_5"] = self.calculate_rolling_std(returns, 5)
        data["volatility_20"] = self.calculate_rolling_std(returns, 20)
        data["volatility_50"] = self.calculate_rolling_std(returns, 50)
        data["volatility_ratio"] = self.safe_divide(
            data["volatility_5"].values, data["volatility_20"].values, 1
        )

        # Volume indicators
        volume_sma_10 = self.calculate_sma(volume, 10)
        volume_sma_20 = self.calculate_sma(volume, 20)
        data["volume_sma_10"] = volume_sma_10
        data["volume_sma_20"] = volume_sma_20
        data["volume_ratio"] = self.safe_divide(volume, volume_sma_20, 1)
        data["price_volume"] = close_prices * volume

        # Volume Price Trend
        price_change_pct = np.concatenate(
            [[0], np.diff(close_prices) / close_prices[:-1]]
        )
        data["volume_price_trend"] = price_change_pct * volume

        # Technical indicators
        data["rsi"] = self.calculate_rsi(close_prices)

        # MACD
        macd_line, signal_line = self.calculate_macd(close_prices)
        data["macd"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_histogram"] = macd_line - signal_line

        # Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger_bands(close_prices)
        data["bb_upper"] = bb_upper
        data["bb_lower"] = bb_lower
        data["bb_width"] = bb_upper - bb_lower

        # Bollinger Band position
        bb_range = bb_upper - bb_lower
        data["bb_position"] = np.where(
            bb_range != 0, (close_prices - bb_lower) / bb_range, 0.5
        )

        # Momentum indicators
        for period in [5, 10, 20]:
            if len(close_prices) > period:
                momentum = np.full(len(close_prices), np.nan)
                roc = np.full(len(close_prices), np.nan)

                momentum[period:] = close_prices[period:] / close_prices[:-period] - 1
                roc[period:] = (
                    close_prices[period:] - close_prices[:-period]
                ) / close_prices[:-period]

                data[f"momentum_{period}"] = momentum
                data[f"roc_{period}"] = roc
            else:
                data[f"momentum_{period}"] = np.nan
                data[f"roc_{period}"] = np.nan

        # Stochastic Oscillator
        stoch_k, stoch_d = self.calculate_stochastic(
            high_prices, low_prices, close_prices
        )
        data["stoch_k"] = stoch_k
        data["stoch_d"] = stoch_d

        # Support and resistance levels
        support_20 = self.calculate_rolling_min(low_prices, 20)
        resistance_20 = self.calculate_rolling_max(high_prices, 20)
        data["support_20"] = support_20
        data["resistance_20"] = resistance_20
        data["support_distance"] = self.safe_divide(
            close_prices - support_20, close_prices, 0
        )
        data["resistance_distance"] = self.safe_divide(
            resistance_20 - close_prices, close_prices, 0
        )

        # Lagged features
        for lag in [1, 2, 3, 5]:
            if len(returns) > lag:
                data[f"returns_lag_{lag}"] = np.concatenate(
                    [np.full(lag, np.nan), returns[:-lag]]
                )
                data[f"volume_lag_{lag}"] = np.concatenate(
                    [np.full(lag, np.nan), volume[:-lag]]
                )
                data[f"rsi_lag_{lag}"] = np.concatenate(
                    [np.full(lag, np.nan), data["rsi"].values[:-lag]]
                )
                data[f"volatility_lag_{lag}"] = np.concatenate(
                    [np.full(lag, np.nan), data["volatility_20"].values[:-lag]]
                )
            else:
                data[f"returns_lag_{lag}"] = np.nan
                data[f"volume_lag_{lag}"] = np.nan
                data[f"rsi_lag_{lag}"] = np.nan
                data[f"volatility_lag_{lag}"] = np.nan

        # Target: next day return
        if len(returns) > 1:
            data["target"] = np.concatenate([returns[1:], [np.nan]])
        else:
            data["target"] = np.nan

        # Replace infinite values and clip extreme values
        data = data.replace([np.inf, -np.inf], np.nan)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = np.clip(data[col], -10, 10)

        return data

    def create_sequences(self, data, sequence_length):
        """Create sequences for time series prediction"""
        # Select only numeric columns and exclude target
        feature_cols = [
            col
            for col in data.columns
            if col != "target" and data[col].dtype in ["float64", "int64"]
        ]

        sequences = []
        targets = []

        for i in range(sequence_length, len(data)):
            sequence_data = data.iloc[i - sequence_length : i][feature_cols]
            target_value = data.iloc[i]["target"]

            # Convert to numpy array for consistent handling
            sequence_array = sequence_data.values

            # Check if sequence has any NaN values or if target is NaN
            # Use numpy methods for consistent checking
            sequence_has_nan = np.isnan(sequence_array).any()
            target_has_nan = (
                np.isnan(target_value) if not pd.isna(target_value) else True
            )

            if not sequence_has_nan and not target_has_nan:
                sequences.append(sequence_array)
                targets.append(target_value)

        return np.array(sequences), np.array(targets)

    def prepare_data(self, sequence_length=30):
        """Main data preparation pipeline"""
        data_dict = self.fetch_data()

        all_features = []
        all_targets = []

        for ticker, df in data_dict.items():
            print(f"Processing {ticker}...")

            # Calculate technical indicators
            processed_df = self.calculate_technical_indicators(df)

            # Remove rows with NaN values
            processed_df = processed_df.dropna()

            if len(processed_df) < sequence_length + 50:
                print(f"Skipping {ticker}: insufficient data after processing")
                continue

            # Create sequences
            sequences, targets = self.create_sequences(processed_df, sequence_length)

            if len(sequences) > 0:
                all_features.append(sequences)
                all_targets.append(targets)
                print(f"✓ {ticker}: {len(sequences)} sequences created")

        if not all_features:
            raise ValueError(
                "No valid sequences created. Check your data and parameters."
            )

        # Combine all data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)

        # Final cleanup
        mask = ~(
            np.isnan(X).any(axis=(1, 2))
            | np.isnan(y)
            | np.isinf(X).any(axis=(1, 2))
            | np.isinf(y)
        )
        X = X[mask]
        y = y[mask]

        print(f"\nFinal dataset:")
        print(f"Total sequences: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        return X, y


class StockPredictor(nn.Module):
    """Advanced neural network for stock prediction"""

    def __init__(
        self, input_dim, sequence_length, hidden_dim=128, num_layers=2, dropout=0.3
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Feed-forward network
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last timestep
        last_hidden = attn_out[:, -1, :]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Final prediction
        output = self.fc_layers(last_hidden)

        return output


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train_model(self):
        """Main training pipeline"""
        # Prepare data
        processor = StockDataProcessor(
            self.config["TICKERS"], self.config["PERIOD"], self.config["INTERVAL"]
        )

        X, y = processor.prepare_data(self.config["SEQUENCE_LENGTH"])

        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = processor.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self.config["TEST_SPLIT"],
            random_state=42,
            shuffle=False,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config["VALIDATION_SPLIT"],
            random_state=42,
            shuffle=False,
        )

        print(
            f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = (
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_t = (
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

        # Initialize model
        model = StockPredictor(
            input_dim=X_train.shape[2],
            sequence_length=self.config["SEQUENCE_LENGTH"],
            dropout=self.config["DROPOUT_RATE"],
        ).to(self.device)

        # Optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(), lr=self.config["LEARNING_RATE"], weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        print("\nStarting training...")
        for epoch in range(self.config["EPOCHS"]):
            model.train()
            train_loss = 0

            # Mini-batch training
            for i in range(0, len(X_train_t), self.config["BATCH_SIZE"]):
                batch_x = X_train_t[i : i + self.config["BATCH_SIZE"]]
                batch_y = y_train_t[i : i + self.config["BATCH_SIZE"]]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                val_r2 = r2_score(y_val_t.cpu().numpy(), val_outputs.cpu().numpy())

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "scaler": processor.scaler,
                        "config": self.config,
                    },
                    self.config["MODEL_PATH"],
                )
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == self.config["EPOCHS"] - 1:
                print(
                    f"Epoch {epoch+1:3d}: Train Loss={train_loss/len(X_train_t)*self.config['BATCH_SIZE']:.6f}, "
                    f"Val Loss={val_loss:.6f}, Val R2={val_r2:.4f}"
                )

            if patience_counter >= 20:
                print("Early stopping triggered")
                break

        # Final evaluation
        print("\nFinal evaluation...")
        checkpoint = torch.load(self.config["MODEL_PATH"])
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t)
            test_r2 = r2_score(y_test_t.cpu().numpy(), test_outputs.cpu().numpy())
            test_mse = mean_squared_error(
                y_test_t.cpu().numpy(), test_outputs.cpu().numpy()
            )

        print(f"Test Results:")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {np.sqrt(test_mse):.6f}")

        return model, processor.scaler


def main():
    """Main execution function"""
    trainer = ModelTrainer(CONFIG)
    model, scaler = trainer.train_model()
    print(f"\nModel and scaler saved to {CONFIG['MODEL_PATH']}")


if __name__ == "__main__":
    main()
