import logging
import warnings
import yaml
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytorch_lightning as L
import torch
import yfinance as yf
from absl.logging import log_every_n

from lightning_modules import StockDataModule, StockPredictor as LightningStockPredictor
from model import AttentionLSTM
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler


class StockPredictorPipeline:
    """Simplified stock prediction pipeline with config loading"""

    def __init__(self, config_path: str = "C:/Users/shahv/OneDrive/Documents/GitHub/Microvest/src/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_environment()

        self.tickers = self.config['data']['tickers']
        self.period = self.config['data']['period']
        self.prediction_days = self.config['data']['prediction_days']
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.use_attention = self.config['model']['use_attention']

        self.scaler = RobustScaler()
        self.model = None

        self.ticker_encodings = {
            ticker: self._create_encoding(i, len(self.tickers))
            for i, ticker in enumerate(self.tickers)
        }

        self.feature_columns = self.config['data']['features'].copy()
        for ticker in self.tickers:
            self.feature_columns.append(f"ticker_{ticker}")

        logging.info(f"Initialized predictor for {len(self.tickers)} tickers with {len(self.feature_columns)} features")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_environment(self):
        """Setup logging, warnings, and random seeds"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Suppress warnings if configured
        if self.config['system']['suppress_warnings']:
            warnings.filterwarnings("ignore")

        # Set random seeds
        seed = self.config['system']['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # PyTorch deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_encoding(self, index: int, total: int) -> np.ndarray:
        """Create one-hot encoding for ticker"""
        encoding = np.zeros(total)
        encoding[index] = 1.0
        return encoding

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe using config parameters with robust NaN handling"""
        # Ensure we have enough data
        min_required_length = max(
            self.config['data']['volatility_window'],
            self.config['data']['volume_ratio_window'],
            self.config['data']['sma_period'],
            self.config['data']['ema_period'],
            self.config['data']['rsi_period'],
            26  # For MACD
        )

        if len(df) < min_required_length:
            logging.warning(f"Insufficient data length: {len(df)} < {min_required_length}")
            return df

        # Price-based features
        df["price_change"] = df["Close"].pct_change()

        # Use config parameters for technical indicators
        volatility_window = self.config['data']['volatility_window']
        volume_ratio_window = self.config['data']['volume_ratio_window']
        sma_period = self.config['data']['sma_period']
        ema_period = self.config['data']['ema_period']
        rsi_period = self.config['data']['rsi_period']

        # Calculate rolling statistics with proper handling
        df["volatility"] = df["price_change"].rolling(window=volatility_window, min_periods=1).std()

        # Volume ratio with safety checks
        volume_ma = df["Volume"].rolling(window=volume_ratio_window, min_periods=1).mean()
        df["volume_ratio"] = df["Volume"] / volume_ma

        # Replace infinite values with NaN, then fill
        df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan)

        # Technical indicators with error handling
        try:
            # SMA
            df.ta.sma(length=sma_period, append=True)
            if f"SMA_{sma_period}" not in df.columns:
                df[f"SMA_{sma_period}"] = df["Close"].rolling(window=sma_period, min_periods=1).mean()

            # EMA
            df.ta.ema(length=ema_period, append=True)
            if f"EMA_{ema_period}" not in df.columns:
                df[f"EMA_{ema_period}"] = df["Close"].ewm(span=ema_period, adjust=False).mean()

            # RSI
            df.ta.rsi(length=rsi_period, append=True)
            if f"RSI_{rsi_period}" not in df.columns:
                # Simple RSI calculation as fallback
                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
                rs = gain / loss
                df[f"RSI_{rsi_period}"] = 100 - (100 / (1 + rs))

            # MACD
            df.ta.macd(append=True)
            if "MACD_12_26_9" not in df.columns:
                # Simple MACD calculation as fallback
                ema12 = df["Close"].ewm(span=12).mean()
                ema26 = df["Close"].ewm(span=26).mean()
                df["MACD_12_26_9"] = ema12 - ema26

            # OBV
            df.ta.obv(append=True)
            if "OBV" not in df.columns:
                # Simple OBV calculation as fallback
                df["OBV"] = (df["Volume"] * df["Close"].diff().apply(
                    lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()

        except Exception as e:
            logging.warning(f"Error calculating technical indicators: {e}")
            # Create fallback indicators
            for col in [f"SMA_{sma_period}", f"EMA_{ema_period}", f"RSI_{rsi_period}", "MACD_12_26_9", "OBV"]:
                if col not in df.columns:
                    df[col] = df["Close"]  # Use close price as fallback

        # Handle NaN values in all columns
        # Forward fill first, then backward fill, then fill remaining with 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Final check for infinite values
        df = df.replace([np.inf, -np.inf], 0)

        return df

    def _prepare_ticker_data(self, ticker: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for a single ticker with robust error handling"""
        try:
            # Fetch data
            data = yf.Ticker(ticker).history(period=self.period)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")

            # Check for minimum data length
            min_length = self.config['model']['sequence_length'] + self.prediction_days + 50  # Extra buffer
            if len(data) < min_length:
                logging.warning(f"Insufficient data for {ticker}: {len(data)} < {min_length}")
                # Try to get more data
                data = yf.Ticker(ticker).history(period="5y")
                if len(data) < min_length:
                    raise ValueError(f"Still insufficient data for {ticker} after trying 5y period")

            # Remove any existing NaN values in raw data
            data = data.dropna()

            # Ensure we have positive volume values
            data = data[data['Volume'] > 0]

            if len(data) < min_length:
                raise ValueError(f"Insufficient valid data for {ticker} after cleaning")

            # Add technical indicators
            data = self._add_technical_indicators(data)

            # Create feature DataFrame with all expected columns
            feature_data = pd.DataFrame(index=data.index)

            # Map config feature names to actual column names
            feature_mapping = {
                "Close": "Close",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Volume": "Volume",
                "price_change": "price_change",
                "volatility": "volatility",
                "volume_ratio": "volume_ratio",
                "SMA_20": f"SMA_{self.config['data']['sma_period']}",
                "EMA_12": f"EMA_{self.config['data']['ema_period']}",
                "RSI_14": f"RSI_{self.config['data']['rsi_period']}",
                "MACD_12_26_9": "MACD_12_26_9",
                "OBV": "OBV"
            }

            # Add base features
            for config_col in self.config['data']['features']:
                actual_col = feature_mapping.get(config_col, config_col)
                if actual_col in data.columns:
                    feature_data[config_col] = data[actual_col]
                else:
                    feature_data[config_col] = 0
                    logging.warning(f"Missing feature {config_col} for {ticker}, filling with 0")

            # Add ticker encodings
            for t in self.tickers:
                feature_data[f"ticker_{t}"] = 1 if t == ticker else 0

            # Final cleanup - remove any remaining NaN values
            feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            feature_data = feature_data.replace([np.inf, -np.inf], 0)

            # Prepare targets
            close_prices = data["Close"].values
            if len(close_prices) < self.prediction_days:
                raise ValueError(f"Not enough data to create targets for {ticker}")

            targets = np.array([
                (close_prices[i + self.prediction_days] - close_prices[i]) / close_prices[i]
                for i in range(len(close_prices) - self.prediction_days)
            ])

            # Ensure feature_data and targets have matching lengths
            min_len = min(len(feature_data), len(targets))
            feature_data = feature_data.iloc[:min_len]
            targets = targets[:min_len]

            # Final validation - check for NaN values
            if feature_data.isnull().any().any():
                logging.error(f"NaN values found in feature data for {ticker}")
                # Drop rows with NaN values
                feature_data = feature_data.dropna()
                targets = targets[:len(feature_data)]

            if np.isnan(targets).any():
                logging.error(f"NaN values found in targets for {ticker}")
                # Remove NaN targets
                valid_idx = ~np.isnan(targets)
                targets = targets[valid_idx]
                feature_data = feature_data.iloc[valid_idx]

            logging.info(f"Prepared {len(feature_data)} samples for {ticker}")

            return feature_data[self.feature_columns].values, targets, close_prices[-min_len:]

        except Exception as e:
            logging.error(f"Error preparing data for {ticker}: {e}")
            raise

    def _combine_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Combine data from all tickers with robust error handling"""
        all_features, all_targets = [], []
        successful_tickers = []

        for ticker in self.tickers:
            try:
                features, targets, _ = self._prepare_ticker_data(ticker)

                # Validate data
                if len(features) == 0 or len(targets) == 0:
                    logging.warning(f"No valid data for {ticker}")
                    continue

                if np.isnan(features).any() or np.isnan(targets).any():
                    logging.warning(f"NaN values detected in {ticker}, skipping")
                    continue

                all_features.append(features)
                all_targets.append(targets)
                successful_tickers.append(ticker)
                logging.info(f"Successfully processed {ticker}: {len(features)} samples")

            except Exception as e:
                logging.warning(f"Failed to process {ticker}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid data found for any ticker")

        # Combine all data
        combined_features = np.concatenate(all_features, axis=0)
        combined_targets = np.concatenate(all_targets, axis=0)

        logging.info(f"Combined data from {len(successful_tickers)} tickers: {combined_features.shape[0]} samples")

        # Verify feature count
        expected_features = len(self.feature_columns)
        if combined_features.shape[1] != expected_features:
            raise ValueError(f"Feature count mismatch! Expected {expected_features}, got {combined_features.shape[1]}")

        # Final validation
        if np.isnan(combined_features).any():
            logging.error("NaN values found in combined features")
            # Remove rows with NaN values
            valid_idx = ~np.isnan(combined_features).any(axis=1)
            combined_features = combined_features[valid_idx]
            combined_targets = combined_targets[valid_idx]

        if np.isnan(combined_targets).any():
            logging.error("NaN values found in combined targets")
            # Remove rows with NaN targets
            valid_idx = ~np.isnan(combined_targets)
            combined_features = combined_features[valid_idx]
            combined_targets = combined_targets[valid_idx]

        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features)

        # Final check after scaling
        if np.isnan(scaled_features).any():
            logging.error("NaN values found after scaling")
            raise ValueError("NaN values in scaled features")

        logging.info(f"Final dataset: {scaled_features.shape[0]} samples, {scaled_features.shape[1]} features")
        return scaled_features, combined_targets

    def train(self) -> Dict:
        """Train the model using config parameters"""
        logging.info("Starting training...")

        # Get config parameters
        model_config = self.config['model']
        epochs = model_config['epochs']
        batch_size = model_config['batch_size']
        learning_rate = model_config['learning_rate']
        validation_split = model_config['validation_split']
        sequence_length = model_config['sequence_length']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']
        dropout = model_config['dropout']
        attention_heads = model_config['attention_heads']
        early_stopping_patience = model_config['early_stopping_patience']

        # Prepare data
        features, targets = self._combine_data()

        # Train/val split
        split_idx = int(len(features) * (1 - validation_split))
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]

        # Create data module
        data_module = StockDataModule(
            train_features, train_targets, val_features, val_targets,
            batch_size, sequence_length
        )

        # Create model
        if self.use_attention:
            model = AttentionLSTM(
                input_size=features.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                attention_heads=attention_heads,
                learning_rate=learning_rate
            )
        else:
            model = LightningStockPredictor(
                input_size=features.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate
            )

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=early_stopping_patience, mode="min"),
            ModelCheckpoint(
                dirpath=self.output_dir / "models",
                filename="best_model",
                monitor="val_loss",
                mode="min"
            )
        ]

        # Train
        trainer = L.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            logger=TensorBoardLogger("runs", name="stock_model"),
            enable_progress_bar=True,
            log_every_n_steps= self.config['model']['log_every_n_steps'],
        )

        trainer.fit(model, data_module)

        # Load best model
        best_model_path = callbacks[1].best_model_path
        if self.use_attention:
            self.model = AttentionLSTM.load_from_checkpoint(best_model_path)
        else:
            self.model = LightningStockPredictor.load_from_checkpoint(best_model_path)

        self.model.eval()

        # Save complete model
        self._save_model()

        return {
            "samples": len(features),
            "features": features.shape[1],
            "val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
            "model_path": best_model_path
        }

    def _save_model(self):
        """Save model with metadata and feature validation"""
        if not self.model:
            return

        # Verify feature counts
        expected_features = len(self.config['data']['features']) + len(self.tickers)
        if self.scaler.n_features_in_ != expected_features:
            logging.error(
                f"Critical: Scaler has {self.scaler.n_features_in_} features but expected {expected_features}")
            raise ValueError("Feature count mismatch in saved model")

        model_dir = Path(self.config['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)

        save_path = model_dir / self.config['paths']['complete_model_file']

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hyperparameters": self.model.hparams,
            "scaler": self.scaler,
            "tickers": self.tickers,
            "ticker_encodings": self.ticker_encodings,
            "feature_columns": self.feature_columns,
            "use_attention": self.use_attention,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "feature_count_verification": expected_features
        }, save_path)

        logging.info(f"Model saved to {save_path} with {expected_features} features")

    def evaluate(self) -> Dict:
        """Evaluate model performance with robust error handling"""
        if not self.model:
            raise ValueError("No trained model found")

        logging.info("Evaluating model...")
        sequence_length = self.config['model']['sequence_length']

        all_predictions, all_targets = [], []
        ticker_results = {}

        for ticker in self.tickers:
            try:
                features, targets, prices = self._prepare_ticker_data(ticker)

                if len(features) < sequence_length:
                    logging.warning(f"Insufficient data for evaluation of {ticker}")
                    continue

                scaled_features = self.scaler.transform(features)

                # Check for NaN values after scaling
                if np.isnan(scaled_features).any():
                    logging.warning(f"NaN values in scaled features for {ticker}")
                    continue

                # Generate predictions
                predictions, valid_targets = [], []
                for i in range(len(scaled_features) - sequence_length + 1):
                    sequence = torch.FloatTensor(scaled_features[i:i + sequence_length]).unsqueeze(0)
                    target = targets[i + sequence_length - 1]

                    with torch.no_grad():
                        pred = self.model(sequence).squeeze().cpu().numpy()

                        # Check for NaN in prediction
                        if np.isnan(pred) or np.isnan(target):
                            continue

                    predictions.append(pred)
                    valid_targets.append(target)

                if not predictions:
                    logging.warning(f"No valid predictions for {ticker}")
                    continue

                # Calculate metrics
                predictions = np.array(predictions)
                valid_targets = np.array(valid_targets)

                mae = mean_absolute_error(valid_targets, predictions)
                rmse = np.sqrt(mean_squared_error(valid_targets, predictions))
                direction_acc = np.mean(np.sign(predictions) == np.sign(valid_targets))

                # Dollar metrics
                avg_price = np.mean(prices)
                dollar_mae = mae * avg_price
                dollar_rmse = rmse * avg_price

                ticker_results[ticker] = {
                    "mae": mae,
                    "rmse": rmse,
                    "direction_accuracy": direction_acc,
                    "dollar_mae": dollar_mae,
                    "dollar_rmse": dollar_rmse,
                    "avg_price": avg_price,
                    "prediction_count": len(predictions)
                }

                all_predictions.extend(predictions)
                all_targets.extend(valid_targets)

                logging.info(f"Evaluated {ticker}: {len(predictions)} predictions, MAE={mae:.6f}")

            except Exception as e:
                logging.warning(f"Failed to evaluate {ticker}: {e}")
                continue

        # Overall metrics
        if all_predictions:
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)

            overall_mae = mean_absolute_error(all_targets, all_predictions)
            overall_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            overall_direction_acc = np.mean(np.sign(all_predictions) == np.sign(all_targets))

            return {
                "overall_mae": overall_mae,
                "overall_rmse": overall_rmse,
                "overall_direction_accuracy": overall_direction_acc,
                "ticker_results": ticker_results,
                "total_samples": len(all_predictions)
            }

        return {"error": "No successful evaluations"}

    def run_pipeline(self) -> Dict:
        """Run complete training and evaluation pipeline"""
        logging.info("Starting pipeline...")

        # Train
        train_results = self.train()

        # Evaluate
        eval_results = self.evaluate()

        return {
            "training": train_results,
            "evaluation": eval_results,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main execution function"""
    # Create predictor with config
    predictor = StockPredictorPipeline(config_path="C:/Users/shahv/OneDrive/Documents/GitHub/Microvest/src/config.yaml")

    # Run pipeline
    results = predictor.run_pipeline()

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    # Training
    train = results["training"]
    print(f"\nTraining: {train['samples']:,} samples, {train['features']} features")
    print(f"Validation Loss: {train['val_loss']:.6f}")

    # Evaluation
    if "evaluation" in results and "overall_mae" in results["evaluation"]:
        eval_res = results["evaluation"]
        print(f"\nEvaluation:")
        print(f"  MAE: {eval_res['overall_mae']:.6f}")
        print(f"  RMSE: {eval_res['overall_rmse']:.6f}")
        print(f"  Direction Accuracy: {eval_res['overall_direction_accuracy']:.4f}")

        print(f"\nTicker Performance:")
        for ticker, metrics in eval_res["ticker_results"].items():
            print(f"  {ticker}: MAE=${metrics['dollar_mae']:.2f}, "
                  f"RMSE=${metrics['dollar_rmse']:.2f}, "
                  f"Dir Acc={metrics['direction_accuracy']:.4f}, "
                  f"Predictions={metrics['prediction_count']}")

    print("\n" + "=" * 50)
    return results


if __name__ == "__main__":
    main()