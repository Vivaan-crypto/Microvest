import lightning as L
from lightning.pytorch.utilities import grad_norm

from dataset import StockDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from model import lstm_model
from sklearn.metrics import r2_score
import random

class LightningDataModule(L.LightningDataModule):
    def __init__(self, x_path: str, y_path: str, batch_size: int = 32, test_size: float = 0.2, random_seed: int = 42):
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed

    def setup(self, stage):
        x = torch.load(self.x_path)
        y = torch.load(self.y_path)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_seed)
        self.train_dataset = StockDataset(x_train, y_train)
        self.test_dataset = StockDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class LightningModule(L.LightningModule):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = lstm_model()
        self.criterion = torch.nn.MSELoss()

        # Store predictions & targets during validation
        self.val_preds = []
        self.val_targets = []

    def shared_step(self, batch, stage: str):
        features, price = batch
        price = price.squeeze(dim=1)
        output = self.model(features)

        loss = self.criterion(output, price)

        if stage == "val":
            # Store for end-of-epoch logging
            self.val_preds.append(output.detach().cpu())
            self.val_targets.append(price.detach().cpu())

        # Safe R² calculation
        y_true = price.detach().cpu().numpy()
        y_pred = output.detach().cpu().numpy()
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = float("nan")

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/r2", r2, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def on_validation_epoch_end(self):
        # Concatenate all validation batches
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        # Pick 5 random indices
        indices = random.sample(range(len(preds)), min(5, len(preds)))

        # Format text for TensorBoard
        text_lines = ["Predicted vs Target vs Error (% points)"]
        for idx in indices:
            pred_val = preds[idx].item() * 100  # convert to %
            target_val = targets[idx].item() * 100  # convert to %
            error = pred_val - target_val
            text_lines.append(f"Pred: {pred_val:+.2f}%, Target: {target_val:+.2f}%, Err: {error:+.2f}%")

        log_text = "\n".join(text_lines)
        if self.logger:
            self.logger.experiment.add_text(
                tag="val/random_predictions",
                text_string=log_text,
                global_step=self.current_epoch
            )

        # Clear for next epoch
        self.val_preds = []
        self.val_targets = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
