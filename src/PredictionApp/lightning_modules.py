import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from model import StockLSTMModel
from dataset import StockDataset
from lightning.pytorch.loggers import TensorBoardLogger


# ==================================================
# DataModule
# ==================================================
class LightningDateModule(L.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size):
        super().__init__()
        self.train_dataset = StockDataset(X_train, y_train)
        self.val_dataset = StockDataset(X_val, y_val)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


# ==================================================
# Lightning Module
# ==================================================
class LightningModule(L.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = StockLSTMModel()
        self.criterion = nn.SmoothL1Loss(beta=1.0)

        self.val_preds = []
        self.val_targets = []

    # --------------------------------------------------
    def forward(self, x):
        price_seq = x[:, :, :5]
        indicators = x[:, -1, 5:]
        return self.model(price_seq, indicators)

    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y_true = y.squeeze(-1)

        loss = self.criterion(preds, y_true)
        r2 = r2_score(y_true.detach().cpu().numpy(),
                      preds.detach().cpu().numpy())

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/r2", r2, prog_bar=True)

        return loss

    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y_true = y.squeeze(-1)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y_true.detach().cpu())

    # --------------------------------------------------
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        loss = self.criterion(preds, targets)
        r2 = r2_score(targets.numpy(), preds.numpy())

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/r2", r2, prog_bar=True)

        # ---------- TensorBoard extras (VAL ONLY) ----------
        if isinstance(self.logger, TensorBoardLogger):
            tb = self.logger.experiment
            epoch = self.current_epoch

            # ---- Histogram: preds vs truth ----
            tb.add_histogram("val/predictions", preds.numpy(), epoch)
            tb.add_histogram("val/targets", targets.numpy(), epoch)

            # ---- Scatter plot (pred vs truth) ----
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(targets.numpy(), preds.numpy(), alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("True Return (scaled)")
            ax.set_ylabel("Predicted Return (scaled)")
            ax.set_title("Validation: Predicted vs True")

            tb.add_figure("val/pred_vs_true", fig, epoch)
            plt.close(fig)

            # ---- Prediction text samples ----
            n = min(15, len(preds))
            lines = [
                f"Pred: {preds[i].item():+.4f} | True: {targets[i].item():+.4f}"
                for i in range(n)
            ]
            tb.add_text("val/sample_predictions", "\n".join(lines), epoch)

        self.val_preds.clear()
        self.val_targets.clear()

    # --------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
