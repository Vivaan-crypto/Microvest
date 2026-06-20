import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall
import seaborn as sns
from model import StockTransformerModel, StockLSTMModel
from dataset import StockDataset
from sklearn.utils.class_weight import compute_class_weight
from metrics import evaluate_signal
import numpy as np

CLASS_NAMES = ("Short", "NoTrade", "Long")


class LightningDateModule(L.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size):
        super().__init__()
        self.train_dataset = StockDataset(X_train, y_train)
        self.val_dataset = StockDataset(X_val, y_val)
        self.batch_size = batch_size

    def train_dataloader(self):
        # Natural distribution + shuffle. Imbalance is handled once, in the loss.
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, drop_last=True, num_workers=5, pin_memory=False, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self):
        # shuffle=False keeps prediction order aligned with the ranking side-data.
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class LightningModule(L.LightningModule):
    def __init__(self, y, lr=1e-3, weight_decay=1e-3, val_fwd_ret=None, val_dates=None):
        super().__init__()
        self.save_hyperparameters(ignore=["y", "val_fwd_ret", "val_dates"])

        # Ranking side-data, aligned 1:1 with the val set.
        self.val_fwd_ret = None if val_fwd_ret is None else np.asarray(val_fwd_ret, dtype=float)
        self.val_dates = None if val_dates is None else np.asarray(val_dates)

        #self.model = StockTransformerModel(d_model=64, transformer_layers=1)
        self.model = torch.compile(StockLSTMModel())
        # Macro-F1 (collapse-sensitive) + per-class recall = per-class accuracy.
        self.train_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
        self.train_acc_pc = MulticlassRecall(num_classes=3, average=None)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=3)

        class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y).astype(np.float32)
        print(f"Class weights: {class_weights}")
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

        self.val_preds, self.val_targets = [], []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y_true = y.squeeze(-1)
        loss = self.criterion(preds, y_true)

        # Epoch-aggregated so train/f1 is directly comparable to val/f1.
        if self.global_step % 5 == 0:
            self.train_f1.update(preds, y_true)
            self.train_acc_pc.update(preds, y_true)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.val_preds.append(self(x).detach().cpu())
        self.val_targets.append(y.squeeze(-1).detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        pred_cls = preds.argmax(1)

        self.log("val/loss", self.criterion(preds, targets), prog_bar=True)
        self.log("val/f1", self.val_f1(preds, targets), prog_bar=True)
        self.log("val/accuracy", (pred_cls == targets).float().mean())

        # Per-class accuracy (correct / total per class) for train and val.
        val_acc = self._per_class_acc(pred_cls, targets)
        for name, a in zip(CLASS_NAMES, val_acc):
            self.log(f"val/acc_{name}", a)

        train_acc = None
        if not self.trainer.sanity_checking:
            train_acc = self.train_acc_pc.compute().cpu().numpy()
            self.train_acc_pc.reset()
            for name, a in zip(CLASS_NAMES, train_acc):
                self.log(f"train/acc_{name}", float(a))

        self._log_figure("confusion_matrix", self._plot_confusion(
            self.confusion_matrix(preds, targets)))
        self._log_figure("per_class_accuracy", self._plot_per_class_acc(train_acc, val_acc))

        # Ranking metrics: the real objective. IC ignores calibration, so it can
        # keep rising even as CE val/loss climbs (sharper, over-confident softmax).
        if self.val_fwd_ret is not None and preds.shape[0] == len(self.val_fwd_ret):
            rep = evaluate_signal(self.val_fwd_ret, proba=torch.softmax(preds, 1).numpy(),
                                  dates=self.val_dates, verbose=False)
            for src, dst in (("ic_spearman", "val/ic_spearman"),
                             ("decile_spread", "val/decile_spread"),
                             ("ls_sharpe", "val/ls_sharpe")):
                v = rep.get(src, float("nan"))
                if np.isfinite(v):
                    self.log(dst, float(v), prog_bar=(dst == "val/ic_spearman"))

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr * 0.01)
        return {"optimizer": opt, "lr_scheduler": sched}

    @staticmethod
    def _per_class_acc(pred_cls, targets):
        out = []
        for c in range(3):
            mask = targets == c
            out.append(float((pred_cls[mask] == c).float().mean()) if mask.any() else float("nan"))
        return out

    def _log_figure(self, name, fig):
        self.logger.experiment.add_figure(f"val/{name}", fig, global_step=self.current_epoch)
        plt.close(fig)

    def _plot_confusion(self, cm):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Validation Confusion Matrix")
        fig.tight_layout()
        return fig

    def _plot_per_class_acc(self, train_acc, val_acc):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(3)
        w = 0.38
        if train_acc is not None:
            ax.bar(x - w / 2, train_acc, w, label="train", color="#4C72B0")
            ax.bar(x + w / 2, val_acc, w, label="val", color="#DD8452")
        else:
            ax.bar(x, val_acc, w, label="val", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy (correct / total in class)")
        ax.set_title("Per-class Accuracy")
        ax.legend()
        fig.tight_layout()
        return fig
