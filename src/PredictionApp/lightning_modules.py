import config
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dataset import StockDataset
from metrics import evaluate_signal
from model import StockLSTMModel
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall

CLASS_NAMES = ("Short", "NoTrade", "Long")


class LightningDateModule(L.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size):
        super().__init__()
        self.train_dataset = StockDataset(X_train, y_train)
        self.val_dataset = StockDataset(X_val, y_val)
        self.batch_size = batch_size

    def _loader_kwargs(self):
        # persistent_workers/prefetch_factor are only valid when num_workers > 0
        # (0 on Windows -- see config.py).
        kw = dict(num_workers=config.DATALOADER_WORKERS)
        if config.DATALOADER_WORKERS > 0:
            kw.update(persistent_workers=True, prefetch_factor=2)
        return kw

    def train_dataloader(self):
        # Natural distribution + shuffle; imbalance is handled in the loss.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        # shuffle=False keeps prediction order aligned with the ranking side-data.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )


class LightningModule(L.LightningModule):
    def __init__(
        self,
        y,
        lr=1e-3,
        weight_decay=1e-3,
        val_fwd_ret=None,
        val_dates=None,
        input_size=26,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["y", "val_fwd_ret", "val_dates"])

        # Ranking side-data aligned 1:1 with the val set (pass fwd_z so IC grades
        # the vol-normalized target the model trains on).
        self.val_fwd_ret = (
            None if val_fwd_ret is None else np.asarray(val_fwd_ret, dtype=float)
        )
        self.val_dates = None if val_dates is None else np.asarray(val_dates)

        self.model = StockLSTMModel(
            input_size=input_size,
            lstm_hidden_size=hidden_size,
            lstm_layers=num_layers,
            dropout_prob=dropout,
        )
        # Macro-F1 (collapse-sensitive) + per-class recall = per-class accuracy.
        self.train_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
        self.train_acc_pc = MulticlassRecall(num_classes=3, average=None)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=3)

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y), y=y
        ).astype(np.float32)
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

        # Loss every step (cheap); heavier torchmetrics subsampled. Epoch-aggregated
        # so train/f1 is comparable to val/f1.
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.global_step % 5 == 0:
            self.train_f1.update(preds, y_true)
            self.train_acc_pc.update(preds, y_true)
            self.log(
                "train/f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # .float(): under bf16-mixed the logits come back bf16; upcast so the loss
        # (float32 class weights) and downstream metrics don't hit a dtype clash.
        self.val_preds.append(self(x).detach().float().cpu())
        self.val_targets.append(y.squeeze(-1).detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        # Accumulated on CPU (validation_step) to avoid growing GPU memory; move the
        # tiny [N,3]/[N] batch back onto the model's device for the loss/metrics.
        preds = torch.cat(self.val_preds).to(self.device)
        targets = torch.cat(self.val_targets).to(self.device)
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

        self._log_figure(
            "confusion_matrix",
            self._plot_confusion(self.confusion_matrix(preds, targets)),
        )
        self._log_figure(
            "per_class_accuracy", self._plot_per_class_acc(train_acc, val_acc)
        )

        # Ranking metrics: the real objective. IC ignores calibration, so it can
        # rise even as CE val/loss climbs.
        if self.val_fwd_ret is not None and preds.shape[0] == len(self.val_fwd_ret):
            rep = evaluate_signal(
                self.val_fwd_ret,
                proba=torch.softmax(preds, 1).cpu().numpy(),
                dates=self.val_dates,
                verbose=False,
            )
            for src, dst in (
                ("ic_spearman", "val/ic_spearman"),
                ("decile_spread", "val/decile_spread"),
                ("ls_sharpe", "val/ls_sharpe"),
            ):
                v = rep.get(src, float("nan"))
                # A degenerate epoch (constant preds -> NaN spearman) logs -1 so
                # mode="max" treats it as terrible instead of erroring mid-run.
                if dst == "val/ic_spearman" and not np.isfinite(v):
                    v = -1.0
                if np.isfinite(v):
                    self.log(dst, float(v), prog_bar=(dst == "val/ic_spearman"))

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        # Weight decay on weight matrices only; decaying biases/LayerNorm gains
        # measurably hurts small models like this one.
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    @staticmethod
    def _per_class_acc(pred_cls, targets):
        out = []
        for c in range(3):
            mask = targets == c
            out.append(
                float((pred_cls[mask] == c).float().mean())
                if mask.any()
                else float("nan")
            )
        return out

    def _log_figure(self, name, fig):
        # Only TensorBoard-style loggers expose add_figure; skip cleanly otherwise.
        exp = getattr(self.logger, "experiment", None) if self.logger else None
        if exp is not None and hasattr(exp, "add_figure"):
            exp.add_figure(f"val/{name}", fig, global_step=self.current_epoch)
        plt.close(fig)

    def _plot_confusion(self, cm):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm.cpu().numpy(),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
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
