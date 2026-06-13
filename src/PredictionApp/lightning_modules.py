import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from torchmetrics import F1Score, Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall
import seaborn as sns
from model import StockTransformerModel, StockLSTMModel
from dataset import StockDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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
        labels = self.train_dataset.targets.long().view(-1)
        class_counts = torch.bincount(labels)
        # Weight each sample inversely proportional to its class frequency
        sample_weights = (1.0 / class_counts.float())[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
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
    def __init__(self, y, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = StockLSTMModel()

        # === Classification Metrics ===
        # macro-averaged F1: every class weighted equally, so majority-class
        # collapse scores poorly (micro F1 == accuracy and hides collapse).
        self.train_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=3, average="macro")

        self.train_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_acc = Accuracy(task="multiclass", num_classes=3)

        self.train_precision = MulticlassPrecision(num_classes=3, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=3, average='macro')

        self.train_recall = MulticlassRecall(num_classes=3, average='macro')
        self.val_recall = MulticlassRecall(num_classes=3, average='macro')

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=3)

        # WeightedRandomSampler balances batches; use plain balanced weights here
        # so the loss still corrects for any residual imbalance without blowing up gradients.
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y).astype(np.float32)

        print(f"Class weights: {class_weights}")

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

        # === Buffering for epoch aggregation ===
        self.val_preds = []
        self.val_targets = []

    # --------------------------------------------------
    def forward(self, x):
        return self.model(x)

    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y_true = y.squeeze(-1)

        loss = self.criterion(preds, y_true)

        # Compute metrics
        f1_score = self.train_f1(preds, y_true)
        accuracy = self.train_acc(preds, y_true)
        precision = self.train_precision(preds, y_true)
        recall = self.train_recall(preds, y_true)

        # Log scalar metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/f1", f1_score, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/precision", precision, on_step=True, on_epoch=False)
        self.log("train/recall", recall, on_step=True, on_epoch=False)

        # Log confusion matrix every 500 steps (reduced from 100 to save time)
        if self.global_step % 500 == 0 and self.global_step > 0:
            cm = self.confusion_matrix(preds, y_true)
            fig = self._plot_confusion_matrix(cm, title="Train Confusion Matrix", figsize=(6, 5))
            self.logger.experiment.add_figure("train/confusion_matrix", fig, global_step=self.global_step)
            plt.close(fig)

            # Log confidence distribution
            probs = torch.softmax(preds, dim=1)
            max_probs = probs.max(dim=1)[0]
            fig_conf = self._plot_confidence_distribution(max_probs.cpu().detach().numpy(), "Training", figsize=(8, 5))
            self.logger.experiment.add_figure("train/confidence_distribution", fig_conf, global_step=self.global_step)
            plt.close(fig_conf)

        return loss

    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        # accumulate predictions and targets on CPU to avoid holding GPU memory
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.squeeze(-1).detach().cpu())

    # --------------------------------------------------
    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return

        device = self.device
        val_preds = torch.cat(self.val_preds, dim=0).to(device)
        val_targets = torch.cat(self.val_targets, dim=0).to(device)

        # Compute loss and metrics
        loss = self.criterion(val_preds, val_targets)
        f1_score = self.val_f1(val_preds, val_targets)
        accuracy = self.val_acc(val_preds, val_targets)
        precision = self.val_precision(val_preds, val_targets)
        recall = self.val_recall(val_preds, val_targets)

        # Log scalar metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/f1", f1_score, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/precision", precision, on_step=False, on_epoch=True)
        self.log("val/recall", recall, on_step=False, on_epoch=True)

        # Log confusion matrix
        cm = self.confusion_matrix(val_preds, val_targets)
        fig = self._plot_confusion_matrix(cm, title="Validation Confusion Matrix")
        self.logger.experiment.add_figure("val/confusion_matrix", fig, global_step=self.current_epoch)
        plt.close(fig)

        # Log confidence distribution
        probs = torch.softmax(val_preds, dim=1)
        max_probs = probs.max(dim=1)[0]
        fig_conf = self._plot_confidence_distribution(max_probs.cpu().detach().numpy(), "Validation")
        self.logger.experiment.add_figure("val/confidence_distribution", fig_conf, global_step=self.current_epoch)
        plt.close(fig_conf)

        # Log per-class metrics visualization
        fig_metrics = self._plot_per_class_metrics(
            val_preds.cpu().numpy(),
            val_targets.cpu().numpy(),
            ["Short", "NoTrade", "Long"]
        )
        self.logger.experiment.add_figure("val/per_class_metrics", fig_metrics, global_step=self.current_epoch)
        plt.close(fig_metrics)

        # Clear buffers
        self.val_preds.clear()
        self.val_targets.clear()

    # --------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr * 0.01
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # --------------------------------------------------
    def on_after_backward(self):
        """Log gradient statistics after backward pass (optimized - only every 50 steps)."""
        # Only log every 50 steps to reduce severe overhead
        if self.global_step % 50 != 0:
            return

        total_norm = 0.0

        # Only iterate through model parameters (more efficient)
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2) ** 2

        total_norm = total_norm ** 0.5

        # Log overall gradient norm
        self.log("train/gradient_norm", total_norm, on_step=True, on_epoch=False)


    def _plot_confusion_matrix(self, cm, title="Confusion Matrix", figsize=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
        ax.set_xlabel("Predicted labels", fontsize=10)
        ax.set_ylabel("True labels", fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        return fig

    def _plot_confidence_distribution(self, confidences, stage="", figsize=(10, 6)):
        """Plot histogram of model's confidence scores."""
        fig, ax = plt.subplots(figsize=figsize)

        # Adaptive bins: use fewer bins if data range is small
        data_range = confidences.max() - confidences.min()
        num_unique = len(np.unique(confidences))

        if data_range < 0.1 or num_unique < 5:
            bins = max(3, min(5, num_unique))  # Use 3-5 bins for small ranges
        else:
            bins = min(20, max(5, num_unique // 2))  # Use 5-20 bins adaptively

        ax.hist(confidences, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Confidence (max softmax)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{stage} Confidence Distribution", fontsize=11)
        ax.axvline(confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidences.mean():.3f}')
        ax.legend(fontsize=9)
        plt.tight_layout()
        return fig

    def _plot_per_class_metrics(self, preds, targets, class_names):
        """Plot per-class precision, recall metrics."""
        from sklearn.metrics import precision_recall_fscore_support

        # Get predicted classes
        pred_classes = np.argmax(preds, axis=1)

        # Compute per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_classes, average=None, labels=[0, 1, 2]
        )

        # Create bar plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.arange(len(class_names))
        width = 0.35

        # Precision
        axes[0].bar(x, precision, width, edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Precision by Class')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(class_names)
        axes[0].set_ylim(0, 1)

        # Recall
        axes[1].bar(x, recall, width, edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('Score')
        axes[1].set_title('Recall by Class')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names)
        axes[1].set_ylim(0, 1)

        # F1
        axes[2].bar(x, f1, width, edgecolor='black', alpha=0.7)
        axes[2].set_ylabel('Score')
        axes[2].set_title('F1 Score by Class')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(class_names)
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        return fig