# lightning_train.py

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import os
import numpy as np
from lightning_modules import LightningModule, LightningDateModule


def main():
    # --------------------------------------------------
    # Load pre-split tensors (already processed)
    # --------------------------------------------------
    # Try loading from CSV directory first, then fall back to Data directory
    data_dir = "Data/CSV" if os.path.exists("Data/CSV/X_train.pt") else "Data"

    X_train = torch.load(f"{data_dir}/X_train.pt")
    y_train = torch.load(f"{data_dir}/y_train.pt")
    X_val = torch.load(f"{data_dir}/X_test.pt")
    y_val = torch.load(f"{data_dir}/y_test.pt")

    print(f"Loaded data from: {data_dir}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Ensure targets are integer class indices in 0..C-1 and dtype long
    # Handle shape [N,1] or [N]
    if y_train.dim() == 2 and y_train.shape[1] == 1:
        y_train = y_train.squeeze(-1)
    if y_val.dim() == 2 and y_val.shape[1] == 1:
        y_val = y_val.squeeze(-1)

    # If labels are floats, convert to long
    y_train = y_train.long()
    y_val = y_val.long()

    # If labels contain negative values (e.g., -1,0,1) for directionality, shift them to non-negative
    # This converts: -1,0,1 -> 0,1,2 (Down, Flat, Up)
    min_label = int(min(int(y_train.min().item()), int(y_val.min().item())))
    if min_label < 0:
        shift = -min_label
        y_train = y_train + shift
        y_val = y_val + shift
        print(f"Shifted labels by {shift} (was in range [{min_label}, {int(max(int(y_train.max().item()), int(y_val.max().item())-shift))}])")

    print(f"Label range: [{int(y_train.min().item())}, {int(y_train.max().item())}]")
    print(f"Number of classes: {int(y_train.max().item()) + 1}")

    # --------------------------------------------------
    # Create Lightning model
    # --------------------------------------------------
    # Compute balanced class weights with increased imbalance handling
    class_counts = np.bincount(y_train.numpy())
    print(f"Class distribution in training data:")
    print(f"  Class 0 (Down): {class_counts[0]} samples ({100*class_counts[0]/len(y_train):.1f}%)")
    if len(class_counts) > 1:
        print(f"  Class 1 (Flat): {class_counts[1]} samples ({100*class_counts[1]/len(y_train):.1f}%)")
    if len(class_counts) > 2:
        print(f"  Class 2 (Up):   {class_counts[2]} samples ({100*class_counts[2]/len(y_train):.1f}%)")
    print()

    data_module = LightningDateModule(X_train, y_train, X_val, y_val, batch_size=32)
    model = LightningModule(y_train.numpy())

    # --------------------------------------------------
    # TensorBoard logger
    # --------------------------------------------------
    logger = TensorBoardLogger(
        "lightning_logs",
        name="stock_prediction_model",
    )

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    # Save best model based on validation F1 score
    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val/f1:.3f}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    # Log learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Early stopping if validation F1 doesn't improve
    early_stopping = EarlyStopping(
        monitor="val/f1",
        patience=20,
        mode="max",
        verbose=True,
        min_delta=0.001,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        max_epochs=200,
        accelerator="cpu",
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    # --------------------------------------------------
    # Train + Validate
    # --------------------------------------------------
    trainer.fit(model, data_module)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    torch.save(model.state_dict(), "model_final.pth")
    print("Model saved to model_final.pth")
    print(f"\nBest model checkpoint directory: {checkpoint_callback.dirpath}")
    print(f"Training logs: {logger.log_dir}")


if __name__ == "__main__":
    main()
