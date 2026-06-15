# lightning_train.py

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import os
import numpy as np
from lightning_modules import LightningModule, LightningDateModule, CLASS_NAMES

# --------------------------------------------------
# Training configuration
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 25
MONITOR = "val_f1"          # slash-free macro-F1 alias (see lightning_modules)
MONITOR_MODE = "max"


def main():
    L.seed_everything(SEED, workers=True)

    # --------------------------    ------------------------
    # Load pre-split tensors (already processed)
    # --------------------------------------------------
    # Try loading from CSV directory first, then fall back to Data directory
    data_dir = "Data"

    X_train = torch.load(f"Data/X_train.pt")
    y_train = torch.load(f"Data/y_train.pt")
    X_val = torch.load(f"Data/X_test.pt")
    y_val = torch.load(f"Data/y_test.pt")

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
    # Report class distribution
    class_counts = np.bincount(y_train.numpy())
    print("Class distribution in training data:")
    for i, name in enumerate(CLASS_NAMES):
        if i < len(class_counts):
            print(f"  Class {i} ({name}): {class_counts[i]} samples "
                  f"({100 * class_counts[i] / len(y_train):.1f}%)")
    print()

    data_module = LightningDateModule(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE)
    model = LightningModule(y_train.numpy(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
    # Save best models on macro-F1 (slash-free alias avoids creating subdirs)
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR,
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val_f1:.3f}",
        save_top_k=3,
        mode=MONITOR_MODE,
        save_last=True,
    )

    # Log learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Early stopping if validation macro-F1 doesn't improve
    early_stopping = EarlyStopping(
        monitor=MONITOR,
        patience=EARLY_STOP_PATIENCE,
        mode=MONITOR_MODE,
        verbose=True,
        min_delta=0.001,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        deterministic=True,
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
    best_score = checkpoint_callback.best_model_score
    print(f"\nBest {MONITOR}: {best_score.item():.4f}" if best_score is not None else "")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Training logs: {logger.log_dir}")
    print(f"View with:  tensorboard --logdir {logger.save_dir}")


if __name__ == "__main__":
    main()
