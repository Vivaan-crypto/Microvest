import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import os
import numpy as np
from datetime import datetime
from lightning_modules import LightningModule, LightningDateModule

#TODO (IDEAS): 1. Switch to HFT or minute candles cause more news resistance. 2.
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

    # Edge / ranking side-data (forward return + decision date per val row), aligned 1:1.
    val_fwd_ret, val_dates = None, None
    # Prefer the v2 vol-normalized target for IC grading when available.
    fwd_path = f"{data_dir}/fwd_z_test.pt"
    if not os.path.exists(fwd_path):
        fwd_path = f"{data_dir}/fwd_ret_test.pt"
    dates_path = f"{data_dir}/dates_test.npy"
    if os.path.exists(fwd_path) and os.path.exists(dates_path):
        val_fwd_ret = torch.load(fwd_path).numpy()
        val_dates = np.load(dates_path)
        print(f"Loaded edge side-data: fwd_ret {val_fwd_ret.shape}, dates {val_dates.shape}")
    else:
        print("Edge side-data not found (run preprocess.py to enable edge/IC metrics)")

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
    print(f"  Class 0 (Short): {class_counts[0]} samples ({100*class_counts[0]/len(y_train):.1f}%)")
    if len(class_counts) > 1:
        print(f"  Class 1 (NoTrade): {class_counts[1]} samples ({100*class_counts[1]/len(y_train):.1f}%)")
    if len(class_counts) > 2:
        print(f"  Class 2 (Long):   {class_counts[2]} samples ({100*class_counts[2]/len(y_train):.1f}%)")
    print()

    data_module = LightningDateModule(X_train, y_train, X_val, y_val, batch_size=32)
    model = LightningModule(y_train.numpy(), val_fwd_ret=val_fwd_ret, val_dates=val_dates,
                            input_size=X_train.shape[-1])

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
    # Select on val/edge (the discrete-signal objective): mean fwd_ret of
    # predicted-Long minus predicted-Short. This grades exactly the rows we act
    # on, unlike F1 (rewards calling the NoTrade majority) or pooled IC (a ranking
    # metric, and inflatable by a time-series effect). Falls back to val/f1 only
    # if no edge side-data is present.
    monitor = "val/edge" if val_fwd_ret is not None else "val/f1"

    # Organize checkpoints by training start time (easier to track runs)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", run_timestamp)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-acc{val/accuracy:.2f}-edge{val/edge:+.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=20,
        mode="max",
        verbose=True,
        min_delta=0.0001,  # edge lives on a smaller scale than F1
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
        num_sanity_val_steps=0,  # partial val batches would desync the edge side-data
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