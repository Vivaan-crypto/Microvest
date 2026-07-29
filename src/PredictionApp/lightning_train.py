import os
from datetime import datetime

import config
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_modules import LightningDateModule, LightningModule


# TODO (IDEAS): 1. Switch to HFT or minute candles cause more news resistance. 2.
def main():
    # Load pre-split tensors (already processed by preprocess.py).
    data_dir = os.path.abspath("src/PredictionApp/Data")

    X_train = torch.load(f"{data_dir}/X_train.pt")
    y_train = torch.load(f"{data_dir}/y_train.pt")
    X_val = torch.load(f"{data_dir}/X_test.pt")
    y_val = torch.load(f"{data_dir}/y_test.pt")

    # Edge/ranking side-data (fwd return + decision date per val row), aligned 1:1.
    # Prefer the v2 vol-normalized target for IC grading.
    val_fwd_ret, val_dates = None, None
    fwd_path = f"{data_dir}/fwd_z_test.pt"
    if not os.path.exists(fwd_path):
        fwd_path = f"{data_dir}/fwd_ret_test.pt"
    dates_path = f"{data_dir}/dates_test.npy"
    if os.path.exists(fwd_path) and os.path.exists(dates_path):
        val_fwd_ret = torch.load(fwd_path).numpy()
        val_dates = np.load(dates_path)
        print(
            f"Loaded edge side-data: fwd_ret {val_fwd_ret.shape}, dates {val_dates.shape}"
        )
    else:
        print("Edge side-data not found (run preprocess.py to enable edge/IC metrics)")

    print(f"Loaded data from: {data_dir}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Targets -> long class indices, shape [N].
    if y_train.dim() == 2 and y_train.shape[1] == 1:
        y_train = y_train.squeeze(-1)
    if y_val.dim() == 2 and y_val.shape[1] == 1:
        y_val = y_val.squeeze(-1)

    y_train = y_train.long()
    y_val = y_val.long()

    # Shift signed labels (-1,0,1 -> 0,1,2) to non-negative class indices.
    min_label = int(min(int(y_train.min().item()), int(y_val.min().item())))
    if min_label < 0:
        shift = -min_label
        y_train = y_train + shift
        y_val = y_val + shift
        print(
            f"Shifted labels by {shift} (was in range [{min_label}, {int(max(int(y_train.max().item()), int(y_val.max().item()) - shift))}])"
        )

    print(f"Label range: [{int(y_train.min().item())}, {int(y_train.max().item())}]")
    print(f"Number of classes: {int(y_train.max().item()) + 1}")

    data_module = LightningDateModule(
        X_train, y_train, X_val, y_val, batch_size=config.TRAIN_BATCH_SIZE
    )
    model = LightningModule(
        y_train.numpy(),
        val_fwd_ret=val_fwd_ret,
        val_dates=val_dates,
        input_size=X_train.shape[-1],
    )

    logger = TensorBoardLogger("lightning_logs")

    monitor = "val/f1"

    # Checkpoints grouped by run start time.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", run_timestamp)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=checkpoint_dir,
        filename="best-model-{epoch:02d}-acc{val/accuracy:.2f}-f1{val/f1:+.4f}",
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
        min_delta=0.0001,
    )

    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        max_epochs=200,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,  # partial val batches would desync the edge side-data
    )

    trainer.fit(model, data_module)

    torch.save(model.state_dict(), "model_final.pth")
    print("Model saved to model_final.pth")
    print(f"\nBest model checkpoint directory: {checkpoint_callback.dirpath}")
    print(f"Training logs: {logger.log_dir}")


if __name__ == "__main__":
    main()
