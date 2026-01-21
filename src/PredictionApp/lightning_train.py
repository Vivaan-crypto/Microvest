# lightning_train.py

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from lightning_modules import LightningModule, LightningDateModule


def main():
    # --------------------------------------------------
    # Load pre-split tensors (already processed)
    # --------------------------------------------------
    X_train = torch.load("Data/X_train.pt")
    y_train = torch.load("Data/y_train.pt")
    X_val = torch.load("Data/X_test.pt")
    y_val = torch.load("Data/y_test.pt")

    # --------------------------------------------------
    # Create Lightning model
    # --------------------------------------------------
    data_module = LightningDateModule(X_train, y_train, X_val, y_val, batch_size=32)
    model = LightningModule()

    # --------------------------------------------------
    # TensorBoard logger
    # --------------------------------------------------
    logger = TensorBoardLogger(
        "lightning_logs",
        name="stock_prediction_model",
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = L.Trainer(
        logger=logger,
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


if __name__ == "__main__":
    main()
