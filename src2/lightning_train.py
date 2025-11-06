from lightning_modules import LightningDataModule, LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch


def main():
    model = LightningModule()
    x_path = 'data/x_tensors.pt'
    y_path = 'data/y_tensors.pt'
    data_module = LightningDataModule(x_path, y_path, 32, 0.2, 42)
    logger = TensorBoardLogger("lightning_logs", name="simple")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=25,
        log_every_n_steps=2,
        accelerator='cpu',
    )

    trainer.fit(model, data_module)
    torch.save(model.state_dict(), "model/model.pth")


if __name__ == '__main__':
    main()
