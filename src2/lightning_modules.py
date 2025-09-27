import lightning as L
from lightning.pytorch.utilities import grad_norm

from dataset import StockDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from model import lstm_model
from sklearn.metrics import r2_score


class LightningDataModule(L.LightningDataModule):
    def __init__(self, x_path: str, y_path: str, batch_size: int = 32, test_size: float = 0.2, random_seed: int = 42):
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed

    def setup(self, stage):
        x = torch.load(self.x_path)
        y = torch.load(self.y_path)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_seed)
        self.train_dataset = StockDataset(x_train, y_train)
        self.test_dataset = StockDataset(x_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class LightningModule(L.LightningModule):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = torch.nn.MSELoss()
        self.model = lstm_model()
        self.train_losses = []

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def shared_step(self, batch, stage: str):
        features, price = batch
        price = price.squeeze(dim=1)
        output = self.model(features)
        loss = self.criterion(output, price)

        # Log loss
        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/r2', torch.tensor(r2_score(output.detach().numpy(), price.detach().numpy())), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
