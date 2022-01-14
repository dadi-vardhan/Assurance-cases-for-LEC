import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
from utils import zoom_image

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [   
                zoom_image(),
                transforms.ToTensor(),
                transforms.Resize((64,64)),
                
            ])
        
        self.dims = (1, 64, 64)
        self.num_classes = 10
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False,transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,num_workers=8)