from typing import NoReturn
import cv2 as cv
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import glob
import numpy as np
from utils import zoom_image
import random
from setuptools.namespaces import flatten
import os

DataPath = '/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/robo_cup_tools' 

class RoboCupDataset(Dataset):
    def __init__(self, data_dir=DataPath,train=False,test=False,valid=False,transform=False):
        self.image_paths = []
        self.transform = transform
        self.data_dir = data_dir
        
        #idx_to_class = {i:j for i, j in enumerate(classes)}
        #class_to_idx = {value:key for key,value in idx_to_class.items()}
        self.class_to_idx = {"AXIS": 0, 
                        "BEARING": 1,
                        "BEARING_BOX": 2,
                        "CONTAINER_BOX_BLUE": 3,
                        "CONTAINER_BOX_RED": 4,
                        "F20_20_B": 5,
                        "F20_20_G": 6,
                        "M20": 7,
                        "M20_100": 8,
                        "M30": 9,
                        "MOTOR": 10,
                        "R20": 11,
                        "S40_40_B": 12,
                        "S40_40_G": 13}
        
        if train == True:
            self.image_paths = self.get_data_paths(self.data_dir)[0]
        elif test == True:
            self.image_paths = self.get_data_paths(self.data_dir)[2]
        elif valid == True:
            self.image_paths = self.get_data_paths(self.data_dir)[1]
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv.imread(image_filepath)
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            print(image)
        return image, label
    
    def get_data_paths(self,data_dir):
        image_paths = []
        for path in glob.glob(data_dir + '/*'):
            image_paths.append(glob.glob(path + '/*'))
            
        image_paths = list(flatten(image_paths))
        random.shuffle(image_paths)
        train_image_paths = image_paths[:int(0.7*len(image_paths))]
        valid_image_paths = image_paths[int(0.7*len(image_paths)):int(0.1*len(image_paths))]
        test_image_paths = image_paths[:int(0.8*len(image_paths)):]
        return train_image_paths, valid_image_paths, test_image_paths
    


class RoboCupDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=1028):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [   transforms.Resize((224,224)),
                zoom_image(),
                transforms.ToTensor()
            ])

        self.batch_size = batch_size
    
    # def prepare_data(self):
    #     # load data
    #     RoboCupDataset(self.data_dir, train=True)
    #     RoboCupDataset(self.data_dir, test=True)
    #     RoboCupDataset(self.data_dir, valid=True)
        
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.robo_cup_train = RoboCupDataset(self.data_dir, train=True, transform=self.transform)
            self.robo_cup_val = RoboCupDataset(self.data_dir, valid=True, transform=self.transform) 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.robo_cup_test = RoboCupDataset(self.data_dir, test=True,transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.robo_cup_train, batch_size=self.batch_size,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.robo_cup_val, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.robo_cup_test, batch_size=self.batch_size,num_workers=8)


if __name__ == "__main__":
    DataPath = '/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/robo_cup_tools'
    dm = RoboCupDataset(DataPath,valid=True)
    print(len(dm))  # Number of images in the dataset