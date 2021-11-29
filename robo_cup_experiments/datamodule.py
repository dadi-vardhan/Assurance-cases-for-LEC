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


class RoboCupDataset(Dataset):
    def __init__(self,train=False,test=False,valid=False,transform=None):
        self.image_paths = []
        self.transform = transform
        self.data_dir = '/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/robo_cup_tools'
        
        #idx_to_class = {i:j for i, j in enumerate(classes)}
        #class_to_idx = {value:key for key,value in idx_to_class.items()}
        self.class_to_idx = {"AXIS": 0, 
                        "BEARING": 1,
                        "BEARING_BOX": 2,
                        "CONTAINER_BOX_BLUE": 3,
                        "CONTAINER_BOX_RED": 4,
                        "DISTANCE_TUBE": 5,
                        "F20_20_B": 6,
                        "F20_20_G": 7,
                        "M20": 8,
                        "M20_100": 9,
                        "M30": 10,
                        "MOTOR": 11,
                        "R20": 12,
                        "S40_40_B": 13,
                        "S40_40_G": 14}
        
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
        image = cv.resize(image, (224, 224), interpolation = cv.INTER_AREA)

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)
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
            [   #transforms.Resize((224,224)),
                zoom_image(),
                transforms.ToTensor()
            ])
        self.dims = (1, 224, 224)
        self.num_classes = 15
        self.batch_size = batch_size
        
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.robo_cup_train = RoboCupDataset(train=True, transform=self.transform)
            self.robo_cup_val = RoboCupDataset(valid=True, transform=self.transform) 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.robo_cup_test = RoboCupDataset(self.data_dir, test=True,transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.robo_cup_train, batch_size=self.batch_size,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.robo_cup_val, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.robo_cup_test, batch_size=self.batch_size,num_workers=8)


# if __name__ == "__main__":

#     dm = RoboCupDataModule(os.getcwd())
#     dm.setup(stage="fit")
#     print(len(dm.train_dataloader()))
#     dm.setup(stage="test")
#     test_dataloader = dm.test_dataloader()
#     print(len(test_dataloader))