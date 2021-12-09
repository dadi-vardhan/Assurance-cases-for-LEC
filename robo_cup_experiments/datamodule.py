from typing import NoReturn
import cv2 as cv
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import VisionDataset
import glob
import numpy as np
from utils import zoom_image
import random
from setuptools.namespaces import flatten
import os
from itertools import islice


class RoboCupDataset(VisionDataset):
    def __init__(self,train=False,test=False,valid=False,transform=None):
        self.image_paths = []
        self.transform = transform
        self.data_dir = '/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/tools'
        self.train_size = 0.7
        self.train_size = 0.2
        self.valid_size = 0.1
        
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
        if test == True:
            self.image_paths = self.get_data_paths(self.data_dir)[2]
        if valid == True:
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
            image = self.transform(image)
        self.image = image
        return image, label
    
    def get_data_paths(self,data_dir):
        image_paths = []
        for path in glob.glob(data_dir + '/*'):
            image_paths.append(glob.glob(path + '/*'))
            
        image_paths = list(flatten(image_paths))
        random.shuffle(image_paths)
        total_imgs = len(image_paths)
        splits = [int(self.train_size*total_imgs),
                  int(self.train_size*total_imgs),
                  int(self.valid_size*total_imgs)]
        Output = [list(islice(image_paths, elem))
                        for elem in splits]
        train_image_paths = Output[0]
        valid_image_paths = Output[2]
        test_image_paths = Output[1]
        return train_image_paths, valid_image_paths, test_image_paths
    


class RoboCupDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        #self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                #zoom_image(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.dims = (3, 224, 224)
        self.num_classes = 15
        self.batch_size = batch_size
        
        
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.robo_cup_train = RoboCupDataset(train=True, transform=self.transform)
            self.robo_cup_val = RoboCupDataset(valid=True, transform=self.transform) 

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.robo_cup_test = RoboCupDataset(test=True,transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.robo_cup_train, batch_size=self.batch_size,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.robo_cup_val, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.robo_cup_test, batch_size=self.batch_size,num_workers=8)


if __name__ == "__main__":

    dm = RoboCupDataModule()
    dm.setup(stage="fit")
    print(len(dm.train_dataloader()))
    print(len(dm.val_dataloader()))
    dm.setup(stage="test")
    print(len(dm.test_dataloader()))
    for i, (x, y) in enumerate(dm.val_dataloader()):
        print(x.shape, y.shape)
        
        if i == 5:
            break
    
    # val_set = RoboCupDataset(valid=True, transform=transforms.ToTensor())
    # for i in range(len(val_set)):
    #     x, y = val_set[i]
    #     print(x.shape, y.shape)
    #     if i == 5:
    #         break