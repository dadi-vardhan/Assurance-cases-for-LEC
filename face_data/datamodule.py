import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
import glob
import numpy as np
import random
from setuptools.namespaces import flatten
import os
from itertools import islice
from torchvision.datasets import VisionDataset
import cv2 as cv

# pip install pytorch-lightning==1.5.0


class FaceDataset(VisionDataset):
    def __init__(self,data_dir,train=False,test=False,valid=False,transform=None):
        self.image_paths = []
        self.transform = transform
        self.data_dir = data_dir
        self.train_size = 0.7
        self.train_size = 0.2
        self.valid_size = 0.1
        
        #idx_to_class = {i:j for i, j in enumerate(classes)}
        #class_to_idx = {value:key for key,value in idx_to_class.items()}
        self.class_to_idx = {"a_j__buckley": 0, 
                        "a_r__rahman": 1,
                        "aamir_khan" : 2,
                        "aaron_staton": 3,
                        "aaron_tveit": 4,
                        "aaron_yoo": 5,
                        "abbie_cornish": 6,
                        "abel_ferrara" : 7,
                        "abigail_breslin": 8,
                        "abigail_spencer": 9}
        
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


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
                [   
                transforms.ToTensor(),
                transforms.Resize((64,64)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        self.dims = (1, 64, 64)
        self.num_classes = 10
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        FaceDataset(self.data_dir, train=True)
        FaceDataset(self.data_dir, train=False)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
           self.mnist_train = FaceDataset(self.data_dir, train=True, transform=self.transform)
           self.mnist_val = FaceDataset(self.data_dir, valid=True, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = FaceDataset(self.data_dir, test=True,transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,num_workers=8)
    

# if __name__ == '__main__':
#     data_dir = '/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/Data/train'
#     dm = FaceDataModule(data_dir,batch_size=32)
#     dm.setup(stage="fit")
#     for i, (x,y) in enumerate(dm.train_dataloader()):
#         print(x.shape)
#         print(y.shape)
#         print(y)
        
#         if i == 10:
#             break