import torch
import os
import numpy as np
import pytorch_lightning as pl
from datamodule  import MNISTDataModule
from model import  MnistModel
from torchvision import datasets
from torchvision.transforms import transforms, ToTensor
from utils import get_device
from torchmetrics.functional.classification.accuracy import accuracy
from torchvision import models
from torch.autograd import Variable


dm = MNISTDataModule(os.getcwd())
# model = MnistModel.load_from_checkpoint(checkpoint_path=os.getcwd()+"/sample-mnist-epoch=14-val_loss=-14320.78.ckpt")

model = MnistModel.load_from_checkpoint(
    checkpoint_path="/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/MNIST-lightning/sample-mnist-epoch=04-val_loss=0.39.ckpt")
model.eval()

device =get_device()
# init trainer with whatever options
trainer = pl.Trainer(gpus=1)

# test (pass in the model)
#trainer.fit(model)
#trainer.test(model, dm)
train_data = datasets.MNIST(root = 'data',train = True,transform = ToTensor(),download = True)
test_data = datasets.MNIST(
                                root = 'data', 
                                train = False, 
                                transform = ToTensor()
                                )
preds = []
targets =[]
nums = np.random.randint(0,500,10)
for i in nums: 
    img = test_data.data[i].reshape(1,28,28)
    img = img.unsqueeze_(0)
    print(img)
    target = test_data.targets[i]
    targets.append(target)
    logits = model(img)
    pred = torch.argmax(logits)
    preds.append(pred)
    print(f"pred : {pred} and label : {target}")

# acc = accuracy(preds, targets)
# print("acc: ",acc)