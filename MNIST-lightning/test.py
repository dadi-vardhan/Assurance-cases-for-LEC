import torch
import os
import numpy as np
import pytorch_lightning as pl
from datamodule  import MNISTDataModule
from model import  MnistModel
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils import get_device
from torchmetrics.functional.classification.accuracy import accuracy
from torch.utils.data import DataLoader


dm = MNISTDataModule(os.getcwd())

model = MnistModel.load_from_checkpoint(
    checkpoint_path="/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/lightning_logs/version_406388/checkpoints/epoch=30-step=1673.ckpt")
model.eval()

device =get_device()
# init trainer with whatever options
trainer = pl.Trainer(checkpoint_callback=True,
                     gpus=1)

# test (pass in the model)
# trainer.fit(model)
# trainer.test(model, dm)


# train_data = datasets.MNIST(root = 'data',train = True,transform = ToTensor(),download = True)
# test_data = datasets.MNIST(
#                                 root = 'data', 
#                                 train = False, 
#                                 transform = ToTensor()
#                                 )
# preds = []
# targets =[]
# nums = np.random.randint(0,500,10)
# for i in nums: 
#     img = test_data.data[i].reshape(1,28,28)
#     img = img.unsqueeze_(0)
#     print(img)
#     target = test_data.targets[i]
#     targets.append(target)
#     logits = model(img)
#     pred = torch.argmax(logits)
#     preds.append(pred)
#     print(f"pred : {pred} and label : {target}")

# acc = accuracy(preds, targets)
# print("acc: ",acc)

#testing
import numpy as np

model.freeze()
test_loader = DataLoader(datasets.MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=1028, shuffle=True)

y_true, y_pred = [],[]
for i, (x, y) in enumerate(test_loader):
    y_hat = model.forward(x).argmax(axis=1).cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    y_true.append(y)
    y_pred.append(y_hat)

    if i == len(test_loader):
        break
y_true = np.hstack(y_true)
y_pred = np.hstack(y_pred)
print(f"accuracy: {accuracy(y_true, y_pred)}")

