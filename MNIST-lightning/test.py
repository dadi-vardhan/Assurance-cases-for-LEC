from cv2 import sepFilter2D
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
from eval_metrics import eval_metrics
import matplotlib.pyplot as plt


dm = MNISTDataModule(os.getcwd())

model = MnistModel.load_from_checkpoint(
    checkpoint_path="/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/MNIST-lightning/AC-213/checkpoints/epoch=106-step=5777.ckpt").eval()

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
#                                 download=True,)
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
classes = ('Zero', 'One', 'Two', 'Three', 'Four',
                            'Five', 'Six', 'Seven', 'Eight', 'Nine')

model.freeze()
test_loader = DataLoader(datasets.MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=1028, shuffle=True)

# y_true, y_pred = [],[]
# for i, (x, y) in enumerate(test_loader):
#     print(x)
#     y_hat = model.forward(x).argmax(axis=1).cpu().detach().numpy()
#     y = y.cpu().detach().numpy()

#     y_true.append(y)
#     y_pred.append(y_hat)

#     if i == 2:
#         break
# y_true = np.hstack(y_true)
# y_pred = np.hstack(y_pred)

# em_mon = eval_metrics(y_true, y_pred,classes=classes)
# print(f"accracy-mon:{em_mon.accuracy()}")
# print(em_mon.classify_report())

img1 = test_loader.dataset.data[0]
# y1 = test_loader.dataset.targets[0].cpu().detach().numpy()
img2 = test_loader.dataset.data[1]
# y2 = test_loader.dataset.targets[1].cpu().detach().numpy()

# print(img1)
img = img1+img2
#img = img.reshape(1,28,28)
img = img.to(device)
img = img[None, None]
img = img.type('torch.FloatTensor')
#img =img.unsqueeze_(1)
# print(img)
#img = img.reshape(1,28,28)
#img /= img.max()
# plt.subplot(1,3,1)
# plt.xlabel("digit-"+str(y1))
# plt.imshow(img1.reshape(28,28))
# plt.subplot(1,3,2)
# plt.xlabel("digit-"+str(y2))
# plt.imshow(img2.reshape(28,28))
# plt.subplot(1,3,3)
# plt.xlabel("digit-"+str(y1)+"+"+str(y2))
# plt.imshow(img.reshape(28,28))
# plt.title("out of distribution image")
# plt.savefig("/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/MNIST-lightning/ood.png")
# plt.show()

#image to tensor for prediction
#img = torch.from_numpy(img).float().to(device)
#x = img.unsqueeze_(0)
# trans = ToTensor()
# img = trans(img)
# print(img)
y_hat = model(img)
logits = torch.nn.functional.sigmoid(y_hat)
#logits = torch.exp(y_hat)
pred = y_hat.max(1, keepdim=True)[1]

print("predicted:",logits)
print(pred)



