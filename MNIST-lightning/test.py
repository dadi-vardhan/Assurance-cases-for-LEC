import torch
import os
import numpy as np
import pytorch_lightning as pl
from datamodule  import MNISTDataModule
from model import  MnistModel



dm = MNISTDataModule(os.getcwd())

# model = MnistModel.load_from_checkpoint(checkpoint_path=os.getcwd()+"/sample-mnist-epoch=14-val_loss=-14320.78.ckpt")

# # trainer = pl.Trainer(model,dm)
# # trainer.test(model,dm)
# #print(model.learning_rate)
# model.eval()

model = MnistModel.load_from_checkpoint(
    checkpoint_path="/home/dadi_vardhan/RandD/Experiments_for_LEC/Mnist_Lightning/Untitled/MNIS-2/checkpoints/epoch=13-step=755.ckpt",
    #hparams_file="/home/dadi_vardhan/RandD/Experiments_for_LEC/Mnist_Lightning/lightning_logs/version_1/hparams.yaml",
    map_location=None
)

# init trainer with whatever options
trainer = pl.Trainer(model,gpus=1,
                     resume_from_checkpoint="/home/dadi_vardhan/RandD/Experiments_for_LEC/Mnist_Lightning/Untitled/MNIS-2/checkpoints/epoch=13-step=755.ckpt")

# test (pass in the model)
#trainer.fit(model)
trainer.test(model, dm)
