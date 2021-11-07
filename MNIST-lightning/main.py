import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
#from pytorch_lightning.runs import Neptunerun
from datamodule import MNISTDataModule
from model import MnistModel


# setting global seed
seed_everything(123, workers=True)


# callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience = 3,
    check_finite=True,
)

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     filename="sample-mnist-{epoch:02d}-{val_loss:.2f}")

# Init DataModule
dm = MNISTDataModule(os.getcwd())

# Init model from datamodule's attributes
model = MnistModel(*dm.size())

# Init trainer
trainer = pl.Trainer(default_root_dir=os.getcwd(),
    min_epochs=5,
    max_epochs=500,
    precision = 16,
    weights_summary = "full",
    callbacks=[early_stopping],
    fast_dev_run = False,
    #logger = run,
    progress_bar_refresh_rate=5,
    gpus=1)

trainer.fit(model, dm)
trainer.validate(model,dm)
trainer.test(model,dm)
#trainer.tune(model,dm)