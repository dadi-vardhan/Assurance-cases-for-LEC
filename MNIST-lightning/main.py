import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
#from pytorch_lightning.runs import Neptunerun
from datamodule import MNISTDataModule
from model import MnistModel

# run = neptune.init(project="dadivishnuvardhan/AC-LECS", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lc \
#                 HR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH \
#                 VuZS5haSIsImFwaV9rZXkiOiI5ZWFjYzgzNy03MTkxLTRiNmQ \
#                 tYjE2Yy0xM2RlZDcwNDQ1M2YifQ==")

#os.environ['CUDA_VISIBLE_DEVICES']='2'

# setting global seed
seed_everything(123, workers=True)


# callbacks
early_stopping = EarlyStopping(
    monitor="over_fit",
    patience = 2,
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
    min_epochs=2,
    max_epochs=500,
    precision = 16,
    weights_summary = "full",
    callbacks=[early_stopping],
    fast_dev_run = True,
    #logger = run,
    progress_bar_refresh_rate=5,
    gpus=1)

trainer.fit(model, dm)
trainer.validate(model,dm)
trainer.test(model,dm)
#trainer.tune(model,dm)