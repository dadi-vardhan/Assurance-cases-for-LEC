import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datamodule import RoboCupDataModule
from model import MnistModel
from pytorch_lightning.loggers.neptune import NeptuneLogger


# setting global seed
seed_everything(123, workers=True)


# callbacks
early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience = 3,
                    check_finite=True,
                )   

# neptune logger for logging metrics
neptune_logger = NeptuneLogger(
    api_key=os.environ['NEPTUNE_API_TOKEN'],
    project_name="dadivishnuvardhan/AC-LECS",
    #experiment_name="default",
    #tags=["mnist", "lightning", "pytorch"],
    #upload_source_files=["**/*.py", "*.yaml"]
    )

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     filename="sample-mnist-{epoch:02d}-{val_loss:.2f}")

# Init DataModule
dm = RoboCupDataModule(os.getcwd())

# Init model from datamodule's attributes
model = MnistModel(*dm.size())

# Init trainer
trainer = pl.Trainer(default_root_dir=os.getcwd(),
    min_epochs=5,
    max_epochs=500,
    precision = 16,
    weights_summary = "full",
    callbacks=[early_stopping],
    fast_dev_run = True,
    logger = neptune_logger,
    progress_bar_refresh_rate=5,
    gpus=1)

trainer.fit(model, dm)
trainer.validate(model,dm)
trainer.test(model,dm)
#trainer.tune(model,dm)

# stopping logger
neptune_logger.experiment.stop()
