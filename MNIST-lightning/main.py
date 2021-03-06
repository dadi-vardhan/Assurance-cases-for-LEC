from pytorch_lightning.trainer.trainer import Trainer
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datamodule import MNISTDataModule
from model import MnistModel
from pytorch_lightning.loggers.neptune import NeptuneLogger


# setting global seed
seed_everything(123, workers=True)


# callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    check_finite=True,
)

# neptune logger for logging metrics
neptune_logger = NeptuneLogger(
    api_key=os.environ['NEPTUNE_API_TOKEN'],
    project_name="dadivishnuvardhan/AC-LECS",
    # experiment_name="default",
    tags=["mnist", "mobile-net", "full", "augment",],
    #upload_source_files=["**/*.py", "*.yaml"]
)

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     filename="sample-mnist-{epoch:02d}-{val_loss:.2f}")

# Init DataModule
dm = MNISTDataModule(os.getcwd())
dm.prepare_data()

# Init model from datamodule's attributes
model = MnistModel(*dm.size())

#Init trainer
# trainer = pl.Trainer(default_root_dir=os.getcwd(),
#                      min_epochs=50,
#                      max_epochs=500,
#                      #precision=16,
#                      weights_summary="full",
#                      callbacks=[early_stopping],
#                      fast_dev_run=True,
#                      logger=neptune_logger,
#                      progress_bar_refresh_rate=5,
#                      gpus=1)

# dm.setup(stage="fit")
# trainer.fit(model, dm)
# trainer.validate(model, dm)

# dm.setup(stage="test")
# trainer.test(model, dm)

trainer = Trainer(auto_lr_find=True)

trainer.tune(model,dm)

# stopping logger
neptune_logger.experiment.stop()
