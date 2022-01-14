import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datamodule import RoboCupDataModule
from model import RobocupModel
from pytorch_lightning.loggers.neptune import NeptuneLogger


# setting global seed
seed_everything(123, workers=True)


# callbacks
early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience = 5,
                    
                    check_finite=True,
                )   

# neptune logger for logging metrics
neptune_logger = NeptuneLogger(
    api_key=os.environ['NEPTUNE_API_TOKEN'],
    project_name="dadivishnuvardhan/AC-LECS",
    #experiment_name="default",
    tags=["robo-cup", "zoom", "sqz-net"],
    #upload_source_files=["**/*.py", "*.yaml"]
    )

# Init DataModule
dm = RoboCupDataModule()

# Init model from datamodule's attributes
model = RobocupModel(*dm.size())

# Init trainer
trainer = pl.Trainer(default_root_dir=os.getcwd(),
    min_epochs=100,
    max_epochs=500,
    #precision = 16,
    #auto_lr_find=True,
    #auto_scale_batch_size="binsearch",
    weights_summary = "full",
    callbacks=[early_stopping],
    fast_dev_run = False,
    logger = neptune_logger,
    progress_bar_refresh_rate=5,
    gpus=1)

# trainer = pl.Trainer(auto_lr_find=True)
# dm.setup(stage="fit")
# trainer.tune(model,dm)

dm.setup(stage="fit")
trainer.fit(model, dm)
trainer.validate(model, dm)

dm.setup(stage="test")
trainer.test(model, dm)

# stopping logger
neptune_logger.experiment.stop()
