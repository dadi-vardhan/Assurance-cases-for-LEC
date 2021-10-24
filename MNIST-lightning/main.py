import torch
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from datamodule import MNISTDataModule
from model import MnistModel

#os.environ['CUDA_VISIBLE_DEVICES']='2'

# setting global seed
seed_everything(123, workers=True)


# callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    stopping_threshold=1e-4,
    divergence_threshold=9.0,
    patience = 5,
    check_finite=True,
)

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     dirpath=os.getcwd(),
#     filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
#     save_top_k=3,
#     mode="min",
# )

# loggers
#tb_logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lc \
                HR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH \
                VuZS5haSIsImFwaV9rZXkiOiI5ZWFjYzgzNy03MTkxLTRiNmQ \
                tYjE2Yy0xM2RlZDcwNDQ1M2YifQ==",
    project_name="dadivishnuvardhan/Mnist-lightning-logs",
    # experiment_name="default",  # Optional,
    # params={"max_epochs": 10},  # Optional,
    # tags=["pytorch-lightning", "mlp"],  # Optional,
)


# Init DataModule
dm = MNISTDataModule(os.getcwd())

# Init model from datamodule's attributes
model = MnistModel(*dm.size(),dm.num_classes)

# Init trainer
trainer = pl.Trainer(default_root_dir=os.getcwd(),
    min_epochs=5,
    max_epochs=100,
    precision = 16,
    weights_summary = "full",
    callbacks=[early_stopping],
    logger = neptune_logger,
    weights_save_path=os.getcwd(),
    progress_bar_refresh_rate=2,
    gpus=1
)

trainer.fit(model, dm)
trainer.validate(model,dm)
trainer.test(model,dm)