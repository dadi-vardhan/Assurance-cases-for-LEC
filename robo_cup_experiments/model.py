import torch
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet18,mobilenet_v2,squeezenet1_0
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
from eval_metrics import eval_metrics
from neptune.new.types import File
import neptune.new as neptune



class MnistModel(pl.LightningModule):
    def __init__(self, channels=3, width=224, height=224, learning_rate=0.02):

        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.classes = ("AXIS","BEARING","BEARING_BOX","CONTAINER_BOX_BLUE","CONTAINER_BOX_RED",
                        "F20_20_B","F20_20_G","M20","M20_100","M30","MOTOR","R20","S40_40_B","S40_40_G")
        self.num_classes = len(self.classes)
        self.learning_rate = learning_rate
        self.model = resnet18()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        # #self.run = neptune.init(project="dadivishnuvardhan/AC-LECS", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lc \
        #         HR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH \
        #         VuZS5haSIsImFwaV9rZXkiOiI5ZWFjYzgzNy03MTkxLTRiNmQ \
        #         tYjE2Yy0xM2RlZDcwNDQ1M2YifQ==")

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        #self.run['train/Train_loss'].log(loss)
        #self.train_loss = loss
        self.logger.experiment.log_metric('Train_loss',loss)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        epoch_avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("train_loss", epoch_avg_loss)
        #self.run['train/train_avg_loss'].log(epoch_avg_loss)
        self.logger.experiment.log_metric('Train_avg_loss',epoch_avg_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.logger.experiment.log_metric('Val_loss',loss)
        self.logger.experiment.log_metric('Val_acc',acc)
        #self.run['val/Validation_loss'].log(loss)
        #self.run['val/Validation_accuracy'].log(acc)
        # if loss > self.train_loss:
        #     self.log("over_fit",loss>self.train_loss)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        epoch_avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log(f"val_loss", epoch_avg_loss)
        #self.run["val/avg_val_loss"].log(epoch_avg_loss)
        self.logger.experiment.log_metric('Avg_val_loss',epoch_avg_loss)
        return epoch_avg_loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        #x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        #acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        acc = accuracy(preds,y)
        self.log("test_loss", loss,on_epoch=True, prog_bar=True)
        self.log("test_acc", acc,on_epoch=True, prog_bar=True)
       # self.run['test/Test_loss'].log(loss)
        #self.run['test/Test_accuracy'].log(acc)
        trgts = y
        preds = preds
        eval = eval_metrics(trgts,preds,self.classes)
        cm = eval.plot_conf_matx()
        cm_norm = eval.plot_conf_matx(normalized=True)
        self.logger.experiment.log_image('Confusion Matrix',cm)
        self.logger.experiment.log_image('Normalized Confusion Matrix',cm_norm)
        #self.run["metrics/confusion_matrix"].log(File.as_image(cm))
        #self.run["metrics/confusion_matrix_Normalized"].log(File.as_image(cm_norm))
        cls_report = eval.classify_report()
        #self.run["metrics/calssification Report"].log(cls_report)
        self.logger.experiment.log_text("classification-report",cls_report)
        f1 = eval.f1_score_weighted()
        #self.run['metrics/F1_score'].log(f1)
        self.logger.experiment.log_metric('F1_score',f1)
        recall = eval.recall_weighted()
        #self.run['metrics/Recall'].log(recall)
        self.logger.experiment.log_metric('Recall',recall)
        prec = eval.precision_weighted()
        #self.run['metrics/Precision'].log(prec)
        self.logger.experiment.log_metric('Precision',prec)
        output = {
          'test_loss': loss,
          'test_acc': acc}
        return output
    
    def test_end(self, outputs):
        """[Logging all metrics at the end of the test phase]

        Args:
            outputs ([tensors]): [model predictions]

        Returns:
            [tensors]: [avg test loss]
        """
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        #self.run["test/Avg_test_loss"].log(avg_loss)
        self.logger.experiment.log_metric('Avg_test_loss',avg_loss)
        return avg_loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer