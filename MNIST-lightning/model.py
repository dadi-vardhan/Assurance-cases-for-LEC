import torch
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet18, mobilenet_v2,squeezenet1_0
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
from eval_metrics import eval_metrics
import matplotlib.pyplot as plt
from neptune.new.types import File
import neptune.new as neptune


class MnistModel(pl.LightningModule):
    def __init__(self, channels=1, width=28, height=28,hidden_size=32, learning_rate=0.02):

        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        #self.hparams = hparam
        self.channels = channels
        self.width = width
        self.height = height
        self.classes = ('Zero', 'One', 'Two', 'Three', 'Four',
                            'Five', 'Six', 'Seven', 'Eight', 'Nine')
        self.num_classes = len(self.classes)
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.model = resnet18()
        self.model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # resnet
        #self.model.features[0][0] = torch.nn.Conv2d(1,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) #mobilenet
        #self.model.features[0] = torch.nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2)) #sqeezenet
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.run = neptune.init(project="dadivishnuvardhan/AC-LECS", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lc \
                HR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH \
                VuZS5haSIsImFwaV9rZXkiOiI5ZWFjYzgzNy03MTkxLTRiNmQ \
                tYjE2Yy0xM2RlZDcwNDQ1M2YifQ==")
        self.train_loss = None

    def forward(self,x):
        x = x.float()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.run['train/Train_loss'].log(loss)
        self.train_loss = loss
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        epoch_avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("train_loss", epoch_avg_loss)
        self.run['train/train_avg_loss'].log(epoch_avg_loss)

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x= x.float()
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.run['val/Validation_loss'].log(loss)
        self.run['val/Validation_accuracy'].log(acc)
        if loss > self.train_loss:
            self.log("over_fit",loss>self.train_loss)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        epoch_avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log(f"val_loss", epoch_avg_loss)
        self.run["val/avg_val_loss"].log(epoch_avg_loss)
        return epoch_avg_loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        x=x.float()
        #x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        #acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        acc = accuracy(preds,y)
        self.log("test_loss", loss,on_epoch=True, prog_bar=True)
        self.log("test_acc", acc,on_epoch=True, prog_bar=True)
        self.run['test/Test_loss'].log(loss)
        self.run['test/Test_accuracy'].log(acc)
        output = {
          'test_loss': loss,
          'test_acc': acc,
          'preds': preds,
          'trgts':y}
        return output
    
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.run["test/Avg_test_loss"].log(avg_loss)
        trgts = outputs["trgts"]
        preds = outputs["preds"]
        eval = eval_metrics(trgts,preds,self.classes)
        cm = self.eval_metrics.plot_conf_matx()
        cm_norm = eval.plot_conf_matx(normalized=True)
        self.run["metrics/confusion_matrix"].log(File.as_image(cm))
        self.run["metrics/confusion matrix"].log(File.as_image(cm_norm))
        cls_report = eval.classify_report()
        self.run["metrics/calssification Report"].log(cls_report)
        f1 = eval.f1_score_weighted()
        self.run['test/F1_score'].log(f1)
        recall = eval.recall_score_weighted()
        self.run['test/Recall'].log(recall)
        prec = eval.precision_score_weighted()
        self.run['test/Precision'].log(prec)
        return avg_loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer