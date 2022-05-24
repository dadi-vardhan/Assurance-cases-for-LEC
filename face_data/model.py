import torch
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet18,mobilenet_v2,squeezenet1_0,resnet50,mobilenet_v3_large,vgg16,resnet152
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
from torchvision.models.squeezenet import squeezenet1_1
from eval_metrics import eval_metrics
#from neptune.types import File
#import neptune.new as neptune


class FaceModel(pl.LightningModule):
    def __init__(self, channels=3, width=64, height=64, learning_rate=0.0022908676527677745):

        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.classes = ("a_j__buckley", 
                        "a_r__rahman",
                        "aamir_khan" ,
                        "aaron_staton",
                        "aaron_tveit",
                        "aaron_yoo",
                        "abbie_cornish",
                        "abel_ferrara",
                        "abigail_breslin",
                        "abigail_spencer")
        self.num_classes = len(self.classes)
        self.learning_rate = learning_rate
        self.model = resnet50()
        
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True) # resnet50
        ##self.model.classifier[1] = torch.nn.Linear(in_features=1280,out_features=10,bias=True) #mobilenet_v2
        #self.model.classifier[1] = torch.nn.Conv2d(512,15,kernel_size=(1,1),stride=(1,1)) #squeezenet1_1
        #self.model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=15, bias=True)# mobilenet_v3 large
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        

    def forward(self,x):
        x = self.model(x)
        # x = torch.nn.Linear(in_features=1000, out_features=13, bias=True,device='cuda')(x)
        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.log_softmax(x,dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        #self.logger.experiment.log_metric('Train_loss',loss)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        epoch_avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("train_loss", epoch_avg_loss)
        #self.logger.experiment.log_metric('Train_avg_loss',epoch_avg_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        #self.logger.experiment.log_metric('val_loss',loss)
        #self.logger.experiment.log_metric('val_acc',acc)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        epoch_avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log(f"val_loss", epoch_avg_loss)
        #self.logger.experiment.log_metric('Avg_val_loss',epoch_avg_loss)
        return epoch_avg_loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds,y)
        self.log("test_loss", loss,on_epoch=True, prog_bar=True)
        self.log("test_acc", acc,on_epoch=True, prog_bar=True)
        
        eval = eval_metrics(y,preds,self.classes)
        cm = eval.plot_conf_matx()
        cm_norm = eval.plot_conf_matx(normalized=True)
        #self.log('Confusion Matrix',cm)
        #self.logger.experiment.log_image('Confusion Matrix',cm)
        #self.logger.experiment.log_image('Normalized Confusion Matrix',cm_norm)
        #cls_report = eval.classify_report()
        #self.logger.experiment.log_text("classification-report",cls_report)
        #self.log('Classification Report',cls_report)
        f1 = eval.f1_score_weighted()
        #self.logger.experiment.log_metric('F1_score',f1)
        self.log('F1 Score',f1)
        recall = eval.recall_weighted()
        #self.logger.experiment.log_metric('Recall',recall)
        self.log('Recall',recall)
        prec = eval.precision_weighted()
        #self.logger.experiment.log_metric('Precision',prec)
        self.log('Precision',prec)
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
        #self.logger.experiment.log_metric('Avg_test_loss',avg_loss)
        self.log('Avg Test Loss',avg_loss)
        return avg_loss
    
    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
    #     return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer