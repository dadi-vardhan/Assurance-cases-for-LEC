import torch
import pytorch_lightning as pl
from torchvision.models import resnet18
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional import f1, precision, recall, specificity
from eval_metrics import createConfusionMatrix



class MnistModel(pl.LightningModule):
    def __init__(self, channels=1, width=28, height=28, num_classes=10, hidden_size=64, learning_rate=6.918309709189363e-05):

        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        #self.hparams = hparam
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.save_hyperparameters()
        self.model = resnet18()
        self.model.conv1 = torch.nn.Conv2d(self.channels,self.hidden_size, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.logger.experiment.log_metric('Train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.logger.experiment.log_metric('Validation_loss', loss)
        self.logger.experiment.log_metric('Validation_accuracy', acc)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        #x = x.view(x.size(0), -1)
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        #acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        acc = accuracy(preds,y)
        output = dict({
          'test_loss': loss,
          'test_acc': acc})
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.logger.experiment.log_metric('Test_loss', loss)
        self.logger.experiment.log_metric('Test_accuracy', acc)
        cm_fig = createConfusionMatrix(batch,self.model)
        self.logger.experiment.log_image('confusion_matrix', cm_fig)
        f1_score = f1(preds,y,num_classes=self.num_classes)
        self.logger.experiment.log_metric('F1_score',f1_score)
        prec = precision(preds, y, average='macro', num_classes=self.num_classes)
        self.logger.experiment.log_metric('Precision',prec)
        rec =recall(preds, y, average='macro', num_classes=self.num_classes)
        self.log.experiment.log_metric("Precision_recall",rec)
        spe = specificity(preds, y, average='macro', num_classes=self.num_classes)
        self.logger.log_hyperparams(self.model.hparams)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer