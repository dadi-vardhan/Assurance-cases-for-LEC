import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, squeezenet1_0
from data_processing import train_loader, valid_loader, test_loader, classes
from utils import gpu_check

class MnistModel():
    
    def __init__(self,model,criterion,dataset:list,device=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.criterion = criterion
        self.train_loader = dataset[0]
        self.valid_loader = dataset[1]
        self.test_loader = dataset[2]
        self.train_on_gpu = device

    def load_model(self):
        """[Moves the model to GPU if available and prints the model]
        """
        #Changing ResNet input channels
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        #print(self.model)
        
        # move tensors to GPU if CUDA is available
        if self.train_on_gpu:
            self.model.cuda()

    def train_model(self,n_epochs):
        """[Train and validate the model]

        Args:
            n_epochs ([int]): [Number of epochs for training the model.]
        """
        
        # track change in validation loss
        valid_loss_min = np.Inf 
        
        for epoch in range(1, n_epochs+1):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            # training the model
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
             
            # validating the model
            self.model.eval()
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
            
            # calculate average losses
            train_loss = train_loss/len(self.train_loader.sampler)
            valid_loss = valid_loss/len(self.valid_loader.sampler)
                
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(self.model.state_dict(), 'model_augmented.pt')
                valid_loss_min = valid_loss
            
if __name__ == '__main__':
    
    num_epochs = 2
    num_classes = len(classes)
    
    model = MnistModel(model=resnet18(num_classes),
                       criterion=nn.CrossEntropyLoss(),
                       dataset=[train_loader,valid_loader,test_loader],
                       device=gpu_check())
    
    model.load_model()
    
    model.train_model(num_epochs)
    
    
