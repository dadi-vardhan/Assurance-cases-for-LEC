from cv2 import sepFilter2D
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils import zoom_image, get_device
from losses import relu_evidence
from model import  MnistModel
from torchvision.transforms import transforms, ToTensor
from torch.autograd import Variable

class Evidential_zoom_monitor():
    
    def __init__(self,model,num_classes, uncertainty = True):
        self.zoomed_imgs = zoom_image(monitor=True)
        self.device = get_device()
        self.num_classes = num_classes
        self.uncertainty = uncertainty
        self.model = model
        self.threshold = 0.2
        self.ldeg = []
        self.lp = []
        self.lu = []
        self.classifications = []
        self.scores = np.zeros((1, self.num_classes))
        self.filename = "evidentail_zoom.png"
    
    def monitor(self,img):
        imgs, ss_list = self.zoomed_imgs(img)
        self.ldeg = ss_list
        for img in imgs:
            trans = transforms.ToTensor()
            img_tensor = trans(img)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor)
            #img_variable = img_variable.to(self.device)
    
            if self.uncertainty:
                output = self.model(img_variable)
                evidence = relu_evidence(output)
                alpha = evidence + 1
                uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
                _, preds = torch.max(output, 1)
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                output = output.flatten()
                prob = prob.flatten()
                preds = preds.flatten()
                self.classifications.append(preds[0].item())
                self.lu.append(uncertainty.mean())

            else:
                output = self.model(img_variable)
                _, preds = torch.max(output, 1)
                prob = F.softmax(output, dim=1)
                output = output.flatten()
                prob = prob.flatten()
                preds = preds.flatten()
                self.classifications.append(preds[0].item())

            self.scores += prob.detach().cpu().numpy() >= self.threshold
            self.lp.append(prob.tolist())
            
        labels = np.arange(10)[self.scores[0].astype(bool)]
        lp = np.array(self.lp)[:, labels]
        c = ["blue", "red", "brown", "purple", "cyan"]
        marker = ["s", "^", "o"]*2
        labels = labels.tolist()
        fig = plt.figure(figsize=[6, 5])
        fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [5, 1, 12]})

        for i in range(len(labels)):
            axs[2].plot(self.ldeg, lp[:, i], marker=marker[i], c=c[i])

        if uncertainty:
            labels += ["uncertainty"]
            axs[2].plot(self.ldeg, self.lu, marker="<", c="black")

    

        #axs[0].set_title("Zoomed \"1\" Digit Classifications")
        axs[0].imshow(1 - self.rimgs, cmap="gray")
        axs[0].axis("off")
        #plt.pause(0.001)

        empty_lst = []
        empty_lst.append(self.classifications)
        axs[1].table(cellText=empty_lst, bbox=[0, 1, 1, 1])
        axs[1].axis("off")

        axs[2].legend(labels)
        axs[2].set_xlim([8, 28])
        axs[2].set_ylim([0, 1])
        axs[2].set_xlabel("Zoom pixels")
        axs[2].set_ylabel("Classification Probability")
        fig.savefig(self.filename)

class Max_monitor():
    
    def __init__(self,model):
        self.zoomed_imgs = zoom_image(monitor=True)
        self.device = get_device()
        self.model = model
        self.model_preds = []
        self.monitor_pred = None
    
    def monitor(self, img):
        imgs, _ = self.zoomed_imgs(img)
        for img in imgs:
            trans = transforms.ToTensor()
            img_tensor = trans(img)
            img_tensor.unsqueeze_(0)
            output = self.model(img_tensor)
            self.model_preds.append(output)
        self.monitor_pred = max(self.model_preds,key=self.model_preds.count)
        return self.monitor_pred
    
    def __str__(self):
        return "Max pred: "+str(self.monitor_pred)
    
class Avg_prob_monitor():
    
    def __init__(self,model):
        self.zoomed_imgs = zoom_image(monitor=True)
        self.device = get_device()
        self.model = model
        self.model_prob_preds = []
        self.mean_prob_pred = None
        self.monitor_pred = None
        
    def monitor(self,img):
        imgs, _ = self.zoomed_imgs(img)
        
        for img in imgs:
            trans = transforms.ToTensor()
            img_tensor = trans(img)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor)
            model_probs = self.model(img_variable)
            self.model_prob_preds.append(model_probs)
        self.mean_prob_pred = np.mean(np.array(self.model_prob_preds), axis=0)
        self.monitor_pred = np.argmax(self.mean_prob_pred)
        return self.monitor_pred
    
    def __str__(self):
        return "Avg pred: "+str(self.monitor_pred)

      



if __name__ == '__main__':
    model = MnistModel.load_from_checkpoint(
    checkpoint_path="/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/MNIST-lightning/sample-mnist-epoch=05-val_loss=0.30.ckpt",
    #hparams_file="",
    map_location=None).eval()
    train_data = datasets.MNIST(root = 'data',train = True,transform = ToTensor(),download = True)
    test_data = datasets.MNIST(
                                root = 'data', 
                                train = False, 
                                transform = ToTensor()
                                )
    img = test_data.data[0]
    target = test_data.targets[0]
    #zoom_analysis = Evidential_zoom_monitor(model=model, num_classes=10)
    #zoom_analysis.monitor(img)
    max_monitor = Max_monitor(model)
    pred = max_monitor.monitor(img)
    print(f"pred : {pred} and label : {target}")
    
        