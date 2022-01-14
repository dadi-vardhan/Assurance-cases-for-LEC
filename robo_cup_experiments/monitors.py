from cv2 import sepFilter2D
from torchvision import datasets
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils import zoom_image, get_device
from losses import relu_evidence
from model import  RobocupModel
from torchvision.transforms import transforms, ToTensor
from torch.autograd import Variable
from eval_metrics import eval_metrics
from datamodule import RoboCupDataModule, RoboCupDataset 
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.neptune import NeptuneLogger
import neptune.new as neptune
from tqdm import tqdm

class Evidential_zoom_monitor():
    
    def __init__(self,model,num_classes, uncertainty = True):
        self.device = get_device()
        self.num_classes = num_classes
        self.zoom_images = 5
        self.uncertainty = uncertainty
        self.model = model
        self.threshold = 0.2
        self.ldeg = []
        self.lp = []
        self.lu = []
        self.classifications = []
        self.scores = np.zeros((1, self.num_classes))
        # self.rand_num = np.random.randint(0, 10000)
        # self.filename = "evd_ops/evidentail_zoom "+str(self.rand_num)+".png"
        self.monitor_pred = None
        self.rimgs = np.zeros((28, 28 * self.zoom_images))
        self.image_sizes = (np.random.dirichlet([25,10,10,10,10])*28).astype(int)
        self.ss = 0
        self.x = int(np.random.rand() * 28)
        self.y = int(np.random.rand() * 28)
        
    def monitor(self,img):
        for i, image_size in enumerate(self.image_sizes):
            self.ss +=image_size
        
            nimg = reduce_img(img, self.ss, [x,y])
            nimg = np.clip(a=nimg, a_min=0, a_max=1)

            self.rimgs[:, i*28:(i+1)*28] = nimg
            trans = transforms.ToTensor()
            img_tensor = trans(nimg)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor)
            img_variable = img_variable.to(self.device)
    
            if self.uncertainty:
                output = self.model.forward(img_tensor).cpu().detach()
                evidence = relu_evidence(output)
                alpha = evidence + 1
                uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
                _, preds = torch.max(output, 1)
                #self.monitor_pred = preds
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                output = output.flatten()
                prob = prob.flatten()
                preds = preds.flatten()
                self.classifications.append(preds[0].item())
                self.lu.append(uncertainty.mean())

            else:
                output = self.model.forward(img_variable).cpu().detach()
                _, preds = torch.max(output, 1)
                #self.monitor_pred = preds
                prob = F.softmax(output, dim=1)
                output = output.flatten()
                prob = prob.flatten()
                preds = preds.flatten()
                self.classifications.append(preds[0].item())

            prob = prob.detach().cpu().numpy()#[0:self.num_classes]
            self.scores += prob >= self.threshold
            self.lp.append(prob.tolist())
        return self.monitor_pred
            
    def plot_evidential_zoom(self):
        labels = np.arange(10)[self.scores[0].astype(bool)]
        self.lp = np.array(self.lp)[:, labels]
        c = ["blue", "red", "brown", "purple", "cyan"]
        marker = ["s", "^", "o"]*2
        labels = labels.tolist()
        fig = plt.figure(figsize=[6, 5])
        fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [5, 1, 12]})

        for i in range(len(labels)):
            axs[2].plot(self.ldeg, self.lp[:, i], marker=marker[i], c=c[i])

        if self.uncertainty:
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
        self.model = model
        self.model_preds = []
        self.monitor_pred = None
    
    def monitor(self, img):
        imgs = self.zoomed_imgs(img)
        for img in imgs:
            trans = transforms.Compose([ transforms.ToTensor(),
                                        #transforms.Resize((64,64))
                                       #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])
            img_tensor = trans(img)
            img_tensor.unsqueeze_(0)
            output = self.model.forward(img_tensor).argmax(axis=1)#.cpu().detach().numpy()
            self.model_preds.append(output)
        self.monitor_pred = max(self.model_preds,key=self.model_preds.count)
        self.model_preds = []
        return self.monitor_pred
    
    def __str__(self):
        return "Max pred: "+str(self.monitor_pred)
    
class Avg_prob_monitor():
    
    def __init__(self,model):
        self.zoomed_imgs = zoom_image(monitor=True)
        self.model = model
        self.model_prob_preds = []
        self.mean_prob_pred = None
        self.monitor_pred = None
        
    def monitor(self,img):
        imgs = self.zoomed_imgs(img)
        for img in imgs:
            trans = transforms.ToTensor()
            img_tensor = trans(img)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor)
            model_probs = self.model(img_variable).cpu().detach().numpy()
            self.model_prob_preds.append(model_probs)
        self.mean_prob_pred = np.mean(np.array(self.model_prob_preds), axis=0)
        self.monitor_pred = np.argmax(self.mean_prob_pred)
        self.model_prob_preds = []
        self.mean_prob_pred = None
        return self.monitor_pred
    
    def __str__(self):
        return "Avg pred: "+str(self.monitor_pred)
    
    
if __name__ == '__main__':
    
    seed_everything(123, workers=True)
    
    # neptune logger for logging metrics
    neptune_logger = neptune.init(
        api_token=os.environ['NEPTUNE_API_TOKEN'],
        project="dadivishnuvardhan/AC-LECS",
        run="AC-230")
    
    model = RobocupModel.load_from_checkpoint(
    checkpoint_path="/home/dadi_vardhan/RandD/Assurance-cases-for-LEC/robo_cup_experiments/Untitled/AC-230/checkpoints/epoch=99-step=1699.ckpt",
                        map_location=None).eval()

    classes = ("AXIS","BEARING","BEARING_BOX","CONTAINER_BOX_BLUE","CONTAINER_BOX_RED","DISTANCE_TUBE",
                        "F20_20_B","F20_20_G","M20","M20_100","M30","MOTOR","R20","S40_40_B","S40_40_G")
    #model.freeze()
    # dm = RoboCupDataModule()
    # dm.setup("test")
    # test_data = dm.test_dataloader()
    test_data = RoboCupDataset(test=True)

    # MAX monitor 
    max_mon = Max_monitor(model)
    y_true_max, y_pred_max = [],[]
    for i, (x, y) in tqdm(enumerate(test_data)):
        y_hat = max_mon.monitor(x)
        y_true_max.append(y)
        y_pred_max.append(y_hat)

        if i == 50:
            break
    y_true_max = np.hstack(y_true_max)
    y_pred_max = np.hstack(y_pred_max)


    em_max_mon = eval_metrics(y_true_max, y_pred_max,classes=classes)
    neptune_logger["Max_monitor/accuracy"] = em_max_mon.accuracy()
    neptune_logger["Max_monitor/precision"] = em_max_mon.precision_weighted()
    neptune_logger["Max_monitor/recall"] = em_max_mon.recall_weighted()
    neptune_logger["Max_monitor/f1"] = em_max_mon.f1_score_weighted()
    neptune_logger["Max_monitor/classification_report"] = em_max_mon.classify_report()
    neptune_logger["Max_monitor/Confusion_matrix"].log(neptune.types.File.as_image(em_max_mon.plot_conf_matx()))
    neptune_logger["Max_monitor/Confusion_matrix_normalized"].log(neptune.types.File.as_image(em_max_mon.plot_conf_matx(normalized=True)))
    
    
    # # AVG monitor
    # avg_mon = Avg_prob_monitor(model)
    # y_true_avg, y_pred_avg = [],[]
    # for i, (x, y) in enumerate(test_data):
    #     y_hat = avg_mon.monitor(x)
    #     y_true_avg.append(y)
    #     y_pred_avg.append(y_hat)

    #     if i == len(test_data):
    #         break
    # y_true_avg = np.hstack(y_true_avg)
    # y_pred_avg = np.hstack(y_pred_avg)

    # em_avg_mon = eval_metrics(y_true_avg, y_pred_avg,classes=classes)
    # neptune_logger["Avg_prob_monitor/accuracy"] = em_avg_mon.accuracy()
    # neptune_logger["Avg_prob_monitor/precision"] = em_avg_mon.precision_weighted()
    # neptune_logger["Avg_prob_monitor/recall"] = em_avg_mon.recall_weighted()
    # neptune_logger["Avg_prob_monitor/f1"] = em_avg_mon.f1_score_weighted()
    # neptune_logger["Avg_prob_monitor/classification_report"] = em_avg_mon.classify_report()
    # neptune_logger["Avg_prob_monitor/Confusion_matrix"].log(neptune.types.File.as_image(em_avg_mon.plot_conf_matx()))
    # neptune_logger["Avg_prob_monitor/Confusion_matrix_normalized"].log(neptune.types.File.as_image(em_avg_mon.plot_conf_matx(normalized=True)))
    
    
    
    # # Evidential Zoom Monitor
    # evd_mon = Evidential_zoom_monitor(model)
    # y_true_evd, y_pred_evd = [],[]
    # for i, (x, y) in enumerate(test_data):
    #     y_hat = evd_mon.monitor(x)
    #     y_true_evd.append(y)
    #     y_pred_evd.append(y_hat)

    #     if i == len(test_data):
    #         break
    # y_true_evd = np.hstack(y_true_evd)
    # y_pred_evd = np.hstack(y_pred_evd)
    
    # em_evd_mon = eval_metrics(y_true_evd, y_pred_evd,classes=classes)
    # neptune_logger["Evidential_monitor/accuracy"] = em_evd_mon.accuracy()
    # neptune_logger["Evidential_monitor/precision"] = em_evd_mon.precision_weighted()
    # neptune_logger["Evidential_monitor/recall"] = em_evd_mon.recall_weighted()
    # neptune_logger["Evidential_monitor/f1"] = em_evd_mon.f1_score_weighted()
    # neptune_logger["Evidential_monitor/classification_report"] = em_evd_mon.classify_report()
    # neptune_logger["Evidential_monitor/Confusion_matrix"].log(neptune.types.File.as_image(em_evd_mon.plot_conf_matx()))
    # neptune_logger["Evidential_monitor/Confusion_matrix_normalized"].log(neptune.types.File.as_image(em_evd_mon.plot_conf_matx(normalized=True)))
    
    
    # stop logging
    neptune_logger.stop()