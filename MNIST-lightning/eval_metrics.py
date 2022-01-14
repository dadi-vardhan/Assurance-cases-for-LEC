import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix



class eval_metrics():
    def __init__(self,targets,preds,classes):
        try:
            self.targets = targets.cpu().numpy()
            self.preds = preds.cpu().numpy()
            self.classes = classes
            self.num_classes = len(self.classes)
        except:
            self.targets = targets
            self.preds = preds
            self.classes = classes
            self.num_classes = len(self.classes)
    
    def plot_conf_matx(self,normalized=False):
        fig, axs = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(self.targets, self.preds, ax=axs,normalize=normalized)
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        plt.savefig(os.path.join(os.getcwd(),'confusion_matrix.png'))
        return fig
    
    def accuracy(self):
        return accuracy_score(self.targets,self.preds,normalize=True)
    
    def f1_score_weighted(self):
        return f1_score(self.targets,self.preds,average='weighted')
    
    def precision_weighted(self):
        return precision_score(self.targets,self.preds,average='weighted')
    
    def recall_weighted(self):
        return recall_score(self.targets,self.preds,average='weighted')
    
    def classify_report(self):
        return classification_report(self.targets,self.preds,target_names=self.classes)
        

