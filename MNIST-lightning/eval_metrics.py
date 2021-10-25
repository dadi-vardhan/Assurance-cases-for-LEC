import torchmetrics as tm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools


def createConfusionMatrix(cm):
    # y_pred = [] # save predction
    # y_true = [] # save ground truth

    # # iterate over data
    # for inputs, labels in loader:
    #     output = net(inputs)  # Feed Network

    #     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    #     y_pred.extend(output)  # save prediction

    #     labels = labels.data.cpu().numpy()
    #     y_true.extend(labels)  # save ground truth

    # constant for classes
    classes = ('Zero', 'One', 'Two', 'Three', 'Four',
               'Five', 'Six', 'Seven', 'Eight', 'Nine')

    # Build confusion matrix
    #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cf_matrix = cm.cpu().numpy()
    print(cf_matrix)
    print(np.shape(cf_matrix))
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    # Create Heatmap
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


def plot_confusion_matrix(cm):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    class_names = ('Zero', 'One', 'Two', 'Three', 'Four',
               'Five', 'Six', 'Seven', 'Eight', 'Nine')
    figure = plt.figure(figsize=(8, 8))
    #plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

