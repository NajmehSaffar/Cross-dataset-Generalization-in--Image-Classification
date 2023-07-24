from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display as render

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss(losses, labels, colors, dirname, filename, display = False):
    plt.style.use('seaborn-whitegrid')
    
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes()
    
    for i in range(len(losses)):
        ax.plot(losses[i], color=colors[i], label=labels[i]);
    plt.legend()
    
    fig.savefig("{}LOSS{}.png".format(dirname, filename),dpi=300, facecolor='white')
    
    if display:
        plt.show()

    
def plot_ROC(lbllist, outlist, dirname, filename, display = False):
    fpr, tpr, _ = metrics.roc_curve(lbllist, outlist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.rcdefaults()
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.grid(False)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=BIGGER_SIZE)
    plt.ylabel('True Positive Rate', fontsize=BIGGER_SIZE)
    plt.title('ROC', fontsize=BIGGER_SIZE)
    plt.xticks(fontsize=BIGGER_SIZE)
    plt.yticks(fontsize=BIGGER_SIZE)
    plt.legend(loc="lower right", fontsize=BIGGER_SIZE)
    fig = plt.gcf() #Grab the figure instance before you call show
    fig.savefig("{}ROCAUC{}.png".format(dirname, filename),dpi=300, facecolor='white')
    if display:
        plt.show()
    return roc_auc
    
    
def plot_CONFMAT(lbllist, predlist, dirname, filename, display = False):
    # Confusion matrix
    conf_mat=confusion_matrix(lbllist, predlist)
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print(classification_report(lbllist, predlist))
    print("\nClass Accuracies: ", class_accuracy, "\n") # Per-class accuracy
    print("Total Accuracy: ", 100 * (accuracy_score(lbllist, predlist)), "\n")
    
    # Confusion matrix
    tlabels = pd.Series(lbllist)
    plabels = pd.Series(predlist)
    df_confusion = pd.crosstab(tlabels,plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)
    # Save the crosstab as a CSV file
    df_confusion.to_csv("{}CONF{}.csv".format(dirname, filename))
    if display:
        render(df_confusion)
    print("\n")

    
def plot_histogram(lbllist, scorelist, dirname, filename, display = False):
    x0 = scorelist[np.where(lbllist==0)]
    x1 = scorelist[np.where(lbllist==1)]
    plt.rcdefaults() 
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    #plt.grid(False)
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    kwargs = dict(alpha=0.5, bins=50, edgecolor='#8e8e8e', linewidth=0.5)
    plt.hist(x1, **kwargs, color='#44aa99', label='Class = 1')
    plt.hist(x0, **kwargs, color='#991100', label='Class = 0')
    plt.gca().set(title='Histogram of scores provided by classifiers', xlabel='Scores', ylabel='Frequency')
    plt.legend(loc="upper right")
    fig = plt.gcf() #Grab the figure instance before you call show
    fig.savefig("{}HIST{}.png".format(dirname, filename),dpi=300)
    if display:
        plt.show()
    
    
def getTime(time):
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return day, hour, minutes, seconds
