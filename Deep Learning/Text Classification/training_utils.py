import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set()

def as_minutes(s):
    return f'{int(s//60)}m {int(s%60)}s'

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f'{as_minutes(s)} (- {as_minutes(rs)})'

def plot_losses(train_loss, val_loss, scale):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss)
    plt.plot([(x + 1) * scale - 1 for x in range(len(val_loss))], val_loss)
    plt.legend(['train loss', 'validation loss'])
    
def print_confusion_matrix(confusion_matrix, class_names=[], figsize = (15,15), fontsize=14, center=1000):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", center=center, cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(targets, predicted, dataset, center=50, unique_labels=None):
    conf_mat = confusion_matrix(targets, predicted)
    den = np.array([max(1, x) for x in conf_mat.sum(axis=1)])
    conf_mat = np.array([conf_mat[i] / den[i] for i, _ in enumerate(conf_mat)])
    conf_mat = (conf_mat * 100).astype(np.int)
    if unique_labels is None:
        unique_labels = set(example.section for example in dataset.examples)
    print_confusion_matrix(conf_mat, sorted(list(unique_labels)), center=center)