from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def merge_classes(data):
    """
    Merges the last two classes into a single class.
    
    Args:
        data: torch_geometric.data.Data object.

    Returns:
        data: transformed torch_geometric.data.Data object.
    """
    data.y[data.y==3] = 2
    return data


def score(y_true, y_pred) -> Tuple[float]:
    """
    Calculates the accuracy, macro F1 score, weighted F1 score and the ROC AUC score.
    
    Args:
        y_true (array like) of shape (n_samples) containing true labels.
        y_pred (array like) of shape (n_samples,n_classes) containing the log softmax probabilities.
    """
    accuracy = accuracy_score(y_true, y_pred.argmax(dim=1, keepdim=True))
    f1_macro = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average='macro')
    f1_weighted = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average='weighted')
    auc = roc_auc_score(y_true, np.exp(y_pred), average='macro', multi_class='ovr')
    return accuracy, f1_macro, f1_weighted, auc


def make_plot(train: Iterable, test: Iterable, plot_type: str, model_name: str) -> None:
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel(plot_type)
    plt.savefig('results/'+model_name+'_'+plot_type+'.pdf')
    plt.close()
