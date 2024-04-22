from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_confusion_matrix(y_pred, y_real, labels, path):
    cm = confusion_matrix(y_real, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/confusion_matrix.png')
    plt.clf()

def plot_accuracy(history, path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/accuracy.png')
    plt.clf()

def plot_loss(history, path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + '/loss.png')
    plt.clf()

def append_accuracy_score(accuracy, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/accuracy.txt', 'a') as file1:
        file1.write(str(accuracy) + '\n')
