import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import itertools


def plot_resultados_regresion(y_train, y_train_predicted,
                              y_test,  y_test_predicted,
                              figsize=(15,5), alpha=1):
    font_size = 14
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(1,2,1)
    plt.plot(y_train, y_train_predicted, 'b.', zorder=1, alpha=alpha)
    lims = [min([y_train.min(), y_test.min()]), max([y_train.max(), y_test.max()])]
    plt.plot(lims, lims, 'k-', zorder=1)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    plt.xlabel('y real', fontsize=font_size)
    plt.ylabel('y predicho', fontsize=font_size)
    plt.title('conjunto de training', fontsize=font_size)
    plt.grid(True)
    plt.axis('equal')
    
    ax2 = plt.subplot(1,2,2)
    plt.plot(y_test, y_test_predicted, 'g.', zorder=1, alpha=alpha)
    plt.plot(lims, lims, 'k-', zorder=1)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    plt.xlabel('y real', fontsize=font_size)
    plt.ylabel('y predicho', fontsize=font_size)
    plt.title('conjunto de test', fontsize=font_size)
    plt.grid(True)
    plt.axis('equal');

def analisis_roc(y, positive_class_scores, POSITIVE_CLASS=1):

    y_is_positive_class = y==POSITIVE_CLASS
    
    fpr, tpr, thresholds = roc_curve(y_is_positive_class,
                                     positive_class_scores,
                                     pos_label=POSITIVE_CLASS)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.plot(fpr, thresholds, color='r', lw=2, linestyle=':', label='Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC. Positive class: '+str(POSITIVE_CLASS), fontsize=16)
    plt.legend(loc="lower right");


def plot_confusion_matrix(y, y_pred,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    aux = np.unique(y)
    tick_marks = aux
    plt.xticks(tick_marks, aux)
    plt.yticks(tick_marks, aux)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.tight_layout();

