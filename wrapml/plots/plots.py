from wrapml.imports.vanilla import List
from wrapml.imports.science import np
from wrapml.imports.science import plt
from wrapml.imports.learn import roc_curve, auc, plot_roc_curve

cmap = plt.cm.Blues
np.set_printoptions(precision=2)


def make_training_history_plot(history,
                               metric: str,
                               model_name: str = None):

    if model_name:
        title = '{} score - {}'.format(metric.capitalize(), model_name)
    else:
        title = '{} score'.format(metric.capitalize())

    plt.plot(history.history[metric])
    plt.plot(history.history[metric])
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    loc = 'center right' if 'loss' not in metric else 'upper right'
    plt.legend(['train', 'valid'], loc=loc)
    plt.show()


def make_confusion_plot(confusion_matrix: np.ndarray,
                        labels: List[str],
                        normalize: bool = False,
                        model_name: str = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if model_name:
        title = 'Confusion Plot - {}'.format(model_name)
    else:
        title = 'Confusion Plot'

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    # fig.tight_layout()
    # return ax
    plt.show()


def make_roc_plot(y_test_pred_prob: np.ndarray,
                  y_test: np.ndarray,
                  pos_label: int or str):

    y_test_pred_prob_max = np.array([np.amax(k) for k in y_test_pred_prob])

    fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob_max, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
