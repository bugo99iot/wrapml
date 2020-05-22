from wrapml.imports.science import plt


def make_training_history_plot(history, metric: str):
    plt.plot(history.history[metric])
    plt.plot(history.history[metric])
    plt.title('Model {} score'.format(metric.capitalize()))
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    loc = 'center right' if 'loss' not in metric else 'upper right'
    plt.legend(['train', 'valid'], loc=loc)
    plt.show()


def make_confusion_matrix_plot():

    return
