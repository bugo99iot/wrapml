from wrapml.imports.vanilla import logger

# ML

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer, \
    roc_auc_score, roc_curve, auc, matthews_corrcoef, cohen_kappa_score, zero_one_loss, plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

# some models don't support parallel jobs
MODEL_CLASSES_NOT_SUPPORTING_PARALLEL_JOBS = [SVC, AdaBoostClassifier, KNeighborsClassifier]
MODEL_CLASSES_NOT_SUPPORTING_RANDOM_STATE = [KNeighborsClassifier]

# NN

import tensorflow as tf

logger.debug("TensorFlow ready. Version:", tf.__version__)

from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow import one_hot

from tensorflow.keras.metrics import Precision as tf_precision_score, Recall as tf_recall_score, Accuracy as tf_accuracy_score, AUC as tf_auc_score

from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')


# define extra metrics for TensorFlow
def tf_recall_score_funk(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def tf_precision_score_funk(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def tf_f1_score_funk(y_true, y_pred):
    precision = tf_precision_score_funk(y_true, y_pred)
    recall = tf_recall_score_funk(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def tf_matthews_score_funk(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
